#ifndef DRIVER_APPLICATION_HPP
#define DRIVER_APPLICATION_HPP

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <sys/time.h>
#include "compatibility.hpp"
#include "runtime_opts.hpp"
#include "stopwatch.hpp"
#include "source.hpp"
#include "multiplexer.hpp"

using namespace std;

class Application {
 public:
  Application(const RuntimeOpts& _opts) : opts(_opts) {
    if (dbtoaster::data_t::kBatchModeActive && opts.batch_size == 0) {
      std::cerr << "Use -b <arg> to specify a valid batch size." << std::endl;
      exit(1);
    }
  }

  void run();

 protected:  
  using Multiplexer = std::conditional<data_t::kOrderedDataset,
                                       OrderedMultiplexer,
                                       RoundRobinMultiplexer>::type;
  using MultiplexerPtr = unique_ptr<Multiplexer>;

  class Visitor {
   public:
    template <class T>
    void addMap(string name, const T& map, bool isTopLevel) {
      std::cout << "Registered: " << name << " " << isTopLevel << std::endl;
    }

    template <class Message, bool HasDeletions>
    void addSource(FileSource source, SourceFormat format) {
      if (format.name == "CSV") {
        char delimiter = ',';
        auto it = format.params.find("delimiter");
        if (it != format.params.end()) {
          delimiter = (it->second)[0];
        }
        if (source.isTable()) {          
          table_iterators.push_back(
            make_unique<CSVFileIterator<Message, 
                                        data_t::kOrderedDataset,
                                        HasDeletions,
                                        false>>(source, delimiter));
        }
        else {
          stream_iterators.push_back(
            make_unique<CSVFileIterator<Message,
                                        data_t::kOrderedDataset,
                                        HasDeletions,
                                        data_t::kBatchModeActive>>(source, delimiter));
        }
      }
      else {
        std::cerr << "Adaptor type not supported: " << format.name << std::endl;
        exit(1);
      }
    }

    MultiplexerPtr getTableMultiplexer(size_t batch_size) {
      return make_unique<Multiplexer>(std::move(table_iterators));
    }

    MultiplexerPtr getStreamMultiplexer(size_t batch_size) {
      for (auto& it : stream_iterators) {
        it->setBatchSize(batch_size);
      }
      return make_unique<Multiplexer>(std::move(stream_iterators));
    }

   private:
    vector<EventIteratorPtr> table_iterators;
    vector<EventIteratorPtr> stream_iterators;
  };

  void init() {
    Visitor visitor;
    dbtoaster::data_t::registerSources(visitor);
    static_multiplexer = visitor.getTableMultiplexer(opts.batch_size);
    dynamic_multiplexer = visitor.getStreamMultiplexer(opts.batch_size);
    static_multiplexer->open();
    dynamic_multiplexer->open();
    static_buffer.clear();
    dynamic_buffer.clear();
  }

  void close() {
    static_multiplexer->close();
    dynamic_multiplexer->close();
    static_buffer.clear();
    dynamic_buffer.clear();
  }

  void preloadInput(dbtoaster::data_t& data) {
    static_buffer.clear();
    Event e = static_multiplexer->next();
    while (!e.isEmpty()) {
      static_buffer.push_back(std::move(e));
      e = static_multiplexer->next();
    }

    dynamic_buffer.clear();
    e = dynamic_multiplexer->next();
    while (!e.isEmpty()) {
      dynamic_buffer.push_back(std::move(e));
      e = dynamic_multiplexer->next();
    }
  }

  void processTables(dbtoaster::data_t& data) {
    if (opts.preload_input) {
      for (auto& e : static_buffer) {
        data.process_table_event(e);
      }
    }
    else {
      Event e = static_multiplexer->next();
      while (!e.isEmpty()) {
        data.process_table_event(std::move(e));
        e = static_multiplexer->next();
      }
    }
  }

  void processOnSystemReady(dbtoaster::data_t& data) {
    data.on_system_ready_event();
  }

  void processStreams(dbtoaster::data_t& data) {
    if (opts.preload_input) {
      for (auto& e : dynamic_buffer) {
        data.process_stream_event(e);
      }
    }
    else {
      Event e = dynamic_multiplexer->next();
      while (!e.isEmpty()) {
        data.process_stream_event(std::move(e));
        e = dynamic_multiplexer->next();
      }
    }
  }

  void processStreamsLogCount(dbtoaster::data_t& data) {
    size_t next_log_count = 0;
    if (opts.preload_input) {
      for (auto& e : dynamic_buffer) {
        logCount(data, next_log_count);       
        data.process_stream_event(e);
      }
      next_log_count = data.tN;
      logCount(data, next_log_count);
    }
    else {
      Event e = dynamic_multiplexer->next();
      while (!e.isEmpty()) {
        logCount(data, next_log_count);
        data.process_stream_event(std::move(e));
        e = dynamic_multiplexer->next();
      }
      next_log_count = data.tN;
      logCount(data, next_log_count);
    }
  }

  void logCount(dbtoaster::data_t& data, size_t& next_log_count) {
    if (data.tN >= next_log_count) {
      struct timeval tp;
      gettimeofday(&tp, nullptr);
      std::cout << data.tN << " tuples processed at " 
                << tp.tv_sec << "s+"
                << tp.tv_usec << "us" << std::endl;
      next_log_count = data.tN + opts.log_count;
    }
  }

  void printResult(dbtoaster::data_t& data) {
    std::cout << "<snap>\n";
    data.serialize(std::cout, 0);
    std::cout << "\n</snap>" << std::endl;
  }

  RuntimeOpts opts;
  MultiplexerPtr static_multiplexer;
  MultiplexerPtr dynamic_multiplexer;
  vector<Event> static_buffer;
  vector<Event> dynamic_buffer;
};

void Application::run() {
  Stopwatch local_time, total_time;

  for (size_t run = 0; run < opts.num_runs; run++) {

    dbtoaster::data_t data;

    init();

    std::cout << "-------------" << std::endl;

    if (opts.preload_input) {
      std::cout << "Preloading input datasets... " << std::flush;
      local_time.restart();
      preloadInput(data);
      local_time.stop();
      std::cout << local_time.elapsedMilliSec() << " ms" << std::endl;
    }

    total_time.restart();

    std::cout << "Processing tables... " << std::flush;
    local_time.restart();
    processTables(data);
    local_time.stop();
    std::cout << local_time.elapsedMilliSec() << " ms" << std::endl;
    
    std::cout << "Executing on_system_ready trigger... " << std::flush;
    local_time.restart();
    processOnSystemReady(data);
    local_time.stop();
    std::cout << local_time.elapsedMilliSec() << " ms" << std::endl;

    if (opts.log_count == 0) {
      std::cout << "Processing streams... " << std::flush;
      local_time.restart();
      processStreams(data);
      local_time.stop();
      std::cout << local_time.elapsedMilliSec() << " ms" << std::endl;
    }
    else {
      std::cout << "Logging tuple count every "
                << opts.log_count << " updates:" << std::endl;
      local_time.restart();
      processStreamsLogCount(data);
      local_time.stop();
      std::cout << "Processing streams with logging... " 
                << local_time.elapsedMilliSec() << " ms" << std::endl;
    }
   
    total_time.stop();

    if (opts.print_result) {
      std::cout << "Final result: " << std::endl;
      printResult(data);
    }

    // Print summary
    std::cout << "    Run: " << run
              << "    Processed: " << data.tN
              << "    Skipped: " << data.tS
              << "    Execution time: " << total_time.elapsedMilliSec() << " ms"
              << (dbtoaster::data_t::kBatchModeActive ? 
                    "    Batch size: " + std::to_string(opts.batch_size) : "")
              << "\n-------------"
              << std::endl;

    // Print statistics for the DBToaster test suite
    std::cout << "SAMPLE = "
              << run << ", " 
              << total_time.elapsedMilliSec() * 1000 << ", " 
              << data.tN << ", " 
              << data.tS << std::endl;

    close();
  }
}

#endif /* DRIVER_APPLICATION_HPP */