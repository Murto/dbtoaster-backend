#ifndef DBTOASTER_NO_WAIT_MEMORY_POOL_HPP
#define DBTOASTER_NO_WAIT_MEMORY_POOL_HPP

#include "macro.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>
#include <omp.h>

namespace dbtoaster {

namespace memory_pool {

template <typename T, std::size_t N = 32>
class NoWaitMemoryPool {
static_assert(N > 0, "Chunk size must be greater than 0");
 public:
  struct Slot {
    union {
      T value;
      Slot* next;
    };
    bool used;
    Slot() noexcept {}
  };

  class Chunk {
   public:
    Chunk* next;
    Slot slots[N];
    Slot& operator[](std::size_t index) {
      return slots[index];
    }
  };

  struct MetaChunkHeader {
    MetaChunkHeader* next;
    std::size_t capacity;
  };

  struct alignas(64) ThreadEntry {
    MetaChunkHeader* meta_chunks{nullptr};
    std::size_t current_chunk{0};
    std::size_t last_chunk;
    
    Chunk* chunks{nullptr};
    Chunk* free_chunks{nullptr};
    
    Slot* current_slot{nullptr};
    Slot* last_slot{nullptr};
  };

  NoWaitMemoryPool(std::size_t thread_count = omp_get_max_threads()) : thread_count_{thread_count} {
    thread_entries_ = new ThreadEntry[thread_count]();
  }

  ~NoWaitMemoryPool() {
    releaseAll();
    for (std::size_t i = 0; i < thread_count_; ++i) {
      ThreadEntry& entry = thread_entries_[i];
      MetaChunkHeader* meta_chunk = entry.meta_chunks;
      while (meta_chunk != nullptr) {
        MetaChunkHeader* next = meta_chunk->next;
        std::free(meta_chunk);
        meta_chunk = next;
      }
    }
    delete[] thread_entries_;
  }

  template <class... Args>
  T* acquire(Args&&... args) {
    std::size_t owner = omp_get_thread_num();
    ThreadEntry& entry = thread_entries_[owner];
    if (entry.current_slot == entry.last_slot) {
      allocateChunk(owner);
    }
    Slot& slot = *(entry.current_slot++);
    slot.used = true;
    new (&slot.value) T(args...);
    return &slot.value;
  }
  
  void release(T* elem) {
    reinterpret_cast<Slot*>(elem)->used = false;
    
  }

  template <class NextFn>
  void releaseChain(T* head, NextFn next_fn) {
    while (head != nullptr) {
      T* next = next_fn(head);
      release(head);
      head = next;
    }
  }

  void releaseAll() {
    #pragma omp simd
    for (std::size_t i = 0; i < thread_count_; ++i) {
      ThreadEntry& entry = thread_entries_[i];
      Chunk* chunk = entry.chunks;
      if (chunk == nullptr) continue;
      for (Slot* slot = &(*chunk)[0]; slot != entry.current_slot; ++slot) {
        if (slot->used) {
          slot->value.~T();
        }
      }
      Chunk* next = chunk->next;
      chunk->next = entry.free_chunks;
      entry.free_chunks = chunk;
      chunk = next;
      while (chunk != nullptr) {
        for (std::size_t j = 0; j < N; ++j) {
          if ((*chunk)[j].used) {
            (*chunk)[j].value.~T();
          }
        }
        next = chunk->next;
        chunk->next = entry.free_chunks;
        entry.free_chunks = chunk;
        chunk = next;
      }
      entry.chunks = nullptr;
      entry.current_slot = nullptr;
      entry.last_slot = nullptr;
    }
  }

  template <class F>
  void foreach(F f) const {
    #pragma omp parallel default(shared)
    {
      #pragma omp master
      {
        for (std::size_t i = 0; i < thread_count_; ++i) {
          ThreadEntry& entry = thread_entries_[i];
          Chunk* chunk = entry.chunks;
          if (chunk == nullptr) continue;
          
          #pragma omp task firstprivate(chunk)
          {
            for (Slot* slot = &(*chunk)[0]; slot != entry.current_slot; ++slot) {
              if (slot->used) {
                slot->value.~T();
              }
            }
          }

          chunk = chunk->next;
          
          while (chunk != nullptr) {
            
            #pragma omp task firstprivate(chunk)
            {
              for (std::size_t i = 0; i < N; ++i) {
                if ((*chunk)[i].used) {
                  f((*chunk)[i].value);
                }
              }
            }

            chunk = chunk->next;
          
          }
        }
      }
    }
  }

 private:

  constexpr static const std::size_t META_CHUNK_HEADER_SIZE = ((sizeof(MetaChunkHeader) + alignof(MetaChunkHeader) - 1) / alignof(MetaChunkHeader)) * alignof(MetaChunkHeader);
  constexpr static const std::size_t CHUNK_SIZE = ((sizeof(Chunk) + alignof(Chunk) - 1) / alignof(Chunk)) * alignof(Chunk);

  // Number of threads we need to serve
  const std::size_t thread_count_;

  // Entires for each thread
  ThreadEntry* thread_entries_;

  void allocateChunk(std::size_t owner);
};

template <typename T, std::size_t N>
void NoWaitMemoryPool<T, N>::allocateChunk(std::size_t owner) {
  ThreadEntry& entry = thread_entries_[owner];
  
  if (entry.free_chunks != nullptr) {
    Chunk* chunk = entry.free_chunks;
    entry.free_chunks = chunk->next;
    chunk->next = entry.chunks;
    entry.chunks = chunk;
    entry.current_slot = &(*chunk)[0];
    entry.last_slot = &(*chunk)[N];
    return;
  }

  if (entry.meta_chunks == nullptr || ++entry.current_chunk == entry.meta_chunks->capacity) {
    std::size_t capacity = entry.meta_chunks == nullptr ? 1 : entry.meta_chunks->capacity << 1;
    MetaChunkHeader* new_meta_chunk = static_cast<MetaChunkHeader*>(std::malloc(META_CHUNK_HEADER_SIZE + capacity * CHUNK_SIZE));
    if (new_meta_chunk == nullptr) {
      std::cerr << "Memory allocation error in memory pool" << std::endl;
      exit(-1);
    }
    new_meta_chunk->next = entry.meta_chunks;
    entry.meta_chunks = new_meta_chunk;
    new_meta_chunk->capacity = capacity;
    entry.current_chunk = 0;
  }

  Chunk* chunk = reinterpret_cast<Chunk*>(reinterpret_cast<unsigned char*>(entry.meta_chunks) + META_CHUNK_HEADER_SIZE) + entry.current_chunk;
  chunk->next = entry.chunks;
  entry.chunks = chunk;
  entry.current_slot = &(*chunk)[0];
  entry.last_slot = &(*chunk)[N];

}

} // namespace memory_pool

} // namespace dbtoaster

#endif // DBTOASTER_NO_WAIT_MEMORY_POOL_HPP
