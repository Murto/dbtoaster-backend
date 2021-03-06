#!/usr/bin/env ruby

require "./#{File.dirname($0)}/util.rb"
require "./#{File.dirname($0)}/db_parser.rb"
require 'getoptlong'
require 'tempfile'

#Takes a file and loads the properties in that file
def readProperties
  @propertiesFile = "#{File.expand_path(File.dirname($0))}/../../conf/ddbt.properties"
  @properties = {}
  IO.foreach(@propertiesFile) do |line|
    @properties[$1.strip] = $2 if line =~ /([^=]*)=(.*)\/\/(.*)/ || line =~ /([^=]*)=(.*)/
  end
end

readProperties()
$dbt_base_rep_path = @properties["ddbt.base_repo"]
$dbt_base_rep_path.strip!
$dbt_path = "#{$dbt_base_rep_path}/dbtoaster/compiler/alpha5"
$dbt_backend_path = "#{File.expand_path(File.dirname($0))}/../.."
$dbt = "./bin/dbtoaster"
$lib_boost_path = @properties["ddbt.lib_boost"]
$ocamlrunparam = "b,l=20M"
$optlevel = "-O3"
$parallel_input = "-p 2"
Dir.chdir $dbt_path

raise "DBToaster is not compiled" unless (File.exists? $dbt)

def results_file(path, delim = /,/, reverse = false)
  File.open(path).readlines.
    delete_if { |l| l.chomp == "" }.  
    map do |l|
      k = l.split(delim).map { |i| i.mirror_chomp.extract_dbt_value }
      [k, k.pop];
    end.map { |k,v| if reverse then [k.reverse,v] else [k,v] end }.to_h
end

def upcast_int_to_float(i)
  if i.is_a? Integer then i.to_f else i end
end

class GenericUnitTest
  attr_reader :runtime, :opts;

  def temp_name(file_name='', ext='', dir=nil)
    id   = Thread.current.hash * Time.now.to_i % 2**32
    name = "%s_%d%s" % [file_name, id, ext]
    dir ? File.join(dir, name) : name
  end

  def query=(q)
    @qname = q
    qdat = File.open("test/unit/queries/#{q}") do |f| 
      eval(f.readlines.join(""), binding) 
    end
    @dataset = if $dataset.nil? then "standard" else $dataset end
    @qpath = qdat[:path];
    
    raise "Invalid dataset" unless qdat[:datasets].has_key? @dataset;
    @toplevels = qdat[:datasets][@dataset][:toplevels].to_a.map do |name,info|
      info[:expected] =
        case info[:type]
          when :singleton then upcast_int_to_float(info[:expected])
          when :onelevel  then info[:expected].to_a.map do |k,v|
            [ k.map { |k_elem| upcast_int_to_float(k_elem) }, 
              upcast_int_to_float(v) ]
          end.to_h
        end
      [name, info]
    end.to_h
    @toplevels

    qfilename = "query_test.sql";
    counter = 1;
    if File.file?(qfilename) then
      qfilename = temp_name('query_test', '.sql');
      counter += 1;
    end

    qfile = File.new(qfilename, "w+");
    at_exit { if File.exist?(qfilename) then qfile.close; File.delete(qfilename) end }
    if qdat[:datasets][@dataset].has_key? :subs then    
      subs = qdat[:datasets][@dataset][:subs]
      qfile.puts(
        File.open(@qpath) do |f| 
          f.readlines.map do |l|
            subs.fold(l) { |curr_l,s| curr_l.gsub(*s).sub("../alpha5",$dbt_path) };
          end.join(""); 
        end
      );
    else
      qfile.puts(
        File.open(@qpath) do |f| 
          f.readlines.map do |l|
            l.sub("../alpha5",$dbt_path)
          end.join(""); 
        end
      );
    end
    qfile.flush;
    @qpath = File.expand_path(qfilename);

    @compiler_flags =
      (qdat.has_key? :compiler_flags) ? qdat[:compiler_flags] : [];
  end
    
  def query
    File.open(@qpath) { |f| f.readlines.join("") };
  end
  
  def diff(e, r)
    if (e == r)                            then "Same"
    elsif (e.is_a? String) || (r.is_a? String) then "Different"
    elsif e == nil && r.abs < $precision   then "Close"
    elsif r == nil && e.abs < $precision   then "Close"
    elsif e == nil                         then "Different"
    elsif r == nil                         then "Different"
    elsif ((e+r).abs > 1) && (((e-r)/(e+r)).abs < $precision)
                                           then "Close"
    elsif ((e+r).abs <= 1) && ((e-r) < $precision )
                                           then "Close"
    else                                        "Different"
    end
  end
  
  def correct?(query = nil)
    if query.nil? then 
      not @toplevels.keys.find { |query| not correct? query }
    else
      raise "Invalid toplevel query #{query}" unless @toplevels.has_key? query
      
      query = @toplevels[query]
      case query[:type]
        when :singleton then 
          (diff(query[:expected], query[:result]) != "Different")
        when :onelevel then
          expected = query[:expected].map { |k,v| [k.join("/"), v] }.to_h;
          result = query[:result].map { |k,v| [k.join("/"), v] }.to_h;
          raise "Got nil result" if result.nil?;
          raise "Metadata has nil expected results" if expected.nil?;
          not ((expected.keys + result.keys).uniq.find do |k|
            diff(expected[k], result[k]) == "Different"
          end)
        else raise "Unknown query result type '#{query[:type]}'"
      end
    end
  end
  
  def dbt_base_cmd
    [ $dbt, @qpath ] +
      (if $depth.nil? then [] else ["--depth", $depth ] end) + 
      @compiler_flags + $compiler_args +
      ($debug_flags.map { |f| ["-d", f]}.flatten(1)) +
      ($opts)
  end
  
  def results(query = @toplevels.keys[0])
    raise "Invalid toplevel query #{query}" unless @toplevels.has_key? query
    query = @toplevels[query]
    expected = query[:expected];
    result = query[:result];
    case query[:type]
      when :singleton then [["*", expected, result, diff(expected, result)]]
      when :onelevel then
        p expected;
        p result;
        (expected.keys + result.keys).uniq.sort.
          map do |k| 
            [ k.join("/"), 
              expected[k], 
              result[k], 
              diff(expected[k], result[k])
            ]
          end
      else raise "Unknown query type '#{query[:type]}'"
    end
  end
end

class CppUnitTest < GenericUnitTest
  def run
    queryName = @qname;
    unless $skip_compile then
      counter = 1;
      if File.file?("bin/queries/#{queryName}.hpp") then
        queryName = temp_name(@qname);
        counter += 1;
      end
      compile_cmd = 
        "OCAMLRUNPARAM='#{$ocamlrunparam}';" +
        "DBT_LIB='#{$lib_boost_path}/lib';" +
        "DBT_HDR='#{$lib_boost_path}/include';" +
        $timeout_compile +
        (dbt_base_cmd + [
        "-l","cpp",
        "-o","bin/queries/#{queryName}.hpp",
        "-c","bin/queries/#{queryName}",
        "-d","mt",
        $optlevel,
      ]).join(" ") + "  2>&1";
      # print compile_cmd
      starttime = Time.now
      system(compile_cmd) or raise "Compilation Error";
      print "(Compile: #{(Time.now - starttime).to_i}s) "
      $stdout.flush;
    end
    return if $compile_only;
    starttime = Time.now;
    IO.popen($timeout_exec + 
             "bin/queries/#{queryName} #{$executable_args.join(" ")}",
             "r") do |qin|
      output = qin.readlines;
      endtime = Time.now;
      output = output.map { |l| l.chomp }.join("");
      @runtime = (endtime - starttime).to_f;
      
      @toplevels.keys.each do |q| 
        if /<#{q}[^>]*>(.*)<\/#{q}>/ =~ output then
          q_results = $1
          case @toplevels[q][:type]
            when :singleton then @toplevels[q][:result] = q_results.to_f
            when :onelevel then  @toplevels[q][:result] = CppDB.new(q_results)
            else nil
            
          end
        else raise "Runtime Error: No result for query #{q}"
        end
      end
    end
  end
  
  def to_s
    "C++ Code Generator"
  end
end

class CppNewBackendUnitTest < GenericUnitTest
    def run
        queryName = @qname;
        unless $skip_compile then
            counter = 1;
            if File.file?("#{$dbt_path}/bin/queries/#{queryName}.hpp") then
              queryName = temp_name(@qname);
              counter += 1;
            end
            compile_cmd =
            "OCAMLRUNPARAM='#{$ocamlrunparam}';" +
            $timeout_compile +
            "(cd #{$dbt_backend_path}; sbt 'toast -l cpp #{$optlevel} -o #{$dbt_path}/bin/queries/#{queryName}.hpp -c #{$dbt_path}/bin/queries/#{queryName} #{@qpath} ')"+
            #(dbt_base_cmd + [
            #"-l","cpp",
            #"-o","bin/queries/#{queryName}.hpp",
            #"-c","bin/queries/#{queryName}",
            #]).join(" ") +
            "  2>&1"
             starttime = Time.now
             system(compile_cmd) or raise "Compilation Error";
             print "(Compile: #{(Time.now - starttime).to_i}s) "
             $stdout.flush;
        end
        return if $compile_only;
        starttime = Time.now;
        IO.popen($timeout_exec +
                 "bin/queries/#{queryName} #{$parallel_input} #{$executable_args.join(" ")}",
                 "r") do |qin|
            output = qin.readlines;
            endtime = Time.now;
            output = output.map { |l| l.chomp }.join("");
            @runtime = (endtime - starttime).to_f;
            
            @toplevels.keys.each do |q|
                if /<#{q}[^>]*>(.*)<\/#{q}>/ =~ output then
                    q_results = $1
                    case @toplevels[q][:type]
                        when :singleton then @toplevels[q][:result] = q_results.to_f
                        when :onelevel then  @toplevels[q][:result] = CppDB.new(q_results)
                        else nil
                        
                    end
                    else raise "Runtime Error: No result for query #{q}"
                end
            end
                 end
    end
    
    def to_s
        "C++ Code Generator"
    end
end


class SCNewBackendUnitTest < GenericUnitTest
    def run
        queryName = @qname;
        unless $skip_compile then
            counter = 1;
            if File.file?("bin/queries/#{queryName}") then
              queryName = temp_name(@qname);
              counter += 1;
            end
            print "#{$dbt_path}/bin/queries/#{queryName}.scala \n"
            compile_cmd =
            "OCAMLRUNPARAM='#{$ocamlrunparam}';" +
            $timeout_compile +
            "(cd #{$dbt_backend_path}/.. ; sbt 'project DDBToaster' 'toast -l scala #{$optlevel} -o #{$dbt_path}/bin/queries/#{queryName}.scala  #{@qpath} '; 
              fsc -J-Xss512m -J-Xmx4G -J-Xms4G -J-XX:-DontCompileHugeMethods -cp /Users/khayyam/Desktop/private/education/Epfl/Pardis-SStore/DDBToaster/storelib/target/scala-2.10/classes:/Users/khayyam/.ivy2/cache/com.typesafe.akka/akka-actor_2.10/jars/akka-actor_2.10-2.2.3.jar:/Users/khayyam/.ivy2/cache/com.typesafe.akka/akka-remote_2.10/jars/akka-remote_2.11-2.2.3.jar:/Users/khayyam/.ivy2/cache/org.uncommons.maths/uncommons-maths/jars/uncommons-maths-1.2.2a.jar:/Users/khayyam/.ivy2/cache/org.scala-lang/scala-actors/jars/scala-actors-2.10.2.jar -d /Users/khayyam/Desktop/private/education/Epfl/Pardis-SStore/DDBToaster/ddbtoaster/target/tmp -deprecation -unchecked -feature -optimise -Yinline-warnings #{$dbt_path}/bin/queries/#{queryName}.scala
              )"+
            #(dbt_base_cmd + [
            #"-l","cpp",
            #"-o","bin/queries/#{queryName}.hpp",
            #"-c","bin/queries/#{queryName}",
            #]).join(" ") +
            "  2>&1"
             starttime = Time.now
             system(compile_cmd) or raise "Compilation Error";
             print "(Compile: #{(Time.now - starttime).to_i}s) "
             $stdout.flush;
        end
        return if $compile_only;
        starttime = Time.now;
        IO.popen($timeout_exec +
                 "(cd #{$dbt_path}/bin/queries ; 
                  java -Xss512m -Xmx4G -Xms4G -XX:-DontCompileHugeMethods -Xbootclasspath/a:/Users/khayyam/.ivy2/cache/org.scala-lang/scala-library/jars/scala-library-2.10.3.jar -cp /Users/khayyam/Desktop/private/education/Epfl/Pardis-SStore/DDBToaster/ddbtoaster/target/tmp:/Users/khayyam/Desktop/private/education/Epfl/Pardis-SStore/DDBToaster/storelib/target/scala-2.10/classes:/Users/khayyam/.ivy2/cache/com.typesafe.akka/akka-actor_2.10/jars/akka-actor_2.10-2.2.3.jar:/Users/khayyam/.ivy2/cache/com.typesafe.akka/akka-remote_2.10/jars/akka-remote_2.11-2.2.3.jar:/Users/khayyam/.ivy2/cache/org.uncommons.maths/uncommons-maths/jars/uncommons-maths-1.2.2a.jar:/Users/khayyam/.ivy2/cache/org.scala-lang/scala-actors/jars/scala-actors-2.10.2.jar:/Users/khayyam/.ivy2/cache/com.typesafe/config/bundles/config-1.2.1.jar ddbt.gen.#{queryName.capitalize} -b0
                  )" + 
                  "  2>&1"
                 ) do |qin|
      output = qin.readlines;
      #print "Here is output : #{output} \n"
      endtime = Time.now;
      output = output.map { |l| l.chomp.strip }.join("");
      @runtime = (endtime - starttime).to_f;
      if /<runtime>(.*)<\/runtime>/ =~ output then
        @runtime = ($1.to_f / 1000000.0).to_f;
      end
      
      @toplevels.keys.each do |q| 
        #print "Here is : #{q}"
        if /<#{q}[^>]*>(.*)<\/#{q}>/ =~ output then
        #if /#{q}(.*)/ =~ output then
          result = $1;
          case @toplevels[q][:type]
            when :singleton then @toplevels[q][:result] = result.strip.to_f;
            when :onelevel then @toplevels[q][:result] = CppDB.new(result);
            else nil
          end
        else raise "Runtime Error"
        end;
      end
    end
  end

  def to_s
    "Scala Code generator"
  end



end


class ScalaUnitTest < GenericUnitTest
  def run
    queryName = @qname;
    unless $skip_compile then
    counter = 1;
    if File::exists?("bin/queries/#{queryName}") then
      queryName = temp_name(@qname);
      counter += 1;
    end
    dir = File.dirname("bin/queries/#{queryName}")
    FileUtils.mkdir_p dir unless File::exists? dir;
    File.delete("bin/queries/#{queryName}.jar") if 
         File::exists?("bin/queries/#{queryName}.jar");
      compile_cmd = 
        "OCAMLRUNPARAM='#{$ocamlrunparam}';" +
        $timeout_compile + 
        (dbt_base_cmd + [
        "-l","scala",
        "-o","bin/queries/#{queryName}.scala",
        "-c","bin/queries/#{queryName}",
        $optlevel,
      ]).join(" ") + "  2>&1";
      starttime = Time.now
      system(compile_cmd) or raise "Compilation Error";
      print "(Compile: #{(Time.now - starttime).to_i}s) "
      $stdout.flush;
    end
    return if $compile_only;
    starttime = Time.now;
    @currentdir = Dir.pwd;
    IO.popen($timeout_exec +
             "scala -J-Xmx100G -J-XX:+HeapDumpOnOutOfMemoryError " +
             "-classpath \"bin/queries/#{queryName}.jar#{$path_delim}" + 
                          "lib/dbt_scala/dbtlib.jar\" " + 
             "org.dbtoaster.RunQuery", "r") do |qin|
      output = qin.readlines;
      endtime = Time.now;
      output = output.map { |l| l.chomp.strip }.join("");
      @runtime = (endtime - starttime).to_f;
      if /<runtime>(.*)<\/runtime>/ =~ output then
        @runtime = ($1.to_f / 1000000.0).to_f;
      end

      @toplevels.keys.each do |q| 
        if /<#{q}[^>]*>(.*)<\/#{q}>/ =~ output then
          result = $1;
          case @toplevels[q][:type]
            when :singleton then @toplevels[q][:result] = result.strip.to_f;
            when :onelevel then @toplevels[q][:result] = CppDB.new(result);
            else nil
          end
        else raise "Runtime Error"
        end;
      end
    end
  end

  def to_s
    "Scala Code generator"
  end
end

class ScalaOptUnitTest < GenericUnitTest
  def run
    queryName = @qname;
    unless $skip_compile then
    counter = 1;
    if File::exists?("bin/queries/#{queryName}") then
      queryName = temp_name(@qname);
      counter += 1;
    end
    dir = File.dirname("bin/queries/#{queryName}")
    FileUtils.mkdir_p dir unless File::exists? dir;
    File.delete("bin/queries/#{queryName}.jar") if 
         File::exists?("bin/queries/#{queryName}.jar");
      compile_cmd = 
        "OCAMLRUNPARAM='#{$ocamlrunparam}';" +
        $timeout_compile + 
        (dbt_base_cmd + [
        "-l","scala",
        "-o","bin/queries/#{queryName}.scala",
        "-O4",
        "-c","bin/queries/#{queryName}",
      ]).join(" ") + "  2>&1";
      starttime = Time.now
      system(compile_cmd) or raise "Compilation Error";
      print "(Compile: #{(Time.now - starttime).to_i}s) "
      $stdout.flush;
    end
    return if $compile_only;
    starttime = Time.now;
    @currentdir = Dir.pwd;
    IO.popen($timeout_exec +
             "scala -J-Xmx2048M -J-XX:+HeapDumpOnOutOfMemoryError " +

             "-J-agentpath:lib/dbt_scala/libyjpagent.jnilib" +
             "=onexit=snapshot" +
             ",logdir=#{@currentdir}/bin/queries/log/#{queryName}" +
             ",dir=#{@currentdir}/bin/queries/snapshot/#{queryName} " +

             "-classpath \"bin/queries/#{queryName}.jar#{$path_delim}" + 
                          "lib/dbt_scala/dbtlib.jar\" " + 
             "org.dbtoaster.RunQuery", "r") do |qin|
      output = qin.readlines;
      endtime = Time.now;
      output = output.map { |l| l.chomp.strip }.join("");
      @runtime = (endtime - starttime).to_f;
      if /<runtime>(.*)<\/runtime>/ =~ output then
        @runtime = ($1.to_f / 1000000.0).to_f;
      end
      
      @toplevels.keys.each do |q| 
        if /<#{q}[^>]*>(.*)<\/#{q}>/ =~ output then
          result = $1;
          case @toplevels[q][:type]
            when :singleton then @toplevels[q][:result] = result.strip.to_f;
            when :onelevel then @toplevels[q][:result] = CppDB.new(result);
            else nil
          end
        else raise "Runtime Error"
        end;
      end
    end
  end

  def to_s
    "Scala Code generator"
  end
end

class InterpreterUnitTest < GenericUnitTest
  def run
    cmd = "OCAMLRUNPARAM='#{$ocamlrunparam}';"+
       "#{$timeout_exec}"+
       "#{dbt_base_cmd.join(" ")}"+
       " -d SINGLE-LINE-MAP-OUTPUT"+
       " -d PARSEABLE-VALUES"+
       " -r 2>&1";
    IO.popen(cmd, "r") do |qin|
      starttime = Time.now;
      @runtime = "unknown";
      qin.each do |l|
        case l
          when /Processing time: ([0-9]+\.?[0-9]*)(e-?[0-9]+)?/ then 
            @runtime = "#{$1}#{$2}".to_f
            print "(Compile: #{(Time.now - starttime - @runtime).to_i}s) "
          when /([a-zA-Z0-9_-]+): (.*)$/ then
            query = $1;
            results = $2;
            raise "Runtime Error: #{results}; #{l}" if query == "error"
            raise "Unexpected result '#{query}'; #{l}" unless 
                @toplevels.has_key? query
            case @toplevels[query][:type]
              when :singleton then @toplevels[query][:result] = results.to_f
              when :onelevel then
                @toplevels[query][:result] = OcamlDB.new(results, false)
            end
        end
      end
    end
  end

  def to_s
    "OcaML Interpreter"
  end
end

tests = [];
$opts = []; 
$debug_flags = [];
$skip_compile = false;
$precision = 1e-4;
$strict = false;
$ret = 0;
$depth = nil;
$verbose = false;
$compile_only = false;
$dataset = nil;
$timeout_exec = "";
$timeout_compile = "";
$compiler_args = [];
$executable_args = [];
$dump_query = false;
$log_detail = false;
$always_return_success = false;
$path_delim = (if RUBY_PLATFORM =~ /cygwin/ then ';' else ':' end);

GetoptLong.new(
  [ '-f',                GetoptLong::REQUIRED_ARGUMENT],
  [ '-t', '--test',      GetoptLong::REQUIRED_ARGUMENT],
  [ '--skip-compile',    GetoptLong::NO_ARGUMENT],
  [ '-p', '--precision', GetoptLong::REQUIRED_ARGUMENT],
  [ '-d',                GetoptLong::REQUIRED_ARGUMENT],
  [ '--depth',           GetoptLong::REQUIRED_ARGUMENT],
  [ '-v', '--verbose',   GetoptLong::NO_ARGUMENT],
  [ '--strict',          GetoptLong::NO_ARGUMENT],
  [ '--compile-only',    GetoptLong::NO_ARGUMENT],
  [ '--dataset',         GetoptLong::REQUIRED_ARGUMENT],
  [ '--timeout',         GetoptLong::REQUIRED_ARGUMENT],  
  [ '--timeout-compile', GetoptLong::REQUIRED_ARGUMENT],    
  [ '--trace',           GetoptLong::OPTIONAL_ARGUMENT],
  [ '--dump-query',      GetoptLong::NO_ARGUMENT],
  [ '--memprofiling',    GetoptLong::NO_ARGUMENT],
  [ '--ignore-errors',   GetoptLong::NO_ARGUMENT],
  [ '--path-delim',      GetoptLong::REQUIRED_ARGUMENT],
  [ '--O1',               GetoptLong::NO_ARGUMENT],
  [ '--O2',               GetoptLong::NO_ARGUMENT],
  [ '--O3',               GetoptLong::NO_ARGUMENT],
  [ '--p0',               GetoptLong::NO_ARGUMENT],
  [ '--p2',               GetoptLong::NO_ARGUMENT]
).each do |opt, arg|
  case opt
    when '-f' then $opts.push(arg)
    when '--skip-compile' then $skip_compile = true;
    when '-p', '--precision' then $precision = 10 ** (-1 * arg.to_i);
    when '-t', '--test' then 
    case arg
        when 'sc' then tests.push SCNewBackendUnitTest
        when 'cppnew'         then tests.push CppNewBackendUnitTest
        when 'cpp'         then tests.push CppUnitTest
        when 'interpreter' then tests.push InterpreterUnitTest
        when 'scala'       then tests.push ScalaUnitTest
        when 'scalaopt'    then tests.push ScalaOptUnitTest
        when 'all'         then tests = [CppNewBackendUnitTest,
                                         CppUnitTest,
                                         InterpreterUnitTest,
                                         ScalaUnitTest,
                                         ScalaOptUnitTest]
      end
    when '-d' then $debug_flags.push(arg)
    when '--depth' then $depth = arg
    when '--strict' then $strict = true;
    when '-v', '--verbose' then $verbose = true;
    when '--compile-only' then $compile_only = true;
    when '--dataset' then $dataset = arg;
    when '--timeout' then 
      if system("which -s timeout") && !arg.nil?
      then $timeout_exec = "timeout #{arg} " 
      end
    when '--timeout-compile' then 
      if system("which -s timeout") && !arg.nil?
      then $timeout_compile = "timeout #{arg} " 
      end
    when '--trace' then $executable_args += ["-t", "QUERY_1_1", "-s", 
                                             if arg.nil? then "100" 
                                                         else arg end ];
                        $log_detail = true;
    when '--dump-query' then $dump_query = true;
    when '--memprofiling' then 
      $compiler_args += ["-l", "cpp:prof", "-g", "^-ltcmalloc"]
    when '--ignore-errors' then $strict = false; $always_return_success = true;
    when '--path-delim' then $path_delim = arg;
    when '--O1' then $optlevel = "-O1";
    when '--O2' then $optlevel = "-O2";
    when '--O3' then $optlevel = "-O3";
    when '--p0' then $parallel_input = "-p 0";
    when '--p2' then $parallel_input = "-p 2";
  end
end

tests.uniq!
tests = [InterpreterUnitTest] if tests.empty?;

queries = ARGV

unless !queries.empty? then
  @dataset = if $dataset.nil? then "standard" else $dataset end
  case @dataset
    when "tiny" then
      queries = [
          "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          # "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          # "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          # "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          # "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          # #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          # "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          # "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    when "tiny_del" then
      queries = [
          "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          # "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          # "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          # "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          # "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          # "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          # #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          # "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          # "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          # "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    when "standard" then
      queries = [
          "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    when "standard_del" then
      queries = [
          "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          # "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          # "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          # "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          # "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          # "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          # #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          # "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          # "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          # "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    when "big" then
      queries = [
          "tpch11a","tpch2","tpch6","tpch4","tpch14","tpch3","tpch10","tpch17a","tpch18a","tpch12",
          "tpch17","tpch18","tpch21","tpch22a","tpch1","tpch22","tpch20","tpch8","tpch9","tpch7",
          "tpch13","ssb4","tpch15",
          "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          
          # "tpch11","tpch19","tpch11c","tpch16","tpch5",
          
          # "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          # "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          # "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          # "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          # "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          # "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          # "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          # #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          # "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          # "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          # "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    when "big_del" then
      queries = [
          "tpch11a","tpch2","tpch6","tpch4","tpch14","tpch3","tpch10","tpch12",
          "tpch17","tpch22a","tpch1","tpch22","tpch20","tpch8","tpch9","tpch7",
          "tpch19","tpch11"
          
          # "tpch17a","tpch21","tpch18a","tpch18","tpch5","tpch11c","ssb4","tpch13","tpch15","tpch16",
          
          # "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          # "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          # "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          # "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          # "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          # "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          # "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          # "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          # #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          # "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          # "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          # "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    when "huge" then
      queries = [
          # "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
          # "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
          # "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
          "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
          # "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
          # "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
          # "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
          # "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
          # #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
          # "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
          # "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
          # "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
        ]
    # when "huge_del" then
    #   queries = [
    #       "tpch1","tpch2","tpch3","tpch4","tpch5","tpch6","tpch7","tpch8",
    #       "tpch9","tpch10","tpch11","tpch11a","tpch11c","tpch12","tpch13","tpch14","tpch15",
    #       "tpch16","tpch17","tpch17a","tpch18","tpch18a","tpch19","tpch20","tpch21","tpch22","tpch22a","ssb4",
    #       "axfinder","brokerspread","brokervariance","pricespread","vwap",#,"missedtrades" => takes too long,"chrissedtrades" => wrong result,
    #       "simple/inequality_selfjoin","simple/invalid_schema_fn","simple/m3k3unable2escalate","simple/miraculous_minus","simple/miraculous_minus2","simple/r_aggcomparison","simple/r_aggofnested","simple/r_aggofnestedagg","simple/r_agtb","simple/r_agtbexists","simple/r_avg","simple/r_bigsumstar","simple/r_btimesa","simple/r_btimesacorrelated","simple/r_count","simple/r_count_of_one","simple/r_count_of_one_prime","simple/r_deepscoping","simple/r_divb","simple/r_existsnestedagg","simple/r_gbasumb","simple/r_gtealldynamic","simple/r_gtesomedynamic","simple/r_gtsomedynamic","simple/r_impossibleineq","simple/r_indynamic","simple/r_ineqandeq","simple/r_instatic","simple/r_lift_of_count","simple/r_ltallagg","simple/r_ltallavg","simple/r_ltallcorravg","simple/r_ltalldynamic","simple/r_multinest","simple/r_natselfjoin","simple/r_nogroupby","simple/r_nonjoineq","simple/r_possibleineq","simple/r_possibleineqwitheq","simple/r_selectstar","simple/r_simplenest","simple/r_smallstar","simple/r_starofnested","simple/r_starofnestedagg","simple/r_sum_gb_all_out_of_aggregate","simple/r_sum_gb_out_of_aggregate","simple/r_sum_out_of_aggregate","simple/r_sumadivsumb","simple/r_sumdivgrp","simple/r_sumnestedintarget","simple/r_sumnestedintargetwitheq","simple/r_sumoutsideofagg","simple/r_sumstar","simple/r_union","simple/r_unique_counts_by_a","simple/rr_ormyself","simple/rs","simple/rs_cmpnest","simple/rs_column_mapping_1","simple/rs_column_mapping_2","simple/rs_column_mapping_3","simple/rs_eqineq","simple/rs_ineqonnestedagg","simple/rs_inequality","simple/rs_ineqwithnestedagg","simple/rs_joinon","simple/rs_joinwithnestedagg","simple/rs_natjoin","simple/rs_natjoinineq","simple/rs_natjoinpartstar","simple/rs_selectconstcmp","simple/rs_selectpartstar","simple/rs_selectstar","simple/rs_simple","simple/rs_stringjoin","simple/rst","simple/rstar","simple/rtt_or_with_stars","simple/singleton_renaming_conflict","simple/ss_math","simple/t_lifttype",
    #       "employee/query01","employee/query01a","employee/query02","employee/query02a","employee/query03","employee/query03a","employee/query04","employee/query04a","employee/query05","employee/query06","employee/query07","employee/query08","employee/query08a","employee/query09","employee/query09a","employee/query10","employee/query10a",
    #       "employee/query11b",#"employee/query11","employee/query11a","employee/query12","employee/query15","employee/query35b","employee/query36b",
    #       "employee/query12a","employee/query13","employee/query14","employee/query16","employee/query16a","employee/query17a","employee/query22","employee/query23a","employee/query24a","employee/query35c","employee/query36c","employee/query37","employee/query38a","employee/query39","employee/query40","employee/query45","employee/query46","employee/query47","employee/query47a","employee/query48","employee/query49","employee/query50","employee/query51",
    #       #"employee/query52","employee/query53","employee/query56","employee/query57","employee/query58","employee/query62","employee/query63","employee/query64","employee/query65","employee/query66","employee/query66a",
    #       "employee/query52a","employee/query53a","employee/query54","employee/query55","employee/query56a","employee/query57a","employee/query58a","employee/query59","employee/query60","employee/query61","employee/query62a","employee/query63a","employee/query64a","employee/query65a",
    #       "zeus/11564068","zeus/12811747","zeus/37494577","zeus/39765730","zeus/48183500","zeus/52548748","zeus/59977251","zeus/75453299","zeus/94384934","zeus/95497049","zeus/96434723",
    #       "mddb1"#,"mddb2" => takes too long time #,"mddb3" => wrong result
    #     ]
  end
end

queries.each do |tquery| 
  tests.each do |test_class|
    t = test_class.new
    opt_terms = 
      (if $debug_flags.empty? then [] 
                              else [["debug flags",$debug_flags]] end)+
      (if $opts.empty? then [] else [["options",$opts]] end)+
      (if $depth.nil? then [] else [["depth", ["#{$depth}"]]] end)+
      (if $compile_only then [["compilation", ["only"]]] else [] end)+
      (if $dataset.nil? then [] else [["dataset", [$dataset]]] end)+
      (if $compiler_args.empty? then [] 
                                else [["compilation args", 
                                      ["'#{$compiler_args.join(" ")}'"]]] 
                                end)+
      (if $executable_args.empty? then [] 
                                  else [["runtime args", 
                                         ["'#{$executable_args.join(" ")}'"]]] 
                                  end)+
      (if $timeout_exec.empty? then []
                               else [["execution timeout",
                                      ["#{$timeout_exec.split(" ")[1]}"]]]
                               end)+
      (if $timeout_compile.empty? then []
                               else [["compile timeout",
                                      ["#{$timeout_compile.split(" ")[1]}"]]]
                               end)                               
    opt_string =
      if opt_terms.empty? then ""
      else " with #{opt_terms.map{|k,tm|"#{k} #{tm.join(", ")}"}.join("; ")}"
      end
    print "Testing query '#{tquery}'#{opt_string} on the #{t.to_s}: ";
    STDOUT.flush;
    t.query = tquery
    puts t.query if $dump_query;
    puts t.dbt_base_cmd.join(" ") if $dump_query;
    begin
      t.run
      unless $compile_only then
        if t.correct? then
          puts "Success (#{t.runtime}s)."
          puts(([["Key", "Expected", "Result", "Different?"], 
                 ["", "", ""]] + t.results).tabulate) if $verbose;
        else
          puts "Failure: Result Mismatch (#{t.runtime}s)."
          puts(([["Key", "Expected", "Result", "Different?"], 
                 ["", "", ""]] + t.results).tabulate);
          $ret = -1;
          exit -1 if $strict;
        end
      else
        puts "Success"
      end
    rescue Exception => e
      puts "Failure: #{e}";
      puts e.backtrace.join("\n");
      $ret = -1;
      exit -1 if $strict;
    end
  end
end
  
exit $ret unless $always_return_success;
