package ddbt.lib

import java.io._

/**
 * This class contains useful functions and variables that are shared across
 * different compiler parts.
 * @author TCK
 */
object Utils {

  // Display a warning to the user
  def warning(msg: String) = {
    val sb = new StringBuilder(("@" * 80) + "\n@" + (" " * 78) + "@\n")
    ("WARNING: " + msg).split("\n").foreach { l => sb.append("@  %-74s  @\n".format(l)) }
    sb.append("@" + (" " * 78) + "@\n" + ("@" * 80))
    System.err.println(sb.toString)
  }

  private val prop_ = new java.util.Properties
  
  try { 
    prop_.load(this.getClass.getResource("/ddbt.properties").openStream()) 
  } catch { case _: Throwable => 
    warning("conf/ddbt.properties does not exist.\n" +
        "Please configure at least ddbt.dbtoaster to dbtoaster_frontend binary path.\n" + 
        "Set ddbt.base_repo if you have access to DBToaster's repository.") 
  }

  def prop(name: String, d: String = "") = prop_.getProperty("ddbt." + name, d)

  def isLMSTurnedOn = prop("lms") == "1"

  val pathFindBin = prop("find_bin", "find")

  val cppOpts = prop("cpp_opts", "").split(" ").toList
  
  val scalacOpts = prop("scalac").split(" +").toList

  // Paths related to DBToaster
  val pathRepo = { 
    prop("base_repo", null)
  }

  val pathDBTBin  = (if (pathRepo != null) pathRepo + "/" else "") + 
    prop("dbtoaster.frontend", "bin/dbtoaster_frontend")

  private lazy val pathJDK = { 
    var p = prop("jdk", null) 
    if (p != null) { 
        val r = prop(p, null)
        if (r != null) p = r
    }
    if (p == null) System.getProperty("java.home") else p 
  }

  private lazy val jClasspaths = 
    System.getProperty("sbt.classpath", 
      System.getProperty("sun.java.command").replaceAll(".*-classpath | .*","") + ":" + 
      System.getProperty("sun.boot.class.path"))
    .split(":")
    .filter(_.matches("(.*)/(\\.(sbt|ivy2)|target)/.*"))
    .filter(_.indexOf("virtualized")== -1)

  private lazy val scalaEmbeddedClasspaths = 
    jClasspaths.filter(_.matches(".*(com.typesafe.akka/akka-actor).*")) ++
    jClasspaths.filter(_.matches(".*(org.scala-lang/scala-library).*")) ++
    jClasspaths.filter(_.matches(".*(storelib/target).*"))

  private lazy val scalacExternalClasspaths =
    jClasspaths.filter(_.matches(".*(storelib/target).*"))

  private lazy val javaRuntimeClasspaths = 
    jClasspaths.filter(_.matches(".*(com.typesafe.akka/akka-actor|com.typesafe/config/bundles/config).*")) ++  
    jClasspaths.filter(_.matches(".*(org.scala-lang/scala-library).*")) ++
    jClasspaths.filter(_.matches(".*(storelib/target).*"))
  
  private lazy val jSparkBootclasspaths = List[String]()

  private lazy val jSparkClasspaths = 
    jClasspaths.filter(_.matches(".*(org.apache.spark/spark-core).*")) ++
    jClasspaths.filter(_.matches(".*(org.apache.hadoop/hadoop-common|org.apache.hadoop/hadoop-annotations).*")) ++
    jClasspaths.filter(_.matches(".*(log4j/log4j|).*")) ++
    jClasspaths.filter(_.matches(".*(storelib/target).*")) ++
    jClasspaths.filter(_.matches(".*(spark/target).*"))

  // Scala compiler wrapper
  def scalaCompiler(dir: File, classpath: String = null, external: Boolean = false): List[String] => Unit = {
    val cp = if (classpath == null || classpath == "") Nil else List(classpath)
    val pathDir = dir.getAbsolutePath

    if (external) externalScalaCompiler(scalacOpts, "", (cp ++ scalacExternalClasspaths).mkString(":"), pathDir)  
    else embeddedScalaCompiler(scalacOpts, "", (cp ++ scalaEmbeddedClasspaths).mkString(":"), pathDir)  
  }

  // Spark compiler wrapper
  def sparkCompiler(dir: File, classpath: String = null, external: Boolean = false): List[String] => Unit = {
    val cp = (if (classpath == null || classpath == "") "" else classpath + ":") + jSparkClasspaths.mkString(":")
    val bootcp = jSparkBootclasspaths.mkString(":")
    val pathDir = dir.getAbsolutePath

    if (external) externalScalaCompiler(scalacOpts, bootcp, cp, pathDir, false)  
    else embeddedScalaCompiler(scalacOpts, bootcp, cp, pathDir)  
  }

  private def embeddedScalaCompiler(opts: List[String], bootcp: String, cp: String, pathDir: String): List[String] => Unit = {
    val s = new scala.tools.nsc.Settings()
    s.processArguments(opts, true)
    s.bootclasspath.value = bootcp
    s.classpath.value = cp //+ " -J-XX:-DontCompileHugeMethods -J-XX:+CMSClassUnloadingEnabled"
    s.outputDirs.setSingleOutput(pathDir)
    val g = new scala.tools.nsc.Global(s)
    (fs: List[String]) => 
      try { (new g.Run).compile(fs) } 
      catch { case t: Throwable => t.printStackTrace }
  }

  private def externalScalaCompiler(opts: List[String], bootcp: String, cp: String, pathDir: String, useFsc: Boolean = true): List[String] => Unit = {
    // FSC external processes
    def execOut(cmd: String) = exec(cmd.split(" +"), prefix = "")
    val jbootcp = if (bootcp == "") "" else "-J-Xbootclasspath/p:" + bootcp
    val jcp = if (cp == null || cp == "") "" else " -cp " + cp
    val args = jbootcp + jcp + " -d " + pathDir + " " + opts.mkString(" ") + " "
    val exe = if (useFsc) "fsc " else "scalac "
    val scalac = exe + prop("jvm").split(" +").map("-J"+_).mkString(" ") + " " + args
    (fs: List[String]) => 
      try { execOut(scalac + fs.mkString(" ")) } 
      catch { case _: IOException => 
        val jbootcp = if (bootcp == "") "" else " -Xbootclasspath/p:" + bootcp
        execOut("java " + prop("jvm") + jbootcp + " scala.tools.nsc.Main " + args + fs.mkString(" ")) 
      }
  }

  // C++ compiler wrapper
  def cppCompiler(out: String, cPath: String, boost: String, cppLibDir: String) = {
    val as = 
      ( List(prop("gpp", "g++"), cppLibDir + "/main.cpp", "-Wall", 
          "-Wno-unused-variable", "-Wno-strict-overflow", "-std=c++11",
          "-include", out, "-o", cPath, "-O3", "-DNDEBUG", "-lpthread", "-ldbtoaster", //"-ljemalloc",
          "-I" + cppLibDir, "-L" + cppLibDir) :::
        (if (boost == null) Nil else 
          List("program_options", "serialization", "system", "filesystem",
               "chrono", "thread").map("-lboost_" + _ + Utils.prop("lib_boost_thread", "")) ::: 
          List("-I" + boost + "/include", "-L" + boost + "/lib")) ::: 
        cppOpts
      ).filter(_ != "")
    //make DBT c++ library
    Utils.exec(Array("make", "-C", cppLibDir))
    Utils.exec(as.toArray)
  }

  // Executes a Scala program
  def scalaExec(cp: List[File], className: String, args: Array[String] = Array(), external: Boolean = false) = {
    if (!external) runMain(cp, className, args)
    else {
      val env = ("JAVA_HOME=" + pathJDK :: 
        scala.collection.JavaConversions.mapAsScalaMap(System.getenv)
          .filter(_._1.toUpperCase != "JAVA_HOME")
          .map(x => x._1 + "=" + x._2).toList).toArray      
      exec(
        ( "java " + prop("jvm") + 
          " -cp " + (
              javaRuntimeClasspaths ++
              cp.map(_.getAbsolutePath)
            ).mkString(":") + " " + 
          className + " " + args.mkString(" ")
        ).split(" +"), 
        env = env, fatal = true, prefix = "")
    }
  }

  // Executes a Scala program
  def sparkExec(cp: List[File], className: String, args: Array[String] = Array(), external: Boolean = false) = {
    if (!external) runMain(cp, className, args)
    else {
      val env = ("JAVA_HOME=" + pathJDK :: 
        scala.collection.JavaConversions.mapAsScalaMap(System.getenv)
          .filter(_._1.toUpperCase != "JAVA_HOME")
          .map(x => x._1 + "=" + x._2).toList).toArray      
      exec(
        ( "java " + prop("jvm") +
          " -cp " + (
              jSparkBootclasspaths ++ 
              jSparkClasspaths ++ 
              cp.map(_.getAbsolutePath)
            ).mkString(":") + " " + 
          className + " " + args.mkString(" ")).split(" +"), 
        env = env, fatal = true, prefix = "")
    }
  }

  def sparkSubmit(className: String, args: Array[String]) = {
    val prop = new java.util.Properties
    prop.load(new java.io.FileInputStream("ddbtoaster/spark/conf/spark.config"))

    val homeDir = prop.getProperty("spark.home.dir")
    val masterURL = prop.getProperty("spark.master.url")
    val driverMemory = prop.getProperty("spark.driver.memory")
    val numExecutors = prop.getProperty("spark.executors.num")
    val execMemory = prop.getProperty("spark.executor.memory")
    val execCores = prop.getProperty("spark.executor.cores")

    val submit = if (homeDir == null || homeDir == "") "spark-submit" 
      else s"${homeDir}/bin/spark-submit"

    val (sMaster, sDeploy) = 
      if (masterURL.matches("yarn-client.*")) 
        ("--master yarn", "--deploy-mode client")
      else if (masterURL.matches("yarn-cluster.*"))
        ("--master yarn", "--deploy-mode cluster")
      else 
        ("--master " + masterURL, "")  
    val sDriverMemory = 
      if (driverMemory == null || driverMemory == "") "" 
      else "--driver-memory " + driverMemory
    val sNumExecutors = 
      if (numExecutors == null || numExecutors == "") ""
      else "--num-executors " + numExecutors
    val sExecMemory = 
      if (execMemory == null || execMemory == "") ""
      else "--executor-memory " + execMemory
    val sExecCores = 
      if (execCores == null || execCores == "") ""
      else "--executor-cores " + execCores
      
    val env = ("JAVA_HOME=" + pathJDK :: 
      scala.collection.JavaConversions.mapAsScalaMap(System.getenv)
        .filter(_._1.toUpperCase != "JAVA_HOME")
        .map(x => x._1 + "=" + x._2).toList).toArray

    val runtimeJars = javaRuntimeClasspaths.filter(_.endsWith(".jar")).mkString(",")

    exec(
      ( s"$submit $sMaster $sDeploy $sDriverMemory $sNumExecutors " + 
        s"$sExecMemory $sExecCores --files ddbtoaster/spark/conf/log4j.properties " +
        s"--conf spark.ui.showConsoleProgress=false " +
        s"--jars $runtimeJars " +
        s"--class $className ./ddbtoaster/pkg/ddbt_gen.jar " + args.mkString(" ")
      ).split(" +"),
      env = env, fatal = true, prefix = "")
  }

  // XXX: use scala.sys.process.Process and scala.sys.process.ProcessLogger instead
  // Gobbles an input stream (lines not matching prefix are sent to out, null matches all, "" matches none)
  private def gobble(in: InputStream, out: PrintStream = null, prefix: String = null) = new Thread {
    private var buf = new StringBuilder
    override def toString = { join; buf.toString.trim }
    override def run {
      val r = new BufferedReader(new InputStreamReader(in))
      var l = r.readLine
      val n = if (prefix == null || out == null) -1 else prefix.length
      while(l != null) {
        if (n== -1) buf.append(l+"\n")
        else if (n > 0 && l.startsWith(prefix)) buf.append(l.substring(n) + "\n")
        else if (out != null) out.println(l)
        l = r.readLine
      }
      r.close
    }
    start
  }

  // Execute arbitrary command, return (out,err)
  def exec(cmd: String, fatal: Boolean): (String, String) = 
    exec(cmd.split(" +"), null, null, fatal)

  def exec(cmd: Array[String], dir: File = null, env: Array[String] = null, 
      fatal: Boolean = true, prefix: String = null): (String, String) = {
    //System.err.println(cmd.mkString(" "))
    val p = Runtime.getRuntime.exec(cmd, env, dir)
    val out = gobble(p.getInputStream, scala.Console.out, prefix)
    val err = gobble(p.getErrorStream, scala.Console.err, prefix)
    val exitVal = p.waitFor
    val o = out.toString
    val e = err.toString
    if (fatal && (e.trim != "" || exitVal != 0)) {
      if(e.trim != "") {
        println("Execution error in: " + cmd.mkString(" "))
        scala.Console.out.print(o)
        scala.Console.err.print(e)
      } else {
        scala.Console.err.print("Error: exit value is " + exitVal)
      }
      System.exit(if (exitVal != 0) exitVal else 1)
    }
    (o, e)
  }

  // Capture console/default output and error streams in two strings
  def captureOut[R](f: () => R, prefix: String = null) : (R, String, String) = { 
    import scala.Console

    val po = new PipedOutputStream
    val out = Console.withOut(new PrintStream(po)) {
      System.setOut(new PrintStream(po))
      gobble(new PipedInputStream(po), Console.out, prefix)
    }
    val pe = new PipedOutputStream
    val err = Console.withErr(new PrintStream(pe)) {
      System.setErr(new PrintStream(pe))  
      gobble(new PipedInputStream(pe), Console.err, prefix)
    }

    val r = 
      try { f() }
      finally { 
        Console.withOut(Console.out) {
          System.setOut(System.out)  
          po.close()
        }
        Console.withErr(Console.err) {
          System.setErr(System.err)
          pe.close()
        }
      }
    (r, out.toString, err.toString)
  }

  // Class loader to run a class with main(args:Array[String]) within the same VM
  def runMain(cp: List[File], cls: String, args: Array[String] = Array()) = {
    try { 
      val l = new CPLoader(cp)
      val c = l.loadClass(cls)
      val m = c.getMethod("main", args.getClass)
      m.invoke(null, args) 
    }
    catch { case t: Throwable => 
      val c = t.getCause
      (if (c != null) c else t).printStackTrace(scala.Console.err) 
    } 
    finally { System.gc; Thread.sleep(50); System.gc }
  }

  import java.net.{URL,URLClassLoader}
  private class CPLoader(cp: List[File]) extends URLClassLoader(cp.map(_.toURI.toURL).toArray, null) {
    override def loadClass(name: String, resolve: Boolean): Class[_] = {
      try { return super.loadClass(name, resolve) }
      catch { case e: ClassNotFoundException => 
        Class.forName(name, resolve, Utils.getClass.getClassLoader) 
      }
    }
  }

  // Files I/O
  def read(file: String) = {
    var reader: FileReader = null
    var content = ""
    try {
      val f = new File(file)
      val charArray = new Array[Char](f.length.toInt)
      reader = new FileReader(f)
      reader.read(charArray)
      content = new String(charArray)
    }
    catch { case ex: Exception => content = "" }
    finally { if (reader != null) reader.close() }
    content
  }

  def write(file: String, data: String) = {
    var writer: PrintWriter = null
    try {
      val f = new File(file)
      val dir = f.getParentFile
      if (dir != null && !dir.exists) dir.mkdirs
      writer = new PrintWriter(f)
      writer.write(data)
    }
    catch { case ex: Exception => }
    finally { if (writer != null) writer.close() }
  }

  // String manipulation
  def ind(s: String, n: Int = 1) = { 
    val i = "  " * n
    i + s.replaceAll("\n? *$", "").replaceAll("\n", "\n" + i) 
  }

  def block(s: String) = 
    if (s == null || s.trim() == "") s.trim() else "\n" + s + "\n"

  def tup(vs: List[String]) = { 
    val v = vs.mkString(", ")
    if (vs.size > 1) "(" + v + ")" else v 
  }


  // Fresh variables name provider
  private val counter = scala.collection.mutable.HashMap[String, Int]()  
  
  def fresh(name: String = "x") = { 
    val c = counter.getOrElse(name, 0) + 1
    counter.put(name, c) 
    name + c 
  }

  def freshClear() = counter.clear


  // Create a temporary directory that will be removed at shutdown
  def makeTempDir(path: String = null, auto_delete: Boolean = false): File = {
    val tmp = if (path != null) new File(path) else new File("ddbtoaster/target/tmp") //File.createTempFile("ddbt",null) deletes folder too early on OracleJVM7/MacOS
    def del(f: File) {
      if (f.isDirectory()) f.listFiles().foreach(del)
      if (!f.delete()) sys.error("Failed to delete file: " + f)
    }
    if (tmp.exists) del(tmp)
    tmp.mkdirs // have a fresh folder
    if (auto_delete) Runtime.getRuntime.addShutdownHook(new Thread{ override def run() = del(tmp) })
    tmp
  }

  // Time measurement
  def ns[T](f: () => T) = { 
    val t0 = System.nanoTime
    var r = f()
    val t1 = System.nanoTime
    (t1 - t0, r) 
  }

  def time(ns: Long, p: Boolean = true) = if (p) { 
      val us = ns/1000
      ("%d.%06d").format(us / 1000000, us % 1000000) 
    } else { 
      val ms = math.round(ns / 1000000.0)
      ("%d.%03d").format(ms / 1000, ms % 1000) 
    }


  def generateNewFileName(qName: String, pathTemplate: String) = {
    var queryName = qName
    var f = new java.io.File(pathTemplate.format(queryName))
    var couldLock = false
    while(!couldLock) {
      var i = 1
      while(f.exists) {
        queryName = qName + "_" + i
        i += 1
        f = new java.io.File(pathTemplate.format(queryName))
      }
      var channel: java.nio.channels.FileChannel = null;
      var lock: java.nio.channels.FileLock = null;
      try {
        channel = new java.io.RandomAccessFile(f, "rw").getChannel();

        // Use the file channel to create a lock on the file.
        // This method blocks until it can retrieve the lock.
        lock = channel.lock

        // Try acquiring the lock without blocking. This method returns
        // null or throws an exception if the file is already locked.
        try {
          lock = channel.tryLock
        } catch {
          case e: java.nio.channels.OverlappingFileLockException =>
          // File is already locked in this thread or virtual machine
        }

        if(lock != null) {

          f.createNewFile

          // Release the lock
          lock.release

          couldLock = true
        }
      } catch {
          case e: Exception => //...
      } finally {
        if(lock != null && lock.isValid) lock.release();
        if(channel != null) channel.close();
      }
    }
    queryName
  }

  def nonNegativeMod(x: Int, mod: Int): Int = {
    val rawMod = x % mod
    rawMod + (if (rawMod < 0) mod else 0)
  }

  def stringIf(flag: Boolean, t: => String, e: => String = "") = if (flag) t else e
}