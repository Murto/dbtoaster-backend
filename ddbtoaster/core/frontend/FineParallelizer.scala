package ddbt.frontend

import ddbt.ast._
import ddbt.ast.M3._
import ddbt.lib.Utils._

object FineParallelizer extends Parallelizer {

  def countMaxThreads(graph: DependencyGraph, group: Set[Statement]=Set()): Int = {
    val valid = graph.filter{
      case (stmt, deps) => graph.filter{
        case (g, _) =>
          group.contains(g)
      }.forall {
        case (gstmt, gdeps) =>
          gstmt != stmt && !deps.contains(gstmt) && !gdeps.contains(stmt)
      }
    }.keySet
    if (valid.nonEmpty) {
      valid.map(v => countMaxThreads(graph - v, group + v)).max
    } else {
      group.size
    }
  }

  def toBlock(graph: DependencyGraph): ParallelBlock = {
    val tasks = toTasks(graph)
    val deps = tasks.flatMap(t => t.inDeps | t.outDeps).toSet
    val threads = if (automaticThreadCount) {
      Some(countMaxThreads(graph))
    } else {
      None
    }
    new ParallelBlock(tasks, deps, threads)
  }

  def apply(t: Trigger): Trigger = {
    val graph = toGraph(t.stmts)
    val (preStmts, optGraph, postStmts) = optimize(graph)
    val tasks = toTasks(optGraph)
    if (tasks.size > 1) {
      val block = toBlock(optGraph)
      val stmts = preStmts ++ (block :: Nil) ++ postStmts
      new Trigger(t.event, stmts)
    } else {
      t
    }
  }

  def apply(s: System): System = {
    val triggers = s.triggers.map(apply)
    new System(s.typeDefs, s.sources, s.maps, s.queries, triggers)
  }

}
