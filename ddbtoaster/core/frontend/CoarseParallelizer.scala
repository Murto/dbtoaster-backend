package ddbt.frontend

import ddbt.ast._
import ddbt.ast.M3._
import ddbt.lib.Utils._

object CoarseParallelizer extends Parallelizer {

  def pruneTop(graph: DependencyGraph): (List[Statement], DependencyGraph) = {
    val top = getTopStmts(graph)
    val (trivial, complex) = top.partition(isTrivial)
    if (trivial.size > 0) {
      val nextGraph = trivial.foldLeft(graph)((g, s) => remove(s, g));
      val (preStmts, finalGraph) = pruneTop(nextGraph)
      (trivial.toList ++ preStmts, finalGraph)
    } else if (complex.size == 1) {
      val nextGraph = complex.foldLeft(graph)((g, s) => remove(s, g));
      val (preStmts, finalGraph) = pruneTop(nextGraph)
      (complex.toList ++ preStmts, finalGraph)
    } else {
      (List(), graph)
    }
  }

  def pruneBottom(graph: DependencyGraph): (DependencyGraph, List[Statement]) = {
    val bottom = getBottomStmts(graph)
    val (trivial, complex) = bottom.partition(isTrivial)
    if (trivial.size > 0) {
      val nextGraph = trivial.foldLeft(graph)((g, s) => remove(s, g));
      val (finalGraph, postStmts) = pruneBottom(nextGraph)
      (finalGraph, postStmts ++ trivial.toList)
    } else if (complex.size == 1) {
      val nextGraph = complex.foldLeft(graph)((g, s) => remove(s, g));
      val (finalGraph, postStmts) = pruneBottom(nextGraph)
      (finalGraph, postStmts ++ complex.toList)
    } else {
      (graph, List())
    }
  }

  def toBlock(tasks: List[ParallelTask]): ParallelBlock = {
    val deps = tasks.flatMap(t => t.inDeps | t.outDeps).toSet
    val threads = if (automaticThreadCount) {
      Some(tasks.size) 
    } else {
      None
    }
    new ParallelBlock(tasks, deps, threads)
  }

  def apply(t: Trigger): Trigger = {
    val graph = toGraph(t.stmts)
    val (preStmts, optGraph, postStmts) = optimize(graph)
    val graphs = splitGraph(optGraph)
    if (graphs.size > 1) {
      val tasks = graphs.map(toTask)
      val block = toBlock(tasks)
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
