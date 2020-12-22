package ddbt

package frontend

import ddbt.ast._
import ddbt.ast.M3._
import ddbt.lib.Utils._

abstract class Parallelizer extends (System => System) {

  sealed abstract class Complexity {

    def +(other: Complexity): Complexity
    
    def *(other: Complexity): Complexity

  }
  
  case object Trivial extends Complexity {

    def +(other: Complexity): Complexity = other

    def *(other: Complexity): Complexity = other

  }

  case object NonTrivial extends Complexity {
    
    def +(other: Complexity): Complexity = NonTrivial

    def *(other: Complexity): Complexity = NonTrivial

  }

  var automaticThreadCount = false
  var optFuseSequential = false
  var optTrimSequential = false
  var optFuseTrivials = false
  var optTrimTrivials = false

  def getCost(e: Expr, keys: Set[String] = Set.empty): (Complexity, Set[String]) = e match {
    case Const(tp, v)        => (Trivial, keys) // [x]
    case e: Ref              => (Trivial, keys) // [x]
    case e: MapRef           => {
      val (ko, ki) = e.keys.zipWithIndex.partition { case((n, _), _) => keys.contains(n) }
      if (e.keys.isEmpty || e.keys.map(_._1).forall(k => keys.contains(k))) { // No new keys to add
        (Trivial, keys)
      } else {
        (NonTrivial, keys ++ e.keys.map(_._1))
      }
    }
    case e: Lift             => getCost(e.e, keys) // [x]
    case e: MapRefConst      => (Trivial, keys) // [x]
    case e: DeltaMapRefConst => (Trivial, keys) // [x]
    case e: AggSum           => {
      if (e.keys.map(_._1).forall(k => keys.contains(k))) {
        getCost(e.e)
      } else {
        (NonTrivial, keys)
      }
      if (e.keys.nonEmpty) (NonTrivial, keys) else getCost(e.e, keys) // [!] May need changing later on
    }
    case e: Mul              => { // Needs changing
      val (leftComplexity,  leftKeys) = getCost(e.l, keys)
      val (rightComplexity, rightKeys) = getCost(e.r, leftKeys)
      (leftComplexity * rightComplexity, rightKeys)
    }
    case e: Add              => { // Needs changing
      val (leftComplexity,  leftKeys) = getCost(e.l, keys)
      val (rightComplexity, rightKeys) = getCost(e.r, leftKeys)
      (leftComplexity + rightComplexity, rightKeys)
    }
    case e: Exists           => getCost(e.e, keys) // [x]
    case e: Apply            => (Trivial, keys)
    //case e: Apply            => e.args.map(getCost).foldLeft(Trivial.asInstanceOf[Complexity])(_+_) // [x] Maybe even always trivial
    case e: Cmp              => {
      val (leftComplexity,  leftKeys) = getCost(e.l, keys)
      val (rightComplexity, rightKeys) = getCost(e.r, leftKeys)
      (leftComplexity * rightComplexity, rightKeys)
    }
    //case e: CmpOrList        => getCost(e.l) + e.r.map(getCost).foldLeft(Trivial.asInstanceOf[Complexity])(_+_) // [x] Though is actually conditional but whatever
    case e: CmpOrList        => (Trivial, keys)
    case e: Tuple            => (Trivial, keys) // [x] Though we never really see this
    case e: TupleLift        => (Trivial, keys) // [x] We never really see this
    case e: Repartition      => getCost(e.e, keys) // [x]
    case e: Gather           => getCost(e.e, keys) // [x]
  }

  def getCost(s: Statement): Complexity = s match {
    case s: TriggerStmt =>
      val init = s.initExpr match {
        case Some(e) => getCost(e, s.target.keys.map(_._1).toSet)._1
        case None => Trivial
      }
      getCost(s.expr, s.target.keys.map(_._1).toSet)._1
    case s: IfStmt =>
      s.thenBlk.map(getCost).foldLeft(Trivial.asInstanceOf[Complexity])(_+_) + s.elseBlk.map(getCost).foldLeft(Trivial.asInstanceOf[Complexity])(_+_)
    case s: SerialBlock =>
      s.stmts.map(getCost).foldLeft(Trivial.asInstanceOf[Complexity])(_+_)
    case s: ParallelBlock =>
      s.stmts.map(getCost).foldLeft(Trivial.asInstanceOf[Complexity])(_+_)
    case s: ParallelTask =>
      getCost(s.stmt)
  }

  def isTrivial(stmt: Statement): Boolean = {
    getCost(stmt) == Trivial
  }

  def isTrivial(graph: DependencyGraph): Boolean = {
    graph.forall {
      case (stmt, _) => isTrivial(stmt)
    }
  }

  def getInDeps(e: Expr): Set[String] = e match {
    case e: Const            => Set.empty[String]
    case e: Ref              => Set.empty[String]
    case e: MapRef           => Set(e.name)
    case e: Lift             => getInDeps(e.e)
    case e: MapRefConst      => Set.empty[String]
    case e: DeltaMapRefConst => Set.empty[String]
    case e: AggSum           => getInDeps(e.e)
    case e: Mul              => Set(e.l, e.r).map(getInDeps).flatten
    case e: Add              => Set(e.l, e.r).map(getInDeps).flatten
    case e: Exists           => getInDeps(e.e)
    case e: Apply            => e.args.map(getInDeps).flatten.toSet
    case e: Cmp              => Set(e.l, e.r).map(getInDeps).flatten
    case e: CmpOrList        => getInDeps(e.l) ++ e.r.map(getInDeps).flatten.toSet
    case e: Tuple            => e.es.map(getInDeps).flatten.toSet
    case e: TupleLift        => getInDeps(e.e)
    case e: Repartition      => getInDeps(e.e)
    case e: Gather           => getInDeps(e.e)
  }

  def getInDeps(s: Statement): Set[String] = s match {
    case s: TriggerStmt =>
      val inDeps = Set(s.target, s.expr).map(getInDeps).flatten
      val outDeps = getOutDeps(s)
      inDeps | outDeps
    case s: IfStmt =>
      getInDeps(s.cond) ++ s.thenBlk.map(getInDeps).flatten.toSet ++ s.elseBlk.map(getInDeps).flatten.toSet
    case s: SerialBlock =>
      s.stmts.flatMap(getInDeps).toSet
    case s: ParallelTask =>
      s.inDeps
    case s: ParallelBlock =>
      s.deps
  }

  def getOutDeps(e: Expr): Set[String] = e match {
    case e: Const            => Set.empty[String]
    case e: Ref              => Set.empty[String]
    case e: MapRef           => Set.empty[String]
    case e: Lift             => Set.empty[String]
    case e: MapRefConst      => Set.empty[String]
    case e: DeltaMapRefConst => Set.empty[String]
    case e: AggSum           => getOutDeps(e.e)
    case e: Mul              => Set(e.l, e.r).map(getOutDeps).flatten
    case e: Add              => Set(e.l, e.r).map(getOutDeps).flatten
    case e: Exists           => getOutDeps(e.e)
    case e: Apply            => e.args.map(getOutDeps).flatten.toSet
    case e: Cmp              => Set(e.l, e.r).map(getOutDeps).flatten
    case e: CmpOrList        => getOutDeps(e.l) ++ e.r.map(getOutDeps).flatten.toSet
    case e: Tuple            => e.es.map(getOutDeps).flatten.toSet
    case e: TupleLift        => Set.empty[String]
    case e: Repartition      => getOutDeps(e.e)
    case e: Gather           => getOutDeps(e.e)
  }

  def getOutDeps(s: Statement): Set[String] = s match {
    case s: TriggerStmt =>
      Set(s.target, s.expr).map(getOutDeps).flatten + s.target.name
    case s: IfStmt =>
      getOutDeps(s.cond) ++ s.thenBlk.map(getOutDeps).flatten.toSet ++ s.elseBlk.map(getOutDeps).flatten.toSet
    case s: SerialBlock =>
      s.stmts.flatMap(getOutDeps).toSet
    case s: ParallelTask =>
      s.outDeps
    case s: ParallelBlock =>
      s.stmts.flatMap(getOutDeps).toSet
  }

  def toGraph(ss: List[Statement]): DependencyGraph = ss match {
    case s :: tail =>
      val dependents = tail.filter(os => (getOutDeps(os) & getInDeps(s)).nonEmpty || (getInDeps(os) & getOutDeps(s)).nonEmpty || (getOutDeps(os) & getOutDeps(s)).nonEmpty).toSet
      toGraph(tail) + (s -> dependents)
    case Nil => Map()
  }

  def getDirectChildren(stmt: Statement, graph: DependencyGraph): Set[Statement] = {
    val children = graph(stmt)
    (graph - stmt).filter{
      case (cstmt, cdeps) =>
        children.contains(cstmt) &&
        children.forall(c => !graph(c).contains(cstmt))
    }.keySet
  }

  def getDirectParents(stmt: Statement, graph: DependencyGraph): Set[Statement] = {
    val parents = graph.filter{
      case (pstmt, pdeps) =>
        pdeps.contains(stmt)
    }.keySet
    (graph - stmt).filter{
      case (pstmt, pdeps) =>
        parents.contains(pstmt) &&
        (parents - pstmt).forall(p => !graph(pstmt).contains(p))
    }.keySet
  }

  def removeRedundantEdges(graph: DependencyGraph): DependencyGraph = {
    graph.map {
      case (stmt, dependents) =>
        (stmt, dependents &~ dependents.flatMap(d => graph(d)).toSet)
    }
  }

  def getConnected(graph: DependencyGraph, stmt: Statement, seen: DependencyGraph = Map()): DependencyGraph = {
    val children = graph(stmt)
    val unseen = graph.filter {
      case (nextStmt, nextChildren) =>
        ((nextChildren + nextStmt) & (children + stmt)).nonEmpty
    } -- seen.keySet
    if (unseen.isEmpty) {
      seen
    } else {
      val nextSeen = seen ++ unseen
      unseen.keySet.map(stmt => getConnected(graph, stmt, nextSeen)).reduce(_ ++ _)
    }
  }

  def splitGraph(graph: DependencyGraph): List[DependencyGraph] = {
    if (graph.isEmpty) {
      Nil
    } else {
      val (stmt, _) = graph.head
      val subGraph = getConnected(graph, stmt)
      subGraph :: splitGraph(graph -- subGraph.keySet)
    }
  }

  def remove(stmt: Statement, graph: DependencyGraph): DependencyGraph = {
    (graph - stmt).map {
      case (s, dependents) =>
        (s, dependents - stmt)
    }
  }
 
  def getTopStmts(graph: DependencyGraph): Set[Statement] = {
    graph.filter {
      case (stmt, _) =>
        (graph - stmt).forall {
          case (_, dependents) => !dependents.contains(stmt)
        }
    }.keySet
  }

  def getBottomStmts(graph: DependencyGraph): Set[Statement] = {
    graph.filter { case (_, stmts) => stmts.isEmpty }.keySet
  }

  def toTasks(graph: DependencyGraph): List[ParallelTask] = {
    toStmts(graph).map(toTask)
  }

  def toTask(stmts: List[Statement]): ParallelTask = {
    val serial = new SerialBlock(stmts)
    toTask(serial)
  }
  
  def toTask(stmt: Statement): ParallelTask = {
    val inDeps = getInDeps(stmt)
    val outDeps = getOutDeps(stmt)
    new ParallelTask(stmt, inDeps, outDeps)
  }

  def toTask(graph: DependencyGraph): ParallelTask = {
    toTask(toStmts(graph))
  }

  def toStmts(graph: DependencyGraph, seen: Set[Statement] = Set()): List[Statement] = graph match {
    case graph: DependencyGraph if graph.nonEmpty =>
      val top = getTopStmts(graph).toList
      top ++ toStmts(graph -- top)
    case _ => Nil
  }

  def fuseSequential(graph: DependencyGraph): DependencyGraph = {
    // Get pairs that can be fused
    val pairs = graph.flatMap{
      case (stmt, deps) =>
        val partners = graph.filter{
          case (pstmt, pdeps) =>
            deps.contains(pstmt) &&
            (deps - pstmt) == pdeps && 
            !(graph - stmt).exists{
              case (ostmt, odeps) =>
                !odeps.contains(stmt) && odeps.contains(pstmt)
            }
        }
        if (partners.nonEmpty) {
          val (partner, deps) = partners.head
          Some((stmt, partner, deps))
        } else {
          None
        }
    }

    // Fuse pairs until no pairs remain to fuse
    if (pairs.nonEmpty) {
      val(from, to, deps) = pairs.head
      val serial = new SerialBlock(from :: to :: Nil)
      val fusedGraph = (graph - (from, to)).map{
        case (stmt, deps) =>
          if (deps.contains(from)) {
            (stmt, deps - (from, to) + serial)
          } else {
            (stmt, deps)
          }
      } + (serial -> deps)
      fuseSequential(fusedGraph)
    } else {
      graph
    }
  }


  def fuseTrivials(graph: DependencyGraph): DependencyGraph = {
    val pairs = graph.flatMap{
      case (stmt, deps) =>
        val parents = getDirectParents(stmt, graph)
        val (trivialParents, _) = parents.partition(isTrivial)
        val children = getDirectChildren(stmt, graph)
        val (trivialChildren, _) = children.partition(isTrivial)
        val parentPairs = trivialParents.collect{
          case p if isTrivial(p) && getDirectChildren(p, graph).size == 1 => 
            (p, stmt, graph(p) - stmt)
        }
        val childPairs = trivialChildren.collect{
          case c if isTrivial(c) && getDirectParents(c, graph).size == 1 =>
            (stmt, c, deps - c)
        }
        parentPairs ++ childPairs
    }

    // Fuse pairs until no pairs remain to use
    if (pairs.nonEmpty) {
      val (from, to, deps) = pairs.head
      val serial = new SerialBlock(from :: to :: Nil)
      val fusedGraph = (graph - (from, to)).map{
        case (stmt, deps) =>
          if (deps.contains(from) || deps.contains(to)) {
            (stmt, (deps - (from, to)) + serial)
          } else {
            (stmt, deps)
          }
      } + (serial -> deps)
      fuseTrivials(fusedGraph)
    } else {
      graph
    }
  }

  def trimTrivials(graph: DependencyGraph): (List[Statement], DependencyGraph, List[Statement]) = {
    val top = getTopStmts(graph).toList
    val preStmts = top.filter(isTrivial)
    val bottom = getBottomStmts(graph).toList
    val postStmts = bottom.filter(isTrivial)
    val trimmedGraph = (preStmts ++ postStmts).foldLeft(graph)((g, s) => remove(s, g))
    if (trimmedGraph != graph) {
      val (nextPreStmts, nextStrippedGraph, nextPostStmts) = trimTrivials(trimmedGraph)
      (preStmts ++ nextPreStmts, nextStrippedGraph, nextPostStmts ++ postStmts)
    } else {
      (preStmts, trimmedGraph, postStmts)
    }
  }

  def trimSequential(graph: DependencyGraph): (List[Statement], DependencyGraph, List[Statement]) = {
    val top = getTopStmts(graph) &~ getBottomStmts(graph)
    val bottom = getBottomStmts(graph) &~ getTopStmts(graph)
    val preStmts = if (top.size == 1) {
      top.toList
    } else {
      Nil
    }
    val postStmts = if (bottom.size == 1) {
      bottom.toList
    } else {
      Nil
    }
    val trimmedGraph = (preStmts ++ postStmts).foldLeft(graph)((g, s) => remove(s, g))
    if (trimmedGraph != graph) {
      val (nextPreStmts, nextStrippedGraph, nextPostStmts) = trimSequential(trimmedGraph)
      (preStmts ++ nextPreStmts, nextStrippedGraph, nextPostStmts ++ postStmts)
    } else {
      (preStmts, trimmedGraph, postStmts)
    }
  }

  def optimize(graph: DependencyGraph): (List[Statement], DependencyGraph, List[Statement]) = {
    var preStmts = List.empty[Statement]
    var postStmts = List.empty[Statement]
    var optGraph = graph
    
    if (optFuseSequential) {
      optGraph = fuseSequential(optGraph)
    }

    if (optFuseTrivials) {
      optGraph = fuseTrivials(optGraph)
    }
    
    if (optTrimTrivials) {
      val (preTrivials, trimmed, postTrivials) = trimTrivials(optGraph)
      preStmts = preStmts ++ preTrivials
      postStmts = postTrivials ++ postStmts
      optGraph = trimmed
    }

    if (optTrimSequential) {
      val (preSeq, trimmed, postSeq) = trimSequential(optGraph)
      preStmts = preStmts ++ preSeq
      postStmts = postSeq ++ postStmts
      optGraph = trimmed
    }

    if (graph != optGraph) {
      val (nextPreStmts, nextOptGraph, nextPostStmts) = optimize(optGraph)
      (preStmts ++ nextPreStmts, nextOptGraph, nextPostStmts ++ postStmts)
    } else {
      (Nil, graph, Nil)
    }
  }

  def resetOpts() = {
    automaticThreadCount = false
    optFuseSequential = false
    optFuseTrivials = false
    optTrimTrivials = false
    optTrimSequential = false
  }

  def setOpt(option: String) = option match {
    case "automaticthreadcount" => automaticThreadCount = true
    case "fusesequential" => optFuseSequential = true
    case "fusetrivials" => optFuseTrivials = true
    case "trimtrivials" => optTrimTrivials = true
    case "trimsequential" => optTrimSequential = true
    case _ => throw new IllegalArgumentException(s"Unsupported option $option")
  }

  def setOptLevel(level: Int) = {
    if (level >= 1) {
      optFuseSequential = true
      optTrimSequential = true
    }
    if (level >= 2) {
      optTrimTrivials = true
      optFuseTrivials = true
    }
  }

  def apply(s: System): System

}
