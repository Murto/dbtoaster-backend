package ddbt.codegen
import ddbt.codegen.lms._

import ddbt.ast._
import ddbt.lib._
//import toasterbooster._
import scala.virtualization.lms.internal._
import scala.virtualization.lms.common._

class LMSGen(cls:String="Query") extends ScalaGen(cls) {
  import ddbt.ast.M3._
  val impl = ScalaExpGen
  
  /*
  We need specific LMS nodes for
  - K3Map / K3Var / K3Temp : pass key as list of symbols, specific name as a string (automatic for temp)
    + foreach,get,add/set, ...
    ==> Need to be adapted from previous version / try to provide common interface for all of them
  - Named fresh (for function arguments)
  - Mirror of the user-defined functions library => some functions should be inlined, others not
  */

  override def genTrigger(t:Trigger):String = {
    //val outStream = new java.io.StringWriter
    //val impl = new Impl(new java.io.PrintWriter(outStream), "org.dbtoaster", false) with DSL
    //val impl = new Impl
    
    val (name,args) = t.evt match {
      case EvtReady => ("SystemReady",Nil)
      case EvtAdd(Schema(n,cs)) => ("Add"+n,cs)
      case EvtDel(Schema(n,cs)) => ("Del"+n,cs)
    }

    type LMSContext = Map[String, impl.Rep[_]]
    val ctx0 = Map[String, impl.Rep[_]]() // empty context

    var exprrrr = ""
    // the ctx argument contains all the symbols that are available in the current context
    // the returned context is only _NEW_ symbols that have been added to the original context
    
    // XXX: we need a continuation here somewhere that is just one operation done on the variable
    def expr(ex:Expr,ctx:LMSContext):(impl.Rep[_],LMSContext) = ex match {
      case Ref(n) => (ctx(n),ctx0)
      case Const(tp,v) => (ex.tp match {
        case TypeLong => impl.unit(v.toLong)
        case TypeDouble => impl.unit(v.toDouble)
        case TypeString => impl.unit(v)
        case TypeDate => impl.unit(new java.util.Date()) // XXX: set correct value
      },ctx0)
      case Mul(l,r) => val (vl,cl)=expr(l,ctx); val (vr,cr)=expr(r,ctx++cl); (ex.tp match {
        case TypeLong => numeric_times[Long](vl,vr)
        case TypeDouble => numeric_times[Double](vl,vr)
        case _ => sys.error("Mul(l,r) only allowed on numeric types")
      },cl++cr)
      case Add(l,r) => val (vl,cl)=expr(l,ctx); val (vr,cr)=expr(r,ctx); (ex.tp match {
        case TypeLong   => numeric_plus[Long](vl,vr)
        case TypeDouble => numeric_plus[Double](vl,vr)
        case _ => sys.error("Add(l,r) only allowed on numeric types")
      },cl++cr)
      case Cmp(l,r,op) => val (vl,cl)=expr(l,ctx); val (vr,cr)=expr(r,cl); (ex.tp match {
        case TypeLong   => cmp[Long](vl,op,vr)
        case TypeDouble => cmp[Double](vl,op,vr)
        case _ => sys.error("Cmp(l,r,op) is not possible for types (%s, %s)".format(l.tp, r.tp))
      },cl++cr)
      case Exists(e) => val (ve,ce)=expr(e,ctx); (impl.__ifThenElse(impl.notequals(ve,impl.unit(0)),impl.unit(1),impl.unit(0)),ce)
      case Lift(n,e) => val (ve,ce) = expr(e,ctx); if (!ctx.contains(n)) (impl.unit(1), ce + (n -> ve)) else (ex.tp match {
        case TypeLong => cmp[Long](ctx(n),OpEq,ve)
        case TypeDouble => cmp[Double](ctx(n),OpEq,ve)
        case _ => sys.error("lift")
      },ce)
      case Apply(fn,tp,as) =>
        val (vs,cn:LMSContext) = ((List[impl.Rep[_]](),ctx0) /: as) { case ((vs,c1),e) => val (v,c2) = expr(e,ctx); (vs:::v::Nil,c1++c2) }
        ( /* Apply(fn,vs) */ impl.unit(-10),cn) // XXX: call appropriate LMS node

      //case AggSum(ks,e) => ks.toSet
      //case MapRef(n,tp,ks) => ks.toSet
      case _ => exprrrr = exprrrr + "ex = " + ex + "\n";(impl.fresh[Any], ctx) //sys.error("Unimplemented")
    }

    def numeric_times[T:Numeric:Manifest](l: impl.Rep[_], r: impl.Rep[_]) = impl.numeric_times[T](l.asInstanceOf[impl.Rep[T]],r.asInstanceOf[impl.Rep[T]])
    def numeric_plus[T:Numeric:Manifest](l: impl.Rep[_], r: impl.Rep[_]) = impl.numeric_plus[T](l.asInstanceOf[impl.Rep[T]],r.asInstanceOf[impl.Rep[T]])
    def cmp[T:Ordering:Manifest](l: impl.Rep[_], op: OpCmp, r: impl.Rep[_]): impl.Rep[Boolean] = op match {
      case OpEq => impl.equals[T,T](l.asInstanceOf[impl.Rep[T]],r.asInstanceOf[impl.Rep[T]])
      case OpNe => impl.notequals[T,T](l.asInstanceOf[impl.Rep[T]],r.asInstanceOf[impl.Rep[T]])
      case OpGt => impl.ordering_gt[T](l.asInstanceOf[impl.Rep[T]],r.asInstanceOf[impl.Rep[T]])
      case OpGe => impl.ordering_gteq[T](l.asInstanceOf[impl.Rep[T]],r.asInstanceOf[impl.Rep[T]])
    }

    val ctxTrigger: LMSContext=(
          maps.map{ case MapDef(name,_,keys,_) => (name, if (keys.size==0) impl.fresh[K3Var[_]] else impl.fresh[K3Map[_,_]]) } union // XXX: we need here name-based k3maps
          args.map{ case (name,tp) => (name,freshRef(/*impl,*/ tp)) }).toMap // we need here name-based references of correct type

    t.stmts.foreach { s =>  s match {
        case StmtMap(m,e,op,oi) => 
          //val fop=op match { case OpAdd => "add" case OpSet => "set" }
          //val clear = op match { case OpAdd => "" case OpSet => if (m.keys.size>0) m.name+".clear()\n" else "" }
          //val init = (oi match {
          //  case Some(ie) => cpsExpr(ie,b,(i:String)=>"if ("+m.name+".get("+(if (m.keys.size==0) "" else tup(m.keys))+")==0) "+m.name+".set("+(if (m.keys.size==0) "" else tup(m.keys)+",")+i+");")+"\n"
          //  case None => ""
          //})
          //clear+init+cpsExpr(e,b,(v:String) => m.name+"."+fop+"("+(if (m.keys.size==0) "" else tup(m.keys)+",")+v+");")+"\n"
          expr(e, ctxTrigger)
        case _ => sys.error("Unimplemented") // we leave room for other type of events
      }
    }

    //maps.map{ case MapDef(name,_,keys,_) => (name, if (keys.size==0) name+"[] ++ " else name+"["+keys+"] ++ ") }+
    //"\n\nHiii\n\n%s\n\niiiH\n\n".format(outStream.toString) +
    "def on"+name+"("+args.map{a=>a._1+":"+a._2.toScala} .mkString(", ")+") {\n"+
    "  "+impl.emit(impl.unit(1.0))+
    "  hello2"+ //ind(t.stmts.map{s=>genStmt(s,b)}.mkString)
    "\n}"
  }

  def freshRef(tp: Type): impl.Sym[_] = freshRefManifest(typeManifest(tp))

  def freshRefManifest[T:Manifest](mf: Manifest[T]): impl.Sym[T] = impl.fresh[T]

  def typeManifest(tp:Type):Manifest[_] = tp match {
    case TypeLong => manifest[Long]
    case TypeDouble => manifest[Double]
    case _ => sys.error("No manifest for "+tp)
  }

/*
  def typeManifest(tp:Type, orTp:Type):Manifest[_] = tp match {
    case TypeLong => typeManifest(orTp)
    case _ => typeManifest(tp)
  }
  
  def tupleManifest(ts:List[Type]) = {
    val ms:List[Manifest[_]] = ts map typeManifest
    val cls:java.lang.Class[_] = Class.forName("scala.Tuple"+ts.size)
    scala.reflect.ManifestFactory.classType(cls,ms.head,ms.tail:_*)
  }
*/

  var maps = List[MapDef]() // global maps, to be replaced by a Map[String,LMS_K3Map]
  override def genSystem(s0:System):String = {
    maps=s0.maps; val r=super.genSystem(s0); maps=Nil; r
  }
}
