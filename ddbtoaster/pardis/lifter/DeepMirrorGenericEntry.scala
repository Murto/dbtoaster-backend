/* Generated by Purgatory 2014-2015 */

package ddbt.lib.store.deep

import ch.epfl.data.sc.pardis
import pardis.ir._
import pardis.types.PardisTypeImplicits._
import pardis.effects._
import pardis.deep._
import pardis.deep.scalalib._
import pardis.deep.scalalib.collection._
import pardis.deep.scalalib.io._
trait GenericEntryOps extends Base  {  
  // Type representation
  val GenericEntryType = GenericEntryIRs.GenericEntryType
  implicit val typeGenericEntry: TypeRep[GenericEntry] = GenericEntryType
  implicit class GenericEntryRep(self : Rep[GenericEntry]) {
     def update(i : Rep[Int], v : Rep[Any]) : Rep[Unit] = genericEntryUpdate(self, i, v)
     def increase(i : Rep[Int], v : Rep[Any]) : Rep[Unit] = genericEntryIncrease(self, i, v)
     def +=(i : Rep[Int], v : Rep[Any]) : Rep[Unit] = genericEntry$plus$eq(self, i, v)
     def decrease(i : Rep[Int], v : Rep[Any]) : Rep[Unit] = genericEntryDecrease(self, i, v)
     def -=(i : Rep[Int], v : Rep[Any]) : Rep[Unit] = genericEntry$minus$eq(self, i, v)
     def get[E](i : Rep[Int])(implicit typeE : TypeRep[E]) : Rep[E] = genericEntryGet[E](self, i)(typeE)
     def cmp(e : Rep[GenericEntry]) : Rep[Int] = genericEntryCmp(self, e)
  }
  object GenericEntry {

  }
  // constructors
   def __newGenericEntry() : Rep[GenericEntry] = genericEntryNew()
  // IR defs
  val GenericEntryNew = GenericEntryIRs.GenericEntryNew
  type GenericEntryNew = GenericEntryIRs.GenericEntryNew
  val GenericEntryUpdate = GenericEntryIRs.GenericEntryUpdate
  type GenericEntryUpdate = GenericEntryIRs.GenericEntryUpdate
  val GenericEntryIncrease = GenericEntryIRs.GenericEntryIncrease
  type GenericEntryIncrease = GenericEntryIRs.GenericEntryIncrease
  val GenericEntry$plus$eq = GenericEntryIRs.GenericEntry$plus$eq
  type GenericEntry$plus$eq = GenericEntryIRs.GenericEntry$plus$eq
  val GenericEntryDecrease = GenericEntryIRs.GenericEntryDecrease
  type GenericEntryDecrease = GenericEntryIRs.GenericEntryDecrease
  val GenericEntry$minus$eq = GenericEntryIRs.GenericEntry$minus$eq
  type GenericEntry$minus$eq = GenericEntryIRs.GenericEntry$minus$eq
  val GenericEntryGet = GenericEntryIRs.GenericEntryGet
  type GenericEntryGet[E] = GenericEntryIRs.GenericEntryGet[E]
  val GenericEntryCmp = GenericEntryIRs.GenericEntryCmp
  type GenericEntryCmp = GenericEntryIRs.GenericEntryCmp
  // method definitions
   def genericEntryNew() : Rep[GenericEntry] = GenericEntryNew()
   def genericEntryUpdate(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) : Rep[Unit] = GenericEntryUpdate(self, i, v)
   def genericEntryIncrease(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) : Rep[Unit] = GenericEntryIncrease(self, i, v)
   def genericEntry$plus$eq(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) : Rep[Unit] = GenericEntry$plus$eq(self, i, v)
   def genericEntryDecrease(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) : Rep[Unit] = GenericEntryDecrease(self, i, v)
   def genericEntry$minus$eq(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) : Rep[Unit] = GenericEntry$minus$eq(self, i, v)
   def genericEntryGet[E](self : Rep[GenericEntry], i : Rep[Int])(implicit typeE : TypeRep[E]) : Rep[E] = GenericEntryGet[E](self, i)
   def genericEntryCmp(self : Rep[GenericEntry], e : Rep[GenericEntry]) : Rep[Int] = GenericEntryCmp(self, e)
  type GenericEntry = ddbt.lib.store.GenericEntry
}
object GenericEntryIRs extends Base {
  // Type representation
  case object GenericEntryType extends TypeRep[GenericEntry] {
    def rebuild(newArguments: TypeRep[_]*): TypeRep[_] = GenericEntryType
    val name = "GenericEntry"
    val typeArguments = Nil
    
    val typeTag = scala.reflect.runtime.universe.typeTag[GenericEntry]
  }
      implicit val typeGenericEntry: TypeRep[GenericEntry] = GenericEntryType
  // case classes
  case class GenericEntryNew() extends ConstructorDef[GenericEntry](List(), "GenericEntry", List(List())){
    override def curriedConstructor = (x: Any) => copy()
  }

  case class GenericEntryUpdate(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) extends FunctionDef[Unit](Some(self), "update", List(List(i,v))){
    override def curriedConstructor = (copy _).curried
  }

  case class GenericEntryIncrease(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) extends FunctionDef[Unit](Some(self), "increase", List(List(i,v))){
    override def curriedConstructor = (copy _).curried
  }

  case class GenericEntry$plus$eq(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) extends FunctionDef[Unit](Some(self), "+=", List(List(i,v))){
    override def curriedConstructor = (copy _).curried
  }

  case class GenericEntryDecrease(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) extends FunctionDef[Unit](Some(self), "decrease", List(List(i,v))){
    override def curriedConstructor = (copy _).curried
  }

  case class GenericEntry$minus$eq(self : Rep[GenericEntry], i : Rep[Int], v : Rep[Any]) extends FunctionDef[Unit](Some(self), "-=", List(List(i,v))){
    override def curriedConstructor = (copy _).curried
  }

  case class GenericEntryGet[E](self : Rep[GenericEntry], i : Rep[Int])(implicit val typeE : TypeRep[E]) extends FunctionDef[E](Some(self), "get", List(List(i)), List(typeE)){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class GenericEntryCmp(self : Rep[GenericEntry], e : Rep[GenericEntry]) extends FunctionDef[Int](Some(self), "cmp", List(List(e))){
    override def curriedConstructor = (copy _).curried
  }

  type GenericEntry = ddbt.lib.store.GenericEntry
}
trait GenericEntryImplicits extends GenericEntryOps { 
  // Add implicit conversions here!
}
trait GenericEntryPartialEvaluation extends GenericEntryComponent with BasePartialEvaluation {  
  // Immutable field inlining 

  // Mutable field inlining 
  // Pure function partial evaluation
}
trait GenericEntryComponent extends GenericEntryOps with GenericEntryImplicits {  }
