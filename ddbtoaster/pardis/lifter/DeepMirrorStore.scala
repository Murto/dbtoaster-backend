/* Generated by Purgatory 2014-2017 */

package ddbt.lib.store.deep

import ch.epfl.data.sc.pardis
import pardis.ir._
import pardis.types.PardisTypeImplicits._
import pardis.effects._
import pardis.deep._
import pardis.deep.scalalib._
import pardis.deep.scalalib.collection._
import pardis.deep.scalalib.io._

trait StoreOps extends Base with ArrayOps with EntryIdxOps with IdxOps {  
  // Type representation
  val StoreType = StoreIRs.StoreType
  type StoreType[E <: ddbt.lib.store.Entry] = StoreIRs.StoreType[E]
  implicit def typeStore[E <: ddbt.lib.store.Entry: TypeRep]: TypeRep[Store[E]] = StoreType(implicitly[TypeRep[E]])
  implicit class StoreRep[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) {
     def unsafeInsert(idx : Rep[Int], e : Rep[E]) : Rep[Unit] = storeUnsafeInsert[E](self, idx, e)(typeE)
     def insert(e : Rep[E]) : Rep[Unit] = storeInsert[E](self, e)(typeE)
     def update(e : Rep[E]) : Rep[Unit] = storeUpdate[E](self, e)(typeE)
     def updateCopy(e : Rep[E]) : Rep[Unit] = storeUpdateCopy[E](self, e)(typeE)
     def updateCopyDependent(e : Rep[E]) : Rep[Unit] = storeUpdateCopyDependent[E](self, e)(typeE)
     def delete(e : Rep[E])(implicit overload1 : Overloaded1) : Rep[Unit] = storeDelete1[E](self, e)(typeE)
     def deleteCopy(e : Rep[E]) : Rep[Unit] = storeDeleteCopy[E](self, e)(typeE)
     def deleteCopyDependent(e : Rep[E]) : Rep[Unit] = storeDeleteCopyDependent[E](self, e)(typeE)
     def get(idx : Rep[Int], key : Rep[E]) : Rep[E] = storeGet[E](self, idx, key)(typeE)
     def getCopy(idx : Rep[Int], key : Rep[E]) : Rep[E] = storeGetCopy[E](self, idx, key)(typeE)
     def getCopyDependent(idx : Rep[Int], key : Rep[E]) : Rep[E] = storeGetCopyDependent[E](self, idx, key)(typeE)
     def foreach(f : Rep[(E => Unit)]) : Rep[Unit] = storeForeach[E](self, f)(typeE)
     def slice(idx : Rep[Int], key : Rep[E], f : Rep[(E => Unit)]) : Rep[Unit] = storeSlice[E](self, idx, key, f)(typeE)
     def sliceCopy(idx : Rep[Int], key : Rep[E], f : Rep[(E => Unit)]) : Rep[Unit] = storeSliceCopy[E](self, idx, key, f)(typeE)
     def sliceCopyDependent(idx : Rep[Int], key : Rep[E], f : Rep[(E => Unit)]) : Rep[Unit] = storeSliceCopyDependent[E](self, idx, key, f)(typeE)
     def range(idx : Rep[Int], min : Rep[E], max : Rep[E], withMin : Rep[Boolean], withMax : Rep[Boolean], f : Rep[(E => Unit)]) : Rep[Unit] = storeRange[E](self, idx, min, max, withMin, withMax, f)(typeE)
     def delete(idx : Rep[Int], key : Rep[E])(implicit overload2 : Overloaded2) : Rep[Unit] = storeDelete2[E](self, idx, key)(typeE)
     def clear : Rep[Unit] = storeClear[E](self)(typeE)
     def index(idx : Rep[Int], idxType : Rep[String], uniq : Rep[Boolean], otherIdx : Rep[Int]) : Rep[Idx[E]] = storeIndex[E](self, idx, idxType, uniq, otherIdx)(typeE)
     def copyIntoPool(e : Rep[E]) : Rep[E] = storeCopyIntoPool[E](self, e)(typeE)
     def n : Rep[Int] = store_Field_N[E](self)(typeE)
     def ops : Rep[Array[EntryIdx[E]]] = store_Field_Ops[E](self)(typeE)
     def idxs : Rep[Array[Idx[E]]] = store_Field_Idxs[E](self)(typeE)
  }
  object Store {

  }
  // constructors
   def __newStore[E <: ddbt.lib.store.Entry](idxs : Rep[Array[Idx[E]]], ops : Rep[Array[EntryIdx[E]]])(implicit overload1 : Overloaded1, typeE : TypeRep[E]) : Rep[Store[E]] = storeNew1[E](idxs, ops)(typeE)
   def __newStore[E <: ddbt.lib.store.Entry]()(implicit cE : Manifest[E], overload2 : Overloaded2, typeE : TypeRep[E]) : Rep[Store[E]] = storeNew2[E]()(typeE, cE)
   def __newStore[E <: ddbt.lib.store.Entry](n : Rep[Int], ops : Rep[Array[EntryIdx[E]]])(implicit cE : Manifest[E], overload3 : Overloaded3, typeE : TypeRep[E]) : Rep[Store[E]] = storeNew3[E](n, ops)(typeE, cE)
  // IR defs
  val StoreNew1 = StoreIRs.StoreNew1
  type StoreNew1[E <: ddbt.lib.store.Entry] = StoreIRs.StoreNew1[E]
  val StoreNew2 = StoreIRs.StoreNew2
  type StoreNew2[E <: ddbt.lib.store.Entry] = StoreIRs.StoreNew2[E]
  val StoreNew3 = StoreIRs.StoreNew3
  type StoreNew3[E <: ddbt.lib.store.Entry] = StoreIRs.StoreNew3[E]
  val StoreUnsafeInsert = StoreIRs.StoreUnsafeInsert
  type StoreUnsafeInsert[E <: ddbt.lib.store.Entry] = StoreIRs.StoreUnsafeInsert[E]
  val StoreInsert = StoreIRs.StoreInsert
  type StoreInsert[E <: ddbt.lib.store.Entry] = StoreIRs.StoreInsert[E]
  val StoreUpdate = StoreIRs.StoreUpdate
  type StoreUpdate[E <: ddbt.lib.store.Entry] = StoreIRs.StoreUpdate[E]
  val StoreUpdateCopy = StoreIRs.StoreUpdateCopy
  type StoreUpdateCopy[E <: ddbt.lib.store.Entry] = StoreIRs.StoreUpdateCopy[E]
  val StoreUpdateCopyDependent = StoreIRs.StoreUpdateCopyDependent
  type StoreUpdateCopyDependent[E <: ddbt.lib.store.Entry] = StoreIRs.StoreUpdateCopyDependent[E]
  val StoreDelete1 = StoreIRs.StoreDelete1
  type StoreDelete1[E <: ddbt.lib.store.Entry] = StoreIRs.StoreDelete1[E]
  val StoreDeleteCopy = StoreIRs.StoreDeleteCopy
  type StoreDeleteCopy[E <: ddbt.lib.store.Entry] = StoreIRs.StoreDeleteCopy[E]
  val StoreDeleteCopyDependent = StoreIRs.StoreDeleteCopyDependent
  type StoreDeleteCopyDependent[E <: ddbt.lib.store.Entry] = StoreIRs.StoreDeleteCopyDependent[E]
  val StoreGet = StoreIRs.StoreGet
  type StoreGet[E <: ddbt.lib.store.Entry] = StoreIRs.StoreGet[E]
  val StoreGetCopy = StoreIRs.StoreGetCopy
  type StoreGetCopy[E <: ddbt.lib.store.Entry] = StoreIRs.StoreGetCopy[E]
  val StoreGetCopyDependent = StoreIRs.StoreGetCopyDependent
  type StoreGetCopyDependent[E <: ddbt.lib.store.Entry] = StoreIRs.StoreGetCopyDependent[E]
  val StoreForeach = StoreIRs.StoreForeach
  type StoreForeach[E <: ddbt.lib.store.Entry] = StoreIRs.StoreForeach[E]
  val StoreSlice = StoreIRs.StoreSlice
  type StoreSlice[E <: ddbt.lib.store.Entry] = StoreIRs.StoreSlice[E]
  val StoreSliceCopy = StoreIRs.StoreSliceCopy
  type StoreSliceCopy[E <: ddbt.lib.store.Entry] = StoreIRs.StoreSliceCopy[E]
  val StoreSliceCopyDependent = StoreIRs.StoreSliceCopyDependent
  type StoreSliceCopyDependent[E <: ddbt.lib.store.Entry] = StoreIRs.StoreSliceCopyDependent[E]
  val StoreRange = StoreIRs.StoreRange
  type StoreRange[E <: ddbt.lib.store.Entry] = StoreIRs.StoreRange[E]
  val StoreDelete2 = StoreIRs.StoreDelete2
  type StoreDelete2[E <: ddbt.lib.store.Entry] = StoreIRs.StoreDelete2[E]
  val StoreClear = StoreIRs.StoreClear
  type StoreClear[E <: ddbt.lib.store.Entry] = StoreIRs.StoreClear[E]
  val StoreIndex = StoreIRs.StoreIndex
  type StoreIndex[E <: ddbt.lib.store.Entry] = StoreIRs.StoreIndex[E]
  val StoreCopyIntoPool = StoreIRs.StoreCopyIntoPool
  type StoreCopyIntoPool[E <: ddbt.lib.store.Entry] = StoreIRs.StoreCopyIntoPool[E]
  val Store_Field_N = StoreIRs.Store_Field_N
  type Store_Field_N[E <: ddbt.lib.store.Entry] = StoreIRs.Store_Field_N[E]
  val Store_Field_Ops = StoreIRs.Store_Field_Ops
  type Store_Field_Ops[E <: ddbt.lib.store.Entry] = StoreIRs.Store_Field_Ops[E]
  val Store_Field_Idxs = StoreIRs.Store_Field_Idxs
  type Store_Field_Idxs[E <: ddbt.lib.store.Entry] = StoreIRs.Store_Field_Idxs[E]
  // method definitions
   def storeNew1[E <: ddbt.lib.store.Entry](idxs : Rep[Array[Idx[E]]], ops : Rep[Array[EntryIdx[E]]])(implicit typeE : TypeRep[E]) : Rep[Store[E]] = StoreNew1[E](idxs, ops)
   def storeNew2[E <: ddbt.lib.store.Entry]()(implicit typeE : TypeRep[E], cE : Manifest[E]) : Rep[Store[E]] = StoreNew2[E]()
   def storeNew3[E <: ddbt.lib.store.Entry](n : Rep[Int], ops : Rep[Array[EntryIdx[E]]])(implicit typeE : TypeRep[E], cE : Manifest[E]) : Rep[Store[E]] = StoreNew3[E](n, ops)
   def storeUnsafeInsert[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreUnsafeInsert[E](self, idx, e)
   def storeInsert[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreInsert[E](self, e)
   def storeUpdate[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreUpdate[E](self, e)
   def storeUpdateCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreUpdateCopy[E](self, e)
   def storeUpdateCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreUpdateCopyDependent[E](self, e)
   def storeDelete1[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreDelete1[E](self, e)
   def storeDeleteCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreDeleteCopy[E](self, e)
   def storeDeleteCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreDeleteCopyDependent[E](self, e)
   def storeGet[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit typeE : TypeRep[E]) : Rep[E] = StoreGet[E](self, idx, key)
   def storeGetCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit typeE : TypeRep[E]) : Rep[E] = StoreGetCopy[E](self, idx, key)
   def storeGetCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit typeE : TypeRep[E]) : Rep[E] = StoreGetCopyDependent[E](self, idx, key)
   def storeForeach[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], f : Rep[((E) => Unit)])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreForeach[E](self, f)
   def storeSlice[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E], f : Rep[((E) => Unit)])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreSlice[E](self, idx, key, f)
   def storeSliceCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E], f : Rep[((E) => Unit)])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreSliceCopy[E](self, idx, key, f)
   def storeSliceCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E], f : Rep[((E) => Unit)])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreSliceCopyDependent[E](self, idx, key, f)
   def storeRange[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], min : Rep[E], max : Rep[E], withMin : Rep[Boolean], withMax : Rep[Boolean], f : Rep[((E) => Unit)])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreRange[E](self, idx, min, max, withMin, withMax, f)
   def storeDelete2[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreDelete2[E](self, idx, key)
   def storeClear[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Unit] = StoreClear[E](self)
   def storeIndex[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], idxType : Rep[String], uniq : Rep[Boolean], otherIdx : Rep[Int])(implicit typeE : TypeRep[E]) : Rep[Idx[E]] = StoreIndex[E](self, idx, idxType, uniq, otherIdx)
   def storeCopyIntoPool[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit typeE : TypeRep[E]) : Rep[E] = StoreCopyIntoPool[E](self, e)
   def store_Field_N[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Int] = Store_Field_N[E](self)
   def store_Field_Ops[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Array[EntryIdx[E]]] = Store_Field_Ops[E](self)
   def store_Field_Idxs[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Array[Idx[E]]] = Store_Field_Idxs[E](self)
  type Store[E <: ddbt.lib.store.Entry] = ddbt.lib.store.Store[E]
}
object StoreIRs extends Base {
  import ArrayIRs._
  import EntryIdxIRs._
  import IdxIRs._
  // Type representation
  case class StoreType[E <: ddbt.lib.store.Entry](typeE: TypeRep[E]) extends TypeRep[Store[E]] {
    def rebuild(newArguments: TypeRep[_]*): TypeRep[_] = StoreType(newArguments(0).asInstanceOf[TypeRep[_ <: ddbt.lib.store.Entry]])
    val name = s"Store[${typeE.name}]"
    val typeArguments = List(typeE)
  }
      implicit def typeStore[E <: ddbt.lib.store.Entry: TypeRep]: TypeRep[Store[E]] = StoreType(implicitly[TypeRep[E]])
  // case classes
  case class StoreNew1[E <: ddbt.lib.store.Entry](idxs : Rep[Array[Idx[E]]], ops : Rep[Array[EntryIdx[E]]])(implicit val typeE : TypeRep[E]) extends ConstructorDef[Store[E]](List(typeE), "Store", List(List(idxs,ops))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreNew2[E <: ddbt.lib.store.Entry]()(implicit val typeE : TypeRep[E], val cE : Manifest[E]) extends ConstructorDef[Store[E]](List(typeE), "Store", List(List())){
    override def curriedConstructor = (x: Any) => copy[E]()
  }

  case class StoreNew3[E <: ddbt.lib.store.Entry](n : Rep[Int], ops : Rep[Array[EntryIdx[E]]])(implicit val typeE : TypeRep[E], val cE : Manifest[E]) extends ConstructorDef[Store[E]](List(typeE), "Store", List(List(n,ops))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreUnsafeInsert[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "unsafeInsert", List(List(idx,e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreInsert[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "insert", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreUpdate[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "update", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreUpdateCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "updateCopy", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreUpdateCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "updateCopyDependent", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreDelete1[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "delete", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreDeleteCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "deleteCopy", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreDeleteCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "deleteCopyDependent", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreGet[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[E](Some(self), "get", List(List(idx,key))){
    override def curriedConstructor = (copy[E] _).curried
    override def effect = Read(self)
  }

  case class StoreGetCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[E](Some(self), "getCopy", List(List(idx,key))){
    override def curriedConstructor = (copy[E] _).curried
    override def effect = Read(self)
  }

  case class StoreGetCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[E](Some(self), "getCopyDependent", List(List(idx,key))){
    override def curriedConstructor = (copy[E] _).curried
    override def effect = Read(self)
  }

  case class StoreForeach[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], f : Rep[((E) => Unit)])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "foreach", List(List(f))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreSlice[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E], f : Rep[((E) => Unit)])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "slice", List(List(idx,key,f))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreSliceCopy[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E], f : Rep[((E) => Unit)])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "sliceCopy", List(List(idx,key,f))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreSliceCopyDependent[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E], f : Rep[((E) => Unit)])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "sliceCopyDependent", List(List(idx,key,f))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreRange[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], min : Rep[E], max : Rep[E], withMin : Rep[Boolean], withMax : Rep[Boolean], f : Rep[((E) => Unit)])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "range", List(List(idx,min,max,withMin,withMax,f))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreDelete2[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], key : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "delete", List(List(idx,key))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreClear[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit val typeE : TypeRep[E]) extends FunctionDef[Unit](Some(self), "clear", List()){
    override def curriedConstructor = (copy[E] _)
  }

  case class StoreIndex[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], idx : Rep[Int], idxType : Rep[String], uniq : Rep[Boolean], otherIdx : Rep[Int])(implicit val typeE : TypeRep[E]) extends FunctionDef[Idx[E]](Some(self), "index", List(List(idx,idxType,uniq,otherIdx))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class StoreCopyIntoPool[E <: ddbt.lib.store.Entry](self : Rep[Store[E]], e : Rep[E])(implicit val typeE : TypeRep[E]) extends FunctionDef[E](Some(self), "copyIntoPool", List(List(e))){
    override def curriedConstructor = (copy[E] _).curried
  }

  case class Store_Field_N[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit val typeE : TypeRep[E]) extends FieldDef[Int](self, "n"){
    override def curriedConstructor = (copy[E] _)
    override def isPure = true

  }

  case class Store_Field_Ops[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit val typeE : TypeRep[E]) extends FieldDef[Array[EntryIdx[E]]](self, "ops"){
    override def curriedConstructor = (copy[E] _)
    override def isPure = true

    override def partiallyEvaluate(children: Any*): Array[EntryIdx[E]] = {
      val self = children(0).asInstanceOf[Store[E]]
      self.ops
    }
    override def partiallyEvaluable: Boolean = true

  }

  case class Store_Field_Idxs[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit val typeE : TypeRep[E]) extends FieldDef[Array[Idx[E]]](self, "idxs"){
    override def curriedConstructor = (copy[E] _)
    override def isPure = true

    override def partiallyEvaluate(children: Any*): Array[Idx[E]] = {
      val self = children(0).asInstanceOf[Store[E]]
      self.idxs
    }
    override def partiallyEvaluable: Boolean = true

  }

  type Store[E <: ddbt.lib.store.Entry] = ddbt.lib.store.Store[E]
}
trait StoreImplicits extends StoreOps { 
  // Add implicit conversions here!
}
trait StoreComponent extends StoreOps with StoreImplicits {  }

trait StorePartialEvaluation extends StoreComponent with BasePartialEvaluation {  
  // Immutable field inlining 
  override def store_Field_N[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Int] = self match {
    case Def(node: StoreNew3[_]) => node.n
    case _ => super.store_Field_N[E](self)(typeE)
  }
  override def store_Field_Ops[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Array[EntryIdx[E]]] = self match {
    case Def(node: StoreNew1[_]) => node.ops
    case Def(node: StoreNew3[_]) => node.ops
    case _ => super.store_Field_Ops[E](self)(typeE)
  }
  override def store_Field_Idxs[E <: ddbt.lib.store.Entry](self : Rep[Store[E]])(implicit typeE : TypeRep[E]) : Rep[Array[Idx[E]]] = self match {
    case Def(node: StoreNew1[_]) => node.idxs
    case _ => super.store_Field_Idxs[E](self)(typeE)
  }

  // Mutable field inlining 
  // Pure function partial evaluation
}


