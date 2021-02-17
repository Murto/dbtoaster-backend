#ifndef DBTOASTER_MULTIMAP_HPP
#define DBTOASTER_MULTIMAP_HPP

#include <iostream>
#include <functional>
#include <string>
#include <cmath>
#include "macro.hpp"
#include "types.hpp"
#include "serialization.hpp"
#include "no_wait_memory_pool.hpp"
#include "no_wait_trivial_memory_pool.hpp"
#include "singleton.hpp"
#include "lock_omp.hpp"

using namespace dbtoaster;

namespace dbtoaster {

template <typename T>
using MemoryPool = memory_pool::NoWaitMemoryPool<T>;

template <typename T>
using TrivialMemoryPool = memory_pool::NoWaitTrivialMemoryPool<T>;

constexpr bool isPowerOfTwo(std::size_t v) {
  return v && ((v & (v - 1)) == 0);
}

constexpr std::size_t kDefaultChunkSize = 32;  // 2^N

template <typename T, typename IDX_FN = T>
class PrimaryHashIndex {
 private: 
  struct IdxNode {
    T* obj{nullptr};
    HashType hash;
    IdxNode* next{nullptr};
  };

  using TMemoryPool = MemoryPool<T>;

  using IdxNodeMemoryPool = TrivialMemoryPool<IdxNode>;
  
  IdxNodeMemoryPool* pool_;
  IdxNode* buckets_;
  std::size_t bucket_count_;
  std::size_t entry_count_;
  std::size_t index_mask_;     // derived value
  std::size_t threshold_;      // derived value
  const double load_factor_;
  mutable Lock lock_;

  void resize_lockless(std::size_t new_size);

 public:
  
  class Transaction {
   private:
    PrimaryHashIndex& parent_;
    Lock& lock_;
    const HashType hash_;

   public:
    Transaction(PrimaryHashIndex& parent, Lock& lock, const HashType hash)
      : parent_{parent}, lock_{lock}, hash_{hash} {}

    ~Transaction() {
      lock_.unlock();
    }

    T* get(const T& key) {
      IdxNode* curr_idx = parent_.buckets_ + (hash_ & parent_.index_mask_);
      if (curr_idx->obj == nullptr) return nullptr;
      do {
        if (hash_ == curr_idx->hash && IDX_FN::equals(key, *curr_idx->obj)) {
          return curr_idx->obj;
        }
        curr_idx = curr_idx->next;
      } while (curr_idx != nullptr);
      return nullptr;
    }
    
    void insert(T* obj) {
      if (parent_.entry_count_ > parent_.threshold_) { parent_.resize_lockless(parent_.bucket_count_ << 1); }

      IdxNode& head = parent_.buckets_[hash_ & parent_.index_mask_];
      if (head.obj == nullptr) {
        head.obj = obj;
        head.hash = hash_;
      } else {
        IdxNode* new_node = parent_.pool_->acquire();
        new_node->obj = obj;
        new_node->hash = hash_;
        new_node->next = head.next;
        head.next = new_node;
      }
      #pragma omp atomic update
      parent_.entry_count_ += 1;
    }
    
    void erase(const T* obj) {
      assert(obj);
      IdxNode& head = parent_.buckets_[hash_ & parent_.index_mask_];
      if (head.obj == nullptr) return;
      if (head.obj == obj) {
        IdxNode* curr = head.next;
        if (curr != nullptr) {
          head.obj = curr->obj;
          head.hash = curr->hash;
          head.next = curr->next;
          parent_.pool_->release(curr);
        } else {
          head.obj = {};
        }
        #pragma omp atomic update
        parent_.entry_count_ -= 1;
      } else {
        IdxNode** curr = &head.next;
        while (*curr != nullptr) {
          IdxNode*& next = (*curr)->next;
          if ((*curr)->obj == obj) { // * comparison sufficient
            parent_.pool_->release(*curr);
            *curr = next;
            #pragma omp atomic update
            parent_.entry_count_ -= 1;
            break;
          }
          curr = &next;
        }
      }
    }

  };
  
  PrimaryHashIndex(std::size_t init_size = kDefaultChunkSize, double load_factor = 0.75)
        : pool_(Singleton<IdxNodeMemoryPool>().acquire()),
        buckets_(nullptr), bucket_count_(0), entry_count_(0),
        index_mask_(0), threshold_(0), load_factor_(load_factor), lock_{} {
    assert(isPowerOfTwo(init_size));
    resize_lockless(init_size);
  }

  ~PrimaryHashIndex();

  void clear();

  std::size_t size() const {
    std::size_t current_entry_count;
    #pragma omp atomic read
    current_entry_count = entry_count_;
    return current_entry_count;
  }

  HashType computeHash(const T& key) const { 
    return IDX_FN::hash(key); 
  }

  Transaction transaction(const HashType h) {
    lock_.lock();
    return {*this, lock_, h};
  }

  T* get(const T& key) const {
    return get(key, IDX_FN::hash(key));
  }

  // Returns the first matching element or nullptr if not found
  T* get(const T& key, const HashType h) const {
    IdxNode* curr_idx = buckets_ + (h & index_mask_);
    if (curr_idx == nullptr) return nullptr;
    do {
      if (h == curr_idx->hash && IDX_FN::equals(key, *curr_idx->obj)) {
        return curr_idx->obj;
      }
      curr_idx = curr_idx->next;
    } while (curr_idx != nullptr);
    return nullptr;
  }

  void insert(T* obj) {
    if (obj != nullptr) insert(obj, IDX_FN::hash(*obj)); 
  }

  // Inserts regardless of whether element already exists
  void insert(T* obj, const HashType h) {
    lock_.lock();
    insert_lockless(obj, h);
    lock_.unlock();
  }
  
  void insert_lockless(T* obj, const HashType h) {
    assert(obj);

    if (entry_count_ > threshold_) { resize_lockless(bucket_count_ << 1); }

    IdxNode& head = buckets_[h & index_mask_];
    if (head.obj != nullptr) {
      IdxNode* new_node = pool_->acquire();
      new_node->obj = obj;
      new_node->hash = h;
      new_node->next = head.next;
      head.next = new_node;
    } else {
      head.obj = obj;
      head.hash = h;
    }
    #pragma omp atomic update
    entry_count_ += 1;
  }

  void insert_lockless(IdxNode* handle) {
    assert(handle);

    if (entry_count_ > threshold_) { resize_lockless(bucket_count_ << 1); }

    IdxNode& head = buckets_[handle->hash & index_mask_];
    if (head.obj == nullptr) {
      head.obj = handle->obj;
      head.hash = handle->hash;
      pool_->release(handle);
    } else {
      handle->next = head.next;
      head.next = handle;
    }
    #pragma omp atomic update
    entry_count_ += 1;
  }

  void erase(const T* obj) {
    if (obj != nullptr) erase(obj, IDX_FN::hash(*obj));
  }

  void erase(const T* obj, const HashType h) {
    lock_.lock();
    erase_lockless(obj, h);
    lock_.unlock();
  }
  
  void erase_lockless(const T* obj, const HashType h) {
    assert(obj);
    IdxNode& head = buckets_[h & index_mask_];
    IdxNode** curr = &head.next;
    if (head.obj == nullptr) {
      return;
    }
    if (head.obj == obj) {
      IdxNode* curr = head.next;
      if (curr != nullptr) {
        head.obj = curr->obj;
        head.hash = curr->hash;
        head.next = curr->next;
        pool_->release(curr);
      } else {
        head.obj = {};
      }
      #pragma omp atomic update
      entry_count_ -= 1;
    } else {
      IdxNode** curr = &head.next;
      while (*curr) {
        IdxNode*& next = (*curr)->next;
        if ((*curr)->obj == obj) { // * comparison sufficient
          pool_->release(*curr);
          *curr = next;
          #pragma omp atomic update
          entry_count_ -= 1;
          break;
        }
        curr = &next;
      }
    }
  }

  template <typename U, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES> 
  friend class MultiHashMap;

};

template <typename T, typename IDX_FN>
PrimaryHashIndex<T, IDX_FN>::~PrimaryHashIndex() {
  clear();
  delete[] buckets_;
  buckets_ = nullptr;
  Singleton<IdxNodeMemoryPool>().release(pool_);
  pool_ = nullptr;
}

template <typename T, typename IDX_FN>
void PrimaryHashIndex<T, IDX_FN>::clear() {
  if (entry_count_ == 0) return;

  for (std::size_t i = 0; i < bucket_count_; ++i) {
    pool_->releaseChain(buckets_[i].next, [](const IdxNode* t) { return t->next; });
    buckets_[i] = {};
  }
  entry_count_ = 0;
}

template <typename T, typename IDX_FN>
void PrimaryHashIndex<T, IDX_FN>::resize_lockless(std::size_t new_size) {
  IdxNode* old_buckets = buckets_;
  std::size_t old_bucket_count = bucket_count_;
  buckets_ = new IdxNode[new_size];

  bucket_count_ = new_size;
  index_mask_ = bucket_count_ - 1;
  threshold_ = bucket_count_ * load_factor_;

  // Rehash entries
  entry_count_ = 0;
  for (std::size_t i = 0; i < old_bucket_count; ++i) {
    IdxNode& head = old_buckets[i];
    if (head.obj != nullptr) {
      insert_lockless(head.obj, head.hash);
      IdxNode* curr = head.next;
      while (curr) {
        IdxNode* next = curr->next;
        insert_lockless(curr);
        curr = next;
      }
    }
  }

  delete[] old_buckets;
}

template <typename T>
struct LinkedNodeBase {
  T* obj{nullptr};
  LinkedNodeBase* next{nullptr};
};

template <typename T, typename IDX_FN = T>
class SecondaryHashIndex {
 public:
 
  using LinkedNode = LinkedNodeBase<T>;

  struct IdxNode {
    LinkedNode node;
    HashType hash;
    IdxNode* next{nullptr};
  };
  
  using TMemoryPool = MemoryPool<T>;

  using IdxNodeMemoryPool = TrivialMemoryPool<IdxNode>;
 
  using LinkedNodeMemoryPool = TrivialMemoryPool<LinkedNode>;

 private:
  IdxNodeMemoryPool* idxNodePool_;
  LinkedNodeMemoryPool* linkedNodePool_;
  IdxNode* buckets_;
  std::size_t bucket_count_;
  std::size_t entry_count_;
  std::size_t index_mask_;   // derived value
  std::size_t threshold_;    // derived value
  double load_factor_;
  mutable Lock lock_;

  void resize_lockless(std::size_t new_size);
  void deleteBucket(IdxNode& bucket);

 public:
  SecondaryHashIndex(std::size_t size = kDefaultChunkSize, double load_factor = 0.75)
      : idxNodePool_(Singleton<IdxNodeMemoryPool>().acquire()),
        linkedNodePool_(Singleton<LinkedNodeMemoryPool>().acquire()),
        buckets_(nullptr), bucket_count_(0), entry_count_(0), index_mask_(0),
        threshold_(0), load_factor_(load_factor), lock_{} {
    resize_lockless(size);
  }

  ~SecondaryHashIndex();

  void clear();

  std::size_t size() const {
    std::size_t current_entry_count;
    #pragma omp atomic read
    current_entry_count = entry_count_;
    return current_entry_count;
  }

  LinkedNode* slice(const T& key) const {
    return slice(key, IDX_FN::hash(key));
  }

  // returns the first matching node or nullptr if not found
  // NOTE: Assumes that there is at least one linked node for each idx node
  LinkedNode* slice(const T& key, const HashType h) const {
    lock_.lock();
    auto result = slice_lockless(key, h);
    lock_.unlock();
    return result;
  }
  
  LinkedNode* slice_lockless(const T& key, const HashType h) const {
    IdxNode* head_idx = buckets_ + (h & index_mask_);
    IdxNode* curr_idx = head_idx->node.obj ? head_idx : nullptr;
    while (curr_idx != nullptr) {
      if (h == curr_idx->hash && IDX_FN::equals(key, *curr_idx->node.obj)) {
        return &curr_idx->node;
      }
      curr_idx = curr_idx->next ? &(*curr_idx->next) : nullptr;
    }
    return nullptr;
  }

  void insert(T* obj) {
    if (obj != nullptr) insert(obj, IDX_FN::hash(*obj));
  }

  // Inserts regardless of whether element already exists
  void insert(T* obj, const HashType h) {
    lock_.lock();
    insert_lockless(obj, h);
    lock_.unlock();
  }
  
  void insert_lockless(T* obj, const HashType h) {
    assert(obj);

    if (entry_count_ > threshold_) { resize_lockless(bucket_count_ << 1); }

    IdxNode& head = buckets_[h & index_mask_];

    LinkedNode* slice_node = slice(*obj, h);
    if (slice_node != nullptr) {
      LinkedNode* new_node = linkedNodePool_->acquire();
      new_node->obj = obj;
      new_node->next = slice_node->next;
      slice_node->next = new_node;
    } else {
      IdxNode& head = buckets_[h & index_mask_];
      if (head.node.obj != nullptr) {
        IdxNode* idx_node = idxNodePool_->acquire();
        idx_node->node.obj = obj;
        idx_node->node.next = {};
        idx_node->hash = h;
        idx_node->next = head.next;
        head.next = idx_node;
      } else {
        head.node.obj = obj;
        head.hash = h;
      }
      #pragma omp atomic update
      entry_count_ += 1;       // Count only distinct elements for non-unique index
    }
  }

  // NOTE: Assumes that the hash does not already exist in the map
  void insert_lockless(LinkedNode&& node, const HashType h) {

    IdxNode& head = buckets_[h & index_mask_];
    if (head.node.obj != nullptr) {
      IdxNode* idx_node = idxNodePool_->acquire();
      idx_node->next = head.next;
      idx_node->node = std::move(node);
      head.next = idx_node;
    } else {
      head.node = std::move(node);
      head.hash = h;
    }
    #pragma omp atomic update
    entry_count_ += 1;
  }

  void insert_lockless(IdxNode* handle) {
    assert(handle);

    IdxNode& head = buckets_[handle->hash & index_mask_];
    if (head.node.obj != nullptr) {
      handle->next = head.next;
      head.next = handle;
    } else {
      head.node = std::move(handle->node);
      head.hash = handle->hash;
      idxNodePool_->release(handle);
    }
    #pragma omp atomic update
    entry_count_ += 1;
  }

  void erase(const T* obj) {
    if (obj != nullptr) erase(obj, IDX_FN::hash(*obj));
  }

  // Deletes an existing element
  void erase(const T* obj, const HashType h) {
    lock_.lock();
    erase_lockless(obj, h);
    lock_.unlock();
  }
  
  void erase_lockless(const T* obj, const HashType h) {
    assert(obj);

    IdxNode* head_idx = buckets_ + (h & index_mask_);
    IdxNode* curr_idx = head_idx->node.obj ? head_idx : nullptr;
    IdxNode** curr_idx_handle;
    while (curr_idx != nullptr) {
      if (curr_idx->hash == h) {
        LinkedNode* head_node = &curr_idx->node;
        LinkedNode* curr_node = head_node;
        LinkedNode** curr_node_handle = nullptr;
        while (curr_node != nullptr) {
          if (IDX_FN::equals(*obj, *curr_node->obj)) {

            // Case: linked node is in a chain
            if (head_node != curr_node) {
              LinkedNode* next_handle = curr_node->next;
              linkedNodePool_->release(*curr_node_handle);
              *curr_node_handle = next_handle;

            // Case: linked node is head of chain, but idx node is not head of bucket
            } else if (head_idx != curr_idx) {
              IdxNode* next_handle = curr_idx->next;
              idxNodePool_->release(*curr_idx_handle);
              *curr_idx_handle = next_handle;
              #pragma omp atomic update
              entry_count_ -= 1;

            // Case linked node is head of chain and idx node is head of bucket
            } else {
              curr_idx->node.obj = {};
              if (curr_idx->next != nullptr) {
                IdxNode* next_next = curr_idx->next->next;
                T* next_obj = curr_idx->next->node.obj;
                LinkedNode* next_node = curr_idx->next->node.next;
                idxNodePool_->release(curr_idx->next);
                curr_idx->next = next_next;
                curr_idx->node.obj = next_obj;
                curr_idx->node.next = next_node;
              }
              #pragma omp atomic update
              entry_count_ -= 1;

            }
            return; // Erased from index
          }
          curr_node_handle = &curr_node->next;
          curr_node = curr_node->next ? &(*curr_node->next) : nullptr;
        }
        return; // Does not exist in index
      }
      curr_idx_handle = &curr_idx->next;
      curr_idx = curr_idx->next ? &(*curr_idx->next) : nullptr;
    }
  }
};

template <typename T, typename IDX_FN>
SecondaryHashIndex<T, IDX_FN>::~SecondaryHashIndex() {
  clear();
  delete[] buckets_;
  buckets_ = nullptr;
  Singleton<IdxNodeMemoryPool>().release(idxNodePool_);
  idxNodePool_ = nullptr;
  Singleton<LinkedNodeMemoryPool>().release(linkedNodePool_);
  linkedNodePool_ = nullptr;  
}

template <typename T, typename IDX_FN>
void SecondaryHashIndex<T, IDX_FN>::resize_lockless(std::size_t new_size) {
  IdxNode* old_buckets = buckets_;
  std::size_t old_bucket_count = bucket_count_;
  buckets_ = new IdxNode[new_size];

  bucket_count_ = new_size;
  index_mask_ = bucket_count_ - 1;
  threshold_ = bucket_count_ * load_factor_;
  
  entry_count_ = 0;
  for (std::size_t i = 0; i < old_bucket_count; ++i) {
    IdxNode& head = old_buckets[i];
    if (head.node.obj != nullptr) {
      insert_lockless(std::move(head.node), head.hash);
      IdxNode* curr = head.next;
      while (curr) {
        IdxNode* next = curr->next;
        insert_lockless(curr);
        curr = next;
      }
    }
  }

  delete[] old_buckets;
}

template <typename T, typename IDX_FN>
void SecondaryHashIndex<T, IDX_FN>::deleteBucket(IdxNode& head) {
  linkedNodePool_->releaseChain(head.node.next, [](const LinkedNode* t) { return t->next; });
  IdxNode* curr = head.next;
  while (curr) {
    IdxNode* next = curr->next;
    linkedNodePool_->releaseChain(curr->node.next, [](const LinkedNode* t) { return t->next; });    
    idxNodePool_->release(curr);
    curr = next;
  }
  head.node = {};
  head.next = {};
}

template <typename T, typename IDX_FN>
void SecondaryHashIndex<T, IDX_FN>::clear() {
  if (entry_count_ == 0) return;

  for (std::size_t i = 0; i < bucket_count_; ++i) {
    deleteBucket(buckets_[i]);
  }
  entry_count_ = 0;
}

template <typename T, typename... SECONDARY_INDEXES>
struct SecondaryIndexList;

template <typename T, typename SECONDARY_INDEX, typename... SECONDARY_INDEXES>
struct SecondaryIndexList<T, SECONDARY_INDEX, SECONDARY_INDEXES...> {
  SECONDARY_INDEX index;
  SecondaryIndexList<T, SECONDARY_INDEXES...> next;

  SecondaryIndexList() = default;

  SecondaryIndexList(std::size_t init_capacity) : index{init_capacity}, next{init_capacity} {}

  typename SECONDARY_INDEX::LinkedNode* slice(const T& key, std::size_t idx) const {
    if (idx == 0) {
      return index.slice(key);
    } else {
      return next.slice(key, idx - 1);
    }
  }

  template <typename H>
  void insert(H obj) {
    index.insert(obj);
    next.insert(obj);
  }

  template <typename H>
  void erase(const H obj) {
    index.erase(obj);
    next.erase(obj);
  }

  void clear() {
    index.clear();
    next.clear();
  }
};

template <typename T, typename SECONDARY_INDEX>
struct SecondaryIndexList<T, SECONDARY_INDEX> {
  SECONDARY_INDEX index;

  SecondaryIndexList() = default;

  SecondaryIndexList(std::size_t init_capacity) : index{init_capacity} {}

  typename SECONDARY_INDEX::LinkedNode* slice(const T& key, std::size_t idx) const {
    return index.slice(key);
  }

  template <typename H>
  void insert(H obj) {
    index.insert(obj);
  }

  template <typename H>
  void erase(const H obj) {
    index.erase(obj);
  }

  void clear() {
    index.clear();
  }
};

template <typename T>
struct SecondaryIndexList<T> {
  SecondaryIndexList() = default;

  SecondaryIndexList(std::size_t init_capacity) {}

  template <typename H>
  void insert(H obj) {}

  template <typename H>
  void erase(const H obj) {}

  void clear() {}
};

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES> 
class MultiHashMap {
 private:
  using TMemoryPool = MemoryPool<T>;

  TMemoryPool pool_;
  PRIMARY_INDEX primary_index_;
  SecondaryIndexList<T, SECONDARY_INDEXES...> secondary_indexes_;
  
  void insert(const T& elem, HashType h) {
    T* curr = pool_.acquire(elem);
    primary_index_.insert(curr, h);
    secondary_indexes_.insert(curr);
  }

  void erase(T* elem, HashType h) { // assume the element is already in the map and mainIdx=0
    assert(elem);    // and elem is in the map

    primary_index_.erase(elem, h);
    secondary_indexes_.erase(elem);
    pool_.release(elem);
  }

  void erase(const T& k) {
    HashType h = primary_index_.computeHash(k);
    primary_index_.get(k, h, [&] (auto elem) {
      erase(elem, h);
    }, [] {});
  }

  void insert(const T& k) {
    insert(k, primary_index_.computeHash(k));
  }

 public:

  MultiHashMap() = default;

  MultiHashMap(std::size_t init_capacity) : primary_index_{init_capacity}, secondary_indexes_{init_capacity} {}

  MultiHashMap(const MultiHashMap &) = delete;

  ~MultiHashMap();

  void clear();

  std::size_t size() const {
    return primary_index_.size();
  }

  const T* get(const T& key) const {
    return primary_index_.get(key);
  }

  const V& getValueOrDefault(const T& key) const {
    T* elem = primary_index_.get(key);
    return elem == nullptr ? Value<V>::zero : elem->__av;
  }
  
  template <typename F>
  void foreach(F f) const {
    pool_.foreach(f);
  }

  template <typename F>
  void slice(const T& k, std::size_t idx, F f) {
    auto* curr = secondary_indexes_.slice(k, idx);
    while (curr != nullptr) {
      T* elem = curr->obj;
      f(*elem);
      curr = curr->next ? &(*curr->next) : nullptr;
    }
  }    
  
  void add(T& k, const V& v) {
    if (Value<V>::isZero(v)) return;

    HashType h = primary_index_.computeHash(k);
    auto transaction = primary_index_.transaction(k);
    T* elem = transaction.get();
    if (elem != nullptr) {
      elem->__av += v; 
    } else {
      k.__av = v;
      T* elem = pool_.acquire(k);
      transaction.insert(elem);
      secondary_indexes_.insert(elem);
    }
  }

  void addOrDelOnZero(T& k, const V& v) {
    if (Value<V>::isZero(v)) return;

    HashType h = primary_index_.computeHash(k);
    auto transaction = primary_index_.transaction(h);
    T* elem = transaction.get(k);
    if (elem != nullptr) {
      elem->__av += v;
      if (Value<V>::isZero(elem->__av))  {
        transaction.erase(elem);
        pool_.release(elem);
      }
    } else {
      k.__av += v;
      T* elem = pool_.acquire(k);
      transaction.insert(elem);
      secondary_indexes_.insert(elem);
    }
  }

  void setOrDelOnZero(T& k, const V& v) {
    HashType h = primary_index_.computeHash(k);
    auto transaction = primary_index_.transaction();
    T* elem = transaction.get(k);
    if (elem != nullptr) {
      if (Value<V>::isZero(v)) {
        transaction.erase(elem);
        secondary_indexes_.erase(elem);
        pool_.release(elem);
      } else { 
        elem->__av = v; 
      }
    } else {
      if (!Value<V>::isZero(v)) {
        k.__av = v;
        T* elem = pool_.acquire(k);
        transaction.insert(elem);
        secondary_indexes_.insert(elem);
      }
    };
  }

  template <class Output>
  void serialize(Output &out) const;
};

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES> 
MultiHashMap<T, V, PRIMARY_INDEX, SECONDARY_INDEXES...>::~MultiHashMap() {
  clear();
}

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES> 
void MultiHashMap<T, V, PRIMARY_INDEX, SECONDARY_INDEXES...>::clear() {
  if (primary_index_.size() == 0) return;

  primary_index_.clear();
  secondary_indexes_.clear();
  pool_.releaseAll();
}

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES>
template <class Output>
void MultiHashMap<T, V, PRIMARY_INDEX, SECONDARY_INDEXES...>::serialize(Output &out) const {
  out << "\n\t\t";
  dbtoaster::serialization::serialize(out, size(),  "count");
  pool_.foreach([&](const auto& elem) {
    out << "\n";
    dbtoaster::serialization::serialize(out, elem, "item", "\t\t");
  });
}

}

#endif /* DBTOASTER_MULTIMAP_HPP */
