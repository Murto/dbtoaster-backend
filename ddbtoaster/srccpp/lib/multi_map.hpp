#ifndef DBTOASTER_MULTIMAP_HPP
#define DBTOASTER_MULTIMAP_HPP

#include <iostream>
#include <functional>
#include <string>
#include <cmath>
#include "macro.hpp"
#include "types.hpp"
#include "serialization.hpp"
#include "uniform_memory_pool.hpp"
#include "trivial_uniform_memory_pool.hpp"
#include "singleton.hpp"

using namespace dbtoaster;

namespace dbtoaster {

template <typename T>
using MemoryPool = memory_pool::UniformMemoryPool<T>;

template <typename T>
using TrivialMemoryPool = memory_pool::TrivialUniformMemoryPool<T>;

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

  void resize(std::size_t new_size);

 public:
  PrimaryHashIndex(std::size_t init_size = kDefaultChunkSize, double load_factor = 0.75)
      : pool_(Singleton<IdxNodeMemoryPool>().acquire()),
        buckets_(nullptr), bucket_count_(0), entry_count_(0),
        index_mask_(0), threshold_(0), load_factor_(load_factor) {
    assert(isPowerOfTwo(init_size));
    resize(init_size);
  }

  ~PrimaryHashIndex();

  void clear();

  FORCE_INLINE std::size_t size() const {
    return entry_count_;
  }

  FORCE_INLINE HashType computeHash(const T& key) { 
    return IDX_FN::hash(key); 
  }

  FORCE_INLINE T* get(const T& key) const {
    return get(key, IDX_FN::hash(key));
  }

  // Returns the first matching element or nullptr if not found
  T* get(const T& key, const HashType h) const {
    IdxNode* head_idx = buckets_ + (h & index_mask_);
    IdxNode* curr_idx = head_idx->obj ? head_idx : nullptr;
    while (curr_idx != nullptr) {
      if (h == curr_idx->hash && IDX_FN::equals(key, *curr_idx->obj)) {
        return curr_idx->obj;
      }
      curr_idx = curr_idx->next ? &(*curr_idx->next) : nullptr;
    }
    return {};
  }

  FORCE_INLINE void insert(T* obj) {
    if (obj) insert(obj, IDX_FN::hash(*obj)); 
  }

  // Inserts regardless of whether element already exists
  FORCE_INLINE void insert(T* obj, const HashType h) {
    assert(obj);

    if (entry_count_ > threshold_) { resize(bucket_count_ << 1); }

    IdxNode& head = buckets_[h & index_mask_];
    if (head.obj) {
      IdxNode* new_node = pool_->acquire();
      new_node->obj = obj;
      new_node->hash = h;
      new_node->next = head.next;
      head.next = new_node;
    } else {
      head.obj = obj;
      head.hash = h;
    }
    ++entry_count_;
  }

  FORCE_INLINE void insert(IdxNode* handle) {
    assert(handle);

    if (entry_count_ > threshold_) { resize(bucket_count_ << 1); }

    IdxNode& head = buckets_[handle->hash & index_mask_];
    if (!head.obj) {
      head.obj = handle->obj;
      head.hash = handle->hash;
      pool_->release(handle);
    } else {
      handle->next = head.next;
      head.next = handle;
    }
    ++entry_count_;
  }

  void erase(const T* obj) {
    if (obj) erase(obj, IDX_FN::hash(*obj));
  }

  void erase(const T* obj, const HashType h) {
    assert(obj);
    IdxNode& head = buckets_[h & index_mask_];
    IdxNode** curr = &head.next;
    if (!head.obj) {
      return;
    }
    if (head.obj == obj) {
      if (*curr) {
        head.obj = (*curr)->obj;
        head.hash = (*curr)->hash;
        head.next = (*curr)->next;
        pool_->release(*curr);
      } else {
        head.obj = {};
      }
      --entry_count_;
    } else {
      while (*curr) {
        IdxNode*& next = (*curr)->next;
        if ((*curr)->obj == obj) { // * comparison sufficient
          pool_->release(*curr);
          *curr = next;
          --entry_count_;
          break;
        }
        curr = &next;
      }
    }
  }

  template <typename U, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES> 
  friend class MultiHashMap;

  void dumpStatistics(bool verbose = false) const;
  std::size_t bucketSize(std::size_t bucket_id) const;
  double avgEntriesPerBucket() const;
  double stdevEntriesPerBucket() const;
  std::size_t maxEntriesPerBucket() const;
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
void PrimaryHashIndex<T, IDX_FN>::resize(std::size_t new_size) {
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
    if (head.obj) {
      insert(head.obj, head.hash);
      IdxNode* curr = head.next;
      while (curr) {
        IdxNode* next = curr->next;
        insert(curr);
        curr = next;
      }
    }
  }

  delete[] old_buckets;
}

template <typename T, typename IDX_FN>
void PrimaryHashIndex<T, IDX_FN>::dumpStatistics(bool verbose) const {
  std::cout << "# of entries: " << entry_count_ << '\n';
  std::cout << "# of buckets: " << bucket_count_ << '\n';
  std::cout << "avg # of entries per bucket " << avgEntriesPerBucket() << '\n';
  std::cout << "stdev # of entries per bucket " << stdevEntriesPerBucket() << '\n';
  std::cout << "max # of entries per bucket " << maxEntriesPerBucket() << '\n';
  if (verbose) {
    for (std::size_t i = 0; i < bucket_count_; ++i) {
      std::cout << "bucket[" << i << "] = " << bucketSize(i) << '\n';  
    }
  }
  std::cout << std::flush;
}

template <typename T, typename IDX_FN>
std::size_t PrimaryHashIndex<T, IDX_FN>::bucketSize(std::size_t bucket_id) const {
  assert(0 <= bucket_id && bucket_id < bucket_count_);
  IdxNode& head = buckets_[bucket_id];
  if (!head.obj) {
    return 0;
  }
  std::size_t cnt = 1;
  IdxNode* n = head.next;
  while (n) {
    ++cnt;
    n = n->next;
  }
  return cnt;
}

template <typename T, typename IDX_FN>
double PrimaryHashIndex<T, IDX_FN>::avgEntriesPerBucket() const {
  return static_cast<double>(entry_count_) / bucket_count_;
}

template <typename T, typename IDX_FN>
double PrimaryHashIndex<T, IDX_FN>::stdevEntriesPerBucket() const {
  double avg = avgEntriesPerBucket();
  double sum = 0.0;
  for (std::size_t i = 0; i < bucket_count_; ++i) {
    std::size_t cnt = bucketSize(i);
    sum += (cnt - avg) * (cnt - avg);
  }
  return sqrt(sum / bucket_count_);
}

template <typename T, typename IDX_FN>
std::size_t PrimaryHashIndex<T, IDX_FN>::maxEntriesPerBucket() const {
  std::size_t max = 0;
  for (std::size_t i = 0; i < bucket_count_; ++i) {
    std::size_t cnt = bucketSize(i);
    if (cnt > max) { max = cnt; }
  }
  return max;
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

  void resize(std::size_t new_size);
  void deleteBucket(IdxNode& bucket);

 public:
  SecondaryHashIndex(std::size_t size = kDefaultChunkSize, double load_factor = 0.75)
      : idxNodePool_(Singleton<IdxNodeMemoryPool>().acquire()),
        linkedNodePool_(Singleton<LinkedNodeMemoryPool>().acquire()),
        buckets_(nullptr), bucket_count_(0), entry_count_(0), index_mask_(0),
        threshold_(0), load_factor_(load_factor) {
    resize(size);
  }

  virtual ~SecondaryHashIndex();

  void clear();

  FORCE_INLINE std::size_t size() const {
    return entry_count_;
  }

  LinkedNode* slice(const T& key) {
    return slice(key, IDX_FN::hash(key));
  }

  // returns the first matching node or nullptr if not found
  // NOTE: Assumes that there is at least one linked node for each idx node
  LinkedNode* slice(const T& key, const HashType h) const {
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

  FORCE_INLINE void insert(T* obj) {
    if (obj) insert(obj, IDX_FN::hash(*obj));
  }

  // Inserts regardless of whether element already exists
  FORCE_INLINE void insert(T* obj, const HashType h) {
    assert(obj);

    if (entry_count_ > threshold_) { resize(bucket_count_ << 1); }

    IdxNode& head = buckets_[h & index_mask_];

    LinkedNode* slice_node = slice(*obj, h);
    if (slice_node != nullptr) {
      LinkedNode* new_node = linkedNodePool_->acquire();
      new_node->obj = obj;
      new_node->next = slice_node->next;
      slice_node->next = new_node;
    } else {
      IdxNode& head = buckets_[h & index_mask_];
      if (head.node.obj) {
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
      ++entry_count_;       // Count only distinct elements for non-unique index
    }
  }

  // NOTE: Assumes that the hash does not already exist in the map
  FORCE_INLINE void insert(LinkedNode&& node, const HashType h) {
    if (entry_count_ > threshold_) { resize(bucket_count_ << 1); }

    IdxNode& head = buckets_[h & index_mask_];
    if (head.node.obj) {
      IdxNode* idx_node = idxNodePool_->acquire();
      idx_node->next = head.next;
      idx_node->node = std::move(node);
      head.next = idx_node;
    } else {
      head.node = std::move(node);
      head.hash = h;
    }
    ++entry_count_;
  }

  FORCE_INLINE void insert(IdxNode* handle) {
    assert(handle);

    if (entry_count_ > threshold_) { resize(bucket_count_ << 1); }

    IdxNode& head = buckets_[handle->hash & index_mask_];
    if (head.node.obj) {
      handle->next = head.next;
      head.next = handle;
    } else {
      head.node = std::move(handle->node);
      head.hash = handle->hash;
      idxNodePool_->release(handle);
    }
    ++entry_count_;
  }

  void erase(const T* obj) {
    if (obj) erase(obj, IDX_FN::hash(*obj));
  }

  // Deletes an existing element
  void erase(const T* obj, const HashType h) {
    assert(obj);

    std::cout << "ERASE" << std::endl;

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
              --entry_count_;

            // Case linked node is head of chain and idx node is head of bucket
            } else {
              curr_idx->node.obj = {};
              if (curr_idx->next) {
                IdxNode* next_next = curr_idx->next->next;
                T* next_obj = curr_idx->next->node.obj;
                LinkedNode* next_node = curr_idx->next->node.next;
                idxNodePool_->release(curr_idx->next);
                curr_idx->next = next_next;
                curr_idx->node.obj = next_obj;
                curr_idx->node.next = next_node;
              }
              --entry_count_;

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
void SecondaryHashIndex<T, IDX_FN>::resize(std::size_t new_size) {
  IdxNode* old_buckets = buckets_;
  std::size_t old_bucket_count = bucket_count_;
  buckets_ = new IdxNode[new_size];

  bucket_count_ = new_size;
  index_mask_ = bucket_count_ - 1;
  threshold_ = bucket_count_ * load_factor_;
  
  entry_count_ = 0;
  for (std::size_t i = 0; i < old_bucket_count; ++i) {
    IdxNode& head = old_buckets[i];
    if (head.node.obj) {
      insert(std::move(head.node), head.hash);
      IdxNode* curr = head.next;
      while (curr) {
        IdxNode* next = curr->next;
        insert(curr);
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

  FORCE_INLINE typename SECONDARY_INDEX::LinkedNode* slice(const T& key, std::size_t idx) {
    if (idx == 0) {
      return index.slice(key);
    } else {
      return next.slice(key, idx - 1);
    }
  }

  template <typename H>
  FORCE_INLINE void insert(H obj) {
    index.insert(obj);
    next.insert(obj);
  }

  template <typename H>
  FORCE_INLINE void erase(const H obj) {
    index.erase(obj);
    next.erase(obj);
  }

  FORCE_INLINE void clear() {
    index.clear();
    next.clear();
  }
};

template <typename T, typename SECONDARY_INDEX>
struct SecondaryIndexList<T, SECONDARY_INDEX> {
  SECONDARY_INDEX index;

  SecondaryIndexList() = default;

  SecondaryIndexList(std::size_t init_capacity) : index{init_capacity} {}

  FORCE_INLINE typename SECONDARY_INDEX::LinkedNode* slice(const T& key, std::size_t idx) {
    return index.slice(key);
  }

  template <typename H>
  FORCE_INLINE void insert(H obj) {
    index.insert(obj);
  }

  template <typename H>
  FORCE_INLINE void erase(const H obj) {
    index.erase(obj);
  }

  FORCE_INLINE void clear() {
    index.clear();
  }
};

template <typename T>
struct SecondaryIndexList<T> {
  SecondaryIndexList() = default;

  SecondaryIndexList(std::size_t init_capacity) {}

  template <typename H>
  FORCE_INLINE void insert(H obj) {}

  template <typename H>
  FORCE_INLINE void erase(const H obj) {}

  FORCE_INLINE void clear() {}
};

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES> 
class MultiHashMap {
 private:
  using TMemoryPool = MemoryPool<T>;

  TMemoryPool pool_;
  PRIMARY_INDEX primary_index_;
  SecondaryIndexList<T, SECONDARY_INDEXES...> secondary_indexes_;
  
  FORCE_INLINE void insert(const T& elem, HashType h) {
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
    T* elem = primary_index_.get(k, h);
    if (elem) erase(elem, h);
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

  FORCE_INLINE std::size_t size() const {
    return primary_index_.size();
  }

  FORCE_INLINE const T* get(const T& key) const {
    return primary_index_.get(key);
  }

  FORCE_INLINE const V& getValueOrDefault(const T& key) const {
    T* elem = primary_index_.get(key);
    if (elem) return elem->__av;
    return Value<V>::zero;
  }
  
  template <typename F>
  FORCE_INLINE void foreach(F f) const {
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
    T* elem = primary_index_.get(k, h);
    if (elem) { 
      elem->__av += v; 
    }
    else {
      k.__av = v;
      insert(k, h);
    }
  }

  FORCE_INLINE void addOrDelOnZero(T& k, const V& v) {
    if (Value<V>::isZero(v)) return;

    HashType h = primary_index_.computeHash(k);
    T* elem = primary_index_.get(k, h);
    if (elem) {
      elem->__av += v;
      if (Value<V>::isZero(elem->__av)) erase(elem, h);
    }
    else {
      k.__av = v;
      insert(k, h);
    }
  }

  FORCE_INLINE void setOrDelOnZero(T& k, const V& v) {
    HashType h = primary_index_.computeHash(k);
    T* elem = primary_index_.get(k, h);
    if (elem) {
      if (Value<V>::isZero(v)) { erase(elem, h); }
      else { elem->__av = v; }
    }
    else if (!Value<V>::isZero(v)) {
      k.__av = v;
      insert(k, h);
    }
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
