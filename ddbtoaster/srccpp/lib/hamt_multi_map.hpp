#ifndef DBTOASTER_HAMT_MULTIMAP_HPP
#define DBTOASTER_HAMT_MULTIMAP_HPP

#include <climits>
#include <iostream>
#include <functional>
#include <string>
#include <cmath>
#include <type_traits>
#include "macro.hpp"
#include "types.hpp"
#include "serialization.hpp"
#include "singleton.hpp"
#include "pointer_sum.hpp"

#ifdef _OPENMP
#include "no_wait_memory_pool.hpp"
#include "no_wait_trivial_memory_pool.hpp"
#else
#include "uniform_memory_pool.hpp"
#include "trivial_uniform_memory_pool.hpp"
#endif

using namespace dbtoaster;

namespace dbtoaster {

#ifdef _OPENMP

template <typename T>
using MemoryPool = memory_pool::NoWaitMemoryPool<T>;

template <typename T>
using TrivialMemoryPool = memory_pool::NoWaitTrivialMemoryPool<T>;

#else

template <typename T>
using MemoryPool = memory_pool::UniformMemoryPool<T>;

template <typename T>
using TrivialMemoryPool = memory_pool::TrivialUniformMemoryPool<T>;

#endif

constexpr bool isPowerOfTwo(std::size_t v) {
  return v && ((v & (v - 1)) == 0);
}

template <std::size_t N>
struct PowerOfTwo {
  constexpr static const std::size_t value = 1 + PowerOfTwo<N / 2>::value;
};

template <>
struct PowerOfTwo<1> {
  constexpr static const std::size_t value = 0;
};

constexpr std::size_t kDefaultArraySize = 8;  // 2^N

template <typename T, typename IDX_FN = T, std::size_t N = kDefaultArraySize>
class PrimaryHashIndex {
 private:
  struct None {};
  struct Lock {};

  struct IdxNode {
    T obj;
    HashType hash;
    IdxNode* next;

    template <typename... Args>
    IdxNode(Args&&... args) : obj{args...} {}
  };

  struct HashArray {
    using Slot = typename std::aligned_storage<sizeof(PointerSum<None, IdxNode, Lock, HashArray>), alignof(PointerSum<None, IdxNode, Lock, HashArray>)>::type;
    Slot slots[N];
    HashArray() {
      #pragma omp simd
      for (std::size_t i = 0; i < N; ++i) {
        new (slots + i) PointerSum<None, IdxNode, Lock, HashArray>(static_cast<None*>(nullptr));
      }
    }
    PointerSum<None, IdxNode, Lock, HashArray>& operator[](std::size_t index) noexcept {
      return *reinterpret_cast<PointerSum<None, IdxNode, Lock, HashArray>*>(slots + index);
    }
  };

  struct alignas(64) Counter {
    std::size_t count{0};
  };

  using IdxNodeMemoryPool = MemoryPool<IdxNode>;
  using ArrayMemoryPool = TrivialMemoryPool<HashArray>;
  using PointerType = PointerSum<None, IdxNode, Lock, HashArray>;

  constexpr static const std::size_t INDEX_MASK = N - 1;
  constexpr static const std::size_t SHIFT = PowerOfTwo<N>::value;
  constexpr static const std::size_t MAX_SHIFT = std::numeric_limits<HashType>::digits - SHIFT;

  IdxNodeMemoryPool idxNodePool_;
  ArrayMemoryPool arrayPool_;
  HashArray array_;
  const std::size_t thread_count_;
  Counter* counters_;

 public:

  class Transaction {
   private:
    PrimaryHashIndex& parent_;
    PointerType* lock_;
    Counter& counter_;
    PointerType ptr_;
    std::size_t shift_;
    const HashType hash_;

   public:
    Transaction(PrimaryHashIndex& parent, PointerType* lock, PointerType ptr, Counter& counter, std::size_t shift, const HashType hash)
      : parent_{parent}, lock_{lock}, counter_{counter}, ptr_{ptr}, shift_{shift}, hash_{hash} {}

    ~Transaction() {
      *lock_ = ptr_;
    }

    T* get(const T& key) {
      PointerType* curr = &ptr_;
      if (curr->template contains<IdxNode>()) {
        IdxNode* node = curr->template get<IdxNode>();
        do {
          if (node->hash == hash_ && IDX_FN::equals(key, node->obj)) {
            return &node->obj;
          }
          node = node->next;
        } while (node != nullptr);
      }
      return nullptr;
    }

    template <typename... Args>
    T* insert(Args&&... args) {
      IdxNode* node = parent_.idxNodePool_.acquire(args...);
      node->hash = hash_;
      node->next = nullptr;
      while (shift_ < MAX_SHIFT) {
        switch (ptr_.tag()) {
          case PointerType::template tag_of<None>(): {
            counter_.count += 1;
            ptr_ = node;
            return &node->obj;
          }
          case PointerType::template tag_of<IdxNode>(): {
            IdxNode* displaced_node = ptr_.template get<IdxNode>();
            HashArray* next_array = parent_.arrayPool_.acquire();
            PointerType& displaced_dest = (*next_array)[(displaced_node->hash >> shift_) & INDEX_MASK];
            displaced_dest = displaced_node;
            ptr_ = next_array;
            [[fallthrough]];
          }
          case PointerType::template tag_of<HashArray>(): {
            HashArray* next_array = ptr_.template get<HashArray>();
            PointerType prev_ptr = ptr_;
            PointerType* prev_lock = lock_;
            lock_ = &(*next_array)[(hash_ >> shift_) & INDEX_MASK];
            ptr_ = *lock_;
            *prev_lock = prev_ptr;
            shift_ += SHIFT;
          }
        }
      }
      if (ptr_.template contains<IdxNode>()) {
        node->next = ptr_.template get<IdxNode>();
      }
      ptr_ = node;
      counter_.count += 1;
      return &node->obj;
    }

    void erase(T* obj) {
      if (ptr_.template contains<IdxNode>()) {
        IdxNode* head = ptr_.template get<IdxNode>();
        IdxNode** curr = &head;
        do {
          if ((*curr)->hash == hash_ && IDX_FN::equals(*obj, (*curr)->obj)) {
            IdxNode* to_release = *curr;
            if (*curr == head && head->next == nullptr) {
              ptr_ = static_cast<None*>(nullptr);
            } else {
              *curr = (*curr)->next;
              ptr_ = head;
            }
            parent_.idxNodePool_.release(to_release);
            counter_.count -= 1;
            return;
          }
          curr = &(*curr)->next;
        } while ((*curr) != nullptr);
      }
    }

  };

  PrimaryHashIndex() : thread_count_{omp_get_max_threads()} {
    counters_ = new Counter[thread_count_];
  }

  ~PrimaryHashIndex();

  void clear();
  void clear(PointerType);

  std::size_t size() const {
    std::size_t total_entry_count{0};
    #pragma omp simd
    for (std::size_t i = 0; i < thread_count_; ++i) {
      total_entry_count += counters_[i].count;
    }
    return total_entry_count;
  }

  HashType computeHash(const T& key) {
    return IDX_FN::hash(key);
  }

  Transaction transaction(const HashType h) {
    Counter& counter = counters_[omp_get_thread_num()];
    PointerType array = &array_;
    PointerType* curr = &array;
    std::size_t shift = 0;
    for (;;) {
      switch (curr->tag()) {
        case PointerType::template tag_of<HashArray>(): {
          HashArray* next_array = curr->template get<HashArray>();
          curr = &(*next_array)[(h >> shift) & INDEX_MASK];
          shift += SHIFT;
        }
        break;
        case PointerType::template tag_of<None>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (curr->template compare_and_swap<None>(locked)) {
            return {*this, curr, locked, counter, shift, h};
          }
        }
        break;
        case PointerType::template tag_of<IdxNode>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (curr->template compare_and_swap<IdxNode>(locked)) {
            return {*this, curr, locked, counter, shift, h};
          }
        }
        break;
      }
    }
  }


  T* get(const T& key) {
    return get(key, IDX_FN::hash(key));
  }

  // Returns the first matching element or nullptr if not found
  T* get(const T& key, const HashType h) {
    PointerType array = &array_;
    PointerType* curr = &array;
    std::size_t shift = 0;
    for (;;) {
      switch (curr->tag()) {
        case PointerType::template tag_of<None>(): {
          return nullptr;
        }
        case PointerType::template tag_of<IdxNode>(): {
          IdxNode* node = curr->template get<IdxNode>();
          do {
            if (node->hash == h && IDX_FN::equals(key, node->obj)) {
              return &node->obj;
            }
            node = node->next;
          } while (node != nullptr);
          return nullptr;
        }
        case PointerType::template tag_of<HashArray>(): {
          HashArray* next_array = curr->template get<HashArray>();
          curr = &(*next_array)[(h >> shift) & INDEX_MASK];
          shift += SHIFT;
          break;
        }
      }
    }
  }

  template <typename F>
  void foreach(F f) const {
    idxNodePool_.foreach([f] (const IdxNode& node) {
      f(node.obj);
    });
  }

  template <typename U, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES>
  friend class MultiHashMap;
};

template <typename T, typename IDX_FN, std::size_t N>
PrimaryHashIndex<T, IDX_FN, N>::~PrimaryHashIndex() {
  clear();
  delete[] counters_;
}

template <typename T, typename IDX_FN, std::size_t N>
void PrimaryHashIndex<T, IDX_FN, N>::clear() {
  
  idxNodePool_.releaseAll();
  arrayPool_.releaseAll();
  #pragma omp simd
  for (std::size_t i = 0; i < N; ++i) {
    array_[i] = static_cast<None*>(nullptr);
  }
  for (std::size_t i = 0; i < thread_count_; ++i) {
    counters_[i].count = 0;
  }
}

template <typename T>
struct LinkedNodeBase {
  T* obj;
  LinkedNodeBase* next;
};

template <typename T, typename IDX_FN = T, std::size_t N = kDefaultArraySize>
class SecondaryHashIndex {
 public:

  using LinkedNode = LinkedNodeBase<T>;

  struct None {};
  struct Lock {};

  struct IdxNode {
    LinkedNode node;
    HashType hash;
    IdxNode* next;
  };

  struct HashArray {
    using Slot = typename std::aligned_storage<sizeof(PointerSum<None, IdxNode, Lock, HashArray>), alignof(PointerSum<None, IdxNode, Lock, HashArray>)>::type;
    Slot slots[N];
    HashArray() {
      #pragma omp simd
      for (std::size_t i = 0; i < N; ++i) {
        new (slots + i) PointerSum<None, IdxNode, Lock, HashArray>(static_cast<None*>(nullptr));
      }
    }
    PointerSum<None, IdxNode, Lock, HashArray>& operator[](std::size_t index) noexcept {
      return *reinterpret_cast<PointerSum<None, IdxNode, Lock, HashArray>*>(slots + index);
    }
  };

  using IdxNodeMemoryPool = TrivialMemoryPool<IdxNode>;
  using LinkedNodeMemoryPool = TrivialMemoryPool<LinkedNode>;
  using ArrayMemoryPool = TrivialMemoryPool<HashArray>;
  using PointerType = PointerSum<None, IdxNode, Lock, HashArray>;

  void deleteBucket(IdxNode*);

 private:
  constexpr static const std::size_t INDEX_MASK = N - 1;
  constexpr static const std::size_t SHIFT = PowerOfTwo<N>::value;
  constexpr static const std::size_t MAX_SHIFT = std::numeric_limits<HashType>::digits - SHIFT;

  IdxNodeMemoryPool* idxNodePool_;
  LinkedNodeMemoryPool* linkedNodePool_;
  ArrayMemoryPool* arrayPool_;
  HashArray array_;
  std::size_t entry_count_;

 public:
  SecondaryHashIndex()
      : idxNodePool_(Singleton<IdxNodeMemoryPool>().acquire()),
        linkedNodePool_(Singleton<LinkedNodeMemoryPool>().acquire()),
        arrayPool_(Singleton<ArrayMemoryPool>().acquire()) {}

  ~SecondaryHashIndex();

  void clear();
  void clear(PointerType);

  std::size_t size() const noexcept {
    std::size_t count;
    #pragma omp atomic read
    count = entry_count_;
    return count;
  }

  LinkedNode* slice(const T& key) {
    return slice(key, IDX_FN::hash(key));
  }

  LinkedNode* slice(const T& key, const HashType h) {
    PointerType array = &array_;
    PointerType* curr = &array;
    std::size_t shift = 0;
    for (;;) {
      switch (curr->tag()) {
        case PointerType::template tag_of<None>(): {
          return nullptr;
        }
        case PointerType::template tag_of<IdxNode>(): {
          IdxNode* node = curr->template get<IdxNode>();
          do {
            if (node->hash == h && IDX_FN::equals(key, *node->node.obj)) {
              return &node->node;
            }
            node = node->next;
          } while (node != nullptr);
          return nullptr;
        }
        case PointerType::template tag_of<HashArray>(): {
          HashArray* next_array = curr->template get<HashArray>();
          curr = &(*next_array)[(h >> shift) & INDEX_MASK];
          shift += SHIFT;
        }
      }
    }
  }

  void insert(T* obj) {
    if (obj) insert(obj, IDX_FN::hash(*obj));
  }

  // Inserts regardless of whether element already exists
  void insert(T* obj, const HashType h) {

    PointerType array = &array_;
    PointerType* curr = &array;
    std::size_t shift = 0;
    while (shift < MAX_SHIFT) {
      switch (curr->tag()) {
        case PointerType::template tag_of<None>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (!curr->template compare_and_swap<None>(locked)) {
            break;
          }
          insert(locked, obj, h);
          *curr = locked;
          return;
        }
        case PointerType::template tag_of<IdxNode>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (!curr->template compare_and_swap<IdxNode>(locked)) {
            break;
          }
          IdxNode* displaced_node = locked.template get<IdxNode>();
          HashArray* next_array = arrayPool_->acquire();
          PointerType& displaced_dest = (*next_array)[(displaced_node->hash >> shift) & INDEX_MASK];
          displaced_dest = displaced_node;
          *curr = next_array;
          [[fallthrough]];
        }
        case PointerType::template tag_of<HashArray>(): {
          HashArray* next_array = curr->template get<HashArray>();
          curr = &(*next_array)[(h >> shift) & INDEX_MASK];
          shift += SHIFT;
        }
      }
    }
    PointerType locked{static_cast<Lock*>(nullptr)};
    do {
      curr->swap(locked);
    } while (locked.template contains<Lock>());
    insert(locked, obj, h);
    *curr = locked;
  }

  void insert(PointerType& ptr, T* obj, const HashType h) {
    IdxNode* head = nullptr;
    if (ptr.template contains<IdxNode>()) {
      head = ptr.template get<IdxNode>();
      IdxNode* curr = head;
      do {
        if (curr->hash == h && IDX_FN::equals(*obj, *curr->node.obj)) {
          LinkedNode* node = linkedNodePool_->acquire();
          node->obj = obj;
          node->next = curr->node.next;
          curr->node.next = node;
          return;
        }
        curr = curr->next;
      } while (curr != nullptr);
    }
    IdxNode* node = idxNodePool_->acquire();
    node->node.obj = obj;
    node->node.next = nullptr;
    node->hash = h;
    node->next = head;
    ptr = node;
    #pragma omp atomic update
    entry_count_ += 1;
  }

  void insert(IdxNode* elem) {
    PointerType array = &array_;
    PointerType* curr = &array;
    std::size_t shift = 0;
    while (shift < MAX_SHIFT) {
      switch (curr->tag()) {
        case PointerType::template tag_of<None>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (!curr->template compare_and_swap<None>(locked)) {
            break;
          }
          *curr = elem;
          #pragma omp atomic update
          entry_count_ += 1;
          return;
        }
        case PointerType::template tag_of<IdxNode>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (!curr->template compare_and_swap<IdxNode>(locked)) {
            break;
          }
          IdxNode* displaced_node = locked.template get<IdxNode>();
          HashArray* next_array = arrayPool_->acquire();
          PointerType& displaced_dest = (*next_array)[(displaced_node->hash >> shift) & INDEX_MASK];
          displaced_dest = displaced_node;
          *curr = next_array;
          break;
        }
        case PointerType::template tag_of<HashArray>(): {
          HashArray* next_array = curr->template get<HashArray>();
          curr = &(*next_array)[(elem->hash >> shift) & INDEX_MASK];
          shift += SHIFT;
        }
      }
    }
    PointerType locked{static_cast<Lock*>(nullptr)};
    do {
      curr->swap(locked);
    } while (locked.template contains<Lock>());
    if (locked.template contains<None>()) {
      *curr = elem;
    } 
    else if (locked.template contains<IdxNode>()) {
      elem->next = locked.template get<IdxNode>();
      *curr = elem;
    }
    *curr = locked;
  }

  void erase(const T* obj) {
    if (obj) erase(obj, IDX_FN::hash(*obj));
  }

  // Deletes an existing element
  void erase(const T* obj, const HashType h) {
    PointerType array = &array_;
    PointerType* curr = &array;
    std::size_t shift = 0;
    while (shift < MAX_SHIFT) {
      switch (curr->tag()) {
        case PointerType::template tag_of<None>(): {
          return;
        }
        case PointerType::template tag_of<IdxNode>(): {
          PointerType locked{static_cast<Lock*>(nullptr)};
          if (!curr->template compare_and_swap<IdxNode>(locked)) {
            continue;
          }
          erase(locked, obj, h);
          *curr = locked;
          break;
        }
        case PointerType::template tag_of<HashArray>(): {
          HashArray* next_array = curr->template get<HashArray>();
          curr = &(*next_array)[(h >> shift) & INDEX_MASK];
          shift += SHIFT;
        }
      }
    }
    if (curr->template contains<None>()) {
      return;
    }
    PointerType locked{static_cast<Lock*>(nullptr)};
    do {
      curr->swap(locked);
    } while (locked.template contains<Lock>());
    erase(locked, obj, h);
    *curr = locked;
  }

  void erase(PointerType& ptr, const T* obj, const HashType h) {
    IdxNode* head = ptr.template get<IdxNode>();
    IdxNode** curr = &head;
    do {
      IdxNode** next = &(*curr)->next;
      if ((*curr)->hash == h) {
        erase(&(*curr)->node, obj);
        if ((*curr)->node.obj == nullptr) {
          idxNodePool_->release(*curr);
          *curr = *next;
          #pragma omp atomic update
          entry_count_ -= 1;
          ptr = head;
          return;
        }
      }
      curr = next;
    } while (curr != nullptr);
  }

  void erase(LinkedNode* head, const T* obj) {
    LinkedNode** curr = &head;
    do {
      LinkedNode** next = &(*curr)->next;
      if (IDX_FN::equals(*obj, *(*curr)->obj)) {
        if ((*curr) == head) {
          if (head->next == nullptr) {
            head->obj = nullptr;
          } else {
            head->obj = (*next)->obj;
            head->next = (*next)->next;
            linkedNodePool_->release(*next);
          }
        } else {
          linkedNodePool_->release(*curr);
          *curr = *next;
        }
        return;
      }
      curr = next;
    } while (curr != nullptr);
  }

};

template <typename T, typename IDX_FN, std::size_t N>
SecondaryHashIndex<T, IDX_FN, N>::~SecondaryHashIndex() {
  clear();
  Singleton<IdxNodeMemoryPool>().release(idxNodePool_);
  Singleton<LinkedNodeMemoryPool>().release(linkedNodePool_);
  Singleton<ArrayMemoryPool>().release(arrayPool_);
  idxNodePool_ = nullptr;
  linkedNodePool_ = nullptr;
  arrayPool_ = nullptr;
}


template <typename T, typename IDX_FN, std::size_t N>
void SecondaryHashIndex<T, IDX_FN, N>::clear() {
  if (entry_count_ == 0) return;
  for (std::size_t i = 0; i < N; ++i) {
    clear(array_[i]);
    array_[i] = static_cast<None*>(nullptr);
  }
  entry_count_ = 0;
}

template <typename T, typename IDX_FN, std::size_t N>
void SecondaryHashIndex<T, IDX_FN, N>::clear(PointerType ptr) {
  if (ptr.template contains<IdxNode>()) {
    deleteBucket(ptr.template get<IdxNode>());
  } else if (ptr.template contains<HashArray>()) {
    HashArray* array = ptr.template get<HashArray>();
    for (std::size_t i = 0; i < N; ++i) {
      clear((*array)[i]);
    }
  }
}

template <typename T, typename IDX_FN, std::size_t N>
void SecondaryHashIndex<T, IDX_FN, N>::deleteBucket(IdxNode* head) {
  while (head != nullptr) {
    IdxNode* next = head->next;
    linkedNodePool_->releaseChain(head->node.next, [] (LinkedNode* node) { return node->next; });
    idxNodePool_->release(head);
    head = next;
  }
}

template <typename T, typename... SECONDARY_INDEXES>
struct SecondaryIndexList;

template <typename T, typename SECONDARY_INDEX, typename... SECONDARY_INDEXES>
struct SecondaryIndexList<T, SECONDARY_INDEX, SECONDARY_INDEXES...> {
  SECONDARY_INDEX index;
  SecondaryIndexList<T, SECONDARY_INDEXES...> next;

  SecondaryIndexList() = default;

  SecondaryIndexList(std::size_t init_capacity) : index{init_capacity}, next{init_capacity} {}

  typename SECONDARY_INDEX::LinkedNode* slice(const T& key, std::size_t idx) {
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

  typename SECONDARY_INDEX::LinkedNode* slice(const T& key, std::size_t) {
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

  SecondaryIndexList(std::size_t) {}

  template <typename H>
  void insert(H) {}

  template <typename H>
  void erase(const H) {}

  void clear() {}
};

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES>
class MultiHashMap {
 private:
  PRIMARY_INDEX primary_index_;
  SecondaryIndexList<T, SECONDARY_INDEXES...> secondary_indexes_;

 public:

  MultiHashMap() = default;

  MultiHashMap(std::size_t init_capacity) : primary_index_{init_capacity}, secondary_indexes_{init_capacity} {}

  MultiHashMap(const MultiHashMap &) = delete;

  ~MultiHashMap();

  void clear();

  std::size_t size() const {
    return primary_index_.size();
  }

  const T* get(const T& key) {
    return primary_index_.get(key);
  }

  const V& getValueOrDefault(const T& key) {
    T* elem = primary_index_.get(key);
    if (elem != nullptr) return elem->__av;
    return Value<V>::zero;
  }

  template <typename F>
  void foreach(F f) const {
    primary_index_.foreach(f);
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
    auto transaction = primary_index_.transaction(h);
    T* elem = transaction.get(k);
    if (elem != nullptr) {
      elem->__av += v;
    }
    else {
      k.__av = v;
      T* elem = transaction.insert(k);
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
      if (Value<V>::isZero(elem->__av)) {
        transaction.erase(elem);
        secondary_indexes_.erase(elem);
      }
    } else {
      k.__av = v;
      T* elem = transaction.insert(k);
      secondary_indexes_.insert(elem);
    }
  }

  void setOrDelOnZero(T& k, const V& v) {
    HashType h = primary_index_.computeHash(k);
    auto transaction = primary_index_.transaction(h);
    T* elem = transaction.get(k);
    if (elem != nullptr) {
      if (Value<V>::isZero(v)) {
        transaction.erase(elem);
        secondary_indexes_.erase(elem);
      } else {
        elem->__av = v;
      }
    } else if (!Value<V>::isZero(v)) {
      k.__av = v;
      T* elem = transaction.insert(k);
      secondary_indexes_.insert(elem);
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
}

template <typename T, typename V, typename PRIMARY_INDEX, typename... SECONDARY_INDEXES>
template <class Output>
void MultiHashMap<T, V, PRIMARY_INDEX, SECONDARY_INDEXES...>::serialize(Output &out) const {
  out << "\n\t\t";
  dbtoaster::serialization::serialize(out, size(),  "count");
  primary_index_.foreach([&](const auto& elem) {
    out << "\n";
    dbtoaster::serialization::serialize(out, elem, "item", "\t\t");
  });
}

}

#endif /* DBTOASTER_HAMT_MULTIMAP_HPP */
