#ifndef DBTOASTER_TS_UNIFORM_MEMORY_POOL_HPP
#define DBTOASTER_TS_UNIFORM_MEMORY_POOL_HPP

#include "lock_omp.hpp"
#include "macro.hpp"
#include "singleton.hpp"
#include <algorithm>
#include <cassert>
#include <csignal>
#include <iostream>
#include <type_traits>
#include <omp.h>

namespace dbtoaster {

namespace memory_pool {

template <typename T, std::size_t N = 128> // Should really use a named constant here
class TSUniformMemoryPool {
static_assert(N > 0, "Chunk size must be greater than 0");
 public:
  struct Slot {
    union {
      T value;
      Slot* next;
    };
    bool used;
    Slot() noexcept {}
  };

  class alignas(64) Chunk {
   public:
    Chunk* next{nullptr};
    alignas(64) char slots[N * sizeof(Slot)];

    Slot* operator[](std::size_t index) {
      return reinterpret_cast<Slot*>(slots + index * sizeof(Slot));
    }
  };

  struct alignas(64) MetaChunkHeader {
    MetaChunkHeader* next;
    std::size_t count;
  };

  class Handle {
    friend class TSUniformMemoryPool;
   public:
    Handle() = default;

    Handle(Slot* slot) noexcept : slot_{slot} {}

    FORCE_INLINE operator bool() const noexcept {
      return slot_ != nullptr;
    }

    bool operator==(const Handle& other) const noexcept {
      return slot_ == other.slot_;
    }

    bool operator!=(const Handle& other) const noexcept {
      return slot_ != other.slot_;
    }

    FORCE_INLINE T& operator*() noexcept {
      return slot_->value;
    }

    FORCE_INLINE const T& operator*() const noexcept {
      return slot_->value;
    }

    FORCE_INLINE T* operator->() noexcept {
      return &slot_->value;
    }

    FORCE_INLINE const T* operator->() const noexcept {
      return &slot_->value;
    }

   private:
    Slot* slot_{nullptr};
  };

  TSUniformMemoryPool() {
    max_threads_ = omp_get_max_threads();
  }

  ~TSUniformMemoryPool() {

    // Destruct constructed values
    if (lastChunk_ != nullptr) {
      for (std::size_t i = 0; i < currentSlot_; ++i) {
        if ((*lastChunk_)[i]->used) {
          (*lastChunk_)[i]->value.~T();
        }
      }
      lastChunk_ = lastChunk_->next;
    }
    while (lastChunk_ != nullptr) {
      for (std::size_t i = 0; i < N; ++i) {
        if ((*lastChunk_)[i]->used) {
          (*lastChunk_)[i]->value.~T();
        }
      }
      lastChunk_ = lastChunk_->next;        
    }

    // Free metachunks
    while (lastMetaChunk_ != nullptr) {
      MetaChunkHeader* next = lastMetaChunk_->next;
      std::free(lastMetaChunk_);
      lastMetaChunk_ = next;
    }

    freeSlots_ = nullptr;
  }

  FORCE_INLINE T* acquire() {
    Slot* slot;

    // Free slot path (fast)
    {
      free_lock_.lock();
      if (freeSlots_ != nullptr) {
        slot = freeSlots_;
        freeSlots_ = freeSlots_->next;
        slot->used = true;
        new (&slot->value) T();
        free_lock_.unlock();
        return &slot->value;
      }
      free_lock_.unlock();
    }

    // New slot path (slow)
    {
      chunk_lock_.lock();
      if (lastChunk_ == nullptr || currentSlot_ >= N) {
        allocateChunk();
      }
      slot = (*lastChunk_)[currentSlot_];
      ++currentSlot_;
      chunk_lock_.unlock();
      slot->used = true;
      new (&slot->value) T();
      return &slot->value;
    }
  }

  template <class... Args>
  FORCE_INLINE T* acquire(Args&&... args) {
    T* value = acquire();
    *value = T(std::forward<Args>(args)...);
    return value;
  }

  // Assumes value is valid
  FORCE_INLINE void release(T* value) {
    Slot* slot = reinterpret_cast<Slot*>(value);
    assert(slot != nullptr);
    slot->used = false;
    slot->value.~T();
    free_lock_.lock();
    slot->next = freeSlots_;
    freeSlots_ = slot;
    free_lock_.unlock();
  }

  template <class NextFn>
  FORCE_INLINE void releaseChain(T* head, NextFn next_fn) {
    while (head) {
      T* next = next_fn(head);
      release(head);
      head = next;
    }
  }

  FORCE_INLINE void releaseAll() {
    if (lastChunk_ != nullptr) {
      for (std::size_t i = 0; i < currentSlot_; ++i) {
        if ((*lastChunk_)[i]->used) {
          (*lastChunk_)[i]->value.~T();
        }
      }
      Chunk* next = lastChunk_->next;
      lastChunk_->next = freeChunks_;
      freeChunks_ = lastChunk_;
      lastChunk_ = next;
    }
    while (lastChunk_ != nullptr) {
      for (std::size_t i = 0; i < N; ++i) {
        if ((*lastChunk_)[i]->used) {
          (*lastChunk_)[i]->value.~T();
        }
      }
      Chunk* next = lastChunk_->next;
      lastChunk_->next = freeChunks_;
      freeChunks_ = lastChunk_;
      lastChunk_ = next;
    }
    freeSlots_ = nullptr;
    currentSlot_ = N;
    chunkCount_ = 0;
  }

  template <class F>
  FORCE_INLINE void foreach(F f) const {
    Chunk* curr = lastChunk_;
    if (curr == nullptr) {
      return;
    }


    for (std::size_t i = 0; i < currentSlot_; ++i) {
      if ((*curr)[i]->used) {
        f((*curr)[i]->value);
      }
    }
    curr = curr->next;

    while (curr != nullptr) {
      #pragma omp task firstprivate(curr)
      {
        for (std::size_t i = 0; i < N; ++i) {
          if ((*curr)[i]->used) {
            f((*curr)[i]->value);
          }
        }
      }
      curr = curr->next;
    }

    /*
    while (curr != nullptr) {
      for (std::size_t i = 0; i < N; ++i) {
        if ((*curr)[i]->used) {
          f((*curr)[i]->value);
        }
      }
      curr = curr->next;
    }
    */

  }

 private:

  std::size_t max_threads_;
  
  Slot* freeSlots_{nullptr};
  Chunk* freeChunks_{nullptr};
 
  std::size_t chunkCount_{0};
  Chunk* lastChunk_{nullptr};
  MetaChunkHeader* lastMetaChunk_{nullptr};
  
  std::size_t currentSlot_{0};
  std::size_t currentChunk_{0};

  mutable Lock free_lock_;
  mutable Lock chunk_lock_;

  void allocateChunk();
};

template <typename T, std::size_t N>
void TSUniformMemoryPool<T, N>::allocateChunk() {
  // precondition: no available elements

  assert(freeSlots_ == nullptr);

  chunkCount_ += 1;

  // If a chunk was allocated in the past, then released, then use it
  if (freeChunks_ != nullptr) {
    Chunk* next = freeChunks_->next;
    freeChunks_->next = lastChunk_;
    lastChunk_ = freeChunks_;
    freeChunks_ = next;
    currentSlot_ = 0;
    return;
  }

  // Allocate new metachunk
  if (lastMetaChunk_ == nullptr || lastMetaChunk_->count == ++currentChunk_) {
    std::size_t new_count = lastMetaChunk_ == nullptr ? 1 : (lastMetaChunk_->count << 1);
    MetaChunkHeader* new_meta_chunk = static_cast<MetaChunkHeader*>(std::malloc(sizeof(MetaChunkHeader) + new_count * sizeof(Chunk)));
    if (new_meta_chunk == nullptr) {
      std::cerr << "Memory allocation error in memory pool" << std::endl;
      exit(-1);
    }
    new_meta_chunk->next = lastMetaChunk_;
    new_meta_chunk->count = new_count;
    lastMetaChunk_ = new_meta_chunk;
    currentChunk_ = 0;
  }

  // Allocate new chunk
  Chunk* new_chunk = static_cast<Chunk*>(static_cast<void*>(lastMetaChunk_ + 1)) + currentChunk_;
  new_chunk->next = lastChunk_;
  lastChunk_ = new_chunk;
  currentSlot_ = 0;

}

} // namespace memory_pool

} // namespace dbtoaster

#endif /* DBTOASTER_TS_UNIFORM_MEMORY_POOL_HPP */
