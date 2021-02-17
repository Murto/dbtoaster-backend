#ifndef DBTOASTER_TS_TRIVIAL_UNIFORM_MEMORY_POOL_HPP
#define DBTOASTER_TS_TRIVIAL_UNIFORM_MEMORY_POOL_HPP

#include "lock_omp.hpp"
#include "macro.hpp"
#include <cassert>
#include <csignal>
#include <iostream>
#include <mutex>
#include <type_traits>

namespace dbtoaster {

namespace memory_pool {

template <typename T, std::size_t N = 128> // Should really use a named constant here
class TSTrivialUniformMemoryPool {
static_assert(N > 0, "Chunk size must be greater than 0");
 public:

  struct Slot {
    union {
      T value;
      Slot* next;
    };
  };

  class Chunk {
   public:
    Chunk* next{nullptr};
    typename std::aligned_storage<sizeof(Slot), alignof(Slot)>::type slots[N];

    Slot* operator[](std::size_t index) {
      return reinterpret_cast<Slot*>(slots + index);
    }
  };

  struct MetaChunkHeader {
    MetaChunkHeader* next;
    std::size_t count;
  };

  class Handle {
    friend class TSTrivialUniformMemoryPool;
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

    Slot* slot_{nullptr};
   private:
  };

  // Assumes serial execution
  TSTrivialUniformMemoryPool() = default;

  // Assumes serial execution
  ~TSTrivialUniformMemoryPool() {
    MetaChunkHeader* curr = lastMetaChunk_;
    while (curr != nullptr) {
      MetaChunkHeader* next = curr->next;
      std::free(curr);
      curr = next;
    }
  }

  FORCE_INLINE Handle acquire() {
    Slot* slot;

    // Free slot path (fast)
    {
      free_lock_.lock();
      if (freeSlots_ != nullptr) {
        slot = freeSlots_;
        freeSlots_ = freeSlots_->next;
        free_lock_.unlock();
        return {slot};
      }
      free_lock_.unlock();
    }

    // New slot path (slow)
    {
      std::unique_lock<std::mutex> lock{chunk_mutex_};
      if (lastChunk_ == nullptr || currentSlot_ >= N) {
        allocateChunk();
      }
      slot = (*lastChunk_)[currentSlot_];
      ++currentSlot_;
      return {slot};
    }
  }

  template <class... Args>
  FORCE_INLINE Handle acquire(Args&&... args) {
    Handle handle = acquire();
    *handle = T(std::forward<Args>(args)...);
    return handle;
  }

  // Assumes same handle is never released twice and that all handles are valid
  // Assumes acquires don't happen at the same time
  FORCE_INLINE void release(Handle handle) {
    Slot* slot = handle.slot_;
    assert(slot != nullptr);
    free_lock_.lock();
    slot->next = freeSlots_;
    freeSlots_ = slot;
    free_lock_.unlock();
  }

  // Assumes same handle is never released twice and that all handles are valid
  template <class NextFn>
  FORCE_INLINE void releaseChain(Handle head, NextFn next_fn) {
    while (head) {
      Handle next = next_fn(head);
      release(head);
      head = next;
    }
  }

 private:

  Slot* freeSlots_{nullptr};
  std::size_t currentSlot_{0};
  Chunk* lastChunk_{nullptr};
  
  std::size_t currentChunk_{0};
  MetaChunkHeader* lastMetaChunk_{nullptr};

  mutable Lock free_lock_;
  mutable std::mutex chunk_mutex_;

  void allocateChunk();
};

// Assumes release is not called at the same time
template <typename T, std::size_t N>
void TSTrivialUniformMemoryPool<T, N>::allocateChunk() {
  // precondition: no available elements
  
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

#endif /* DBTOASTER_TS_TRIVIAL_UNIFORM_MEMORY_POOL_HPP */
