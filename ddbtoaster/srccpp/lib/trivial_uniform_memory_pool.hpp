#ifndef DBTOASTER_TRIVIAL_UNIFORM_MEMORY_POOL_HPP
#define DBTOASTER_TRIVIAL_UNIFORM_MEMORY_POOL_HPP

#include "macro.hpp"
#include <atomic>
#include <cassert>
#include <csignal>
#include <iostream>
#include <mutex>
#include <type_traits>

namespace dbtoaster {

namespace memory_pool {

template <typename T, std::size_t N = 128> // Should really use a named constant here
class TrivialUniformMemoryPool {
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
    alignas(Slot) char slots[N * sizeof(Slot)];

    Slot* operator[](std::size_t index) {
      return static_cast<Slot*>(static_cast<void*>(&slots[0])) + index;
    }
  };

  struct MetaChunkHeader {
    MetaChunkHeader* next;
    std::size_t count;
  };

  // Assumes serial execution
  TrivialUniformMemoryPool() = default;

  // Assumes serial execution
  ~TrivialUniformMemoryPool() {
    MetaChunkHeader* curr = lastMetaChunk_;
    while (curr != nullptr) {
      MetaChunkHeader* next = curr->next;
      std::free(curr);
      curr = next;
    }
  }

  // Assumes releases don't happen at the same time
  template <class... Args>
  FORCE_INLINE T* acquire(Args&&... args) {
    Slot* slot;
    if (freeSlots_ != nullptr) {
      slot = freeSlots_;
      freeSlots_ = freeSlots_->next;
    } else {
      if (lastChunk_ == nullptr || currentSlot_ >= N) {
        allocateChunk();
      }
      slot = (*lastChunk_)[currentSlot_];
      ++currentSlot_;
    }
    new (&slot->value) T(args...);
    return &slot->value;
  }

  // Assumes same value is never released twice and that all values are valid
  // Assumes acquires don't happen at the same time
  FORCE_INLINE void release(T* value) {
    Slot* slot = reinterpret_cast<Slot*>(value);
    slot->next = freeSlots_;
    freeSlots_ = slot;
  }

  // Assumes same value is never released twice and that all values are valid
  template <class NextFn>
  FORCE_INLINE void releaseChain(T* head, NextFn next_fn) {
    while (head) {
      T* next = next_fn(head);
      release(head);
      head = next;
    }
  }
  
  FORCE_INLINE void releaseAll() {
    while (lastChunk_ != nullptr) {
      Chunk* next = lastChunk_->next;
      lastChunk_->next = freeChunks_;
      freeChunks_ = lastChunk_;
      lastChunk_ = next;
    }
    freeSlots_ = nullptr;
    currentSlot_ = N;
  }



 private:

  Slot* freeSlots_{nullptr};
  Chunk* freeChunks_{nullptr};
  
  Chunk* lastChunk_{nullptr};
  MetaChunkHeader* lastMetaChunk_{nullptr};
  
  std::size_t currentSlot_{0};
  std::size_t currentChunk_{0};

  void allocateChunk();
};

// Assumes release is not called at the same time
template <typename T, std::size_t N>
void TrivialUniformMemoryPool<T, N>::allocateChunk() {
  // precondition: no available elements
  assert(freeSlots_ == nullptr);

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

#endif /* DBTOASTER_TRIVIAL_UNIFORM_MEMORY_POOL_HPP */
