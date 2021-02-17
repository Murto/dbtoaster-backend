#ifndef DBTOASTER_COMPACT_VIRTUAL_MEMORY_POOL_HPP
#define DBTOASTER_COMPACT_VIRTUAL_MEMORY_POOL_HPP

#include "macro.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <type_traits>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

namespace dbtoaster {

namespace memory_pool {

template <typename T>
class CompactVirtualMemoryPool {
 public:
  struct Slot {
    union {
      T value;
      Slot* next;
    };
    bool used;
    Slot() noexcept {}
  };

  CompactVirtualMemoryPool(std::size_t reserve_size = PAGE_SIZE * 1000000) : reserve_size_{reserve_size} {
    slots_ = static_cast<Slot*>(mmap(nullptr, reserve_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0));
    if (slots_ == MAP_FAILED) {
      std::cerr << "Memory allocation error in memory pool" << std::endl;
      std::exit(-1);
    }
    current_slot_ = slots_;
  }

  ~CompactVirtualMemoryPool() {
    releaseAll();
    int status = munmap(slots_, reserve_size_);
    if (status == -1) {
      std::cerr << "Memory deallocation error in memory pool" << std::endl;
      std::exit(-1);
    }
  }

  template <class... Args>
  T* acquire(Args&&... args) {
    Slot* slot = allocateSlot();
    slot->used = true;
    new (&slot->value) T(args...);
    return &slot->value;
  }
  
  void release(T* elem) {
    Slot* slot = reinterpret_cast<Slot*>(elem);
    slot->used = false;
  }

  template <class NextFn>
  void releaseChain(T* head, NextFn next_fn) {
    while (head) {
      T* next = next_fn(head);
      release(head);
      head = next;
    }
  }

  void releaseAll() {
    for (Slot* slot = slots_; slot != current_slot_; ++slot) {
      if (slot->used) {
        slot->value.~T();
        slot->used = false;
      }
    }
    current_slot_ = slots_;
  }

  template <class F>
  void foreach(F f) const {
    #pragma omp parallel for schedule(runtime)
    for (Slot* slot = slots_; slot < current_slot_; ++slot) {
      if (slot->used) {
        f(slot->value);
      }
    }
  }

 private:

  // Page size on current operating system
  static const long PAGE_SIZE;

  // Size of reserved memory
  const std::size_t reserve_size_;

  // Beginning of reserved slots
  Slot* slots_;

  // Current slot for allocation
  Slot* current_slot_;

  Slot* allocateSlot();
};

template <typename T>
const long CompactVirtualMemoryPool<T>::PAGE_SIZE = sysconf(_SC_PAGE_SIZE);

template <typename T>
auto CompactVirtualMemoryPool<T>::allocateSlot() -> Slot* {

  Slot* slot;
  
  #pragma omp atomic capture
  { slot = current_slot_; ++current_slot_; }

  /*
  if (slot - slots_ >= reserve_size_) {
    std::cerr << "Memory allocation error in memory pool" << std::endl;
    exit(-1);
  }
  */
  
  return slot;
}

} // namespace memory_pool

} // namespace dbtoaster

#endif // DBTOASTER_COMPACT_VIRTUAL_MEMORY_POOL_HPP
