#ifndef DBTOASTER_LOCK_HPP
#define DBTOASTER_LOCK_HPP

#include "macro.hpp"
#include <atomic>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dbtoaster {

class alignas(64) Lock {
 public:
  FORCE_INLINE Lock() {
    #ifdef _OPENMP
    omp_init_lock(&lock_);
    #endif
  }

  FORCE_INLINE ~Lock() { 
    #ifdef _OPENMP
    omp_destroy_lock(&lock_);
    #endif
  }

  FORCE_INLINE bool try_lock() {
    #ifdef _OPENMP
    return omp_test_lock(&lock_);
    #else
    return true;
    #endif
  }

  FORCE_INLINE void lock() {
    #ifdef _OPENMP
    omp_set_lock(&lock_);
    #endif
  }

  FORCE_INLINE void unlock() {
    #ifdef _OPENMP
    omp_unset_lock(&lock_);
    #endif
  }

 private:
  omp_lock_t lock_;
};

} // namespace dbtoaster

#endif // DBTOASTER_LOCK_HPP
