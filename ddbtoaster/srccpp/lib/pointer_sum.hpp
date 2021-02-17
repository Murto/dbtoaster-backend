#ifndef POINTER_SUM_HPP
#define POINTER_SUM_HPP

#include "macro.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>

namespace dbtoaster {

namespace {

template <std::size_t N, std::size_t M = 0>
struct Mask {
  constexpr static const std::size_t value = Mask<N / 2, (M * 2) | 1>::value;
};

template <std::size_t M>
struct Mask<0, M> {
  constexpr static const std::size_t value = M;
};

template <typename S, typename... Ts>
struct IndexOf;

template <typename S, typename T, typename... Ts>
struct IndexOf<S, T, Ts...> {
  constexpr static const std::size_t value = 1 + IndexOf<S, Ts...>::value;
};

template <typename S, typename... Ts>
struct IndexOf<S, S, Ts...> {
  constexpr static const std::size_t value = 0;
};

} // anonymous namespace

template <typename... Ts>
class PointerSum {
 private:
  static_assert(sizeof...(Ts) > 0, "Nonzero number of types must be specified");
  constexpr static const std::uintptr_t TAG_MASK = Mask<sizeof...(Ts) - 1>::value;
  constexpr static const std::uintptr_t PTR_MASK = ~TAG_MASK;
  std::uintptr_t ptr_;

 public:

  PointerSum() = delete;

  template <typename T>
  FORCE_INLINE PointerSum(T* ptr) noexcept : ptr_{(reinterpret_cast<std::uintptr_t>(ptr) & PTR_MASK) | IndexOf<T, Ts...>::value} {}

  template <typename T>
  FORCE_INLINE PointerSum& operator=(const PointerSum& other) {
    #pragma omp atomic write
    ptr_ = other.ptr_;
    return *this;
  }

  template <typename T>
  FORCE_INLINE PointerSum& operator=(T* ptr) noexcept {
    std::uintptr_t next_ptr = (reinterpret_cast<std::uintptr_t>(ptr) & PTR_MASK) | IndexOf<T, Ts...>::value;
    #pragma omp atomic write 
    ptr_ = next_ptr;
    return *this;
  }

  FORCE_INLINE std::uintptr_t tag() const noexcept {
    std::uintptr_t curr_ptr;
    #pragma omp atomic read 
    curr_ptr = ptr_;
    return curr_ptr & TAG_MASK;
  }

  template <typename T>
  FORCE_INLINE constexpr static std::uintptr_t tag_of() noexcept {
    return IndexOf<T, Ts...>::value;
  }

  FORCE_INLINE void swap(PointerSum& other) noexcept {
    std::uintptr_t temp = other.ptr_;
    #pragma omp atomic capture 
    { other.ptr_ = ptr_; ptr_ = temp; }
  }

  template <typename T>
  FORCE_INLINE bool compare_and_swap(PointerSum& other, bool weak = true) noexcept {
    std::uintptr_t curr_ptr;
    #pragma omp atomic read 
    curr_ptr = ptr_;
    if ((curr_ptr & TAG_MASK) != IndexOf<T, Ts...>::value) return false;
    std::uintptr_t next_ptr = other.ptr_;
    bool success = __atomic_compare_exchange(&ptr_, &curr_ptr, &next_ptr, weak, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
    other.ptr_ = curr_ptr;
    return success;
  }

  template <typename T>
  FORCE_INLINE bool contains() const noexcept {
    std::uintptr_t ptr;
    #pragma omp atomic read 
    ptr = ptr_;
    return (ptr & TAG_MASK) == IndexOf<T, Ts...>::value;
  }

  template <typename T>
  FORCE_INLINE T* get() noexcept {
    return reinterpret_cast<T*>(ptr_ & PTR_MASK);
  }

  template <typename T>
  FORCE_INLINE const T* get() const noexcept {
    std::uintptr_t ptr;
    #pragma omp atomic read 
    ptr = ptr_;
    std::uintptr_t actual = ptr & TAG_MASK;
    std::uintptr_t expected = IndexOf<T, Ts...>::value;
    assert(actual == expected);
    return reinterpret_cast<T*>(ptr & PTR_MASK);
  }

};

} // namespace dbtoaster

#endif // POINTER_SUM_HPP
