//
// NanoRT, single header only modern ray tracing kernel.
//

//
// Notes : The number of primitives are up to 2G. If you want to render large
// data, please split data into chunks(~ 2G prims) and use NanoSG scene graph
// library(`${nanort}/examples/nanosg`).
//

/*
The MIT License (MIT)

Copyright (c) 2015 - Present: Light Transport Entertainment Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef NANORT_H_
#define NANORT_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

// compiler macros
//
// NANORT_USE_CPP11_FEATURE : Enable C++11 feature
// NANORT_ENABLE_PARALLEL_BUILD : Enable parallel BVH build.
// NANORT_ENABLE_SERIALIZATION : Enable serialization feature for built BVH.
//
// Parallelized BVH build is supported on C++11 thread version.
// OpenMP version is not fully tested.
// thus turn off if you face a problem when building BVH in parallel.
// #define NANORT_ENABLE_PARALLEL_BUILD

// Some constants
#define kNANORT_MAX_STACK_DEPTH (512)
#define kNANORT_MIN_PRIMITIVES_FOR_PARALLEL_BUILD (2048)  // Reduced threshold for better parallelization
#define kNANORT_SHALLOW_DEPTH (5)  // Increased for better work distribution (2^5 = 32 subtrees)

#ifdef NANORT_USE_CPP11_FEATURE
// Assume C++11 compiler has thread support.
// In some situation (e.g. embedded system, JIT compilation), thread feature
// may not be available though...
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#define kNANORT_MAX_THREADS (256)

// Simple thread pool for BVH construction
class ThreadPool {
public:
  ThreadPool(size_t threads) : stop_(false) {
    for (size_t i = 0; i < threads; ++i) {
      workers_.emplace_back([this] {
        for (;;) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
          }
          task();
        }
      });
    }
  }

  template<class F>
  void enqueue(F&& f) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (stop_) return;
      tasks_.emplace(std::forward<F>(f));
    }
    condition_.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return tasks_.empty(); });
  }

  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_) {
      worker.join();
    }
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;
};

// Parallel build should work well for C++11 version, thus force enable it.
#ifndef NANORT_ENABLE_PARALLEL_BUILD
#define NANORT_ENABLE_PARALLEL_BUILD
#endif

#endif

namespace nanort {

//
// SIMD Optimizations for BVH traversal, intersection and build
//
// Usage:
//   - Automatically detects and enables SSE2, AVX2, or ARM NEON when available
//   - Compile with: -msse2, -mavx2, or -mfpu=neon flags for optimal performance  
//   - Disable with: #define NANORT_DISABLE_SIMD before including nanort.h
//   - Use SIMD-optimized functions: IntersectRayAABB_SIMD_*, TraverseSIMD_*
//   - Check active SIMD path with: NANORT_SIMD_PATH macro
//

// SIMD feature detection and configuration
// Users can disable SIMD optimizations by defining NANORT_DISABLE_SIMD
#ifndef NANORT_DISABLE_SIMD

  #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define NANORT_ENABLE_SIMD_X86
    #if defined(__SSE2__) || (defined(_MSC_VER) && _MSC_VER >= 1300)
      #define NANORT_ENABLE_SSE2
      #include <emmintrin.h>
    #endif
    #if defined(__AVX2__) || (defined(_MSC_VER) && _MSC_VER >= 1800)
      #define NANORT_ENABLE_AVX2
      #include <immintrin.h>
    #endif
  #elif defined(__ARM_NEON) || defined(__aarch64__)
    #define NANORT_ENABLE_NEON
    #include <arm_neon.h>
  #endif

  // Auto-selection of optimal SIMD path
  #if defined(NANORT_ENABLE_AVX2)
    #define NANORT_SIMD_PATH "AVX2"
    #define NANORT_SIMD_WIDTH 8
  #elif defined(NANORT_ENABLE_SSE2)
    #define NANORT_SIMD_PATH "SSE2"  
    #define NANORT_SIMD_WIDTH 4
  #elif defined(NANORT_ENABLE_NEON)
    #define NANORT_SIMD_PATH "NEON"
    #define NANORT_SIMD_WIDTH 4
  #else
    #define NANORT_SIMD_PATH "scalar"
    #define NANORT_SIMD_WIDTH 1
  #endif

#else
  // SIMD disabled - define fallback macros
  #define NANORT_SIMD_PATH "disabled"
  #define NANORT_SIMD_WIDTH 1
#endif // NANORT_DISABLE_SIMD

// ================================================================================
// SIMD-optimized ray-AABB intersection functions
// ================================================================================
#if !defined(NANORT_DISABLE_SIMD)

#ifdef NANORT_ENABLE_SSE2
// SSE2 version: test ray against 4 AABBs simultaneously
inline int IntersectRayAABB4_SSE2(
    const __m128 ray_org_x, const __m128 ray_org_y, const __m128 ray_org_z,
    const __m128 ray_inv_dir_x, const __m128 ray_inv_dir_y, const __m128 ray_inv_dir_z,
    const __m128 min_t, const __m128 max_t,
    const __m128 bbox_min_x, const __m128 bbox_min_y, const __m128 bbox_min_z,
    const __m128 bbox_max_x, const __m128 bbox_max_y, const __m128 bbox_max_z,
    __m128 *tmin_out, __m128 *tmax_out) {

  // Calculate intersection points
  const __m128 tmin_x = _mm_mul_ps(_mm_sub_ps(bbox_min_x, ray_org_x), ray_inv_dir_x);
  const __m128 tmax_x = _mm_mul_ps(_mm_sub_ps(bbox_max_x, ray_org_x), ray_inv_dir_x);
  const __m128 tmin_y = _mm_mul_ps(_mm_sub_ps(bbox_min_y, ray_org_y), ray_inv_dir_y);
  const __m128 tmax_y = _mm_mul_ps(_mm_sub_ps(bbox_max_y, ray_org_y), ray_inv_dir_y);
  const __m128 tmin_z = _mm_mul_ps(_mm_sub_ps(bbox_min_z, ray_org_z), ray_inv_dir_z);
  const __m128 tmax_z = _mm_mul_ps(_mm_sub_ps(bbox_max_z, ray_org_z), ray_inv_dir_z);

  // Handle ray direction signs (min/max swap)
  const __m128 real_tmin_x = _mm_min_ps(tmin_x, tmax_x);
  const __m128 real_tmax_x = _mm_max_ps(tmin_x, tmax_x);
  const __m128 real_tmin_y = _mm_min_ps(tmin_y, tmax_y);
  const __m128 real_tmax_y = _mm_max_ps(tmin_y, tmax_y);
  const __m128 real_tmin_z = _mm_min_ps(tmin_z, tmax_z);
  const __m128 real_tmax_z = _mm_max_ps(tmin_z, tmax_z);

  // Find overall tmin and tmax
  const __m128 tmin = _mm_max_ps(real_tmin_z, _mm_max_ps(real_tmin_y, _mm_max_ps(real_tmin_x, min_t)));
  const __m128 tmax = _mm_min_ps(real_tmax_z, _mm_min_ps(real_tmax_y, _mm_min_ps(real_tmax_x, max_t)));

  // Test for intersection: tmin <= tmax
  const __m128 hit_mask = _mm_cmple_ps(tmin, tmax);

  *tmin_out = tmin;
  *tmax_out = tmax;

  return _mm_movemask_ps(hit_mask);
}

// SIMD-optimized triangle intersection helpers
inline __m128 cross_product_x_sse2(const __m128 a_y, const __m128 a_z, const __m128 b_y, const __m128 b_z) {
  return _mm_sub_ps(_mm_mul_ps(a_y, b_z), _mm_mul_ps(a_z, b_y));
}

inline __m128 cross_product_y_sse2(const __m128 a_z, const __m128 a_x, const __m128 b_z, const __m128 b_x) {
  return _mm_sub_ps(_mm_mul_ps(a_z, b_x), _mm_mul_ps(a_x, b_z));
}

inline __m128 cross_product_z_sse2(const __m128 a_x, const __m128 a_y, const __m128 b_x, const __m128 b_y) {
  return _mm_sub_ps(_mm_mul_ps(a_x, b_y), _mm_mul_ps(a_y, b_x));
}
#endif

#ifdef NANORT_ENABLE_AVX2
// AVX2 version: test ray against 8 AABBs simultaneously
inline int IntersectRayAABB8_AVX2(
    const __m256 ray_org_x, const __m256 ray_org_y, const __m256 ray_org_z,
    const __m256 ray_inv_dir_x, const __m256 ray_inv_dir_y, const __m256 ray_inv_dir_z,
    const __m256 min_t, const __m256 max_t,
    const __m256 bbox_min_x, const __m256 bbox_min_y, const __m256 bbox_min_z,
    const __m256 bbox_max_x, const __m256 bbox_max_y, const __m256 bbox_max_z,
    __m256 *tmin_out, __m256 *tmax_out) {

  // Calculate intersection points
  const __m256 tmin_x = _mm256_mul_ps(_mm256_sub_ps(bbox_min_x, ray_org_x), ray_inv_dir_x);
  const __m256 tmax_x = _mm256_mul_ps(_mm256_sub_ps(bbox_max_x, ray_org_x), ray_inv_dir_x);
  const __m256 tmin_y = _mm256_mul_ps(_mm256_sub_ps(bbox_min_y, ray_org_y), ray_inv_dir_y);
  const __m256 tmax_y = _mm256_mul_ps(_mm256_sub_ps(bbox_max_y, ray_org_y), ray_inv_dir_y);
  const __m256 tmin_z = _mm256_mul_ps(_mm256_sub_ps(bbox_min_z, ray_org_z), ray_inv_dir_z);
  const __m256 tmax_z = _mm256_mul_ps(_mm256_sub_ps(bbox_max_z, ray_org_z), ray_inv_dir_z);

  // Handle ray direction signs
  const __m256 real_tmin_x = _mm256_min_ps(tmin_x, tmax_x);
  const __m256 real_tmax_x = _mm256_max_ps(tmin_x, tmax_x);
  const __m256 real_tmin_y = _mm256_min_ps(tmin_y, tmax_y);
  const __m256 real_tmax_y = _mm256_max_ps(tmin_y, tmax_y);
  const __m256 real_tmin_z = _mm256_min_ps(tmin_z, tmax_z);
  const __m256 real_tmax_z = _mm256_max_ps(tmin_z, tmax_z);

  // Find overall tmin and tmax
  const __m256 tmin = _mm256_max_ps(real_tmin_z, _mm256_max_ps(real_tmin_y, _mm256_max_ps(real_tmin_x, min_t)));
  const __m256 tmax = _mm256_min_ps(real_tmax_z, _mm256_min_ps(real_tmax_y, _mm256_min_ps(real_tmax_x, max_t)));

  // Test for intersection
  const __m256 hit_mask = _mm256_cmp_ps(tmin, tmax, _CMP_LE_OQ);

  *tmin_out = tmin;
  *tmax_out = tmax;

  return _mm256_movemask_ps(hit_mask);
}

// AVX2 triangle intersection helpers
inline __m256 cross_product_x_avx2(const __m256 a_y, const __m256 a_z, const __m256 b_y, const __m256 b_z) {
  return _mm256_sub_ps(_mm256_mul_ps(a_y, b_z), _mm256_mul_ps(a_z, b_y));
}

inline __m256 cross_product_y_avx2(const __m256 a_z, const __m256 a_x, const __m256 b_z, const __m256 b_x) {
  return _mm256_sub_ps(_mm256_mul_ps(a_z, b_x), _mm256_mul_ps(a_x, b_z));
}

inline __m256 cross_product_z_avx2(const __m256 a_x, const __m256 a_y, const __m256 b_x, const __m256 b_y) {
  return _mm256_sub_ps(_mm256_mul_ps(a_x, b_y), _mm256_mul_ps(a_y, b_x));
}
#endif

#ifdef NANORT_ENABLE_NEON
// ARM NEON version: test ray against 4 AABBs simultaneously
inline int IntersectRayAABB4_NEON(
    const float32x4_t ray_org_x, const float32x4_t ray_org_y, const float32x4_t ray_org_z,
    const float32x4_t ray_inv_dir_x, const float32x4_t ray_inv_dir_y, const float32x4_t ray_inv_dir_z,
    const float32x4_t min_t, const float32x4_t max_t,
    const float32x4_t bbox_min_x, const float32x4_t bbox_min_y, const float32x4_t bbox_min_z,
    const float32x4_t bbox_max_x, const float32x4_t bbox_max_y, const float32x4_t bbox_max_z,
    float32x4_t *tmin_out, float32x4_t *tmax_out) {

  // Calculate intersection points
  const float32x4_t tmin_x = vmulq_f32(vsubq_f32(bbox_min_x, ray_org_x), ray_inv_dir_x);
  const float32x4_t tmax_x = vmulq_f32(vsubq_f32(bbox_max_x, ray_org_x), ray_inv_dir_x);
  const float32x4_t tmin_y = vmulq_f32(vsubq_f32(bbox_min_y, ray_org_y), ray_inv_dir_y);
  const float32x4_t tmax_y = vmulq_f32(vsubq_f32(bbox_max_y, ray_org_y), ray_inv_dir_y);
  const float32x4_t tmin_z = vmulq_f32(vsubq_f32(bbox_min_z, ray_org_z), ray_inv_dir_z);
  const float32x4_t tmax_z = vmulq_f32(vsubq_f32(bbox_max_z, ray_org_z), ray_inv_dir_z);

  // Handle ray direction signs
  const float32x4_t real_tmin_x = vminq_f32(tmin_x, tmax_x);
  const float32x4_t real_tmax_x = vmaxq_f32(tmin_x, tmax_x);
  const float32x4_t real_tmin_y = vminq_f32(tmin_y, tmax_y);
  const float32x4_t real_tmax_y = vmaxq_f32(tmin_y, tmax_y);
  const float32x4_t real_tmin_z = vminq_f32(tmin_z, tmax_z);
  const float32x4_t real_tmax_z = vmaxq_f32(tmin_z, tmax_z);

  // Find overall tmin and tmax
  const float32x4_t tmin = vmaxq_f32(real_tmin_z, vmaxq_f32(real_tmin_y, vmaxq_f32(real_tmin_x, min_t)));
  const float32x4_t tmax = vminq_f32(real_tmax_z, vminq_f32(real_tmax_y, vminq_f32(real_tmax_x, max_t)));

  // Test for intersection
  const uint32x4_t hit_mask = vcleq_f32(tmin, tmax);

  *tmin_out = tmin;
  *tmax_out = tmax;

  // Convert mask to integer result
  static const uint32_t mask_values[4] = {1, 2, 4, 8};
  const uint32x4_t mask_shifts = vld1q_u32(mask_values);
  const uint32x4_t result_mask = vandq_u32(hit_mask, mask_shifts);
  
  uint32_t result[4];
  vst1q_u32(result, result_mask);
  return result[0] | result[1] | result[2] | result[3];
}

// NEON triangle intersection helpers
inline float32x4_t cross_product_x_neon(const float32x4_t a_y, const float32x4_t a_z, 
                                         const float32x4_t b_y, const float32x4_t b_z) {
  return vsubq_f32(vmulq_f32(a_y, b_z), vmulq_f32(a_z, b_y));
}

inline float32x4_t cross_product_y_neon(const float32x4_t a_z, const float32x4_t a_x,
                                         const float32x4_t b_z, const float32x4_t b_x) {
  return vsubq_f32(vmulq_f32(a_z, b_x), vmulq_f32(a_x, b_z));
}

inline float32x4_t cross_product_z_neon(const float32x4_t a_x, const float32x4_t a_y,
                                         const float32x4_t b_x, const float32x4_t b_y) {
  return vsubq_f32(vmulq_f32(a_x, b_y), vmulq_f32(a_y, b_x));
}
#endif // NANORT_ENABLE_NEON

#endif // !NANORT_DISABLE_SIMD

// ================================================================================
// SIMD utility functions for BVH build
// ================================================================================
#if !defined(NANORT_DISABLE_SIMD)

#ifdef NANORT_ENABLE_SSE2
// Compute AABBs for 4 primitives simultaneously
inline void ComputeAABB4_SSE2(const float* vertices, const unsigned int* faces, 
                               unsigned int base_idx, size_t vertex_stride_bytes,
                               __m128 *bmin_x, __m128 *bmin_y, __m128 *bmin_z,
                               __m128 *bmax_x, __m128 *bmax_y, __m128 *bmax_z) {
  const size_t vertex_stride_floats = vertex_stride_bytes / sizeof(float);
  
  // Load vertex data for 4 triangles (12 vertices total)
  __m128 v_x[3], v_y[3], v_z[3];
  
  for (unsigned int tri = 0; tri < 4U; tri++) {
    const unsigned int* face = &faces[3 * (base_idx + tri)];
    
    // Load triangle vertices
    for (unsigned int v = 0; v < 3U; v++) {
      const float* vertex = &vertices[face[v] * vertex_stride_floats];
      reinterpret_cast<float*>(&v_x[v])[tri] = vertex[0];
      reinterpret_cast<float*>(&v_y[v])[tri] = vertex[1]; 
      reinterpret_cast<float*>(&v_z[v])[tri] = vertex[2];
    }
  }
  
  // Find min/max for each triangle
  *bmin_x = _mm_min_ps(v_x[0], _mm_min_ps(v_x[1], v_x[2]));
  *bmax_x = _mm_max_ps(v_x[0], _mm_max_ps(v_x[1], v_x[2]));
  *bmin_y = _mm_min_ps(v_y[0], _mm_min_ps(v_y[1], v_y[2]));
  *bmax_y = _mm_max_ps(v_y[0], _mm_max_ps(v_y[1], v_y[2]));
  *bmin_z = _mm_min_ps(v_z[0], _mm_min_ps(v_z[1], v_z[2]));
  *bmax_z = _mm_max_ps(v_z[0], _mm_max_ps(v_z[1], v_z[2]));
}

// Compute centroids for 4 primitives simultaneously  
inline void ComputeCentroid4_SSE2(const __m128 bmin_x, const __m128 bmin_y, const __m128 bmin_z,
                                   const __m128 bmax_x, const __m128 bmax_y, const __m128 bmax_z,
                                   __m128 *centroid_x, __m128 *centroid_y, __m128 *centroid_z) {
  const __m128 half = _mm_set1_ps(0.5f);
  *centroid_x = _mm_mul_ps(_mm_add_ps(bmin_x, bmax_x), half);
  *centroid_y = _mm_mul_ps(_mm_add_ps(bmin_y, bmax_y), half);
  *centroid_z = _mm_mul_ps(_mm_add_ps(bmin_z, bmax_z), half);
}
#endif // NANORT_ENABLE_SSE2

#endif // !NANORT_DISABLE_SIMD

// RayType
typedef enum {
  RAY_TYPE_NONE = 0x0,
  RAY_TYPE_PRIMARY = 0x1,
  RAY_TYPE_SECONDARY = 0x2,
  RAY_TYPE_DIFFUSE = 0x4,
  RAY_TYPE_REFLECTION = 0x8,
  RAY_TYPE_REFRACTION = 0x10
} RayType;

#ifdef __clang__
#pragma clang diagnostic push
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif

#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#endif

// ----------------------------------------------------------------------------
// Small vector class useful for multi-threaded environment.
//
// stack_container.h
//
// Copyright (c) 2006-2008 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// This allocator can be used with STL containers to provide a stack buffer
// from which to allocate memory and overflows onto the heap. This stack buffer
// would be allocated on the stack and allows us to avoid heap operations in
// some situations.
//
// STL likes to make copies of allocators, so the allocator itself can't hold
// the data. Instead, we make the creator responsible for creating a
// StackAllocator::Source which contains the data. Copying the allocator
// merely copies the pointer to this shared source, so all allocators created
// based on our allocator will share the same stack buffer.
//
// This stack buffer implementation is very simple. The first allocation that
// fits in the stack buffer will use the stack buffer. Any subsequent
// allocations will not use the stack buffer, even if there is unused room.
// This makes it appropriate for array-like containers, but the caller should
// be sure to reserve() in the container up to the stack buffer size. Otherwise
// the container will allocate a small array which will "use up" the stack
// buffer.
template <typename T, size_t stack_capacity>
class StackAllocator : public std::allocator<T> {
 public:
  typedef T* pointer;
  typedef typename std::allocator<T>::size_type size_type;

  // Backing store for the allocator. The container owner is responsible for
  // maintaining this for as long as any containers using this allocator are
  // live.
  struct Source {
    Source() : used_stack_buffer_(false) {}

    // Casts the buffer in its right type.
    T *stack_buffer() { return reinterpret_cast<T *>(stack_buffer_); }
    const T *stack_buffer() const {
      return reinterpret_cast<const T *>(stack_buffer_);
    }

    //
    // IMPORTANT: Take care to ensure that stack_buffer_ is aligned
    // since it is used to mimic an array of T.
    // Be careful while declaring any unaligned types (like bool)
    // before stack_buffer_.
    //

    // The buffer itself. It is not of type T because we don't want the
    // constructors and destructors to be automatically called. Define a POD
    // buffer of the right size instead.
    char stack_buffer_[sizeof(T[stack_capacity])];

    // Set when the stack buffer is used for an allocation. We do not track
    // how much of the buffer is used, only that somebody is using it.
    bool used_stack_buffer_;
  };

  // Used by containers when they want to refer to an allocator of type U.
  template <typename U>
  struct rebind {
    typedef StackAllocator<U, stack_capacity> other;
  };

  // For the straight up copy c-tor, we can share storage.
  StackAllocator(const StackAllocator<T, stack_capacity> &rhs)
      : source_(rhs.source_) {}

  // ISO C++ requires the following constructor to be defined,
  // and std::vector in VC++2008SP1 Release fails with an error
  // in the class _Container_base_aux_alloc_real (from <xutility>)
  // if the constructor does not exist.
  // For this constructor, we cannot share storage; there's
  // no guarantee that the Source buffer of Ts is large enough
  // for Us.
  // TODO(Google): If we were fancy pants, perhaps we could share storage
  // iff sizeof(T) == sizeof(U).
  template <typename U, size_t other_capacity>
  StackAllocator(const StackAllocator<U, other_capacity> &other)
      : source_(NULL) {
    (void)other;
  }

  explicit StackAllocator(Source *source) : source_(source) {}

  // Actually do the allocation. Use the stack buffer if nobody has used it yet
  // and the size requested fits. Otherwise, fall through to the standard
  // allocator.
  pointer allocate(size_type n, void *hint = 0) {
    if (source_ != NULL && !source_->used_stack_buffer_ &&
        n <= stack_capacity) {
      source_->used_stack_buffer_ = true;
      return source_->stack_buffer();
    } else {
#if __cplusplus >= 201703L
      return std::allocator_traits<std::allocator<T>>::allocate(*this, n, hint);
#else
      return std::allocator<T>::allocate(n, hint);
#endif
    }
  }

  // Free: when trying to free the stack buffer, just mark it as free. For
  // non-stack-buffer pointers, just fall though to the standard allocator.
  void deallocate(pointer p, size_type n) {
    if (source_ != NULL && p == source_->stack_buffer())
      source_->used_stack_buffer_ = false;
    else
      std::allocator<T>::deallocate(p, n);
  }

 private:
  Source *source_;
};

// A wrapper around STL containers that maintains a stack-sized buffer that the
// initial capacity of the vector is based on. Growing the container beyond the
// stack capacity will transparently overflow onto the heap. The container must
// support reserve().
//
// WATCH OUT: the ContainerType MUST use the proper StackAllocator for this
// type. This object is really intended to be used only internally. You'll want
// to use the wrappers below for different types.
template <typename TContainerType, int stack_capacity>
class StackContainer {
 public:
  typedef TContainerType ContainerType;
  typedef typename ContainerType::value_type ContainedType;
  typedef StackAllocator<ContainedType, stack_capacity> Allocator;

  // Allocator must be constructed before the container!
  StackContainer() : allocator_(&stack_data_), container_(allocator_) {
    // Make the container use the stack allocation by reserving our buffer size
    // before doing anything else.
    container_.reserve(stack_capacity);
  }

  // Getters for the actual container.
  //
  // Danger: any copies of this made using the copy constructor must have
  // shorter lifetimes than the source. The copy will share the same allocator
  // and therefore the same stack buffer as the original. Use std::copy to
  // copy into a "real" container for longer-lived objects.
  ContainerType &container() { return container_; }
  const ContainerType &container() const { return container_; }

  // Support operator-> to get to the container. This allows nicer syntax like:
  //   StackContainer<...> foo;
  //   std::sort(foo->begin(), foo->end());
  ContainerType *operator->() { return &container_; }
  const ContainerType *operator->() const { return &container_; }

#ifdef UNIT_TEST
  // Retrieves the stack source so that that unit tests can verify that the
  // buffer is being used properly.
  const typename Allocator::Source &stack_data() const { return stack_data_; }
#endif

 protected:
  typename Allocator::Source stack_data_;
  unsigned char pad_[7];
  Allocator allocator_;
  ContainerType container_;

  // DISALLOW_EVIL_CONSTRUCTORS(StackContainer);
  StackContainer(const StackContainer &);
  void operator=(const StackContainer &);
};

// StackVector
//
// Example:
//   StackVector<int, 16> foo;
//   foo->push_back(22);  // we have overloaded operator->
//   foo[0] = 10;         // as well as operator[]
template <typename T, size_t stack_capacity>
class StackVector
    : public StackContainer<std::vector<T, StackAllocator<T, stack_capacity> >,
                            stack_capacity> {
 public:
  StackVector()
      : StackContainer<std::vector<T, StackAllocator<T, stack_capacity> >,
                       stack_capacity>() {}

  // We need to put this in STL containers sometimes, which requires a copy
  // constructor. We can't call the regular copy constructor because that will
  // take the stack buffer from the original. Here, we create an empty object
  // and make a stack buffer of its own.
  StackVector(const StackVector<T, stack_capacity> &other)
      : StackContainer<std::vector<T, StackAllocator<T, stack_capacity> >,
                       stack_capacity>() {
    this->container().assign(other->begin(), other->end());
  }

  StackVector<T, stack_capacity> &operator=(
      const StackVector<T, stack_capacity> &other) {
    this->container().assign(other->begin(), other->end());
    return *this;
  }

  // Vectors are commonly indexed, which isn't very convenient even with
  // operator-> (using "->at()" does exception stuff we don't want).
  T &operator[](size_t i) { return this->container().operator[](i); }
  const T &operator[](size_t i) const {
    return this->container().operator[](i);
  }
};

// ----------------------------------------------------------------------------

template <typename T = float>
class real3 {
 public:
  real3() {}
  real3(T x) {
    v[0] = x;
    v[1] = x;
    v[2] = x;
  }
  real3(T xx, T yy, T zz) {
    v[0] = xx;
    v[1] = yy;
    v[2] = zz;
  }
  explicit real3(const T *p) {
    v[0] = p[0];
    v[1] = p[1];
    v[2] = p[2];
  }

  inline T x() const { return v[0]; }
  inline T y() const { return v[1]; }
  inline T z() const { return v[2]; }

  real3 operator*(T f) const { return real3(x() * f, y() * f, z() * f); }
  real3 operator-(const real3 &f2) const {
    return real3(x() - f2.x(), y() - f2.y(), z() - f2.z());
  }
  real3 operator*(const real3 &f2) const {
    return real3(x() * f2.x(), y() * f2.y(), z() * f2.z());
  }
  real3 operator+(const real3 &f2) const {
    return real3(x() + f2.x(), y() + f2.y(), z() + f2.z());
  }
  real3 &operator+=(const real3 &f2) {
    v[0] += f2.x();
    v[1] += f2.y();
    v[2] += f2.z();
    return (*this);
  }
  real3 operator/(const real3 &f2) const {
    return real3(x() / f2.x(), y() / f2.y(), z() / f2.z());
  }
  real3 operator-() const { return real3(-x(), -y(), -z()); }
  T operator[](int i) const { return v[i]; }
  T &operator[](int i) { return v[i]; }

  T v[3];
  // T pad;  // for alignment (when T = float)
};

template <typename T>
inline real3<T> operator*(T f, const real3<T> &v) {
  return real3<T>(v.x() * f, v.y() * f, v.z() * f);
}

template <typename T>
inline real3<T> vneg(const real3<T> &rhs) {
  return real3<T>(-rhs.x(), -rhs.y(), -rhs.z());
}

template <typename T>
inline T vlength(const real3<T> &rhs) {
  return std::sqrt(rhs.x() * rhs.x() + rhs.y() * rhs.y() + rhs.z() * rhs.z());
}

template <typename T>
inline real3<T> vnormalize(const real3<T> &rhs) {
  real3<T> v = rhs;
  T len = vlength(rhs);
  if (std::fabs(len) > std::numeric_limits<T>::epsilon()) {
    T inv_len = static_cast<T>(1.0) / len;
    v.v[0] *= inv_len;
    v.v[1] *= inv_len;
    v.v[2] *= inv_len;
  }
  return v;
}

template <typename T>
inline real3<T> vcross(const real3<T> a, const real3<T> b) {
  real3<T> c;
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

template <typename T>
inline T vdot(const real3<T> a, const real3<T> b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline real3<T> vsafe_inverse(const real3<T> v) {
  real3<T> r;

#ifdef NANORT_USE_CPP11_FEATURE

  if (std::fabs(v[0]) < std::numeric_limits<T>::epsilon()) {
    r[0] = std::numeric_limits<T>::infinity() *
           std::copysign(static_cast<T>(1), v[0]);
  } else {
    r[0] = static_cast<T>(1.0) / v[0];
  }

  if (std::fabs(v[1]) < std::numeric_limits<T>::epsilon()) {
    r[1] = std::numeric_limits<T>::infinity() *
           std::copysign(static_cast<T>(1), v[1]);
  } else {
    r[1] = static_cast<T>(1.0) / v[1];
  }

  if (std::fabs(v[2]) < std::numeric_limits<T>::epsilon()) {
    r[2] = std::numeric_limits<T>::infinity() *
           std::copysign(static_cast<T>(1), v[2]);
  } else {
    r[2] = static_cast<T>(1.0) / v[2];
  }
#else

  if (std::fabs(v[0]) < std::numeric_limits<T>::epsilon()) {
    T sgn = (v[0] < static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1);
    r[0] = std::numeric_limits<T>::infinity() * sgn;
  } else {
    r[0] = static_cast<T>(1.0) / v[0];
  }

  if (std::fabs(v[1]) < std::numeric_limits<T>::epsilon()) {
    T sgn = (v[1] < static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1);
    r[1] = std::numeric_limits<T>::infinity() * sgn;
  } else {
    r[1] = static_cast<T>(1.0) / v[1];
  }

  if (std::fabs(v[2]) < std::numeric_limits<T>::epsilon()) {
    T sgn = (v[2] < static_cast<T>(0)) ? static_cast<T>(-1) : static_cast<T>(1);
    r[2] = std::numeric_limits<T>::infinity() * sgn;
  } else {
    r[2] = static_cast<T>(1.0) / v[2];
  }
#endif

  return r;
}

template <typename real>
inline const real *get_vertex_addr(const real *p, const size_t idx,
                                   const size_t stride_bytes) {
  return reinterpret_cast<const real *>(
      reinterpret_cast<const unsigned char *>(p) + idx * stride_bytes);
}

template <typename T = float>
class Ray {
 public:
  Ray()
      : min_t(static_cast<T>(0.0)),
        max_t(std::numeric_limits<T>::max()),
        type(RAY_TYPE_NONE) {
    org[0] = static_cast<T>(0.0);
    org[1] = static_cast<T>(0.0);
    org[2] = static_cast<T>(0.0);
    dir[0] = static_cast<T>(0.0);
    dir[1] = static_cast<T>(0.0);
    dir[2] = static_cast<T>(-1.0);
  }

  T org[3];           // must set
  T dir[3];           // must set
  T min_t;            // minimum ray hit distance.
  T max_t;            // maximum ray hit distance.
  unsigned int type;  // ray type

  // TODO(LTE): Align sizeof(Ray)
};

template <typename T = float>
class BVHNode {
 public:
  BVHNode() {}
  BVHNode(const BVHNode &rhs) {
    bmin[0] = rhs.bmin[0];
    bmin[1] = rhs.bmin[1];
    bmin[2] = rhs.bmin[2];
    flag = rhs.flag;

    bmax[0] = rhs.bmax[0];
    bmax[1] = rhs.bmax[1];
    bmax[2] = rhs.bmax[2];
    axis = rhs.axis;

    data[0] = rhs.data[0];
    data[1] = rhs.data[1];
  }

  BVHNode &operator=(const BVHNode &rhs) {
    bmin[0] = rhs.bmin[0];
    bmin[1] = rhs.bmin[1];
    bmin[2] = rhs.bmin[2];
    flag = rhs.flag;

    bmax[0] = rhs.bmax[0];
    bmax[1] = rhs.bmax[1];
    bmax[2] = rhs.bmax[2];
    axis = rhs.axis;

    data[0] = rhs.data[0];
    data[1] = rhs.data[1];

    return (*this);
  }

  ~BVHNode() {}

  T bmin[3];
  T bmax[3];

  int flag;  // 1 = leaf node, 0 = branch node
  int axis;

  // leaf
  //   data[0] = npoints
  //   data[1] = index
  //
  // branch
  //   data[0] = child[0]
  //   data[1] = child[1]
  unsigned int data[2];
};

template <class H>
class IntersectComparator {
 public:
  bool operator()(const H &a, const H &b) const { return a.t < b.t; }
};

/// BVH build option.
template <typename T = float>
struct BVHBuildOptions {
  T cost_t_aabb;
  unsigned int min_leaf_primitives;
  unsigned int max_tree_depth;
  unsigned int bin_size;
  unsigned int shallow_depth;
  unsigned int min_primitives_for_parallel_build;

  // Cache bounding box computation.
  // Requires more memory, but BVHbuild can be faster.
  bool cache_bbox;
  unsigned char pad[3];

  // Set default value: Taabb = 0.2
  BVHBuildOptions()
      : cost_t_aabb(static_cast<T>(0.2)),
        min_leaf_primitives(4),
        max_tree_depth(256),
        bin_size(64),
        shallow_depth(kNANORT_SHALLOW_DEPTH),
        min_primitives_for_parallel_build(
            kNANORT_MIN_PRIMITIVES_FOR_PARALLEL_BUILD),
        cache_bbox(false) {}
};

/// BVH build statistics.
class BVHBuildStatistics {
 public:
  unsigned int max_tree_depth;
  unsigned int num_leaf_nodes;
  unsigned int num_branch_nodes;
  float build_secs;

  // Set default value: Taabb = 0.2
  BVHBuildStatistics()
      : max_tree_depth(0),
        num_leaf_nodes(0),
        num_branch_nodes(0),
        build_secs(0.0f) {}
};

///
/// @brief BVH trace option.
///
class BVHTraceOptions {
 public:
  // Hit only for face IDs in indexRange.
  // This feature is good to mimic something like glDrawArrays()
  unsigned int prim_ids_range[2];

  // Prim ID to skip for avoiding self-intersection
  // -1 = no skipping
  unsigned int skip_prim_id;

  bool cull_back_face;
  
  // Profiling counters (mutable to allow modification in const methods)
  mutable unsigned long bbox_intersections;
  mutable unsigned long primitive_intersections;

  BVHTraceOptions() {
    prim_ids_range[0] = 0;
    prim_ids_range[1] = 0x7FFFFFFF;  // Up to 2G face IDs.

    skip_prim_id = static_cast<unsigned int>(-1);
    cull_back_face = false;
    bbox_intersections = 0;
    primitive_intersections = 0;
  }
  
  // Reset profiling counters
  void ResetCounters() const {
    bbox_intersections = 0;
    primitive_intersections = 0;
  }
};

///
/// @brief Bounding box.
///
template <typename T>
class BBox {
 public:
  real3<T> bmin;
  real3<T> bmax;

  BBox() {
    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<T>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<T>::max();
  }
};

///
/// @brief Hit class for traversing nodes.
///
/// Stores hit information of node traversal.
/// Node traversal is used for two-level ray tracing(efficient ray traversal of a scene hierarchy)
///
template <typename T>
class NodeHit {
 public:
  NodeHit()
      : t_min(std::numeric_limits<T>::max()),
        t_max(-std::numeric_limits<T>::max()),
        node_id(static_cast<unsigned int>(-1)) {}

  NodeHit(const NodeHit<T> &rhs) {
    t_min = rhs.t_min;
    t_max = rhs.t_max;
    node_id = rhs.node_id;
  }

  NodeHit &operator=(const NodeHit<T> &rhs) {
    t_min = rhs.t_min;
    t_max = rhs.t_max;
    node_id = rhs.node_id;

    return (*this);
  }

  ~NodeHit() {}

  T t_min;
  T t_max;
  unsigned int node_id;
};

///
/// @brief Comparator object for NodeHit.
///
/// Comparator object for finding nearest hit point in node traversal.
///
template <typename T>
class NodeHitComparator {
 public:
  inline bool operator()(const NodeHit<T> &a, const NodeHit<T> &b) {
    return a.t_min < b.t_min;
  }
};

///
/// @brief Bounding Volume Hierarchy acceleration.
///
/// BVHAccel is central part of ray tracing(ray traversal).
/// BVHAccel takes an input geometry(primitive) information and build a data structure
/// for efficient ray tracing(`O(log2 N)` in theory, where N is the number of primitive in the scene).
///
/// @tparam T real value type(float or double).
///
template <typename T>
class BVHAccel {
 public:
  BVHAccel() : pad0_(0) { (void)pad0_; }
  ~BVHAccel() {}

  ///
  /// Build BVH for input primitives.
  ///
  /// @tparam Prim Primitive(e.g. Triangle) accessor class.
  /// @tparam Pred Predicator(comparator class object for `Prim` class to find nearest hit point)
  ///
  /// @param[in] num_primitives The number of primitive.
  /// @param[in] p Primitive accessor class object.
  /// @param[in] pred Predicator object.
  ///
  /// @return true upon success.
  ///
  template <class Prim, class Pred>
  bool Build(const unsigned int num_primitives, const Prim &p, const Pred &pred,
             const BVHBuildOptions<T> &options = BVHBuildOptions<T>());

  ///
  /// Get statistics of built BVH tree. Valid after `Build()`
  ///
  /// @return BVH build statistics.
  ///
  BVHBuildStatistics GetStatistics() const { return stats_; }

#if defined(NANORT_ENABLE_SERIALIZATION)
  ///
  /// Dump built BVH to the file.
  ///
  bool Dump(const char *filename) const;
  bool Dump(FILE *fp) const;

  ///
  /// Load BVH binary
  ///
  bool Load(const char *filename);
  bool Load(FILE *fp);
#endif

  void Debug();

  ///
  /// @brief Traverse into BVH along ray and find closest hit point & primitive if
  /// found
  ///
  /// @tparam I Intersector class
  /// @tparam H Hit class
  ///
  /// @param[in] ray Input ray
  /// @param[in] intersector Intersector object. This object is called for each possible intersection of ray and BVH during traversal.
  /// @param[out] isect Intersection point information(filled when closest hit point was found)
  /// @param[in] options Traversal options.
  ///
  /// @return true if the closest hit point found.
  ///
  template <class I, class H>
  bool Traverse(const Ray<T> &ray, const I &intersector, H *isect,
                const BVHTraceOptions &options = BVHTraceOptions()) const;

#if 0
  /// Multi-hit ray traversal
  /// Returns `max_intersections` frontmost intersections
  template<class I, class H, class Comp>
  bool MultiHitTraverse(const Ray<T> &ray,
                        int max_intersections,
                        const I &intersector,
                        StackVector<H, 128> *isects,
                        const BVHTraceOptions &options = BVHTraceOptions()) const;
#endif

  ///
  /// List up nodes which intersects along the ray.
  /// This function is useful for two-level BVH traversal.
  /// See `examples/nanosg` for example.
  ///
  /// @tparam I Intersection class
  ///
  ///
  ///
  template <class I>
  bool ListNodeIntersections(const Ray<T> &ray, int max_intersections,
                             const I &intersector,
                             StackVector<NodeHit<T>, 128> *hits) const;

  const std::vector<BVHNode<T> > &GetNodes() const { return nodes_; }
  const std::vector<unsigned int> &GetIndices() const { return indices_; }

  ///
  /// Returns bounding box of built BVH.
  ///
  void BoundingBox(T bmin[3], T bmax[3]) const {
    if (nodes_.empty()) {
      bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<T>::max();
      bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<T>::max();
    } else {
      bmin[0] = nodes_[0].bmin[0];
      bmin[1] = nodes_[0].bmin[1];
      bmin[2] = nodes_[0].bmin[2];
      bmax[0] = nodes_[0].bmax[0];
      bmax[1] = nodes_[0].bmax[1];
      bmax[2] = nodes_[0].bmax[2];
    }
  }

  bool IsValid() const { return nodes_.size() > 0; }

 private:
#if defined(NANORT_ENABLE_PARALLEL_BUILD)
  typedef struct {
    unsigned int left_idx;
    unsigned int right_idx;
    unsigned int offset;
  } ShallowNodeInfo;

  // Used only during BVH construction
  std::vector<ShallowNodeInfo> shallow_node_infos_;

  /// Builds shallow BVH tree recursively.
  template <class P, class Pred>
  unsigned int BuildShallowTree(std::vector<BVHNode<T> > *out_nodes,
                                unsigned int left_idx, unsigned int right_idx,
                                unsigned int depth,
                                unsigned int max_shallow_depth, const P &p,
                                const Pred &pred);
#endif

  /// Builds BVH tree recursively.
  template <class P, class Pred>
  unsigned int BuildTree(BVHBuildStatistics *out_stat,
                         std::vector<BVHNode<T> > *out_nodes,
                         unsigned int left_idx, unsigned int right_idx,
                         unsigned int depth, const P &p, const Pred &pred);

  template <class I>
  bool TestLeafNode(const BVHNode<T> &node, const Ray<T> &ray,
                    const I &intersector, const BVHTraceOptions &options) const;

  template <class I>
  bool TestLeafNodeIntersections(
      const BVHNode<T> &node, const Ray<T> &ray, const int max_intersections,
      const I &intersector,
      std::priority_queue<NodeHit<T>, std::vector<NodeHit<T> >,
                          NodeHitComparator<T> > *isect_pq) const;

#if 0
  template<class I, class H, class Comp>
  bool MultiHitTestLeafNode(std::priority_queue<H, std::vector<H>, Comp> *isect_pq,
                            int max_intersections,
                            const BVHNode<T> &node, const Ray<T> &ray,
                            const I &intersector) const;
#endif

  std::vector<BVHNode<T> > nodes_;
  std::vector<unsigned int> indices_;  // max 4G triangles.
  std::vector<BBox<T> > bboxes_;
  BVHBuildOptions<T> options_;
  BVHBuildStatistics stats_;
  unsigned int pad0_;
};

template <typename T>
struct CWBVHNode {
  T bmin[3];
  T bmax[3];
  
  union {
    struct {
      unsigned int child_index;
      unsigned int num_children;
    } internal;
    
    struct {
      unsigned int primitive_count;
      unsigned int primitive_index;
    } leaf;
  } data;
  
  unsigned char meta_data;
  unsigned char num_children;
  unsigned char axis;
  unsigned char flags;
  
  bool IsLeaf() const { return (flags & 1) != 0; }
  void SetLeaf() { flags |= 1; }
  void SetInternal() { flags &= ~1; }
};

template <typename T>
struct CWBVHBuildOptions {
  unsigned int branching_factor;
  unsigned int min_leaf_primitives;
  unsigned int max_tree_depth;
  T compression_threshold;
  bool enable_compression;
  
  CWBVHBuildOptions()
      : branching_factor(8),
        min_leaf_primitives(4),
        max_tree_depth(64),
        compression_threshold(static_cast<T>(0.1)),
        enable_compression(true) {}
};

template <typename T>
class CWBVHAccel {
 public:
  CWBVHAccel() {}
  ~CWBVHAccel() {}

  template <class Prim, class Pred>
  bool Build(const unsigned int num_primitives, const Prim &p, const Pred &pred,
             const CWBVHBuildOptions<T> &options = CWBVHBuildOptions<T>());

  template <class I, class H>
  bool Traverse(const Ray<T> &ray, const I &intersector, H *isect,
               const BVHTraceOptions &options = BVHTraceOptions()) const;

#if !defined(NANORT_DISABLE_SIMD)
  #ifdef NANORT_ENABLE_SSE2
    template <class I, class H>
    bool TraverseSIMD_SSE2(const Ray<T> &ray, const I &intersector, H *isect,
                          const BVHTraceOptions &options = BVHTraceOptions()) const;
  #endif

  #ifdef NANORT_ENABLE_AVX2
    template <class I, class H>
    bool TraverseSIMD_AVX2(const Ray<T> &ray, const I &intersector, H *isect,
                          const BVHTraceOptions &options = BVHTraceOptions()) const;
  #endif
#endif // !NANORT_DISABLE_SIMD

  void BoundingBox(T bmin[3], T bmax[3]) const {
    if (nodes_.empty()) {
      bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<T>::max();
      bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<T>::max();
    } else {
      bmin[0] = nodes_[0].bmin[0];
      bmin[1] = nodes_[0].bmin[1];
      bmin[2] = nodes_[0].bmin[2];
      bmax[0] = nodes_[0].bmax[0];
      bmax[1] = nodes_[0].bmax[1];
      bmax[2] = nodes_[0].bmax[2];
    }
  }

  bool IsValid() const { return nodes_.size() > 0; }

 private:
  template <class Prim>
  unsigned int BuildWideTree(std::vector<CWBVHNode<T> > *out_nodes,
                            const std::vector<BVHNode<T> > &binary_nodes,
                            unsigned int node_index,
                            const Prim &p);

  template <class Prim>
  void CompressNode(CWBVHNode<T> *node, const Prim &p);

  template <class I>
  bool TestCWBVHLeafNode(const CWBVHNode<T> &node, const Ray<T> &ray,
                        const I &intersector, const BVHTraceOptions &options) const;

  std::vector<CWBVHNode<T> > nodes_;
  std::vector<unsigned int> indices_;
  CWBVHBuildOptions<T> options_;
};

template <typename T>
template <class Prim, class Pred>
bool CWBVHAccel<T>::Build(const unsigned int num_primitives, const Prim &p,
                         const Pred &pred, const CWBVHBuildOptions<T> &options) {
  options_ = options;
  nodes_.clear();
  indices_.clear();

  if (num_primitives == 0) {
    return false;
  }

  BVHAccel<T> binary_bvh;
  BVHBuildOptions<T> binary_options;
  binary_options.min_leaf_primitives = options.min_leaf_primitives;
  binary_options.max_tree_depth = options.max_tree_depth;
  
  if (!binary_bvh.Build(num_primitives, p, pred, binary_options)) {
    return false;
  }

  const std::vector<BVHNode<T> > &binary_nodes = binary_bvh.GetNodes();
  const std::vector<unsigned int> &binary_indices = binary_bvh.GetIndices();
  
  indices_ = binary_indices;

  BuildWideTree(&nodes_, binary_nodes, 0, p);

  if (options.enable_compression) {
    for (size_t i = 0; i < nodes_.size(); ++i) {
      CompressNode(&nodes_[i], p);
    }
  }

  return true;
}

template <typename T>
template <class Prim>
unsigned int CWBVHAccel<T>::BuildWideTree(std::vector<CWBVHNode<T> > *out_nodes,
                                          const std::vector<BVHNode<T> > &binary_nodes,
                                          unsigned int node_index,
                                          const Prim &p) {
  if (node_index >= binary_nodes.size()) {
    return static_cast<unsigned int>(-1);
  }

  const BVHNode<T> &binary_node = binary_nodes[node_index];
  unsigned int current_index = static_cast<unsigned int>(out_nodes->size());
  
  CWBVHNode<T> wide_node;
  wide_node.bmin[0] = binary_node.bmin[0];
  wide_node.bmin[1] = binary_node.bmin[1];
  wide_node.bmin[2] = binary_node.bmin[2];
  wide_node.bmax[0] = binary_node.bmax[0];
  wide_node.bmax[1] = binary_node.bmax[1];
  wide_node.bmax[2] = binary_node.bmax[2];
  wide_node.axis = static_cast<unsigned char>(binary_node.axis);
  wide_node.flags = 0;
  wide_node.meta_data = 0;

  if (binary_node.flag == 1) {
    wide_node.SetLeaf();
    wide_node.data.leaf.primitive_count = binary_node.data[0];
    wide_node.data.leaf.primitive_index = binary_node.data[1];
    wide_node.num_children = 0;
  } else {
    wide_node.SetInternal();
    
    std::vector<unsigned int> children;
    std::queue<unsigned int> to_process;
    to_process.push(binary_node.data[0]);
    to_process.push(binary_node.data[1]);
    
    while (!to_process.empty() && children.size() < options_.branching_factor) {
      unsigned int child_idx = to_process.front();
      to_process.pop();
      
      if (child_idx >= binary_nodes.size()) continue;
      
      const BVHNode<T> &child = binary_nodes[child_idx];
      
      if (child.flag == 1 || children.size() >= (options_.branching_factor - 2)) {
        children.push_back(child_idx);
      } else {
        to_process.push(child.data[0]);
        to_process.push(child.data[1]);
      }
    }
    
    wide_node.data.internal.child_index = static_cast<unsigned int>(out_nodes->size() + 1);
    wide_node.data.internal.num_children = static_cast<unsigned int>(children.size());
    wide_node.num_children = static_cast<unsigned char>(children.size());
  }

  out_nodes->push_back(wide_node);

  if (!wide_node.IsLeaf()) {
    std::vector<unsigned int> children;
    std::queue<unsigned int> to_process;
    to_process.push(binary_node.data[0]);
    to_process.push(binary_node.data[1]);
    
    while (!to_process.empty() && children.size() < options_.branching_factor) {
      unsigned int child_idx = to_process.front();
      to_process.pop();
      
      if (child_idx >= binary_nodes.size()) continue;
      
      const BVHNode<T> &child = binary_nodes[child_idx];
      
      if (child.flag == 1 || children.size() >= (options_.branching_factor - 2)) {
        children.push_back(child_idx);
      } else {
        to_process.push(child.data[0]);
        to_process.push(child.data[1]);
      }
    }
    
    for (unsigned int child_idx : children) {
      BuildWideTree(out_nodes, binary_nodes, child_idx, p);
    }
  }

  return current_index;
}

template <typename T>
template <class Prim>
void CWBVHAccel<T>::CompressNode(CWBVHNode<T> *node, const Prim &) {
  if (node->IsLeaf()) return;
  
  T extent[3] = {
    node->bmax[0] - node->bmin[0],
    node->bmax[1] - node->bmin[1],
    node->bmax[2] - node->bmin[2]
  };
  
  T max_extent = (std::max)(extent[0], extent[1]);
  max_extent = (std::max)(max_extent, extent[2]);
  
  if (max_extent < options_.compression_threshold) {
    node->meta_data |= 0x80;
  }
}

template <typename T>
template <class I, class H>
bool CWBVHAccel<T>::Traverse(const Ray<T> &ray, const I &intersector, H *isect,
                            const BVHTraceOptions &options) const {
  if (nodes_.empty()) return false;

  T hit_t = ray.max_t;
  
  std::vector<unsigned int> stack;
  stack.reserve(64);
  stack.push_back(0);

  // Init isect info as no hit
  intersector.Update(hit_t, static_cast<unsigned int>(-1));
  intersector.PrepareTraversal(ray, options);

  int dir_sign[3];
  dir_sign[0] = ray.dir[0] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[1] = ray.dir[1] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[2] = ray.dir[2] < static_cast<T>(0.0) ? 1 : 0;

  real3<T> ray_inv_dir;
  real3<T> ray_dir;
  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  ray_inv_dir = vsafe_inverse(ray_dir);

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  while (!stack.empty()) {
    unsigned int node_index = stack.back();
    stack.pop_back();

    if (node_index >= nodes_.size()) continue;

    const CWBVHNode<T> &node = nodes_[node_index];

    T min_t = std::numeric_limits<T>::max();
    T max_t = -std::numeric_limits<T>::max();

    bool bbox_hit = IntersectRayAABB(&min_t, &max_t, ray.min_t, hit_t, 
                                    node.bmin, node.bmax, ray_org, ray_inv_dir, dir_sign);

    // Count bounding box intersection tests
    options.bbox_intersections++;

    if (bbox_hit) {
      if (node.IsLeaf()) {
        if (TestCWBVHLeafNode(node, ray, intersector, options)) {
          hit_t = intersector.GetT();
        }
      } else {
        for (unsigned char i = 0; i < node.num_children; ++i) {
          stack.push_back(node.data.internal.child_index + i);
        }
      }
    }
  }

  bool hit = (intersector.GetT() < ray.max_t);
  intersector.PostTraversal(ray, hit, isect);

  return hit;
}

// ================================================================================
// SIMD-optimized BVH traversal implementations  
// ================================================================================
#if !defined(NANORT_DISABLE_SIMD)

#ifdef NANORT_ENABLE_SSE2
// SIMD-optimized CWBVH traversal using SSE2 
// Tests 4 child nodes simultaneously for better performance
template <typename T>
template <class I, class H>
bool CWBVHAccel<T>::TraverseSIMD_SSE2(const Ray<T> &ray, const I &intersector, H *isect,
                                     const BVHTraceOptions &options) const {
  if (nodes_.empty()) return false;

  T hit_t = ray.max_t;
  
  std::vector<unsigned int> stack;
  stack.reserve(64);
  stack.push_back(0);

  intersector.Update(hit_t, static_cast<unsigned int>(-1));
  intersector.PrepareTraversal(ray, options);

  int dir_sign[3];
  dir_sign[0] = ray.dir[0] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[1] = ray.dir[1] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[2] = ray.dir[2] < static_cast<T>(0.0) ? 1 : 0;

  real3<T> ray_inv_dir = vsafe_inverse({ray.dir[0], ray.dir[1], ray.dir[2]});
  real3<T> ray_org = {ray.org[0], ray.org[1], ray.org[2]};

  // Load ray data into SIMD registers for vectorized AABB tests
  const __m128 ray_org_x = _mm_set1_ps(ray_org[0]);
  const __m128 ray_org_y = _mm_set1_ps(ray_org[1]); 
  const __m128 ray_org_z = _mm_set1_ps(ray_org[2]);
  const __m128 ray_inv_dir_x = _mm_set1_ps(ray_inv_dir[0]);
  const __m128 ray_inv_dir_y = _mm_set1_ps(ray_inv_dir[1]);
  const __m128 ray_inv_dir_z = _mm_set1_ps(ray_inv_dir[2]);
  const __m128 min_t = _mm_set1_ps(ray.min_t);
  const __m128 max_t_vec = _mm_set1_ps(hit_t);

  while (!stack.empty()) {
    const unsigned int node_index = stack.back();
    stack.pop_back();
    
    const CWBVHNode<T> &node = nodes_[node_index];

    if (node.IsLeaf()) {
      if (TestCWBVHLeafNode(node, ray, intersector, options)) {
        hit_t = intersector.GetT();
      }
    } else {
      // For internal nodes with multiple children, use SIMD to test up to 4 AABBs at once
      const unsigned int num_children = node.num_children;
      const unsigned int base_child = node.data.internal.child_index;
      
      // Process children in groups of 4 using SIMD
      for (unsigned int child_group = 0; child_group < num_children; child_group += 4) {
        const unsigned int children_in_group = std::min(4U, num_children - child_group);
        
        if (children_in_group == 4) {
          // Load 4 child AABBs for SIMD intersection test
          __m128 bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z;
          
          for (unsigned int i = 0; i < 4U; i++) {
            const CWBVHNode<T> &child = nodes_[base_child + child_group + i];
            reinterpret_cast<float*>(&bbox_min_x)[i] = static_cast<float>(child.bmin[0]);
            reinterpret_cast<float*>(&bbox_min_y)[i] = static_cast<float>(child.bmin[1]); 
            reinterpret_cast<float*>(&bbox_min_z)[i] = static_cast<float>(child.bmin[2]);
            reinterpret_cast<float*>(&bbox_max_x)[i] = static_cast<float>(child.bmax[0]);
            reinterpret_cast<float*>(&bbox_max_y)[i] = static_cast<float>(child.bmax[1]);
            reinterpret_cast<float*>(&bbox_max_z)[i] = static_cast<float>(child.bmax[2]);
          }
          
          __m128 tmin_out, tmax_out;
          int hit_mask = IntersectRayAABB4_SSE2(
            ray_org_x, ray_org_y, ray_org_z,
            ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z,
            min_t, max_t_vec,
            bbox_min_x, bbox_min_y, bbox_min_z,
            bbox_max_x, bbox_max_y, bbox_max_z,
            &tmin_out, &tmax_out);
          
          // Add hit children to stack
          for (int i = 0; i < 4; i++) {
            if (hit_mask & (1 << i)) {
              stack.push_back(base_child + child_group + static_cast<unsigned int>(i));
            }
          }
        } else {
          // Handle remaining children with scalar code
          for (unsigned int i = 0; i < children_in_group; i++) {
            T min_t_out, max_t_out;
            const CWBVHNode<T> &child = nodes_[base_child + child_group + i];
            
            if (IntersectRayAABB(&min_t_out, &max_t_out, ray.min_t, hit_t,
                                 child.bmin, child.bmax, ray_org, ray_inv_dir, dir_sign)) {
              stack.push_back(base_child + child_group + i);
            }
          }
        }
      }
    }
  }

  bool hit = (intersector.GetT() < ray.max_t);
  intersector.PostTraversal(ray, hit, isect);
  return hit;
}
#endif

#ifdef NANORT_ENABLE_AVX2  
// AVX2-optimized CWBVH traversal for 8-way SIMD
template <typename T>
template <class I, class H>
bool CWBVHAccel<T>::TraverseSIMD_AVX2(const Ray<T> &ray, const I &intersector, H *isect,
                                     const BVHTraceOptions &options) const {
  if (nodes_.empty()) return false;

  T hit_t = ray.max_t;
  
  std::vector<unsigned int> stack;
  stack.reserve(64);
  stack.push_back(0);

  intersector.Update(hit_t, static_cast<unsigned int>(-1));
  intersector.PrepareTraversal(ray, options);

  real3<T> ray_inv_dir = vsafe_inverse({ray.dir[0], ray.dir[1], ray.dir[2]});
  real3<T> ray_org = {ray.org[0], ray.org[1], ray.org[2]};

  // Load ray data into AVX registers for 8-way AABB tests
  const __m256 ray_org_x = _mm256_set1_ps(ray_org[0]);
  const __m256 ray_org_y = _mm256_set1_ps(ray_org[1]); 
  const __m256 ray_org_z = _mm256_set1_ps(ray_org[2]);
  const __m256 ray_inv_dir_x = _mm256_set1_ps(ray_inv_dir[0]);
  const __m256 ray_inv_dir_y = _mm256_set1_ps(ray_inv_dir[1]);
  const __m256 ray_inv_dir_z = _mm256_set1_ps(ray_inv_dir[2]);
  const __m256 min_t = _mm256_set1_ps(ray.min_t);
  const __m256 max_t_vec = _mm256_set1_ps(hit_t);

  while (!stack.empty()) {
    const unsigned int node_index = stack.back();
    stack.pop_back();
    
    const CWBVHNode<T> &node = nodes_[node_index];

    if (node.IsLeaf()) {
      if (TestCWBVHLeafNode(node, ray, intersector, options)) {
        hit_t = intersector.GetT();
      }
    } else {
      const unsigned int num_children = node.num_children;
      const unsigned int base_child = node.data.internal.child_index;
      
      // Process children in groups of 8 using AVX2
      for (unsigned int child_group = 0; child_group < num_children; child_group += 8) {
        const unsigned int children_in_group = std::min(8U, num_children - child_group);
        
        if (children_in_group == 8) {
          // Load 8 child AABBs for AVX2 intersection test
          __m256 bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z;
          
          for (unsigned int i = 0; i < 8U; i++) {
            const CWBVHNode<T> &child = nodes_[base_child + child_group + i];
            reinterpret_cast<float*>(&bbox_min_x)[i] = static_cast<float>(child.bmin[0]);
            reinterpret_cast<float*>(&bbox_min_y)[i] = static_cast<float>(child.bmin[1]); 
            reinterpret_cast<float*>(&bbox_min_z)[i] = static_cast<float>(child.bmin[2]);
            reinterpret_cast<float*>(&bbox_max_x)[i] = static_cast<float>(child.bmax[0]);
            reinterpret_cast<float*>(&bbox_max_y)[i] = static_cast<float>(child.bmax[1]);
            reinterpret_cast<float*>(&bbox_max_z)[i] = static_cast<float>(child.bmax[2]);
          }
          
          __m256 tmin_out, tmax_out;
          int hit_mask = IntersectRayAABB8_AVX2(
            ray_org_x, ray_org_y, ray_org_z,
            ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z,
            min_t, max_t_vec,
            bbox_min_x, bbox_min_y, bbox_min_z,
            bbox_max_x, bbox_max_y, bbox_max_z,
            &tmin_out, &tmax_out);
          
          // Add hit children to stack
          for (int i = 0; i < 8; i++) {
            if (hit_mask & (1 << i)) {
              stack.push_back(base_child + child_group + static_cast<unsigned int>(i));
            }
          }
        } else {
          // Handle remaining children with scalar fallback
          int dir_sign[3];
          dir_sign[0] = ray.dir[0] < static_cast<T>(0.0) ? 1 : 0;
          dir_sign[1] = ray.dir[1] < static_cast<T>(0.0) ? 1 : 0;
          dir_sign[2] = ray.dir[2] < static_cast<T>(0.0) ? 1 : 0;

          for (unsigned int i = 0; i < children_in_group; i++) {
            T min_t_out, max_t_out;
            const CWBVHNode<T> &child = nodes_[base_child + child_group + i];
            
            if (IntersectRayAABB(&min_t_out, &max_t_out, ray.min_t, hit_t,
                                 child.bmin, child.bmax, ray_org, ray_inv_dir, dir_sign)) {
              stack.push_back(base_child + child_group + i);
            }
          }
        }
      }
    }
  }

  bool hit = (intersector.GetT() < ray.max_t);
  intersector.PostTraversal(ray, hit, isect);
  return hit;
}
#endif // NANORT_ENABLE_AVX2

#endif // !NANORT_DISABLE_SIMD

template <typename T>
template <class I>
bool CWBVHAccel<T>::TestCWBVHLeafNode(const CWBVHNode<T> &node, const Ray<T> &,
                                     const I &intersector, const BVHTraceOptions &options) const {
  bool hit = false;

  unsigned int num_primitives = node.data.leaf.primitive_count;
  unsigned int offset = node.data.leaf.primitive_index;

  for (unsigned int i = 0; i < num_primitives; i++) {
    unsigned int prim_idx = indices_[i + offset];

    // Count primitive intersection tests
    options.primitive_intersections++;

    if (intersector.Intersect(prim_idx)) {
      hit = true;
    }
  }

  return hit;
}

// Predefined SAH predicator for triangle.
template <typename T = float>
class TriangleSAHPred {
 public:
  TriangleSAHPred(
      const T *vertices, const unsigned int *faces,
      size_t vertex_stride_bytes)  // e.g. 12 for sizeof(float) * XYZ
      : axis_(0),
        pos_(static_cast<T>(0.0)),
        vertices_(vertices),
        faces_(faces),
        vertex_stride_bytes_(vertex_stride_bytes) {}

  TriangleSAHPred(const TriangleSAHPred<T> &rhs)
      : axis_(rhs.axis_),
        pos_(rhs.pos_),
        vertices_(rhs.vertices_),
        faces_(rhs.faces_),
        vertex_stride_bytes_(rhs.vertex_stride_bytes_) {}

  void Set(int axis, T pos) const {
    axis_ = axis;
    pos_ = pos;
  }

  bool operator()(unsigned int i) const {
    int axis = axis_;
    T pos = pos_;

    unsigned int i0 = faces_[3 * i + 0];
    unsigned int i1 = faces_[3 * i + 1];
    unsigned int i2 = faces_[3 * i + 2];

    real3<T> p0(get_vertex_addr<T>(vertices_, i0, vertex_stride_bytes_));
    real3<T> p1(get_vertex_addr<T>(vertices_, i1, vertex_stride_bytes_));
    real3<T> p2(get_vertex_addr<T>(vertices_, i2, vertex_stride_bytes_));

    T center = p0[axis] + p1[axis] + p2[axis];
    return (center < pos * static_cast<T>(3.0));
  }

 private:
  mutable int axis_;
  mutable T pos_;
  const T *vertices_;
  const unsigned int *faces_;
  const size_t vertex_stride_bytes_;
};

// Predefined Triangle mesh geometry.
template <typename T = float>
class TriangleMesh {
 public:
  TriangleMesh(
      const T *vertices, const unsigned int *faces,
      const size_t vertex_stride_bytes)  // e.g. 12 for sizeof(float) * XYZ
      : vertices_(vertices),
        faces_(faces),
        vertex_stride_bytes_(vertex_stride_bytes) {}

  /// Compute bounding box for `prim_index`th triangle.
  /// This function is called for each primitive in BVH build.
  void BoundingBox(real3<T> *bmin, real3<T> *bmax,
                   unsigned int prim_index) const {
    unsigned vertex = faces_[3 * prim_index + 0];

    (*bmin)[0] = get_vertex_addr(vertices_, vertex, vertex_stride_bytes_)[0];
    (*bmin)[1] = get_vertex_addr(vertices_, vertex, vertex_stride_bytes_)[1];
    (*bmin)[2] = get_vertex_addr(vertices_, vertex, vertex_stride_bytes_)[2];
    (*bmax)[0] = get_vertex_addr(vertices_, vertex, vertex_stride_bytes_)[0];
    (*bmax)[1] = get_vertex_addr(vertices_, vertex, vertex_stride_bytes_)[1];
    (*bmax)[2] = get_vertex_addr(vertices_, vertex, vertex_stride_bytes_)[2];

    // remaining two vertices of the primitive
    for (unsigned int i = 1; i < 3; i++) {
      // xyz
      for (int k = 0; k < 3; k++) {
        T coord = get_vertex_addr<T>(vertices_, faces_[3 * prim_index + i],
                                     vertex_stride_bytes_)[k];

        (*bmin)[k] = std::min((*bmin)[k], coord);
        (*bmax)[k] = std::max((*bmax)[k], coord);
      }
    }
  }

  void BoundingBoxAndCenter(real3<T>* bmin, real3<T>* bmax, real3<T>* center, unsigned int prim_index) const {
    unsigned int i0 = faces_[3 * prim_index + 0];
    unsigned int i1 = faces_[3 * prim_index + 1];
    unsigned int i2 = faces_[3 * prim_index + 2];

    real3<T> p0(get_vertex_addr<T>(vertices_, i0, vertex_stride_bytes_));
    real3<T> p1(get_vertex_addr<T>(vertices_, i1, vertex_stride_bytes_));
    real3<T> p2(get_vertex_addr<T>(vertices_, i2, vertex_stride_bytes_));
    for (int k = 0; k < 3; ++k) {
      (*bmin)[k] = std::min(p0[k], std::min(p1[k], p2[k]));
      (*bmax)[k] = std::max(p0[k], std::max(p1[k], p2[k]));
    }
    *center = (p0 + p1 + p2) * (T(1) / T(3));
  }

  const T *vertices_;
  const unsigned int *faces_;
  const size_t vertex_stride_bytes_;

  //
  // Accessors
  //
  const T *GetVertices() const {
    return vertices_;
  }

  const unsigned int *GetFaces() const {
    return faces_;
  }

  size_t GetVertexStrideBytes() const {
    return vertex_stride_bytes_;
  }
};

///
/// Stores intersection point information for triangle geometry.
///
template <typename T = float>
class TriangleIntersection {
 public:
  T u;
  T v;

  // Required member variables.
  T t;
  unsigned int prim_id;
};

///
/// Intersector is a template class which implements intersection method and stores
/// intesection point information(`H`)
///
/// @tparam T Precision(float or double)
/// @tparam H Intersection point information struct
///
template <typename T = float, class H = TriangleIntersection<T> >
class TriangleIntersector {
 public:

  // Initialize from mesh object.
  // M: mesh class
  template<class M>
  TriangleIntersector(const M &m)
      : vertices_(m.GetVertices()),
        faces_(m.GetFaces()),
        vertex_stride_bytes_(m.GetVertexStrideBytes()) {}

  template<class M>
  TriangleIntersector(const M *m)
      : vertices_(m->GetVertices()),
        faces_(m->GetFaces()),
        vertex_stride_bytes_(m->GetVertexStrideBytes()) {}

  TriangleIntersector(const T *vertices, const unsigned int *faces,
                      const size_t vertex_stride_bytes)  // e.g.
                                                         // vertex_stride_bytes
                                                         // = 12 = sizeof(float)
                                                         // * 3
      : vertices_(vertices),
        faces_(faces),
        vertex_stride_bytes_(vertex_stride_bytes) {}

  // For Watertight Ray/Triangle Intersection.
  typedef struct {
    T Sx;
    T Sy;
    T Sz;
    int kx;
    int ky;
    int kz;
  } RayCoeff;

  /// Do ray intersection stuff for `prim_index` th primitive and return hit
  /// distance `t`, barycentric coordinate `u` and `v`.
  /// Returns true if there's intersection.
  bool Intersect(T *t_inout, const unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    // Self-intersection test.
    if (prim_index == trace_options_.skip_prim_id) {
      return false;
    }

    const unsigned int f0 = faces_[3 * prim_index + 0];
    const unsigned int f1 = faces_[3 * prim_index + 1];
    const unsigned int f2 = faces_[3 * prim_index + 2];

    const real3<T> p0(get_vertex_addr(vertices_, f0 + 0, vertex_stride_bytes_));
    const real3<T> p1(get_vertex_addr(vertices_, f1 + 0, vertex_stride_bytes_));
    const real3<T> p2(get_vertex_addr(vertices_, f2 + 0, vertex_stride_bytes_));

    const real3<T> A = p0 - ray_org_;
    const real3<T> B = p1 - ray_org_;
    const real3<T> C = p2 - ray_org_;

    const T Ax = A[ray_coeff_.kx] - ray_coeff_.Sx * A[ray_coeff_.kz];
    const T Ay = A[ray_coeff_.ky] - ray_coeff_.Sy * A[ray_coeff_.kz];
    const T Bx = B[ray_coeff_.kx] - ray_coeff_.Sx * B[ray_coeff_.kz];
    const T By = B[ray_coeff_.ky] - ray_coeff_.Sy * B[ray_coeff_.kz];
    const T Cx = C[ray_coeff_.kx] - ray_coeff_.Sx * C[ray_coeff_.kz];
    const T Cy = C[ray_coeff_.ky] - ray_coeff_.Sy * C[ray_coeff_.kz];

    T U = Cx * By - Cy * Bx;
    T V = Ax * Cy - Ay * Cx;
    T W = Bx * Ay - By * Ax;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
#endif

    // Fall back to test against edges using double precision.
    if (U == static_cast<T>(0.0) || V == static_cast<T>(0.0) ||
        W == static_cast<T>(0.0)) {
      double CxBy = static_cast<double>(Cx) * static_cast<double>(By);
      double CyBx = static_cast<double>(Cy) * static_cast<double>(Bx);
      U = static_cast<T>(CxBy - CyBx);

      double AxCy = static_cast<double>(Ax) * static_cast<double>(Cy);
      double AyCx = static_cast<double>(Ay) * static_cast<double>(Cx);
      V = static_cast<T>(AxCy - AyCx);

      double BxAy = static_cast<double>(Bx) * static_cast<double>(Ay);
      double ByAx = static_cast<double>(By) * static_cast<double>(Ax);
      W = static_cast<T>(BxAy - ByAx);
    }

    // Check for mixed signs (invalid intersection)
    if ((U < static_cast<T>(0.0) || V < static_cast<T>(0.0) || W < static_cast<T>(0.0)) &&
        (U > static_cast<T>(0.0) || V > static_cast<T>(0.0) || W > static_cast<T>(0.0))) {
      return false;
    }

    // Handle backface culling
    if (trace_options_.cull_back_face && 
        U < static_cast<T>(0.0) && V < static_cast<T>(0.0) && W < static_cast<T>(0.0)) {
      return false;
    }

    T det = U + V + W;
    if (det == static_cast<T>(0.0)) return false;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

    const T Az = ray_coeff_.Sz * A[ray_coeff_.kz];
    const T Bz = ray_coeff_.Sz * B[ray_coeff_.kz];
    const T Cz = ray_coeff_.Sz * C[ray_coeff_.kz];
    const T D = U * Az + V * Bz + W * Cz;

    const T rcpDet = static_cast<T>(1.0) / det;
    T tt = D * rcpDet;

    if (tt > (*t_inout)) {
      return false;
    }

    if (tt < t_min_) {
      return false;
    }

    (*t_inout) = tt;
    // Use Möller-Trumbore style barycentric coordinates
    // U + V + W = 1.0 and interp(p) = U * p0 + V * p1 + W * p2
    // We want interp(p) = (1 - u - v) * p0 + u * v1 + v * p2;
    // => u = V, v = W.
    u_ = V * rcpDet;
    v_ = W * rcpDet;

    return true;
  }

  // ================================================================================
  // SIMD-optimized watertight triangle intersection methods
  // ================================================================================
#if !defined(NANORT_DISABLE_SIMD)
  #if defined(NANORT_ENABLE_SSE2) && defined(__GNUC__) && !defined(__clang__)
    // GCC-specific SSE2 optimizations for triangle intersection
  inline bool IntersectSIMD_SSE2(T *t_inout, const unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    if (prim_index == trace_options_.skip_prim_id) {
      return false;
    }

    const unsigned int f0 = faces_[3 * prim_index + 0];
    const unsigned int f1 = faces_[3 * prim_index + 1];
    const unsigned int f2 = faces_[3 * prim_index + 2];

    const real3<T> p0(get_vertex_addr(vertices_, f0 + 0, vertex_stride_bytes_));
    const real3<T> p1(get_vertex_addr(vertices_, f1 + 0, vertex_stride_bytes_));
    const real3<T> p2(get_vertex_addr(vertices_, f2 + 0, vertex_stride_bytes_));

    // Load triangle vertices into SIMD registers for vectorized computation
    const __m128 v0 = _mm_set_ps(0.0f, p0[2], p0[1], p0[0]);
    const __m128 v1 = _mm_set_ps(0.0f, p1[2], p1[1], p1[0]);
    const __m128 v2 = _mm_set_ps(0.0f, p2[2], p2[1], p2[0]);
    const __m128 ray_org = _mm_set_ps(0.0f, ray_org_[2], ray_org_[1], ray_org_[0]);

    // Vectorized computation of A, B, C vectors
    const __m128 A_vec = _mm_sub_ps(v0, ray_org);
    const __m128 B_vec = _mm_sub_ps(v1, ray_org);
    const __m128 C_vec = _mm_sub_ps(v2, ray_org);

    // Extract individual components for the watertight algorithm
    // (The core watertight intersection logic still needs scalar computation
    // due to its specialized nature, but we vectorize the setup)
    float A_arr[4], B_arr[4], C_arr[4];
    _mm_store_ps(A_arr, A_vec);
    _mm_store_ps(B_arr, B_vec);
    _mm_store_ps(C_arr, C_vec);
    
    const real3<T> A = {static_cast<T>(A_arr[0]), static_cast<T>(A_arr[1]), static_cast<T>(A_arr[2])};
    const real3<T> B = {static_cast<T>(B_arr[0]), static_cast<T>(B_arr[1]), static_cast<T>(B_arr[2])};
    const real3<T> C = {static_cast<T>(C_arr[0]), static_cast<T>(C_arr[1]), static_cast<T>(C_arr[2])};

    // Continue with the standard watertight intersection algorithm
    const T Ax = A[ray_coeff_.kx] - ray_coeff_.Sx * A[ray_coeff_.kz];
    const T Ay = A[ray_coeff_.ky] - ray_coeff_.Sy * A[ray_coeff_.kz];
    const T Bx = B[ray_coeff_.kx] - ray_coeff_.Sx * B[ray_coeff_.kz];
    const T By = B[ray_coeff_.ky] - ray_coeff_.Sy * B[ray_coeff_.kz];
    const T Cx = C[ray_coeff_.kx] - ray_coeff_.Sx * C[ray_coeff_.kz];
    const T Cy = C[ray_coeff_.ky] - ray_coeff_.Sy * C[ray_coeff_.kz];

    // Vectorized cross product computation using SIMD
    const __m128 cross_factors = _mm_set_ps(static_cast<float>(By * Ax - Bx * Ay),  // W
                                            static_cast<float>(Ax * Cy - Ay * Cx),  // V
                                            static_cast<float>(Cx * By - Cy * Bx),  // U
                                            0.0f);
    
    float cross_arr[4];
    _mm_store_ps(cross_arr, cross_factors);
    
    T U = static_cast<T>(cross_arr[2]); // U
    T V = static_cast<T>(cross_arr[1]); // V
    T W = static_cast<T>(cross_arr[0]); // W

    // Edge case handling with double precision fallback
    if (U == static_cast<T>(0.0) || V == static_cast<T>(0.0) || W == static_cast<T>(0.0)) {
      double CxBy = static_cast<double>(Cx) * static_cast<double>(By);
      double CyBx = static_cast<double>(Cy) * static_cast<double>(Bx);
      U = static_cast<T>(CxBy - CyBx);

      double AxCy = static_cast<double>(Ax) * static_cast<double>(Cy);
      double AyCx = static_cast<double>(Ay) * static_cast<double>(Cx);
      V = static_cast<T>(AxCy - AyCx);

      double BxAy = static_cast<double>(Bx) * static_cast<double>(Ay);
      double ByAx = static_cast<double>(By) * static_cast<double>(Ax);
      W = static_cast<T>(BxAy - ByAx);
    }

    // Check for mixed signs (invalid intersection)
    if ((U < static_cast<T>(0.0) || V < static_cast<T>(0.0) || W < static_cast<T>(0.0)) &&
        (U > static_cast<T>(0.0) || V > static_cast<T>(0.0) || W > static_cast<T>(0.0))) {
      return false;
    }

    // Handle backface culling
    if (trace_options_.cull_back_face && 
        U < static_cast<T>(0.0) && V < static_cast<T>(0.0) && W < static_cast<T>(0.0)) {
      return false;
    }

    T det = U + V + W;
    if (det == static_cast<T>(0.0)) return false;

    const T Az = ray_coeff_.Sz * A[ray_coeff_.kz];
    const T Bz = ray_coeff_.Sz * B[ray_coeff_.kz];
    const T Cz = ray_coeff_.Sz * C[ray_coeff_.kz];
    const T D = U * Az + V * Bz + W * Cz;

    const T rcpDet = static_cast<T>(1.0) / det;
    T tt = D * rcpDet;

    if (tt > (*t_inout) || tt < t_min_) {
      return false;
    }

    (*t_inout) = tt;
    u_ = V * rcpDet;
    v_ = W * rcpDet;

    return true;
  }
#endif

#if defined(NANORT_ENABLE_NEON)
  // ARM NEON optimized triangle intersection
  inline bool IntersectSIMD_NEON(T *t_inout, const unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    if (prim_index == trace_options_.skip_prim_id) {
      return false;
    }

    const unsigned int f0 = faces_[3 * prim_index + 0];
    const unsigned int f1 = faces_[3 * prim_index + 1];
    const unsigned int f2 = faces_[3 * prim_index + 2];

    const real3<T> p0(get_vertex_addr(vertices_, f0 + 0, vertex_stride_bytes_));
    const real3<T> p1(get_vertex_addr(vertices_, f1 + 0, vertex_stride_bytes_));
    const real3<T> p2(get_vertex_addr(vertices_, f2 + 0, vertex_stride_bytes_));

    // Load triangle vertices into NEON registers
    const float32x4_t v0 = {p0[0], p0[1], p0[2], 0.0f};
    const float32x4_t v1 = {p1[0], p1[1], p1[2], 0.0f};
    const float32x4_t v2 = {p2[0], p2[1], p2[2], 0.0f};
    const float32x4_t ray_org = {ray_org_[0], ray_org_[1], ray_org_[2], 0.0f};

    // Vectorized computation of A, B, C vectors
    const float32x4_t A_vec = vsubq_f32(v0, ray_org);
    const float32x4_t B_vec = vsubq_f32(v1, ray_org);
    const float32x4_t C_vec = vsubq_f32(v2, ray_org);

    // Extract components for watertight intersection
    const real3<T> A = {vgetq_lane_f32(A_vec, 0), vgetq_lane_f32(A_vec, 1), vgetq_lane_f32(A_vec, 2)};
    const real3<T> B = {vgetq_lane_f32(B_vec, 0), vgetq_lane_f32(B_vec, 1), vgetq_lane_f32(B_vec, 2)};
    const real3<T> C = {vgetq_lane_f32(C_vec, 0), vgetq_lane_f32(C_vec, 1), vgetq_lane_f32(C_vec, 2)};

    // Continue with standard watertight intersection algorithm
    const T Ax = A[ray_coeff_.kx] - ray_coeff_.Sx * A[ray_coeff_.kz];
    const T Ay = A[ray_coeff_.ky] - ray_coeff_.Sy * A[ray_coeff_.kz];
    const T Bx = B[ray_coeff_.kx] - ray_coeff_.Sx * B[ray_coeff_.kz];
    const T By = B[ray_coeff_.ky] - ray_coeff_.Sy * B[ray_coeff_.kz];
    const T Cx = C[ray_coeff_.kx] - ray_coeff_.Sx * C[ray_coeff_.kz];
    const T Cy = C[ray_coeff_.ky] - ray_coeff_.Sy * C[ray_coeff_.kz];

    // Vectorized cross product computation
    const float32x4_t cross_vec = {Cx * By - Cy * Bx,  // U
                                   Ax * Cy - Ay * Cx,  // V
                                   Bx * Ay - By * Ax,  // W
                                   0.0f};

    T U = vgetq_lane_f32(cross_vec, 0);
    T V = vgetq_lane_f32(cross_vec, 1);
    T W = vgetq_lane_f32(cross_vec, 2);

    // Edge case handling with double precision fallback
    if (U == static_cast<T>(0.0) || V == static_cast<T>(0.0) || W == static_cast<T>(0.0)) {
      double CxBy = static_cast<double>(Cx) * static_cast<double>(By);
      double CyBx = static_cast<double>(Cy) * static_cast<double>(Bx);
      U = static_cast<T>(CxBy - CyBx);

      double AxCy = static_cast<double>(Ax) * static_cast<double>(Cy);
      double AyCx = static_cast<double>(Ay) * static_cast<double>(Cx);
      V = static_cast<T>(AxCy - AyCx);

      double BxAy = static_cast<double>(Bx) * static_cast<double>(Ay);
      double ByAx = static_cast<double>(By) * static_cast<double>(Ax);
      W = static_cast<T>(BxAy - ByAx);
    }

    // Check for mixed signs (invalid intersection)
    if ((U < static_cast<T>(0.0) || V < static_cast<T>(0.0) || W < static_cast<T>(0.0)) &&
        (U > static_cast<T>(0.0) || V > static_cast<T>(0.0) || W > static_cast<T>(0.0))) {
      return false;
    }

    // Handle backface culling
    if (trace_options_.cull_back_face && 
        U < static_cast<T>(0.0) && V < static_cast<T>(0.0) && W < static_cast<T>(0.0)) {
      return false;
    }

    T det = U + V + W;
    if (det == static_cast<T>(0.0)) return false;

    const T Az = ray_coeff_.Sz * A[ray_coeff_.kz];
    const T Bz = ray_coeff_.Sz * B[ray_coeff_.kz];
    const T Cz = ray_coeff_.Sz * C[ray_coeff_.kz];
    const T D = U * Az + V * Bz + W * Cz;

    const T rcpDet = static_cast<T>(1.0) / det;
    T tt = D * rcpDet;

    if (tt > (*t_inout) || tt < t_min_) {
      return false;
    }

    (*t_inout) = tt;
    u_ = V * rcpDet;
    v_ = W * rcpDet;

    return true;
  }
  #endif // NANORT_ENABLE_NEON
#endif // !NANORT_DISABLE_SIMD

  /// Returns the nearest hit distance.
  T GetT() const { return t_; }

  /// Update is called when initializing intersection and nearest hit is found.
  void Update(T t, unsigned int prim_idx) const {
    t_ = t;
    prim_id_ = prim_idx;
  }

  /// Prepare BVH traversal (e.g. compute inverse ray direction)
  /// This function is called only once in BVH traversal.
  void PrepareTraversal(const Ray<T> &ray,
                        const BVHTraceOptions &trace_options) const {
    ray_org_[0] = ray.org[0];
    ray_org_[1] = ray.org[1];
    ray_org_[2] = ray.org[2];

    // Calculate dimension where the ray direction is maximal.
    ray_coeff_.kz = 0;
    T absDir = std::fabs(ray.dir[0]);
    if (absDir < std::fabs(ray.dir[1])) {
      ray_coeff_.kz = 1;
      absDir = std::fabs(ray.dir[1]);
    }
    if (absDir < std::fabs(ray.dir[2])) {
      ray_coeff_.kz = 2;
      absDir = std::fabs(ray.dir[2]);
    }

    ray_coeff_.kx = ray_coeff_.kz + 1;
    if (ray_coeff_.kx == 3) ray_coeff_.kx = 0;
    ray_coeff_.ky = ray_coeff_.kx + 1;
    if (ray_coeff_.ky == 3) ray_coeff_.ky = 0;

    // Swap kx and ky dimension to preserve winding direction of triangles.
    if (ray.dir[ray_coeff_.kz] < static_cast<T>(0.0))
      std::swap(ray_coeff_.kx, ray_coeff_.ky);

    // Calculate shear constants.
    ray_coeff_.Sx = ray.dir[ray_coeff_.kx] / ray.dir[ray_coeff_.kz];
    ray_coeff_.Sy = ray.dir[ray_coeff_.ky] / ray.dir[ray_coeff_.kz];
    ray_coeff_.Sz = static_cast<T>(1.0) / ray.dir[ray_coeff_.kz];

    trace_options_ = trace_options;

    t_min_ = ray.min_t;

    u_ = static_cast<T>(0.0);
    v_ = static_cast<T>(0.0);
  }

  /// Post BVH traversal stuff.
  /// Fill `isect` if there is a hit.
  void PostTraversal(const Ray<T> &ray, bool hit, H *isect) const {
    if (hit && isect) {
      (*isect).t = t_;
      (*isect).u = u_;
      (*isect).v = v_;
      (*isect).prim_id = prim_id_;
    }
    (void)ray;
  }

 private:
  const T *vertices_;
  const unsigned int *faces_;
  const size_t vertex_stride_bytes_;

  mutable real3<T> ray_org_;
  mutable RayCoeff ray_coeff_;
  mutable BVHTraceOptions trace_options_;
  mutable T t_min_;

  mutable T t_;
  mutable T u_;
  mutable T v_;
  mutable unsigned int prim_id_;
};

//
// Robust BVH Ray Traversal : http://jcgt.org/published/0002/02/02/paper.pdf
//

// NaN-safe min and max function.
template <class T>
const T &safemin(const T &a, const T &b) {
  return (a < b) ? a : b;
}
template <class T>
const T &safemax(const T &a, const T &b) {
  return (a > b) ? a : b;
}

//
// SAH functions
//
template <typename T>
struct Bin {
  BBox<T> bbox;
  size_t  count;
  T cost;

  Bin()
    : count(0), cost(0)
  {
    // Note: bbox is initialized to be empty
  }
};

template <typename T>
struct BinBuffer {
  explicit BinBuffer(unsigned int size) {
    bin_size = size;
    bin.resize(3 * size);
    clear();
  }

  void clear() {
    std::fill(bin.begin(), bin.end(), Bin<T>());
  }

  std::vector<Bin<T> > bin;
  unsigned int bin_size;
  unsigned int pad0;
};

template <typename T>
inline T CalculateSurfaceArea(const real3<T> &min, const real3<T> &max) {
  real3<T> box = max - min;
  return static_cast<T>(2.0) *
         (box[0] * box[1] + box[1] * box[2] + box[2] * box[0]);
}

template <typename T>
inline void GetBoundingBoxOfTriangle(real3<T> *bmin, real3<T> *bmax,
                                     const T *vertices,
                                     const unsigned int *faces,
                                     unsigned int index) {
  unsigned int f0 = faces[3 * index + 0];
  unsigned int f1 = faces[3 * index + 1];
  unsigned int f2 = faces[3 * index + 2];

  real3<T> p[3];

  p[0] = real3<T>(&vertices[3 * f0]);
  p[1] = real3<T>(&vertices[3 * f1]);
  p[2] = real3<T>(&vertices[3 * f2]);

  (*bmin) = p[0];
  (*bmax) = p[0];

  for (int i = 1; i < 3; i++) {
    (*bmin)[0] = std::min((*bmin)[0], p[i][0]);
    (*bmin)[1] = std::min((*bmin)[1], p[i][1]);
    (*bmin)[2] = std::min((*bmin)[2], p[i][2]);

    (*bmax)[0] = std::max((*bmax)[0], p[i][0]);
    (*bmax)[1] = std::max((*bmax)[1], p[i][1]);
    (*bmax)[2] = std::max((*bmax)[2], p[i][2]);
  }
}

template <typename T, class P>
inline void ContributeBinBuffer(BinBuffer<T> *bins,  // [out]
                                const real3<T> &scene_min,
                                const real3<T> &scene_max,
                                unsigned int *indices, unsigned int left_idx,
                                unsigned int right_idx, const P &p) {
  T bin_size = static_cast<T>(bins->bin_size);

  // Calculate extent
  real3<T> scene_size, scene_inv_size;
  scene_size = scene_max - scene_min;

  for (int i = 0; i < 3; ++i) {
    assert(scene_size[i] >= static_cast<T>(0.0));

    if (scene_size[i] > static_cast<T>(0.0)) {
      scene_inv_size[i] = bin_size / scene_size[i];
    } else {
      scene_inv_size[i] = static_cast<T>(0.0);
    }
  }

  bins->clear();

  for (size_t i = left_idx; i < right_idx; i++) {
    //
    // Quantize the center position into [0, BIN_SIZE)
    //
    // q[i] = (int)(p[i] - scene_bmin) / scene_size
    //

    real3<T> bmin, bmax, center;
    p.BoundingBoxAndCenter(&bmin, &bmax, &center, indices[i]);
    real3<T> quantized_center = (center - scene_min) * scene_inv_size;

    for (int j = 0; j < 3; ++j) {
      // idx is now in [0, BIN_SIZE)
      unsigned idx = std::min(bins->bin_size - 1, unsigned(std::max(0, int(quantized_center[j]))));

      // Increment bin counter + extend bounding box of bin
      unsigned int bin_idx = static_cast<unsigned int >(j) * bins->bin_size + idx;

      // TODO: assert when out-of-bounds access?.
      if (bin_idx < bins->bin_size) {
        Bin<T>& bin = bins->bin[static_cast<unsigned int>(j) * bins->bin_size + idx];
        bin.count++;
        for (int k = 0; k < 3; ++k) {
          bin.bbox.bmin[k] = std::min(bin.bbox.bmin[k], bmin[k]);
          bin.bbox.bmax[k] = std::max(bin.bbox.bmax[k], bmax[k]);
        }
      }
    }
  }
}

template <typename T>
inline T SAH(size_t ns1, T leftArea, size_t ns2, T rightArea, T invS, T Taabb,
             T Ttri) {
  T sah;

  sah = static_cast<T>(2.0) * Taabb +
        (leftArea * invS) * static_cast<T>(ns1) * Ttri +
        (rightArea * invS) * static_cast<T>(ns2) * Ttri;

  return sah;
}

template <typename T>
inline bool FindCutFromBinBuffer(T *cut_pos,        // [out] xyz
                                 int *minCostAxis,  // [out]
                                 BinBuffer<T> *bins, const real3<T> &bmin,
                                 const real3<T> &bmax) {
  T minCost[3];
  for (int j = 0; j < 3; ++j) {
    minCost[j] = std::numeric_limits<T>::max();

    // Sweep left to accumulate bounding boxes and compute the right-hand side of the cost
    size_t count = 0;
    BBox<T> accumulated_bbox;
    for (size_t i = bins->bin_size - 1; i > 0; --i) {
      Bin<T>& bin = bins->bin[static_cast<unsigned int>(j) * bins->bin_size + i];
      for (int k = 0; k < 3; ++k) {
        accumulated_bbox.bmin[k] = std::min(bin.bbox.bmin[k], accumulated_bbox.bmin[k]);
        accumulated_bbox.bmax[k] = std::max(bin.bbox.bmax[k], accumulated_bbox.bmax[k]);
      }
      count += bin.count;
      bin.cost = T(count) * CalculateSurfaceArea(accumulated_bbox.bmin, accumulated_bbox.bmax);
    }

    // Sweep right to compute the full cost
    count = 0;
    accumulated_bbox = BBox<T>();
    size_t minBin = 1;
    for (size_t i = 0; i < bins->bin_size - 1; i++) {
      Bin<T>& bin = bins->bin[static_cast<unsigned int>(j) * bins->bin_size + i];
      Bin<T>& next_bin = bins->bin[static_cast<unsigned int>(j) * bins->bin_size + i + 1];
      for (int k = 0; k < 3; ++k) {
        accumulated_bbox.bmin[k] = std::min(bin.bbox.bmin[k], accumulated_bbox.bmin[k]);
        accumulated_bbox.bmax[k] = std::max(bin.bbox.bmax[k], accumulated_bbox.bmax[k]);
      }
      count += bin.count;
      // Traversal cost and intersection cost are irrelevant for minimization
      T cost = T(count) * CalculateSurfaceArea(accumulated_bbox.bmin, accumulated_bbox.bmax) + next_bin.cost;
      if (cost < minCost[j]) {
        minCost[j] = cost;
        // Store the beginning of the right partition
        minBin = i + 1;
      }
    }
    cut_pos[j] = T(minBin) * ((bmax[j] - bmin[j]) / T(bins->bin_size)) + bmin[j];
  }
  *minCostAxis = 0;
  if (minCost[0] > minCost[1]) *minCostAxis = 1;
  if (minCost[*minCostAxis] > minCost[2]) *minCostAxis = 2;

  return true;
}

#ifdef _OPENMP
template <typename T, class P>
void ComputeBoundingBoxOMP(real3<T> *bmin, real3<T> *bmax,
                           const unsigned int *indices, unsigned int left_index,
                           unsigned int right_index, const P &p) {
  { p.BoundingBox(bmin, bmax, indices[left_index]); }

  T local_bmin[3] = {(*bmin)[0], (*bmin)[1], (*bmin)[2]};
  T local_bmax[3] = {(*bmax)[0], (*bmax)[1], (*bmax)[2]};

  unsigned int n = right_index - left_index;

#pragma omp parallel firstprivate(local_bmin, local_bmax) if (n > (1024 * 128))
  {
#pragma omp parallel for
    // for each face
    for (int i = int(left_index); i < int(right_index); i++) {
      unsigned int idx = indices[i];

      real3<T> bbox_min, bbox_max;

      p.BoundingBox(&bbox_min, &bbox_max, idx);

      // xyz
      for (int k = 0; k < 3; k++) {
        (*bmin)[k] = std::min((*bmin)[k], bbox_min[k]);
        (*bmax)[k] = std::max((*bmax)[k], bbox_max[k]);
      }
    }

#pragma omp critical
    {
      for (int k = 0; k < 3; k++) {
        (*bmin)[k] = std::min((*bmin)[k], local_bmin[k]);
        (*bmax)[k] = std::max((*bmax)[k], local_bmax[k]);
      }
    }
  }
}
#endif

#ifdef NANORT_USE_CPP11_FEATURE
template <typename T, class P>
inline void ComputeBoundingBoxThreaded(real3<T> *bmin, real3<T> *bmax,
                                       const unsigned int *indices,
                                       unsigned int left_index,
                                       unsigned int right_index, const P &p) {
  unsigned int n = right_index - left_index;

  // Use a smaller threshold for threading to benefit from parallelism sooner
  const size_t min_work_per_thread = 64;
  size_t num_threads = std::min(
      size_t(kNANORT_MAX_THREADS),
      std::max(size_t(1), std::min(size_t(std::thread::hardware_concurrency()),
                                   n / min_work_per_thread)));

  if (num_threads <= 1 || n < min_work_per_thread * 2) {
    // Fall back to serial version for small work
    ComputeBoundingBox(bmin, bmax, indices, left_index, right_index, p);
    return;
  }

  // Align memory for better cache performance
  struct alignas(64) LocalBounds {
    T bmin[3];
    T bmax[3];
  };

  std::vector<LocalBounds> local_bounds(num_threads);
  std::vector<std::thread> workers;
  workers.reserve(num_threads);

  const size_t work_per_thread = n / num_threads;

  for (size_t t = 0; t < num_threads; t++) {
    workers.emplace_back([&, t]() {
      const size_t start = left_index + t * work_per_thread;
      const size_t end = (t == num_threads - 1) 
          ? right_index 
          : std::min(left_index + (t + 1) * work_per_thread, size_t(right_index));

      LocalBounds& bounds = local_bounds[t];
      bounds.bmin[0] = bounds.bmin[1] = bounds.bmin[2] = std::numeric_limits<T>::infinity();
      bounds.bmax[0] = bounds.bmax[1] = bounds.bmax[2] = -std::numeric_limits<T>::infinity();

      // Process primitives in chunks for better cache behavior
      constexpr size_t chunk_size = 16;
      for (size_t i = start; i < end; i += chunk_size) {
        const size_t chunk_end = std::min(i + chunk_size, end);
        
        for (size_t j = i; j < chunk_end; j++) {
          unsigned int idx = indices[j];
          real3<T> bbox_min, bbox_max;
          p.BoundingBox(&bbox_min, &bbox_max, idx);

          bounds.bmin[0] = std::min(bounds.bmin[0], bbox_min[0]);
          bounds.bmin[1] = std::min(bounds.bmin[1], bbox_min[1]);
          bounds.bmin[2] = std::min(bounds.bmin[2], bbox_min[2]);
          bounds.bmax[0] = std::max(bounds.bmax[0], bbox_max[0]);
          bounds.bmax[1] = std::max(bounds.bmax[1], bbox_max[1]);
          bounds.bmax[2] = std::max(bounds.bmax[2], bbox_max[2]);
        }
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  // Merge results efficiently
  (*bmin)[0] = (*bmin)[1] = (*bmin)[2] = std::numeric_limits<T>::infinity();
  (*bmax)[0] = (*bmax)[1] = (*bmax)[2] = -std::numeric_limits<T>::infinity();

  for (size_t t = 0; t < num_threads; t++) {
    const LocalBounds& bounds = local_bounds[t];
    (*bmin)[0] = std::min((*bmin)[0], bounds.bmin[0]);
    (*bmin)[1] = std::min((*bmin)[1], bounds.bmin[1]);
    (*bmin)[2] = std::min((*bmin)[2], bounds.bmin[2]);
    (*bmax)[0] = std::max((*bmax)[0], bounds.bmax[0]);
    (*bmax)[1] = std::max((*bmax)[1], bounds.bmax[1]);
    (*bmax)[2] = std::max((*bmax)[2], bounds.bmax[2]);
  }
}
#endif

template <typename T, class P>
inline void ComputeBoundingBox(real3<T> *bmin, real3<T> *bmax,
                               const unsigned int *indices,
                               unsigned int left_index,
                               unsigned int right_index, const P &p) {
  unsigned int idx = indices[left_index];
  p.BoundingBox(bmin, bmax, idx);

  {
    // for each primitive
    for (unsigned int i = left_index + 1; i < right_index; i++) {
      idx = indices[i];
      real3<T> bbox_min, bbox_max;
      p.BoundingBox(&bbox_min, &bbox_max, idx);

      // xyz
      for (int k = 0; k < 3; k++) {
        (*bmin)[k] = std::min((*bmin)[k], bbox_min[k]);
        (*bmax)[k] = std::max((*bmax)[k], bbox_max[k]);
      }
    }
  }
}

template <typename T>
inline void GetBoundingBox(real3<T> *bmin, real3<T> *bmax,
                           const std::vector<BBox<T> > &bboxes,
                           unsigned int *indices, unsigned int left_index,
                           unsigned int right_index) {
  unsigned int i = left_index;
  unsigned int idx = indices[i];

  (*bmin)[0] = bboxes[idx].bmin[0];
  (*bmin)[1] = bboxes[idx].bmin[1];
  (*bmin)[2] = bboxes[idx].bmin[2];
  (*bmax)[0] = bboxes[idx].bmax[0];
  (*bmax)[1] = bboxes[idx].bmax[1];
  (*bmax)[2] = bboxes[idx].bmax[2];

  // for each face
  for (i = left_index + 1; i < right_index; i++) {
    idx = indices[i];

    // xyz
    for (int k = 0; k < 3; k++) {
      (*bmin)[k] = std::min((*bmin)[k], bboxes[idx].bmin[k]);
      (*bmax)[k] = std::max((*bmax)[k], bboxes[idx].bmax[k]);
    }
  }
}

//
// --
//

#if defined(NANORT_ENABLE_PARALLEL_BUILD)
template <typename T>
template <class P, class Pred>
unsigned int BVHAccel<T>::BuildShallowTree(std::vector<BVHNode<T> > *out_nodes,
                                           unsigned int left_idx,
                                           unsigned int right_idx,
                                           unsigned int depth,
                                           unsigned int max_shallow_depth,
                                           const P &p, const Pred &pred) {
  assert(left_idx <= right_idx);

  unsigned int offset = static_cast<unsigned int>(out_nodes->size());

  if (stats_.max_tree_depth < depth) {
    stats_.max_tree_depth = depth;
  }

  real3<T> bmin, bmax;

#if defined(NANORT_USE_CPP11_FEATURE) && defined(NANORT_ENABLE_PARALLEL_BUILD)
  ComputeBoundingBoxThreaded(&bmin, &bmax, &indices_.at(0), left_idx, right_idx,
                             p);
#else
  ComputeBoundingBox(&bmin, &bmax, &indices_.at(0), left_idx, right_idx, p);
#endif

  unsigned int n = right_idx - left_idx;
  if ((n <= options_.min_leaf_primitives) ||
      (depth >= options_.max_tree_depth)) {
    // Create leaf node.
    BVHNode<T> leaf;

    leaf.bmin[0] = bmin[0];
    leaf.bmin[1] = bmin[1];
    leaf.bmin[2] = bmin[2];

    leaf.bmax[0] = bmax[0];
    leaf.bmax[1] = bmax[1];
    leaf.bmax[2] = bmax[2];

    assert(left_idx < std::numeric_limits<unsigned int>::max());

    leaf.flag = 1;  // leaf
    leaf.data[0] = n;
    leaf.data[1] = left_idx;

    out_nodes->push_back(leaf);  // atomic update

    stats_.num_leaf_nodes++;

    return offset;
  }

  //
  // Create branch node.
  //
  if (depth >= max_shallow_depth) {
    // Delay to build tree
    ShallowNodeInfo info;
    info.left_idx = left_idx;
    info.right_idx = right_idx;
    info.offset = offset;
    shallow_node_infos_.push_back(info);

    // Add dummy node.
    BVHNode<T> node;
    node.axis = -1;
    node.flag = -1;
    out_nodes->push_back(node);

    return offset;

  } else {
    //
    // TODO(LTE): multi-threaded SAH computation, or use simple object median or
    // spacial median for shallow tree to speeding up the parallel build.
    //

    //
    // Compute SAH and find best split axis and position
    //
    int min_cut_axis = 0;
    T cut_pos[3] = {0.0, 0.0, 0.0};

    BinBuffer<T> bins(options_.bin_size);
    ContributeBinBuffer(&bins, bmin, bmax, &indices_.at(0), left_idx, right_idx,
                        p);
    FindCutFromBinBuffer(cut_pos, &min_cut_axis, &bins, bmin, bmax);

    // Try all 3 axis until good cut position avaiable.
    unsigned int mid_idx = left_idx;
    int cut_axis = min_cut_axis;

    for (int axis_try = 0; axis_try < 3; axis_try++) {
      unsigned int *begin = &indices_[left_idx];
      unsigned int *end =
          &indices_[right_idx - 1] + 1;  // mimics end() iterator
      unsigned int *mid = 0;

      // try min_cut_axis first.
      cut_axis = (min_cut_axis + axis_try) % 3;

      pred.Set(cut_axis, cut_pos[cut_axis]);
      //
      // Split at (cut_axis, cut_pos)
      // indices_ will be modified.
      //
      mid = std::partition(begin, end, pred);

      mid_idx = left_idx + static_cast<unsigned int>((mid - begin));

      if ((mid_idx == left_idx) || (mid_idx == right_idx)) {
        // Can't split well.
        // Switch to object median (which may create unoptimized tree, but
        // stable)
        mid_idx = left_idx + (n >> 1);

        // Try another axis if there's an axis to try.

      } else {
        // Found good cut. exit loop.
        break;
      }
    }

    BVHNode<T> node;
    node.axis = cut_axis;
    node.flag = 0;  // 0 = branch

    out_nodes->push_back(node);

    unsigned int left_child_index = 0;
    unsigned int right_child_index = 0;

    left_child_index = BuildShallowTree(out_nodes, left_idx, mid_idx, depth + 1,
                                        max_shallow_depth, p, pred);

    right_child_index = BuildShallowTree(out_nodes, mid_idx, right_idx,
                                         depth + 1, max_shallow_depth, p, pred);

    //std::cout << "shallow[" << offset << "] l and r = " << left_child_index << ", " << right_child_index << std::endl;
    (*out_nodes)[offset].data[0] = left_child_index;
    (*out_nodes)[offset].data[1] = right_child_index;

    (*out_nodes)[offset].bmin[0] = bmin[0];
    (*out_nodes)[offset].bmin[1] = bmin[1];
    (*out_nodes)[offset].bmin[2] = bmin[2];

    (*out_nodes)[offset].bmax[0] = bmax[0];
    (*out_nodes)[offset].bmax[1] = bmax[1];
    (*out_nodes)[offset].bmax[2] = bmax[2];
  }

  stats_.num_branch_nodes++;

  return offset;
}
#endif

template <typename T>
template <class P, class Pred>
unsigned int BVHAccel<T>::BuildTree(BVHBuildStatistics *out_stat,
                                    std::vector<BVHNode<T> > *out_nodes,
                                    unsigned int left_idx,
                                    unsigned int right_idx, unsigned int depth,
                                    const P &p, const Pred &pred) {
  assert(left_idx <= right_idx);

  unsigned int offset = static_cast<unsigned int>(out_nodes->size());

  if (out_stat->max_tree_depth < depth) {
    out_stat->max_tree_depth = depth;
  }

  real3<T> bmin, bmax;
  if (!bboxes_.empty()) {
    GetBoundingBox(&bmin, &bmax, bboxes_, &indices_.at(0), left_idx, right_idx);
  } else {
    ComputeBoundingBox(&bmin, &bmax, &indices_.at(0), left_idx, right_idx, p);
  }

  unsigned int n = right_idx - left_idx;
  if ((n <= options_.min_leaf_primitives) ||
      (depth >= options_.max_tree_depth)) {
    // Create leaf node.
    BVHNode<T> leaf;

    leaf.bmin[0] = bmin[0];
    leaf.bmin[1] = bmin[1];
    leaf.bmin[2] = bmin[2];

    leaf.bmax[0] = bmax[0];
    leaf.bmax[1] = bmax[1];
    leaf.bmax[2] = bmax[2];

    assert(left_idx < std::numeric_limits<unsigned int>::max());

    leaf.flag = 1;  // leaf
    leaf.data[0] = n;
    leaf.data[1] = left_idx;

    out_nodes->push_back(leaf);  // atomic update

    out_stat->num_leaf_nodes++;

    return offset;
  }

  //
  // Create branch node.
  //

  //
  // Compute SAH and find best split axis and position
  //
  int min_cut_axis = 0;
  T cut_pos[3] = {0.0, 0.0, 0.0};

  BinBuffer<T> bins(options_.bin_size);
  ContributeBinBuffer(&bins, bmin, bmax, &indices_.at(0), left_idx, right_idx,
                      p);
  FindCutFromBinBuffer(cut_pos, &min_cut_axis, &bins, bmin, bmax);

  // Try all 3 axis until good cut position avaiable.
  unsigned int mid_idx = left_idx;
  int cut_axis = min_cut_axis;

  for (int axis_try = 0; axis_try < 3; axis_try++) {
    unsigned int *begin = &indices_[left_idx];
    unsigned int *end = &indices_[right_idx - 1] + 1;  // mimics end() iterator.
    unsigned int *mid = 0;

    // try min_cut_axis first.
    cut_axis = (min_cut_axis + axis_try) % 3;

    pred.Set(cut_axis, cut_pos[cut_axis]);

    //
    // Split at (cut_axis, cut_pos)
    // indices_ will be modified.
    //
    mid = std::partition(begin, end, pred);

    mid_idx = left_idx + static_cast<unsigned int>((mid - begin));

    if ((mid_idx == left_idx) || (mid_idx == right_idx)) {
      // Can't split well.
      // Switch to object median(which may create unoptimized tree, but
      // stable)
      mid_idx = left_idx + (n >> 1);

      // Try another axis to find better cut.

    } else {
      // Found good cut. exit loop.
      break;
    }
  }

  BVHNode<T> node;
  node.axis = cut_axis;
  node.flag = 0;  // 0 = branch

  out_nodes->push_back(node);

  unsigned int left_child_index = 0;
  unsigned int right_child_index = 0;

  left_child_index =
      BuildTree(out_stat, out_nodes, left_idx, mid_idx, depth + 1, p, pred);

  right_child_index =
      BuildTree(out_stat, out_nodes, mid_idx, right_idx, depth + 1, p, pred);

  {
    (*out_nodes)[offset].data[0] = left_child_index;
    (*out_nodes)[offset].data[1] = right_child_index;

    (*out_nodes)[offset].bmin[0] = bmin[0];
    (*out_nodes)[offset].bmin[1] = bmin[1];
    (*out_nodes)[offset].bmin[2] = bmin[2];

    (*out_nodes)[offset].bmax[0] = bmax[0];
    (*out_nodes)[offset].bmax[1] = bmax[1];
    (*out_nodes)[offset].bmax[2] = bmax[2];
  }

  out_stat->num_branch_nodes++;

  return offset;
}

template <typename T>
template <class Prim, class Pred>
bool BVHAccel<T>::Build(unsigned int num_primitives, const Prim &p,
                        const Pred &pred, const BVHBuildOptions<T> &options) {
  options_ = options;
  stats_ = BVHBuildStatistics();

  nodes_.clear();
  bboxes_.clear();
#if defined(NANORT_ENABLE_PARALLEL_BUILD)
  shallow_node_infos_.clear();
#endif

  assert(options_.bin_size > 1);

  if (num_primitives == 0) {
    return false;
  }

  unsigned int n = num_primitives;

  //
  // 1. Create triangle indices(this will be permutated in BuildTree)
  //
  indices_.resize(n);

#if defined(NANORT_USE_CPP11_FEATURE)
  {
    // Only use threading for initialization if there's enough work
    const size_t min_work_per_thread = 1024;
    size_t num_threads = std::min(
        size_t(kNANORT_MAX_THREADS),
        std::max(size_t(1), std::min(size_t(std::thread::hardware_concurrency()),
                                     n / min_work_per_thread)));

    if (num_threads <= 1 || n < min_work_per_thread * 2) {
      // Serial initialization for small arrays
      for (size_t k = 0; k < n; k++) {
        indices_[k] = static_cast<unsigned int>(k);
      }
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_threads);

      const size_t work_per_thread = n / num_threads;

      for (size_t t = 0; t < num_threads; t++) {
        workers.emplace_back([&, t]() {
          const size_t start = t * work_per_thread;
          const size_t end = (t == num_threads - 1) ? n : (t + 1) * work_per_thread;

          // Vectorized initialization in chunks
          constexpr size_t chunk_size = 64;
          for (size_t i = start; i < end; i += chunk_size) {
            const size_t chunk_end = std::min(i + chunk_size, end);
            for (size_t k = i; k < chunk_end; k++) {
              indices_[k] = static_cast<unsigned int>(k);
            }
          }
        });
      }

      for (auto &worker : workers) {
        worker.join();
      }
    }
  }

#else

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < static_cast<int>(n); i++) {
    indices_[static_cast<size_t>(i)] = static_cast<unsigned int>(i);
  }
#endif  // !NANORT_USE_CPP11_FEATURE

  //
  // 2. Compute bounding box (optional).
  //
  real3<T> bmin, bmax;

  if (options.cache_bbox) {
    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<T>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<T>::max();

    bboxes_.resize(n);

    for (size_t i = 0; i < n; i++) {  // for each primitive
      unsigned int idx = indices_[i];

      BBox<T> bbox;
      p.BoundingBox(&(bbox.bmin), &(bbox.bmax), static_cast<unsigned int>(i));
      bboxes_[idx] = bbox;

      // xyz
      for (int k = 0; k < 3; k++) {
        bmin[k] = std::min(bmin[k], bbox.bmin[k]);
        bmax[k] = std::max(bmax[k], bbox.bmax[k]);
      }
    }

  } else {
#if defined(NANORT_USE_CPP11_FEATURE)
    ComputeBoundingBoxThreaded(&bmin, &bmax, &indices_.at(0), 0, n, p);
#elif defined(_OPENMP)
    ComputeBoundingBoxOMP(&bmin, &bmax, &indices_.at(0), 0, n, p);
#else
    ComputeBoundingBox(&bmin, &bmax, &indices_.at(0), 0, n, p);
#endif
  }

//
// 3. Build tree
//
#if defined(NANORT_ENABLE_PARALLEL_BUILD)
#if defined(NANORT_USE_CPP11_FEATURE)

  // Do parallel build for large enough datasets.
  if (n > options.min_primitives_for_parallel_build) {
    BuildShallowTree(&nodes_, 0, n, /* root depth */ 0, options.shallow_depth,
                     p, pred);  // [0, n)

    assert(shallow_node_infos_.size() > 0);

    // Build deeper tree in parallel with optimized load balancing
    std::vector<std::vector<BVHNode<T> > > local_nodes(
        shallow_node_infos_.size());
    std::vector<BVHBuildStatistics> local_stats(shallow_node_infos_.size());

    size_t num_threads = std::min(
        size_t(kNANORT_MAX_THREADS),
        std::max(size_t(1), size_t(std::thread::hardware_concurrency())));
    if (shallow_node_infos_.size() < num_threads) {
      num_threads = shallow_node_infos_.size();
    }

    // Pre-allocate to reduce memory allocations during parallel execution
    for (size_t i = 0; i < shallow_node_infos_.size(); i++) {
      const size_t work_size = shallow_node_infos_[i].right_idx - shallow_node_infos_[i].left_idx;
      local_nodes[i].reserve(work_size * 2);  // Rough estimate for node count
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    std::atomic<uint32_t> work_counter(0);

    // Use work-stealing approach with better granularity
    for (size_t t = 0; t < num_threads; t++) {
      workers.emplace_back([&]() {
        // Create thread-local copy of Pred once per thread
        const Pred local_pred = pred;
        
        uint32_t idx = 0;
        while ((idx = work_counter.fetch_add(1, std::memory_order_relaxed)) < shallow_node_infos_.size()) {
          const size_t task_idx = size_t(idx);
          const unsigned int left_idx = shallow_node_infos_[task_idx].left_idx;
          const unsigned int right_idx = shallow_node_infos_[task_idx].right_idx;
          
          BuildTree(&(local_stats[task_idx]), &(local_nodes[task_idx]),
                    left_idx, right_idx, options.shallow_depth, p, local_pred);
        }
      });
    }

    for (auto &t : workers) {
      t.join();
    }

    // Join local nodes
    for (size_t ii = 0; ii < local_nodes.size(); ii++) {
      assert(!local_nodes[ii].empty());
      size_t offset = nodes_.size();

      // Add offset to child index (for branch node).
      for (size_t j = 0; j < local_nodes[ii].size(); j++) {
        if (local_nodes[ii][j].flag == 0) {  // branch
          local_nodes[ii][j].data[0] += offset - 1;
          local_nodes[ii][j].data[1] += offset - 1;
        }
      }

      // replace
      nodes_[shallow_node_infos_[ii].offset] = local_nodes[ii][0];

      // Skip root element of the local node.
      nodes_.insert(nodes_.end(), local_nodes[ii].begin() + 1,
                    local_nodes[ii].end());
    }

    // Join statistics
    for (size_t ii = 0; ii < local_nodes.size(); ii++) {
      stats_.max_tree_depth =
          std::max(stats_.max_tree_depth, local_stats[ii].max_tree_depth);
      stats_.num_leaf_nodes += local_stats[ii].num_leaf_nodes;
      stats_.num_branch_nodes += local_stats[ii].num_branch_nodes;
    }

  } else {
    // Single thread.
    BuildTree(&stats_, &nodes_, 0, n,
              /* root depth */ 0, p, pred);  // [0, n)
  }

#elif defined(_OPENMP)

  // Do parallel build for large enough datasets.
  if (n > options.min_primitives_for_parallel_build) {
    BuildShallowTree(&nodes_, 0, n, /* root depth */ 0, options.shallow_depth,
                     p, pred);  // [0, n)

    assert(shallow_node_infos_.size() > 0);

    // Build deeper tree in parallel
    std::vector<std::vector<BVHNode<T> > > local_nodes(
        shallow_node_infos_.size());
    std::vector<BVHBuildStatistics> local_stats(shallow_node_infos_.size());

#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(shallow_node_infos_.size()); i++) {
      unsigned int left_idx = shallow_node_infos_[size_t(i)].left_idx;
      unsigned int right_idx = shallow_node_infos_[size_t(i)].right_idx;
      const Pred local_pred = pred;
      BuildTree(&(local_stats[size_t(i)]), &(local_nodes[size_t(i)]), left_idx,
                right_idx, options.shallow_depth, p, local_pred);
    }

    // Join local nodes
    for (size_t i = 0; i < local_nodes.size(); i++) {
      assert(!local_nodes[size_t(i)].empty());
      size_t offset = nodes_.size();

      // Add offset to child index (for branch node).
      for (size_t j = 0; j < local_nodes[i].size(); j++) {
        if (local_nodes[i][j].flag == 0) {  // branch
          local_nodes[i][j].data[0] += offset - 1;
          local_nodes[i][j].data[1] += offset - 1;
        }
      }

      // replace
      nodes_[shallow_node_infos_[i].offset] = local_nodes[i][0];

      // Skip root element of the local node.
      nodes_.insert(nodes_.end(), local_nodes[i].begin() + 1,
                    local_nodes[i].end());
    }

    // Join statistics
    for (size_t i = 0; i < local_nodes.size(); i++) {
      stats_.max_tree_depth =
          std::max(stats_.max_tree_depth, local_stats[i].max_tree_depth);
      stats_.num_leaf_nodes += local_stats[i].num_leaf_nodes;
      stats_.num_branch_nodes += local_stats[i].num_branch_nodes;
    }

  } else {
    // Single thread
    BuildTree(&stats_, &nodes_, 0, n,
              /* root depth */ 0, p, pred);  // [0, n)
  }

#else  // !NANORT_ENABLE_PARALLEL_BUILD
  {
    BuildTree(&stats_, &nodes_, 0, n,
              /* root depth */ 0, p, pred);  // [0, n)
  }
#endif
#else  // !_OPENMP

  // Single thread BVH build
  {
    BuildTree(&stats_, &nodes_, 0, n,
              /* root depth */ 0, p, pred);  // [0, n)
  }
#endif

  return true;
}

template <typename T>
void BVHAccel<T>::Debug() {
  for (size_t i = 0; i < indices_.size(); i++) {
    printf("index[%d] = %d\n", int(i), int(indices_[i]));
  }

  for (size_t i = 0; i < nodes_.size(); i++) {
    printf("node[%d] : bmin %f, %f, %f, bmax %f, %f, %f\n", int(i),
           nodes_[i].bmin[0], nodes_[i].bmin[1], nodes_[i].bmin[2],
           nodes_[i].bmax[0], nodes_[i].bmax[1], nodes_[i].bmax[2]);
  }
}

#if defined(NANORT_ENABLE_SERIALIZATION)
template <typename T>
bool BVHAccel<T>::Dump(const char *filename) const {
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    // fprintf(stderr, "[BVHAccel] Cannot write a file: %s\n", filename);
    return false;
  }

  size_t numNodes = nodes_.size();
  assert(nodes_.size() > 0);

  size_t numIndices = indices_.size();

  size_t r = 0;
  r = fwrite(&numNodes, sizeof(size_t), 1, fp);
  assert(r == 1);

  r = fwrite(&nodes_.at(0), sizeof(BVHNode<T>), numNodes, fp);
  assert(r == numNodes);

  r = fwrite(&numIndices, sizeof(size_t), 1, fp);
  assert(r == 1);

  r = fwrite(&indices_.at(0), sizeof(unsigned int), numIndices, fp);
  assert(r == numIndices);

  fclose(fp);

  return true;
}

template <typename T>
bool BVHAccel<T>::Dump(FILE *fp) const {
  size_t numNodes = nodes_.size();
  assert(nodes_.size() > 0);

  size_t numIndices = indices_.size();

  size_t r = 0;
  r = fwrite(&numNodes, sizeof(size_t), 1, fp);
  assert(r == 1);

  r = fwrite(&nodes_.at(0), sizeof(BVHNode<T>), numNodes, fp);
  assert(r == numNodes);

  r = fwrite(&numIndices, sizeof(size_t), 1, fp);
  assert(r == 1);

  r = fwrite(&indices_.at(0), sizeof(unsigned int), numIndices, fp);
  assert(r == numIndices);

  return true;
}

template <typename T>
bool BVHAccel<T>::Load(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    // fprintf(stderr, "Cannot open file: %s\n", filename);
    return false;
  }

  size_t numNodes;
  size_t numIndices;

  size_t r = 0;
  r = fread(&numNodes, sizeof(size_t), 1, fp);
  assert(r == 1);
  assert(numNodes > 0);

  nodes_.resize(numNodes);
  r = fread(&nodes_.at(0), sizeof(BVHNode<T>), numNodes, fp);
  assert(r == numNodes);

  r = fread(&numIndices, sizeof(size_t), 1, fp);
  assert(r == 1);

  indices_.resize(numIndices);

  r = fread(&indices_.at(0), sizeof(unsigned int), numIndices, fp);
  assert(r == numIndices);

  fclose(fp);

  return true;
}

template <typename T>
bool BVHAccel<T>::Load(FILE *fp) {
  size_t numNodes;
  size_t numIndices;

  size_t r = 0;
  r = fread(&numNodes, sizeof(size_t), 1, fp);
  assert(r == 1);
  assert(numNodes > 0);

  nodes_.resize(numNodes);
  r = fread(&nodes_.at(0), sizeof(BVHNode<T>), numNodes, fp);
  assert(r == numNodes);

  r = fread(&numIndices, sizeof(size_t), 1, fp);
  assert(r == 1);

  indices_.resize(numIndices);

  r = fread(&indices_.at(0), sizeof(unsigned int), numIndices, fp);
  assert(r == numIndices);

  return true;
}
#endif

template <typename T>
inline bool IntersectRayAABB(T *tminOut,  // [out]
                             T *tmaxOut,  // [out]
                             T min_t, T max_t, const T bmin[3], const T bmax[3],
                             real3<T> ray_org, real3<T> ray_inv_dir,
                             int ray_dir_sign[3]);
template <>
inline bool IntersectRayAABB<float>(float *tminOut,  // [out]
                                    float *tmaxOut,  // [out]
                                    float min_t, float max_t,
                                    const float bmin[3], const float bmax[3],
                                    real3<float> ray_org,
                                    real3<float> ray_inv_dir,
                                    int ray_dir_sign[3]) {
  float tmin, tmax;

  const float min_x = ray_dir_sign[0] ? bmax[0] : bmin[0];
  const float min_y = ray_dir_sign[1] ? bmax[1] : bmin[1];
  const float min_z = ray_dir_sign[2] ? bmax[2] : bmin[2];
  const float max_x = ray_dir_sign[0] ? bmin[0] : bmax[0];
  const float max_y = ray_dir_sign[1] ? bmin[1] : bmax[1];
  const float max_z = ray_dir_sign[2] ? bmin[2] : bmax[2];

  // X
  const float tmin_x = (min_x - ray_org[0]) * ray_inv_dir[0];
  // MaxMult robust BVH traversal(up to 4 ulp).
  // 1.0000000000000004 for double precision.
  const float tmax_x = (max_x - ray_org[0]) * ray_inv_dir[0] * 1.00000024f;

  // Y
  const float tmin_y = (min_y - ray_org[1]) * ray_inv_dir[1];
  const float tmax_y = (max_y - ray_org[1]) * ray_inv_dir[1] * 1.00000024f;

  // Z
  const float tmin_z = (min_z - ray_org[2]) * ray_inv_dir[2];
  const float tmax_z = (max_z - ray_org[2]) * ray_inv_dir[2] * 1.00000024f;

  tmin = safemax(tmin_z, safemax(tmin_y, safemax(tmin_x, min_t)));
  tmax = safemin(tmax_z, safemin(tmax_y, safemin(tmax_x, max_t)));

  if (tmin <= tmax) {
    (*tminOut) = tmin;
    (*tmaxOut) = tmax;

    return true;
  }
  return false;  // no hit
}

template <>
inline bool IntersectRayAABB<double>(double *tminOut,  // [out]
                                     double *tmaxOut,  // [out]
                                     double min_t, double max_t,
                                     const double bmin[3], const double bmax[3],
                                     real3<double> ray_org,
                                     real3<double> ray_inv_dir,
                                     int ray_dir_sign[3]) {
  double tmin, tmax;

  const double min_x = ray_dir_sign[0] ? bmax[0] : bmin[0];
  const double min_y = ray_dir_sign[1] ? bmax[1] : bmin[1];
  const double min_z = ray_dir_sign[2] ? bmax[2] : bmin[2];
  const double max_x = ray_dir_sign[0] ? bmin[0] : bmax[0];
  const double max_y = ray_dir_sign[1] ? bmin[1] : bmax[1];
  const double max_z = ray_dir_sign[2] ? bmin[2] : bmax[2];

  // X
  const double tmin_x = (min_x - ray_org[0]) * ray_inv_dir[0];
  // MaxMult robust BVH traversal(up to 4 ulp).
  const double tmax_x =
      (max_x - ray_org[0]) * ray_inv_dir[0] * 1.0000000000000004;

  // Y
  const double tmin_y = (min_y - ray_org[1]) * ray_inv_dir[1];
  const double tmax_y =
      (max_y - ray_org[1]) * ray_inv_dir[1] * 1.0000000000000004;

  // Z
  const double tmin_z = (min_z - ray_org[2]) * ray_inv_dir[2];
  const double tmax_z =
      (max_z - ray_org[2]) * ray_inv_dir[2] * 1.0000000000000004;

  tmin = safemax(tmin_z, safemax(tmin_y, safemax(tmin_x, min_t)));
  tmax = safemin(tmax_z, safemin(tmax_y, safemin(tmax_x, max_t)));

  if (tmin <= tmax) {
    (*tminOut) = tmin;
    (*tmaxOut) = tmax;

    return true;
  }
  return false;  // no hit
}

// ================================================================================
// SIMD-optimized alternative ray-AABB intersection functions  
// ================================================================================
// These are separate functions to avoid conflicts with the original implementations
#if !defined(NANORT_DISABLE_SIMD)

#ifdef NANORT_ENABLE_SSE2
inline bool IntersectRayAABB_SIMD_SSE2(float *tminOut, float *tmaxOut,
                                       float min_t, float max_t,
                                       const float bmin[3], const float bmax[3],
                                       const float ray_org[3],
                                       const float ray_inv_dir[3]) {
  // Load ray data into SIMD registers
  const __m128 ray_org_simd = _mm_set_ps(0.0f, ray_org[2], ray_org[1], ray_org[0]);
  const __m128 ray_inv_dir_simd = _mm_set_ps(0.0f, ray_inv_dir[2], ray_inv_dir[1], ray_inv_dir[0]);
  
  // Load AABB data  
  const __m128 bmin_simd = _mm_set_ps(0.0f, bmin[2], bmin[1], bmin[0]);
  const __m128 bmax_simd = _mm_set_ps(0.0f, bmax[2], bmax[1], bmax[0]);
  
  // Calculate intersection points
  const __m128 tmin_vals = _mm_mul_ps(_mm_sub_ps(bmin_simd, ray_org_simd), ray_inv_dir_simd);
  const __m128 tmax_vals = _mm_mul_ps(_mm_sub_ps(bmax_simd, ray_org_simd), ray_inv_dir_simd);
  
  // Handle ray direction signs
  const __m128 real_tmin = _mm_min_ps(tmin_vals, tmax_vals);
  const __m128 real_tmax = _mm_max_ps(tmin_vals, tmax_vals);
  
  // Extract components
  float tmin_arr[4], tmax_arr[4];
  _mm_store_ps(tmin_arr, real_tmin);
  _mm_store_ps(tmax_arr, real_tmax);
  
  // Apply robust MaxMult factor
  tmax_arr[0] *= 1.00000024f;
  tmax_arr[1] *= 1.00000024f; 
  tmax_arr[2] *= 1.00000024f;
  
  // Find overall tmin and tmax
  float tmin = safemax(tmin_arr[2], safemax(tmin_arr[1], safemax(tmin_arr[0], min_t)));
  float tmax = safemin(tmax_arr[2], safemin(tmax_arr[1], safemin(tmax_arr[0], max_t)));

  if (tmin <= tmax) {
    (*tminOut) = tmin;
    (*tmaxOut) = tmax;
    return true;
  }
  return false;
}
#endif

#ifdef NANORT_ENABLE_AVX2
inline bool IntersectRayAABB_SIMD_AVX2(float *tminOut, float *tmaxOut,
                                       float min_t, float max_t,
                                       const float bmin[3], const float bmax[3],
                                       const float ray_org[3],
                                       const float ray_inv_dir[3]) {
  // Use 256-bit registers 
  const __m256 ray_org_simd = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, ray_org[2], ray_org[1], ray_org[0]);
  const __m256 ray_inv_dir_simd = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, ray_inv_dir[2], ray_inv_dir[1], ray_inv_dir[0]);
  
  const __m256 bmin_simd = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, bmin[2], bmin[1], bmin[0]);
  const __m256 bmax_simd = _mm256_set_ps(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, bmax[2], bmax[1], bmax[0]);
  
  const __m256 tmin_vals = _mm256_mul_ps(_mm256_sub_ps(bmin_simd, ray_org_simd), ray_inv_dir_simd);
  const __m256 tmax_vals = _mm256_mul_ps(_mm256_sub_ps(bmax_simd, ray_org_simd), ray_inv_dir_simd);
  
  const __m256 real_tmin = _mm256_min_ps(tmin_vals, tmax_vals);
  const __m256 real_tmax = _mm256_max_ps(tmin_vals, tmax_vals);
  
  float tmin_arr[8], tmax_arr[8];
  _mm256_store_ps(tmin_arr, real_tmin);
  _mm256_store_ps(tmax_arr, real_tmax);
  
  tmax_arr[0] *= 1.00000024f;
  tmax_arr[1] *= 1.00000024f;
  tmax_arr[2] *= 1.00000024f;
  
  float tmin = safemax(tmin_arr[2], safemax(tmin_arr[1], safemax(tmin_arr[0], min_t)));
  float tmax = safemin(tmax_arr[2], safemin(tmax_arr[1], safemin(tmax_arr[0], max_t)));

  if (tmin <= tmax) {
    (*tminOut) = tmin;
    (*tmaxOut) = tmax;
    return true;
  }
  return false;
}
#endif

#ifdef NANORT_ENABLE_NEON
inline bool IntersectRayAABB_SIMD_NEON(float *tminOut, float *tmaxOut,
                                       float min_t, float max_t,
                                       const float bmin[3], const float bmax[3],
                                       const float ray_org[3],
                                       const float ray_inv_dir[3]) {
  // Load data into NEON registers
  const float32x4_t ray_org_neon = {ray_org[0], ray_org[1], ray_org[2], 0.0f};
  const float32x4_t ray_inv_dir_neon = {ray_inv_dir[0], ray_inv_dir[1], ray_inv_dir[2], 0.0f};
  const float32x4_t bmin_neon = {bmin[0], bmin[1], bmin[2], 0.0f};
  const float32x4_t bmax_neon = {bmax[0], bmax[1], bmax[2], 0.0f};
  
  const float32x4_t tmin_vals = vmulq_f32(vsubq_f32(bmin_neon, ray_org_neon), ray_inv_dir_neon);
  const float32x4_t tmax_vals = vmulq_f32(vsubq_f32(bmax_neon, ray_org_neon), ray_inv_dir_neon);
  
  const float32x4_t real_tmin = vminq_f32(tmin_vals, tmax_vals);
  const float32x4_t real_tmax = vmaxq_f32(tmin_vals, tmax_vals);
  
  float tmin_arr[4], tmax_arr[4];
  vst1q_f32(tmin_arr, real_tmin);
  vst1q_f32(tmax_arr, real_tmax);
  
  tmax_arr[0] *= 1.00000024f;
  tmax_arr[1] *= 1.00000024f;
  tmax_arr[2] *= 1.00000024f;
  
  float tmin = safemax(tmin_arr[2], safemax(tmin_arr[1], safemax(tmin_arr[0], min_t)));
  float tmax = safemin(tmax_arr[2], safemin(tmax_arr[1], safemin(tmax_arr[0], max_t)));

  if (tmin <= tmax) {
    (*tminOut) = tmin;
    (*tmaxOut) = tmax;
    return true;
  }
  return false;
}
#endif // NANORT_ENABLE_NEON

#endif // !NANORT_DISABLE_SIMD

template <typename T>
template <class I>
inline bool BVHAccel<T>::TestLeafNode(const BVHNode<T> &node, const Ray<T> &ray,
                                      const I &intersector, const BVHTraceOptions &options) const {
  bool hit = false;

  unsigned int num_primitives = node.data[0];
  unsigned int offset = node.data[1];

  T t = intersector.GetT();  // current hit distance

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  real3<T> ray_dir;
  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  for (unsigned int i = 0; i < num_primitives; i++) {
    unsigned int prim_idx = indices_[i + offset];

    // Count primitive intersection tests
    options.primitive_intersections++;

    T local_t = t;
    if (intersector.Intersect(&local_t, prim_idx)) {
      // Update isect state
      t = local_t;

      intersector.Update(t, prim_idx);
      hit = true;
    }
  }

  return hit;
}

#if 0  // TODO(LTE): Implement
template <typename T> template<class I, class H, class Comp>
bool BVHAccel<T>::MultiHitTestLeafNode(
  std::priority_queue<H, std::vector<H>, Comp>  *isect_pq,
  int max_intersections,
  const BVHNode<T> &node,
  const Ray<T> &ray,
  const I &intersector) const {
  bool hit = false;

  unsigned int num_primitives = node.data[0];
  unsigned int offset = node.data[1];

  T t = std::numeric_limits<T>::max();
  if (isect_pq->size() >= static_cast<size_t>(max_intersections)) {
    t = isect_pq->top().t;  // current furthest hit distance
  }

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  real3<T> ray_dir;
  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  for (unsigned int i = 0; i < num_primitives; i++) {
    unsigned int prim_idx = indices_[i + offset];

    T local_t = t, u = 0.0f, v = 0.0f;

    if (intersector.Intersect(&local_t, &u, &v, prim_idx))
    {
      // Update isect state
      if ((local_t > ray.min_t))
      {
        if (isect_pq->size() < static_cast<size_t>(max_intersections))
        {
          H isect;
          t = local_t;
          isect.t = t;
          isect.u = u;
          isect.v = v;
          isect.prim_id = prim_idx;
          isect_pq->push(isect);

          // Update t to furthest distance.
          t = ray.max_t;

          hit = true;
        }
        else if (local_t < isect_pq->top().t)
        {
          // delete furthest intersection and add new intersection.
          isect_pq->pop();

          H hit;
          hit.t = local_t;
          hit.u = u;
          hit.v = v;
          hit.prim_id = prim_idx;
          isect_pq->push(hit);

          // Update furthest hit distance
          t = isect_pq->top().t;

          hit = true;
        }
      }
    }
  }

  return hit;
}
#endif

template <typename T>
template <class I, class H>
bool BVHAccel<T>::Traverse(const Ray<T> &ray, const I &intersector, H *isect,
                           const BVHTraceOptions &options) const {
  const int kMaxStackDepth = 512;
  (void)kMaxStackDepth;

  T hit_t = ray.max_t;

  int node_stack_index = 0;
  unsigned int node_stack[512];
  node_stack[0] = 0;

  // Init isect info as no hit
  intersector.Update(hit_t, static_cast<unsigned int>(-1));

  intersector.PrepareTraversal(ray, options);

  int dir_sign[3];
  dir_sign[0] = ray.dir[0] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[1] = ray.dir[1] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[2] = ray.dir[2] < static_cast<T>(0.0) ? 1 : 0;

  real3<T> ray_inv_dir;
  real3<T> ray_dir;
  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  ray_inv_dir = vsafe_inverse(ray_dir);

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  T min_t = std::numeric_limits<T>::max();
  T max_t = -std::numeric_limits<T>::max();

  while (node_stack_index >= 0) {
    unsigned int index = node_stack[node_stack_index];
    const BVHNode<T> &node = nodes_[index];

    node_stack_index--;

    bool hit = IntersectRayAABB(&min_t, &max_t, ray.min_t, hit_t, node.bmin,
                                node.bmax, ray_org, ray_inv_dir, dir_sign);

    // Count bounding box intersection tests
    options.bbox_intersections++;

    if (hit) {
      // Branch node
      if (node.flag == 0) {
        int order_near = dir_sign[node.axis];
        int order_far = 1 - order_near;

        // Traverse near first.
        node_stack[++node_stack_index] = node.data[order_far];
        node_stack[++node_stack_index] = node.data[order_near];
      } else if (TestLeafNode(node, ray, intersector, options)) {  // Leaf node
        hit_t = intersector.GetT();
      }
    }
  }

  assert(node_stack_index < kNANORT_MAX_STACK_DEPTH);

  bool hit = (intersector.GetT() < ray.max_t);
  intersector.PostTraversal(ray, hit, isect);

  return hit;
}

template <typename T>
template <class I>
inline bool BVHAccel<T>::TestLeafNodeIntersections(
    const BVHNode<T> &node, const Ray<T> &ray, const int max_intersections,
    const I &intersector,
    std::priority_queue<NodeHit<T>, std::vector<NodeHit<T> >,
                        NodeHitComparator<T> > *isect_pq) const {
  bool hit = false;

  unsigned int num_primitives = node.data[0];
  unsigned int offset = node.data[1];

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  real3<T> ray_dir;
  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  intersector.PrepareTraversal(ray);

  for (unsigned int i = 0; i < num_primitives; i++) {
    unsigned int prim_idx = indices_[i + offset];

    T min_t, max_t;

    if (intersector.Intersect(&min_t, &max_t, prim_idx)) {
      // Always add to isect lists.
      NodeHit<T> isect;
      isect.t_min = min_t;
      isect.t_max = max_t;
      isect.node_id = prim_idx;

      if (isect_pq->size() < static_cast<size_t>(max_intersections)) {
        isect_pq->push(isect);
      } else if (min_t < isect_pq->top().t_min) {
        // delete the furthest intersection and add a new intersection.
        isect_pq->pop();

        isect_pq->push(isect);
      }
    }
  }

  return hit;
}

template <typename T>
template <class I>
bool BVHAccel<T>::ListNodeIntersections(
    const Ray<T> &ray, int max_intersections, const I &intersector,
    StackVector<NodeHit<T>, 128> *hits) const {
  const int kMaxStackDepth = 512;

  T hit_t = ray.max_t;

  int node_stack_index = 0;
  unsigned int node_stack[512];
  node_stack[0] = 0;

  // Stores furthest intersection at top
  std::priority_queue<NodeHit<T>, std::vector<NodeHit<T> >,
                      NodeHitComparator<T> >
      isect_pq;

  (*hits)->clear();

  int dir_sign[3];
  dir_sign[0] = ray.dir[0] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[1] = ray.dir[1] < static_cast<T>(0.0) ? 1 : 0;
  dir_sign[2] = ray.dir[2] < static_cast<T>(0.0) ? 1 : 0;

  real3<T> ray_inv_dir;
  real3<T> ray_dir;

  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  ray_inv_dir = vsafe_inverse(ray_dir);

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  T min_t, max_t;

  while (node_stack_index >= 0) {
    unsigned int index = node_stack[node_stack_index];
    const BVHNode<T> &node = nodes_[static_cast<size_t>(index)];

    node_stack_index--;

    bool hit = IntersectRayAABB(&min_t, &max_t, ray.min_t, hit_t, node.bmin,
                                node.bmax, ray_org, ray_inv_dir, dir_sign);

    if (hit) {
      // Branch node
      if (node.flag == 0) {
        int order_near = dir_sign[node.axis];
        int order_far = 1 - order_near;

        // Traverse near first.
        node_stack[++node_stack_index] = node.data[order_far];
        node_stack[++node_stack_index] = node.data[order_near];
      } else {  // Leaf node
        TestLeafNodeIntersections(node, ray, max_intersections, intersector,
                                  &isect_pq);
      }
    }
  }

  assert(node_stack_index < kMaxStackDepth);
  (void)kMaxStackDepth;

  if (!isect_pq.empty()) {
    // Store intesection in reverse order (make it frontmost order)
    size_t n = isect_pq.size();
    (*hits)->resize(n);

    for (size_t i = 0; i < n; i++) {
      const NodeHit<T> &isect = isect_pq.top();
      (*hits)[n - i - 1] = isect;
      isect_pq.pop();
    }

    return true;
  }

  return false;
}

#if 0  // TODO(LTE): Implement
template <typename T> template<class I, class H, class Comp>
bool BVHAccel<T>::MultiHitTraverse(const Ray<T> &ray,
                                         int max_intersections,
                                         const I &intersector,
                                         StackVector<H, 128> *hits,
                                         const BVHTraceOptions& options) const {
  const int kMaxStackDepth = 512;

  T hit_t = ray.max_t;

  int node_stack_index = 0;
  unsigned int node_stack[512];
  node_stack[0] = 0;

  // Stores furthest intersection at top
  std::priority_queue<H, std::vector<H>, Comp>  isect_pq;

  (*hits)->clear();

  // Init isect info as no hit
  intersector.Update(hit_t, static_cast<unsigned int>(-1));

  intersector.PrepareTraversal(ray, options);

  int dir_sign[3];
  dir_sign[0] = ray.dir[0] < static_cast<T>(0.0) ? static_cast<T>(1) : static_cast<T>(0);
  dir_sign[1] = ray.dir[1] < static_cast<T>(0.0) ? static_cast<T>(1) : static_cast<T>(0);
  dir_sign[2] = ray.dir[2] < static_cast<T>(0.0) ? static_cast<T>(1) : static_cast<T>(0);

  real3<T> ray_inv_dir;
  real3<T> ray_dir;

  ray_dir[0] = ray.dir[0];
  ray_dir[1] = ray.dir[1];
  ray_dir[2] = ray.dir[2];

  ray_inv_dir = vsafe_inverse(ray_dir);

  real3<T> ray_org;
  ray_org[0] = ray.org[0];
  ray_org[1] = ray.org[1];
  ray_org[2] = ray.org[2];

  T min_t, max_t;

  while (node_stack_index >= 0)
  {
    unsigned int index = node_stack[node_stack_index];
    const BVHNode<T> &node = nodes_[static_cast<size_t>(index)];

    node_stack_index--;

    bool hit = IntersectRayAABB(&min_t, &max_t, ray.min_t, hit_t, node.bmin,
                                node.bmax, ray_org, ray_inv_dir, dir_sign);

    // branch node
    if(hit)
    {
      if (node.flag == 0)
      {
        int order_near = dir_sign[node.axis];
        int order_far = 1 - order_near;

        // Traverse near first.
        node_stack[++node_stack_index] = node.data[order_far];
        node_stack[++node_stack_index] = node.data[order_near];
      }
      else
      {
        if (MultiHitTestLeafNode(&isect_pq, max_intersections, node, ray, intersector))
        {
          // Only update `hit_t` when queue is full.
          if (isect_pq.size() >= static_cast<size_t>(max_intersections))
          {
            hit_t = isect_pq.top().t;
          }
        }
      }
    }
  }

  assert(node_stack_index < kMaxStackDepth);
  (void)kMaxStackDepth;

  if (!isect_pq.empty())
  {
    // Store intesection in reverse order (make it frontmost order)
    size_t n = isect_pq.size();
    (*hits)->resize(n);

    for (size_t i = 0; i < n; i++)
    {
      const H &isect = isect_pq.top();
      (*hits)[n - i - 1] = isect;
      isect_pq.pop();
    }

    return true;
  }

  return false;
}
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

}  // namespace nanort

#endif  // NANORT_H_
