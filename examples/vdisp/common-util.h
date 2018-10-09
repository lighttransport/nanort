#ifndef EXAMPLE_COMMON_UTIL_H_
#define EXAMPLE_COMMON_UTIL_H_

#define NANORT_USE_CPP11_FEATURE
#define NANORT_ENABLE_PARALLEL_BUILD
#include "../../nanort.h"

#include <cstdint>

namespace example {

typedef nanort::real3<float> float3;

// PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// http://www.pcg-random.org/
typedef struct {
  uint64_t state;
  uint64_t inc;  // not used?
} pcg32_state_t;

#define PCG32_INITIALIZER \
  { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

inline float pcg32_random(pcg32_state_t* rng) {
  uint64_t oldstate = rng->state;
  rng->state = oldstate * uint64_t(6364136223846793005) + rng->inc;
  uint32_t xorshifted = uint32_t(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint32_t rot = oldstate >> 59u;
  uint32_t ret =
      (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));

  return float(double(ret) / double(4294967296.0));
}

inline void pcg32_srandom(pcg32_state_t* rng, uint64_t initstate, uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1U) | 1U;
  pcg32_random(rng);
  rng->state += initstate;
  pcg32_random(rng);
}

const float kPI = 3.141592f;
const float kEPS = 1e-3f;

inline float3 Lerp3(float3 v0, float3 v1, float3 v2, float u, float v) {
  return (1.0f - u - v) * v0 + u * v1 + v * v2;
}

inline void CalcNormal(float3& N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}


//
// Simple random number generator class
//
class RNG
{
 public:
  explicit RNG(uint64_t initstate, uint64_t initseq) {
    pcg32_srandom(&state_, initstate, initseq);
  }

  ~RNG() {}

  ///
  /// Draw random number[0.0, 1.0)
  ///
  inline float Draw() {
    float x = pcg32_random(&state_);
    if (x >= 1.0f) {
      x = std::nextafter(1.0f, 0.0f);
    }
    return x;
  }

 private:
  pcg32_state_t state_;
};


} // namespace example

#endif // EXAMPLE_COMMON_UTIL_H_
