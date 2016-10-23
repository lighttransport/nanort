#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>

#include "shader.h"

#ifdef __clang__
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#endif

using namespace example;

// PowGVenerator from fmath.hpp
union fi {
  float f;
  unsigned int i;
};

static inline unsigned int mask(int x) { return (1U << x) - 1; }

/*
  for given y > 0
  get f_y(x) := pow(x, y) for x >= 0
*/

class PowGenerator {
  enum { N = 11 };
  float tbl0_[256];
  struct {
    float app;
    float rev;
  } tbl1_[1 << N];

 public:
  PowGenerator(float y) {
    for (int i = 0; i < 256; i++) {
      tbl0_[i] = ::powf(2, (i - 127) * y);
    }
    const double e = 1 / double(1 << 24);
    const double h = 1 / double(1 << N);
    const size_t n = 1U << N;
    for (size_t i = 0; i < n; i++) {
      double x = 1 + double(i) / n;
      double a = ::pow(x, static_cast<double>(y));
      tbl1_[i].app = static_cast<float>(a);
      double b = ::pow(x + h - e, static_cast<double>(y));
      tbl1_[i].rev = static_cast<float>(((b - a) / (h - e) / (1 << 23)));
    }
  }
  float get(float x) const {
    fi fi;
    fi.f = x;
    int a = (fi.i >> 23) & mask(8);
    unsigned int b = fi.i & mask(23);
    unsigned int b1 = b & (mask(N) << (23 - N));
    unsigned int b2 = b & mask(23 - N);
    float f;
    int idx = b1 >> (23 - N);
    f = tbl0_[a] * (tbl1_[idx].app + float(b2) * tbl1_[idx].rev);
    return f;
  }
};

// Hair shader based on PBRT-v3

/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#ifdef DEBUG
#define CHECK(x) assert(x)
#define CHECK_LT(a, b) assert((a) < (b))
#define CHECK_GE(a, b) assert((a) >= (b))
#else
#define CHECK(x) (void)(x)
#define CHECK_LT(a, b) \
  {                    \
    (void)(a);         \
    (void)(b);         \
  }
#define CHECK_GE(a, b) \
  {                    \
    (void)(a);         \
    (void)(b);         \
  }
#endif

#define Pi (3.141592f)

// HairBSDF Constants
static const float SqrtPiOver8 = 0.626657069f;

// General Utility Functions
static inline float Sqr(float v) { return v * v; }

// Simple RGB -> Y conversion
static inline float luminance(vec3 v) {
  return 0.299f * v.x + 0.587f * v.y + 0.114f * v.z;
}

template <int n>
static float Pow(float v) {
  static_assert(n > 0, "Power can't be negative");
  float n2 = Pow<n / 2>(v);
  return n2 * n2 * Pow<n & 1>(v);
}

template <>
inline float Pow<1>(float v) {
  return v;
}
template <>
inline float Pow<0>(float v) {
  (void)v;
  return 1;
}

template <typename T, typename U, typename V>
static inline T Clamp(T val, U low, V high) {
  if (val < low)
    return low;
  else if (val > high)
    return high;
  else
    return val;
}

static inline float SafeASin(float x) {
  CHECK(x >= -1.0001f && x <= 1.0001f);
  return std::asin(Clamp(x, -1, 1));
}

static inline float SafeSqrt(float x) {
  CHECK_GE(x, -1e-4);
  return std::sqrt(std::max(0.0f, x));
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
static uint32_t Compact1By1(uint32_t x) {
  // TODO: as of Haswell, the PEXT instruction could do all this in a
  // single instruction.
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x &= 0x55555555;
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >> 1)) & 0x33333333;
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >> 2)) & 0x0f0f0f0f;
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >> 4)) & 0x00ff00ff;
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x >> 8)) & 0x0000ffff;
  return x;
}

static void Demuxfloat(float p[2], float f) {
  CHECK(f >= 0 && f < 1);
  uint64_t v = static_cast<uint64_t>(f * (1ull << 32));
  CHECK_LT(v, 0x100000000);
  uint32_t bits[2] = {Compact1By1(static_cast<uint32_t>(v)),
                      Compact1By1(static_cast<uint32_t>(v >> 1))};
  p[0] = bits[0] / static_cast<float>(1 << 16);
  p[1] = bits[1] / static_cast<float>(1 << 16);
}

inline float AbsCosTheta(const vec3 &w) { return std::abs(w.z); }

// static inline void SigmaAFromConcentration(float sigma_a[3], float ce, float
// cp) {
//    float eumelaninSigmaA[3] = {0.419f, 0.697f, 1.37f};
//    float pheomelaninSigmaA[3] = {0.187f, 0.4f, 1.05f};
//    for (int i = 0; i < 3; ++i) {
//        sigma_a[i] = (ce * eumelaninSigmaA[i] + cp * pheomelaninSigmaA[i]);
//    }
//}

// static inline void SigmaAFromReflectance(float sigma_a[3], const float c[3],
// float beta_n) {
//    for (int i = 0; i < 3; ++i) {
//        sigma_a[i] = Sqr(std::log(c[i]) /
//                         (5.969f - 0.215f * beta_n + 2.532f * Sqr(beta_n) -
//                          10.73f * Pow<3>(beta_n) + 5.574f * Pow<4>(beta_n) +
//                          0.245f * Pow<5>(beta_n)));
//    }
//}

// Hair Local Declarations
static inline float I0(float x), LogI0(float x);

// Hair Local Functions
static float Mp(float cosThetaI, float cosThetaO, float sinThetaI,
                float sinThetaO, float v) {
  float a = cosThetaI * cosThetaO / v;
  float b = sinThetaI * sinThetaO / v;
  float mp =
      (v <= 0.1f)
          ? (std::exp(LogI0(a) - b - 1 / v + 0.6931f + std::log(1 / (2 * v))))
          : (std::exp(-b) * I0(a)) / (std::sinh(1 / v) * 2 * v);
  CHECK(!std::isinf(mp) && !std::isnan(mp));
  return mp;
}

static inline float I0(float x) {
  float val = 0;
  float x2i = 1;
  int ifact = 1;
  int i4 = 1;
  // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
  for (int i = 0; i < 10; ++i) {
    if (i > 1) ifact *= i;
    val += x2i / (i4 * Sqr(ifact));
    x2i *= x * x;
    i4 *= 4;
  }
  return val;
}

static inline float LogI0(float x) {
  if (x > 12)
    return x + 0.5f * (-std::log(2 * Pi) + std::log(1 / x) + 1 / (8 * x));
  else
    return std::log(I0(x));
}

static float FrDielectric(float cosThetaI, float etaI, float etaT) {
  cosThetaI = Clamp(cosThetaI, -1, 1);
  // Potentially swap indices of refraction
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    std::swap(etaI, etaT);
    cosThetaI = std::abs(cosThetaI);
  }

  // Compute _cosThetaT_ using Snell's law
  float sinThetaI = std::sqrt(std::max(0.0f, 1.0f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Handle total internal reflection
  if (sinThetaT >= 1) return 1;
  float cosThetaT = std::sqrt(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));
  float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
                ((etaT * cosThetaI) + (etaI * cosThetaT));
  float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
                ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (Rparl * Rparl + Rperp * Rperp) / 2;
}

static void Ap(vec3 ap[pMax + 1], float cosThetaO, float eta, float h,
               const vec3 T) {
  // std::array<vec3, pMax + 1> ap;
  // Compute $p=0$ attenuation at initial cylinder intersection
  float cosGammaO = SafeSqrt(1 - h * h);
  float cosTheta = cosThetaO * cosGammaO;
  float f = FrDielectric(cosTheta, 1.f, eta);
  ap[0].x = f;
  ap[0].y = f;
  ap[0].z = f;

  // Compute $p=1$ attenuation term
  ap[1].x = Sqr(1 - f) * T.x;
  ap[1].y = Sqr(1 - f) * T.y;
  ap[1].z = Sqr(1 - f) * T.z;

  // Compute attenuation terms up to $p=_pMax_$
  for (int p = 2; p < pMax; ++p) {
    ap[p].x = ap[p - 1].x * T.x * f;
    ap[p].y = ap[p - 1].y * T.y * f;
    ap[p].z = ap[p - 1].z * T.z * f;
  }

  // Compute attenuation term accounting for remaining orders of scattering
  ap[pMax].x = ap[pMax - 1].x * f * T.x / (1.f - T.x * f);
  ap[pMax].y = ap[pMax - 1].y * f * T.y / (1.f - T.y * f);
  ap[pMax].z = ap[pMax - 1].z * f * T.z / (1.f - T.z * f);
  // return ap;
}

inline float Phi(int p, float gammaO, float gammaT) {
  return 2 * p * gammaT - 2 * gammaO + p * Pi;
}

inline float Logistic(float x, float s) {
  x = std::abs(x);
  return std::exp(-x / s) / (s * Sqr(1 + std::exp(-x / s)));
}

inline float LogisticCDF(float x, float s) {
  return 1 / (1 + std::exp(-x / s));
}

inline float TrimmedLogistic(float x, float s, float a, float b) {
  CHECK_LT(a, b);
  return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

inline float Np(float phi, int p, float s, float gammaO, float gammaT) {
  float dphi = phi - Phi(p, gammaO, gammaT);
  // Remap _dphi_ to $[-\pi,\pi]$
  while (dphi > Pi) dphi -= 2 * Pi;
  while (dphi < -Pi) dphi += 2 * Pi;
  return TrimmedLogistic(dphi, s, -Pi, Pi);
}

static float SampleTrimmedLogistic(float u, float s, float a, float b) {
  CHECK_LT(a, b);
  float k = LogisticCDF(b, s) - LogisticCDF(a, s);
  float x = -s * std::log(1 / (u * k + LogisticCDF(a, s)) - 1);
  CHECK(!std::isnan(x));
  return Clamp(x, a, b);
}

void HairBSDF::ComputeApPdf(float pdfs[pMax + 1], float cosThetaO) const {
  // Compute array of $A_p$ values for _cosThetaO_
  float sinThetaO = SafeSqrt(1 - cosThetaO * cosThetaO);

  // Compute $\cos \thetat$ for refracted ray
  float sinThetaT = sinThetaO / eta_;
  float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

  // Compute $\gammat$ for refracted ray
  float etap = std::sqrt(eta_ * eta_ - Sqr(sinThetaO)) / cosThetaO;
  float sinGammaT = h_ / etap;
  float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
  // float gammaT = SafeASin(sinGammaT);

  // Compute the transmittance _T_ of a single path through the cylinder
  vec3 T;
  T.x = std::exp(-sigma_a_.x * (2 * cosGammaT / cosThetaT));
  T.y = std::exp(-sigma_a_.y * (2 * cosGammaT / cosThetaT));
  T.z = std::exp(-sigma_a_.z * (2 * cosGammaT / cosThetaT));

  vec3 ap[pMax + 1];
  Ap(ap, cosThetaO, eta_, h_, T);

  // Compute $A_p$ PDF from individual $A_p$ terms
  float sumY = 0.0f;
  for (int i = 0; i <= pMax; ++i) {
    sumY += luminance(ap[i]);
  }
  for (int i = 0; i <= pMax; ++i) {
    pdfs[i] = luminance(ap[i]) / sumY;
  }
}

// HairBSDF Method Definitions
HairBSDF::HairBSDF(float h, float eta, const vec3 &sigma_a, float beta_m,
                   float beta_n, float alpha)
    : h_(h),
      gammaO_(SafeASin(h)),
      eta_(eta),
      sigma_a_(sigma_a),
      beta_m_(beta_m),
      beta_n_(beta_n),
      alpha_(alpha) {
  CHECK(h >= -1 && h <= 1);
  CHECK(beta_m >= 0 && beta_m <= 1);
  CHECK(beta_n >= 0 && beta_n <= 1);
  // Compute longitudinal variance from $\beta_m$
  static_assert(
      pMax >= 3,
      "Longitudinal variance code must be updated to handle low pMax");
  v_[0] =
      Sqr(0.726f * beta_m_ + 0.812f * Sqr(beta_m_) + 3.7f * Pow<20>(beta_m_));
  v_[1] = .25f * v_[0];
  v_[2] = 4 * v_[0];
  for (int p = 3; p <= pMax; ++p)
    // TODO: is there anything better here?
    v_[p] = v_[2];

  // Compute azimuthal logistic scale factor from $\beta_n$
  s_ = SqrtPiOver8 *
       (0.265f * beta_n_ + 1.194f * Sqr(beta_n_) + 5.372f * Pow<22>(beta_n_));
  CHECK(!std::isnan(s_));

  // Compute $\alpha$ terms for hair scales
  sin2kAlpha_[0] = std::sin(alpha_);
  cos2kAlpha_[0] = SafeSqrt(1 - Sqr(sin2kAlpha_[0]));
  for (int i = 1; i < 3; ++i) {
    sin2kAlpha_[i] = 2 * cos2kAlpha_[i - 1] * sin2kAlpha_[i - 1];
    cos2kAlpha_[i] = Sqr(cos2kAlpha_[i - 1]) - Sqr(sin2kAlpha_[i - 1]);
  }
}

vec3 HairBSDF::f(const vec3 &wo, const vec3 &wi) const {
  // Compute hair coordinate system terms related to _wo_
  float sinThetaO = wo.x;
  float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
  float phiO = std::atan2(wo.z, wo.y);

  // Compute hair coordinate system terms related to _wi_
  float sinThetaI = wi.x;
  float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
  float phiI = std::atan2(wi.z, wi.y);

  // Compute $\cos \thetat$ for refracted ray
  float sinThetaT = sinThetaO / eta_;
  float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

  // Compute $\gammat$ for refracted ray
  float etap = std::sqrt(eta_ * eta_ - Sqr(sinThetaO)) / cosThetaO;
  float sinGammaT = h_ / etap;
  float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
  float gammaT = SafeASin(sinGammaT);

  // Compute the transmittance _T_ of a single path through the cylinder
  vec3 T;
  T.x = std::exp(-sigma_a_.x * (2 * cosGammaT / cosThetaT));
  T.y = std::exp(-sigma_a_.y * (2 * cosGammaT / cosThetaT));
  T.z = std::exp(-sigma_a_.z * (2 * cosGammaT / cosThetaT));

  // Evaluate hair BSDF
  float phi = phiI - phiO;
  // std::array<vec3, pMax + 1> ap = Ap(cosThetaO, eta_, h_, T);
  vec3 ap[pMax + 1];
  Ap(ap, cosThetaO, eta_, h_, T);
  vec3 fsum(0.0f);
  for (int p = 0; p < pMax; ++p) {
    // Compute $\sin \thetai$ and $\cos \thetai$ terms accounting for scales
    float sinThetaIp, cosThetaIp;
    if (p == 0) {
      sinThetaIp = sinThetaI * cos2kAlpha_[1] + cosThetaI * sin2kAlpha_[1];
      cosThetaIp = cosThetaI * cos2kAlpha_[1] - sinThetaI * sin2kAlpha_[1];
    }

    // Handle remainder of $p$ values for hair scale tilt
    else if (p == 1) {
      sinThetaIp = sinThetaI * cos2kAlpha_[0] - cosThetaI * sin2kAlpha_[0];
      cosThetaIp = cosThetaI * cos2kAlpha_[0] + sinThetaI * sin2kAlpha_[0];
    } else if (p == 2) {
      sinThetaIp = sinThetaI * cos2kAlpha_[2] - cosThetaI * sin2kAlpha_[2];
      cosThetaIp = cosThetaI * cos2kAlpha_[2] + sinThetaI * sin2kAlpha_[2];
    } else {
      sinThetaIp = sinThetaI;
      cosThetaIp = cosThetaI;
    }

    // Handle out-of-range $\cos \thetai$ from scale adjustment
    cosThetaIp = std::abs(cosThetaIp);
    fsum += Mp(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, v_[p]) * ap[p] *
            Np(phi, p, s_, gammaO_, gammaT);
  }

  // Compute contribution of remaining terms after _pMax_
  fsum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v_[pMax]) * ap[pMax] /
          (2.f * Pi);
  if (AbsCosTheta(wi) > 0) fsum /= AbsCosTheta(wi);
  CHECK(!std::isinf(luminance(fsum)) && !std::isnan(luminance(fsum)));
  return fsum;
}

vec3 HairBSDF::Sample_f(const vec3 &wo, vec3 *wi, const float u2[2],
                        float *pdf) const {
  // Compute hair coordinate system terms related to _wo_
  float sinThetaO = wo.x;
  float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
  float phiO = std::atan2(wo.z, wo.y);

  // Derive four random samples from _u2_
  // Point2f u[2] = {Demuxfloat(u2[0]), Demuxfloat(u2[1])};
  float u[2][2];
  Demuxfloat(u[0], u2[0]);
  Demuxfloat(u[1], u2[1]);

  // Determine which term $p$ to sample for hair scattering
  float sinThetaI;
  float cosThetaI;
  float apPdf[pMax + 1];
  float dphi;
  float gammaT;
  {
    ComputeApPdf(apPdf, cosThetaO);
    int p;
    for (p = 0; p < pMax; ++p) {
      if (u[0][0] < apPdf[p]) break;
      u[0][0] -= apPdf[p];
    }

    // Sample $M_p$ to compute $\thetai$
    u[1][0] = std::max(u[1][0], float(1e-5));
    float cosTheta =
        1 + v_[p] * std::log(u[1][0] + (1 - u[1][0]) * std::exp(-2 / v_[p]));
    float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    float cosPhi = std::cos(2 * Pi * u[1][1]);
    sinThetaI = -cosTheta * sinThetaO + sinTheta * cosPhi * cosThetaO;
    cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));

    // Update sampled $\sin \thetai$ and $\cos \thetai$ to account for scales
    float sinThetaIp = sinThetaI, cosThetaIp = cosThetaI;
    if (p == 0) {
      sinThetaIp = sinThetaI * cos2kAlpha_[1] - cosThetaI * sin2kAlpha_[1];
      cosThetaIp = cosThetaI * cos2kAlpha_[1] + sinThetaI * sin2kAlpha_[1];
    } else if (p == 1) {
      sinThetaIp = sinThetaI * cos2kAlpha_[0] + cosThetaI * sin2kAlpha_[0];
      cosThetaIp = cosThetaI * cos2kAlpha_[0] - sinThetaI * sin2kAlpha_[0];
    } else if (p == 2) {
      sinThetaIp = sinThetaI * cos2kAlpha_[2] + cosThetaI * sin2kAlpha_[2];
      cosThetaIp = cosThetaI * cos2kAlpha_[2] - sinThetaI * sin2kAlpha_[2];
    }
    sinThetaI = sinThetaIp;
    cosThetaI = cosThetaIp;

    // Sample $N_p$ to compute $\Delta\phi$

    // Compute $\gammat$ for refracted ray
    float etap = std::sqrt(eta_ * eta_ - Sqr(sinThetaO)) / cosThetaO;
    float sinGammaT = h_ / etap;
    // float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
    gammaT = SafeASin(sinGammaT);
    if (p < pMax)
      dphi =
          Phi(p, gammaO_, gammaT) + SampleTrimmedLogistic(u[0][1], s_, -Pi, Pi);
    else
      dphi = 2 * Pi * u[0][1];

    // Compute _wi_ from sampled hair scattering angles
    float phiI = phiO + dphi;
    *wi =
        vec3(sinThetaI, cosThetaI * std::cos(phiI), cosThetaI * std::sin(phiI));
  }

  // Compute PDF for sampled hair scattering direction _wi_
  *pdf = 0;
  for (int p = 0; p < pMax; ++p) {
    // Compute $\sin \thetai$ and $\cos \thetai$ terms accounting for scales
    float sinThetaIp, cosThetaIp;
    if (p == 0) {
      sinThetaIp = sinThetaI * cos2kAlpha_[1] + cosThetaI * sin2kAlpha_[1];
      cosThetaIp = cosThetaI * cos2kAlpha_[1] - sinThetaI * sin2kAlpha_[1];
    }

    // Handle remainder of $p$ values for hair scale tilt
    else if (p == 1) {
      sinThetaIp = sinThetaI * cos2kAlpha_[0] - cosThetaI * sin2kAlpha_[0];
      cosThetaIp = cosThetaI * cos2kAlpha_[0] + sinThetaI * sin2kAlpha_[0];
    } else if (p == 2) {
      sinThetaIp = sinThetaI * cos2kAlpha_[2] - cosThetaI * sin2kAlpha_[2];
      cosThetaIp = cosThetaI * cos2kAlpha_[2] + sinThetaI * sin2kAlpha_[2];
    } else {
      sinThetaIp = sinThetaI;
      cosThetaIp = cosThetaI;
    }

    // Handle out-of-range $\cos \thetai$ from scale adjustment
    cosThetaIp = std::abs(cosThetaIp);
    *pdf += Mp(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, v_[p]) * apPdf[p] *
            Np(dphi, p, s_, gammaO_, gammaT);
  }
  *pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v_[pMax]) *
          apPdf[pMax] * (1 / (2 * Pi));
  // if (std::abs(wi->x) < .9999) CHECK_NEAR(*pdf, Pdf(wo, *wi), .01);
  return f(wo, *wi);
}

float HairBSDF::Pdf(const vec3 &wo, const vec3 &wi) const {
  // Compute hair coordinate system terms related to _wo_
  float sinThetaO = wo.x;
  float cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
  float phiO = std::atan2(wo.z, wo.y);

  // Compute hair coordinate system terms related to _wi_
  float sinThetaI = wi.x;
  float cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
  float phiI = std::atan2(wi.z, wi.y);

  // Compute $\cos \thetat$ for refracted ray
  // float sinThetaT = sinThetaO / eta_;
  // float cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

  // Compute $\gammat$ for refracted ray
  float etap = std::sqrt(eta_ * eta_ - Sqr(sinThetaO)) / cosThetaO;
  float sinGammaT = h_ / etap;
  // float cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
  float gammaT = SafeASin(sinGammaT);

  // Compute PDF for $A_p$ terms
  float apPdf[pMax + 1];
  ComputeApPdf(apPdf, cosThetaO);

  // Compute PDF sum for hair scattering events
  float phi = phiI - phiO;
  float pdf = 0;
  for (int p = 0; p < pMax; ++p) {
    // Compute $\sin \thetai$ and $\cos \thetai$ terms accounting for scales
    float sinThetaIp, cosThetaIp;
    if (p == 0) {
      sinThetaIp = sinThetaI * cos2kAlpha_[1] + cosThetaI * sin2kAlpha_[1];
      cosThetaIp = cosThetaI * cos2kAlpha_[1] - sinThetaI * sin2kAlpha_[1];
    }

    // Handle remainder of $p$ values for hair scale tilt
    else if (p == 1) {
      sinThetaIp = sinThetaI * cos2kAlpha_[0] - cosThetaI * sin2kAlpha_[0];
      cosThetaIp = cosThetaI * cos2kAlpha_[0] + sinThetaI * sin2kAlpha_[0];
    } else if (p == 2) {
      sinThetaIp = sinThetaI * cos2kAlpha_[2] - cosThetaI * sin2kAlpha_[2];
      cosThetaIp = cosThetaI * cos2kAlpha_[2] + sinThetaI * sin2kAlpha_[2];
    } else {
      sinThetaIp = sinThetaI;
      cosThetaIp = cosThetaI;
    }

    // Handle out-of-range $\cos \thetai$ from scale adjustment
    cosThetaIp = std::abs(cosThetaIp);
    pdf += Mp(cosThetaIp, cosThetaO, sinThetaIp, sinThetaO, v_[p]) * apPdf[p] *
           Np(phi, p, s_, gammaO_, gammaT);
  }
  pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v_[pMax]) *
         apPdf[pMax] * (1 / (2 * Pi));
  return pdf;
}
