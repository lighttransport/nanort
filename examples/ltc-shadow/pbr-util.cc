#include "pbr-util.h"
#include "../../nanort.h"  // float3

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include <atomic>
#include <thread>

// for debug
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

namespace example {

constexpr float kPI = 3.141592f;

typedef nanort::real3<float> float3;

static float3 uv_to_dir(float u, float v, float phi_offset);
static void dir_to_uv(float *uu, float *vv, const float3 n);
static void SaveImage(const Image& image, const std::string& filename);

//
// Simple bilinear filtering.
//
static void SampleImage(float* rgba, float u, float v, int width, int height,
                        int channels, const float* texels) {
  float sx = std::floor(u);
  float sy = std::floor(v);

  // Wrap mode = repeat
  float uu = u - sx;
  float vv = v - sy;

  // clamp
  uu = std::max(uu, 0.0f);
  uu = std::min(uu, 1.0f);
  vv = std::max(vv, 0.0f);
  vv = std::min(vv, 1.0f);

  float px = (width - 1) * uu;
  float py = (height - 1) * vv;

  int x0 = std::max(0, std::min((int)px, (width - 1)));
  int y0 = std::max(0, std::min((int)py, (height - 1)));
  int x1 = std::max(0, std::min((x0 + 1), (width - 1)));
  int y1 = std::max(0, std::min((y0 + 1), (height - 1)));

  float dx = px - (float)x0;
  float dy = py - (float)y0;

  float w[4];

  w[0] = (1.0f - dx) * (1.0 - dy);
  w[1] = (1.0f - dx) * (dy);
  w[2] = (dx) * (1.0 - dy);
  w[3] = (dx) * (dy);

  int i00 = channels * (y0 * width + x0);
  int i01 = channels * (y0 * width + x1);
  int i10 = channels * (y1 * width + x0);
  int i11 = channels * (y1 * width + x1);

  float t[4][4]; // Assume channels <= 4

  for (int c = 0; c < channels; c++) {
    t[0][c] = texels[i00 + c];
    t[1][c] = texels[i01 + c];
    t[2][c] = texels[i10 + c];
    t[3][c] = texels[i11 + c];
  }
  
  for (int i = 0; i < channels; i++) {
    rgba[i] = w[0] * t[0][i] + w[1] * t[1][i] + w[2] * t[2][i] + w[3] * t[3][i];
  }
}

// Base on Filament -----------------------

/*
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

static void SaveCubemap(const Cubemap& cubemap, const std::string& basename);

struct CubemapAddress {
  int face;
  float s;
  float t;
};

static inline float saturate(const float f) {
  return std::max(0.0f, std::min(f, 1.0f));
}

static inline float dot(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline float pow5(float x) { return (x * x) * (x * x) * x; }

static inline void hammersley(uint32_t i, float iN, float ret[2]) {
  constexpr float tof = 0.5f / 0x80000000U;
  uint32_t bits = i;
  bits = (bits << 16) | (bits >> 16);
  bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
  bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
  bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
  bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);

  ret[0] = i * iN;
  ret[1] = bits * tof;
}

static void hemisphereImportanceSampleDggx(float u[2], float a, float ret[3]) {
  const float phi = 2 * kPI * u[0];
  // NOTE: (aa-1) == (a-1)(a+1) produces better fp accuracy
  const float cosTheta2 = (1 - u[1]) / (1 + (a + 1) * ((a - 1) * u[1]));
  const float cosTheta = std::sqrt(cosTheta2);
  const float sinTheta = std::sqrt(1 - cosTheta2);

  ret[0] = sinTheta * std::cos(phi);
  ret[1] = sinTheta * std::sin(phi);
  ret[2] = cosTheta;
}

static float Fresnel(float f0, float f90, float LoH) {
  const float Fc = pow5(1 - LoH);
  return f0 * (1 - Fc) + f90 * Fc;
}

static float Visibility(float NoV, float NoL, float a) {
  // Heitz 2014, "Understanding the Masking-Shadowing Function in
  // Microfacet-Based BRDFs" Height-correlated GGX
  const float a2 = a * a;
  const float GGXL = NoV * std::sqrt((NoL - NoL * a2) * NoL + a2);
  const float GGXV = NoL * std::sqrt((NoV - NoV * a2) * NoV + a2);
  return 0.5f / (GGXV + GGXL);
}

// Importance-sampled
static void DFV(float NoV, float roughness, size_t numSamples, float ret[2]) {
  float r[2] = {0, 0};
  const float linearRoughness = roughness * roughness;
  const float V[3] = {std::sqrt(1 - NoV * NoV), 0, NoV};
  for (size_t i = 0; i < numSamples; i++) {
    float u[2];
    hammersley(uint32_t(i), 1.0f / numSamples, u);
    float H[3];
    hemisphereImportanceSampleDggx(u, linearRoughness, H);
    float L[3];
    L[0] = 2 * dot(V, H) * H[0] - V[0];
    L[1] = 2 * dot(V, H) * H[1] - V[1];
    L[2] = 2 * dot(V, H) * H[2] - V[2];
    const float VoH = saturate(dot(V, H));
    const float NoL = saturate(L[2]);
    const float NoH = saturate(H[2]);
    if (NoL > 0) {
      // Note: remember VoH == LoH  (H is half vector)
      const float v = Visibility(NoV, NoL, linearRoughness) * NoL * (VoH / NoH);
      const float Fc = pow5(1 - VoH);
      r[0] += v * (1.0 - Fc);
      r[1] += v * Fc;
    }
  }
  ret[0] = r[0] * (4.0 / numSamples);
  ret[1] = r[1] * (4.0 / numSamples);
}

static void DFV_Multiscatter(float NoV, float roughness, size_t numSamples,
                             float ret[2]) {
  float r[2] = {0, 0};

  const float linearRoughness = roughness * roughness;
  const float V[3] = {std::sqrt(1 - NoV * NoV), 0, NoV};
  for (size_t i = 0; i < numSamples; i++) {
    float u[2];
    hammersley(uint32_t(i), 1.0f / numSamples, u);
    float H[3];
    hemisphereImportanceSampleDggx(u, linearRoughness, H);
    float L[3];
    L[0] = 2 * dot(V, H) * H[0] - V[0];
    L[1] = 2 * dot(V, H) * H[1] - V[1];
    L[2] = 2 * dot(V, H) * H[2] - V[2];
    const float VoH = saturate(dot(V, H));
    const float NoL = saturate(L[2]);
    const float NoH = saturate(H[2]);

    if (NoL > 0) {
      // Note: remember VoH == LoH  (H is half vector)
      const float v = Visibility(NoV, NoL, linearRoughness) * NoL * (VoH / NoH);
      const float Fc = pow5(1 - VoH);

      // This looks different from the computation performed in DFV() but the
      // result is the same in the shader. With DFV() we would traditionally
      // do the following per fragment:
      //
      // vec2 dfg = sampleEnv(...);
      // return f0 * dfg.x + dfg.y;
      //
      // With the multi-scattering formulation we instead do the following per
      // fragment:
      //
      // vec2 dfg = sampleEnv(...);
      // return mix(dfg.xxx, dfg.yyy, f0)
      r[0] += v * Fc;
      r[1] += v;
    }
  }

  ret[0] = r[0] * (4.0f / numSamples);
  ret[1] = r[1] * (4.0f / numSamples);
}

//
// Compute DFG Lut.
// x axis : NdotV, y axis : roughness
//
void BuildDFGLut(const bool multiscatter, const size_t width, Image* output) {
  auto fn = multiscatter ? DFV_Multiscatter : DFV;

  size_t height = width;

  output->width = width;
  output->height = height;

  output->channels = 3;
  output->pixels.resize(width * height * 3);

  // TODO(LTE): multithreading.

  for (size_t y = 0; y < height; y++) {
    const float roughness =
        std::min(1.0, std::max(0.0, (y + 0.5) / float(height)));
    for (size_t x = 0; x < width; x++) {
      const float NoV = std::min(1.0, std::max(0.0, (x + 0.5) / float(width)));
      float r[3];
      fn(NoV, roughness, 1024, r);

      output->pixels[3 * (y * width + x) + 0] = r[0];
      output->pixels[3 * (y * width + x) + 1] = r[1];
      output->pixels[3 * (y * width + x) + 2] = r[2];
    }
  }
}

static float DistributionGGX(float NoH, float linearRoughness) {
  // NOTE: (aa-1) == (a-1)(a+1) produces better fp accuracy
  float a = linearRoughness;
  float f = (a - 1) * ((a + 1) * (NoH * NoH)) + 1.0;
  return (a * a) / (kPI * f * f);
}

template <typename T>
static inline constexpr T log4(T x) {
  // log2(x)/log2(4)
  // log2(x)/2
  return std::log2(x) * T(0.5);
}

inline float3 getDirectionFor(int face_idx, float x, float y, int dim) {
  // map [0, dim] to [-1,1] with (-1,-1) at bottom left
  float scale = 2.0f / dim;
  float cx = (x * scale) - 1.0f;
  float cy = 1.0f - (y * scale);

  float3 dir;
  const float l = std::sqrt(cx * cx + cy * cy + 1.0f);
  switch (face_idx) {
    case 1: // PX
      dir = {1, cy, -cx};
      break;
    case 0: // NX
      dir = {-1, cy, cx};
      break;
    case 3: // PY
      dir = {cx, 1, -cy};
      break;
    case 2: // NY
      dir = {cx, -1, cy};
      break;
    case 5: // PZ
      dir = {cx, cy, 1};
      break;
    case 4: // NZ
      dir = {-cx, cy, -1};
      break;
  }
  return dir * (1 / l);
}

CubemapAddress getAddressFor(const float3& r) {
  CubemapAddress addr;
  double sc, tc, ma;
  const double rx = std::abs(r[0]);
  const double ry = std::abs(r[1]);
  const double rz = std::abs(r[2]);
  if (rx >= ry && rx >= rz) {
    ma = rx;
    if (r[0] >= 0) {
      addr.face = 1;  // PX
      sc = -r[2];
      tc = -r[1];
    } else {
      addr.face = 0;  // NX
      sc = r[2];
      tc = -r[1];
    }
  } else if (ry >= rx && ry >= rz) {
    ma = ry;
    if (r[1] >= 0) {
      addr.face = 3;  // PY
      sc = r[0];
      tc = r[2];
    } else {
      addr.face = 2;  // NY
      sc = r[0];
      tc = -r[2];
    }
  } else {
    ma = rz;
    if (r[2] >= 0) {
      addr.face = 5;  // PZ
      sc = r[0];
      tc = -r[1];
    } else {
      addr.face = 4;  // NZ
      sc = -r[0];
      tc = -r[1];
    }
  }
  // ma is guaranteed to be >= sc and tc
  addr.s = (sc / ma + 1) * 0.5f;
  addr.t = (tc / ma + 1) * 0.5f;
  return addr;
}

float3 sampleAt(const Image& image, size_t x, size_t y) {
  float3 r;
  // Assume channels >= 3
  // std::cout << "x = " << x << ", y = " << y << ", width = " << image.width <<
  // ", chan = " << image.channels << ", len = " << image.pixels.size() <<
  // std::endl;
  r[0] = image.pixels[image.channels * (y * image.width + x) + 0];
  r[1] = image.pixels[image.channels * (y * image.width + x) + 1];
  r[2] = image.pixels[image.channels * (y * image.width + x) + 2];

  return r;
}

inline float3 sampleAt(const Cubemap& cubemap, const float3& direction) {
  CubemapAddress addr(getAddressFor(direction));
  int dim = cubemap.dim();
  const size_t x = std::min(size_t(addr.s * dim), size_t(dim - 1));
  const size_t y = std::min(size_t(addr.t * dim), size_t(dim - 1));
  return sampleAt(cubemap.faces[addr.face], x, y);
}

float3 filterAt(const Image& image, double x, double y) {
  const size_t x0 = size_t(x);
  const size_t y0 = size_t(y);
  // we allow ourselves to read past the width/height of the Image because the
  // data is valid and contain the "seamless" data.
  size_t x1 = x0 + 1;
  size_t y1 = y0 + 1;
  const float u = float(x - x0);
  const float v = float(y - y0);
  const float one_minus_u = 1 - u;
  const float one_minus_v = 1 - v;
  const float3 c0 = sampleAt(image, x0, y0);
  const float3 c1 = sampleAt(image, x1, y0);
  const float3 c2 = sampleAt(image, x0, y1);
  const float3 c3 = sampleAt(image, x1, y1);
  return (one_minus_u * one_minus_v) * c0 + (u * one_minus_v) * c1 +
         (one_minus_u * v) * c2 + (u * v) * c3;
}

inline float3 filterAt(const Cubemap& cubemap, const float3& direction) {
  CubemapAddress addr(getAddressFor(direction));
  int dim = cubemap.dim();
  double s = std::min(addr.s * dim, dim - 1.0f);
  double t = std::min(addr.t * dim, dim - 1.0f);

  return filterAt(cubemap.faces[addr.face], s, t);
}

float3 trilinearFilterAt(const Cubemap& l0, const Cubemap& l1, float lerp,
                         const float3& L) {
  CubemapAddress addr = getAddressFor(L);
  const Image& i0 = l0.faces[addr.face];
  float l0dim = float(l0.dim());
  float l0upperBound = std::nextafter(l0dim, 0.0f);
  float x0 = std::min(addr.s * l0dim, l0upperBound);
  float y0 = std::min(addr.t * l0dim, l0upperBound);

  float3 c0(filterAt(i0, x0, y0));

  if (&l0 != &l1) {
    const Image& i1 = l1.faces[addr.face];
    float l1dim = float(l1.dim());
    float l1upperBound = std::nextafter(l1dim, 0.0f);
    float x1 = std::min(addr.s * l1dim, l1upperBound);
    float y1 = std::min(addr.t * l1dim, l1upperBound);
    c0 += lerp * (filterAt(i1, x1, y1) - c0);
  }
  return c0;
}

void writeAt(Image& image, size_t x, size_t y, float3 L) {
  if ((x >= image.width) || (y >= image.height)) {
    return;
  }

  // Assume RGB
  image.pixels[3 * (y * image.width + x) + 0] = L[0];
  image.pixels[3 * (y * image.width + x) + 1] = L[1];
  image.pixels[3 * (y * image.width + x) + 2] = L[2];
}

static void downsampleCubemapLevelBoxFilter(Cubemap& dst, const Cubemap& src) {
  size_t scale = src.dim() / dst.dim();
  size_t dim = dst.dim();

  for (size_t f = 0; f < 6; f++) {
    for (size_t y = 0; y < dim; y++) {
      for (size_t x = 0; x < dim; ++x) {
        float3 L = filterAt(src.faces[f], x * scale + 0.5, y * scale + 0.5);

        writeAt(dst.faces[f], x, y, L);
      }
    }
  }
}

static void downsampleImageLevelBoxFilter(Image& dst, const Image& src) {
  dst.width = src.width / 2;
  dst.height = src.height / 2;
  dst.channels = src.channels;
  dst.pixels.resize(dst.width * dst.height * dst.channels);

  for (size_t y = 0; y < dst.height; y++) {
    float v = (y + 0.5f) / float(dst.height);
    for (size_t x = 0; x < dst.width; ++x) {
      float rgba[4];
      float u = (x + 0.5f) / float(dst.width);
      SampleImage(rgba, u, v, src.width, src.height, src.channels, src.pixels.data());
      for (size_t c = 0; c < dst.channels; c++) {
        dst.pixels[dst.channels * (y * dst.width + x) + c] = rgba[c];
      }
    }
  }
}

static void* getPixelRef(Image& image, size_t x, size_t y) {
  return static_cast<void*>(
      &image.pixels[image.channels * (y * image.width + x)]);
}

/*
 * We handle "seamless" cubemaps by duplicating a row to the bottom, or column
 * to the right of each faces that don't have an adjacent face in the image (the
 * duplicate is taken from the adjacent face in the cubemap). This is because
 * when accessing an image with bilinear filtering, we always overshoot to the
 * right or bottom. This works well with cubemaps stored as a cross in memory.
 */
static void makeSeamless(Cubemap& cubemap) {
  // Geometry geometry = mGeometry;
  size_t dim = cubemap.dim();

  if (dim < 2) {
    return;
  }

  auto stitch = [&](void* dst, size_t incDst, void const* src, ssize_t incSrc) {
    for (size_t i = 0; i < dim; ++i) {
      *(float3*)dst = *(float3*)src;
      dst = ((uint8_t*)dst + incDst);
      src = ((uint8_t*)src + incSrc);
    }
  };

  const size_t bpr = cubemap.bytesPerRow();
  const size_t bpp = sizeof(float) * 3;

  const int NX = 0;
  const int PX = 1;
  const int NY = 2;
  const int PY = 3;
  const int NZ = 4;
  const int PZ = 5;

  // Assume horizontal cross
  {
    // both for horizontal and vertical
    stitch(getPixelRef(cubemap.faces[NX], 0, dim), bpp,
           getPixelRef(cubemap.faces[NY], 0, dim - 1), -bpr);

    stitch(getPixelRef(cubemap.faces[PY], dim, 0), bpr,
           getPixelRef(cubemap.faces[PX], dim - 1, 0), -bpp);

    stitch(getPixelRef(cubemap.faces[PX], 0, dim), bpp,
           getPixelRef(cubemap.faces[NY], dim - 1, 0), bpr);

    stitch(getPixelRef(cubemap.faces[NY], dim, 0), bpr,
           getPixelRef(cubemap.faces[PX], 0, dim - 1), bpp);

    // horizontal cross
    stitch(getPixelRef(cubemap.faces[NZ], 0, dim), bpp,
           getPixelRef(cubemap.faces[NY], dim - 1, dim - 1), -bpp);

    stitch(getPixelRef(cubemap.faces[NZ], dim, 0), bpr,
           getPixelRef(cubemap.faces[NX], 0, 0), bpr);

    stitch(getPixelRef(cubemap.faces[NY], 0, dim), bpp,
           getPixelRef(cubemap.faces[NZ], dim - 1, dim - 1), -bpp);
  }
}

void generateMipmaps(std::vector<Cubemap>& levels) {
  // Image temp;
  const Cubemap& base(levels[0]);
  size_t dim = base.dim();
  size_t mipLevel = 0;
  while (dim > 1) {
    dim >>= 1;
    Cubemap dst(dim);
    const Cubemap& src(levels[mipLevel++]);
    downsampleCubemapLevelBoxFilter(dst, src);

    makeSeamless(dst);

#if 1  // dbg
    std::string basename = "cubemap_l" + std::to_string(mipLevel);
    SaveCubemap(dst, basename);
#endif

    // images.push_back(std::move(temp));
    levels.push_back(std::move(dst));
  }
}

void generateMipmaps(std::vector<Image>& levels) {
  // Image temp;
  const Image& base(levels[0]);
  size_t dim = base.height;
  size_t mipLevel = 0;
  while (dim > 1) {
    dim >>= 1;

    const Image& src(levels[mipLevel++]);
    Image dst;
    downsampleImageLevelBoxFilter(dst, src);

#if 1  // dbg
    std::string filename = "longlat_mip_l" + std::to_string(mipLevel) + ".exr";
    SaveImage(dst, filename);
#endif

    // images.push_back(std::move(temp));
    levels.push_back(std::move(dst));
  }
}

void roughnessFilter(Cubemap& dst, const std::vector<Cubemap>& levels,
                     float linearRoughness, size_t maxNumSamples) {
  const float numSamples = maxNumSamples;
  const float inumSamples = 1.0f / numSamples;
  const size_t maxLevel = levels.size() - 1;
  const float maxLevelf = maxLevel;
  // const Cubemap& base(levels[0]);
  const size_t dim0 = dst.dim();
  const float omegaP = float((4 * kPI) / (6 * dim0 * dim0));

  if (linearRoughness < std::numeric_limits<float>::epsilon()) {
    // Do simple image copy
    const Cubemap& cm = levels[0];
    for (size_t f = 0; f < 6; ++f) {
      for (size_t y = 0; y < dim0; ++y) {
        for (size_t x = 0; x < dim0; ++x) {
          const float px = x + 0.5f;
          const float py = y + 0.5f;
          const float3 N = getDirectionFor(f, px, py, dim0);
          // FIXME: we should pick the proper LOD here and do trilinear
          // filtering
          const float3 L = sampleAt(cm, N);
          writeAt(dst.faces[f], x, y, L);
        }
      }
    }

    return;
  }

  auto startT = std::chrono::system_clock::now();

  // be careful w/ the size of this structure, the smaller the better
  struct CacheEntry {
    float3 L;
    float brdf_NoL;
    float lerp;
    uint8_t l0;
    uint8_t l1;
  };

  std::vector<CacheEntry> cache;
  cache.reserve(maxNumSamples);

  // precompute everything that only depends on the sample #
  double weight = 0;
  // index of the sample to use
  // our goal is to use maxNumSamples for which NoL is > 0
  // to achieve this, we might have to try more samples than
  // maxNumSamples
  for (size_t sampleIndex = 0; sampleIndex < maxNumSamples; sampleIndex++) {
    /*
     *       (sampling)
     *            L         H (never calculated below)
     *            .\       /.
     *            . \     / .
     *            .  \   /  .
     *            .   \ /   .
     *         ---|----o----|-------> n
     *    cos(2*theta)    cos(theta)
     *       = n.L           = n.H
     *
     * Note: NoH == LoH
     * (H is the half-angle between L and V, and V == N)
     *
     */

    // get Hammersley distribution for the half-sphere
    float u[2];
    hammersley(uint32_t(sampleIndex), inumSamples, u);

    // Importance sampling GGX - Trowbridge-Reitz
    // This corresponds to integrating Dggx()*cos(theta) over the hemisphere
    float H[3];
    hemisphereImportanceSampleDggx(u, linearRoughness, H);

#if 0
        // This produces the same result that the code below using the the non-simplified
        // equation. This let's us see that N == V and that L = -reflect(V, H)
        // Keep this for reference.
        const double3 N = {0, 0, 1};
        const double3 V = N;
        const double3 L = 2 * dot(H, V) * H - V;
        const double NoL = dot(N, L);
        const double NoH = dot(N, H);
        const double NoH2 = NoH * NoH;
        const double NoV = dot(N, V);
        const double LoH = dot(L, H);
#else
    const float NoV = 1;
    const float NoH = H[2];
    const float NoH2 = H[2] * H[2];
    const float NoL = 2 * NoH2 - 1;
    float L[3];
    L[0] = 2 * NoH * H[0];
    L[1] = 2 * NoH * H[1];
    L[2] = NoL;
    const float LoH = dot(L, H);
#endif

    if (NoL > 0) {
      // pre-filtered importance sampling
      // see: "Real-time Shading with Filtered Importance Sampling", Jaroslav
      // Krivanek see: "GPU-Based Importance Sampling, GPU Gems 3", Mark Colbert

      // K is a LOD bias that allows a bit of overlapping between samples
      // log4(K)=1 empirically works well with box-filtered mipmaps
      constexpr float K = 4;

      // OmegaS is is the solid-angle of an important sample (i.e. how much
      // surface of the sphere it represents). It obviously is function of the
      // PDF.
      const float pdf = DistributionGGX(NoH, linearRoughness) / 4;
      const float omegaS = 1 / (numSamples * pdf);

      // The LOD is given by: max[ log4(Os/Op) + K, 0 ]
      const float l = float(log4(omegaS) - log4(omegaP) + log4(K));
      const float mipLevel = std::max(0.0f, std::min(float(l), maxLevelf));

      const float V = Visibility(NoV, NoL, linearRoughness);
      const float F = Fresnel(1, 0, LoH);
      const float brdf_NoL = float(F * V * NoL);

      weight += brdf_NoL;

      uint8_t l0 = uint8_t(mipLevel);
      uint8_t l1 = uint8_t(std::min(maxLevel, size_t(l0 + 1)));
      float lerp = mipLevel - l0;

      CacheEntry entry;
      entry.L[0] = L[0];
      entry.L[1] = L[1];
      entry.L[2] = L[2];
      entry.brdf_NoL = brdf_NoL;
      entry.lerp = lerp;
      entry.l0 = l0;
      entry.l1 = l1;
      cache.push_back(entry);
    }
  }

  std::for_each(cache.begin(), cache.end(),
                [weight](CacheEntry& entry) { entry.brdf_NoL /= weight; });

  // we can sample the cubemap in any order, sort by the weight, it could
  // improve fp precision
  std::sort(cache.begin(), cache.end(),
            [](CacheEntry const& lhs, CacheEntry const& rhs) {
              return lhs.brdf_NoL < rhs.brdf_NoL;
            });

  for (size_t f = 0; f < 6; ++f) {
    std::cout << "f " << f << std::endl;

    std::vector<std::thread> workers;
    std::atomic<size_t> i(0);

    int num_threads = std::thread::hardware_concurrency();

    num_threads = std::max(1, num_threads);

    for (uint32_t t = 0; t < num_threads; t++) {
      workers.emplace_back(std::thread([&]() {
        size_t y = 0;

        for ((y = i++); y < dim0; ++y) {
          float3 R[3];
          const size_t numSamples = cache.size();
          for (size_t x = 0; x < dim0; ++x) {
            float p[2] = {x + 0.5f, y + 0.5f};
            // getDirectionFor(f, p[0], [1]);
            const float3 N = getDirectionFor(f, p[0], p[1], dim0);

            // center the cone around the normal (handle case of normal close to
            // up)
            float3 up;
            if (std::abs(N[2] < 0.999f)) {
              up = float3(0, 0, 1);
            } else {
              up = float3(1, 0, 0);
            }

            R[0] = vnormalize(vcross(up, N));
            R[1] = vcross(N, R[0]);
            R[2] = N;

            float3 Li = 0;
            for (size_t sample = 0; sample < numSamples; sample++) {
              const CacheEntry& e = cache[sample];
              float3 L;
              L[0] = R[0][0] * e.L[0] + R[1][0] * e.L[1] + R[2][0] * e.L[2];
              L[1] = R[0][1] * e.L[0] + R[1][1] * e.L[1] + R[2][1] * e.L[2];
              L[2] = R[0][2] * e.L[0] + R[1][2] * e.L[1] + R[2][2] * e.L[2];
              const Cubemap& cmBase = levels[e.l0];
              const Cubemap& next = levels[e.l1];
              const float3 c0 = trilinearFilterAt(cmBase, next, e.lerp, L);
              Li += c0 * e.brdf_NoL;
              // HACK
              //const float3 c0 =
              //    trilinearFilterAt(cmBase, next, 1.0f /*e.lerp*/, );
              //Li += c0 * e.brdf_NoL;
            }
            // Cubemap::writeAt(data, Cubemap::Texel(Li));
            // HACK
            writeAt(dst.faces[f], x, y, Li);
            // dst.faces[f].pixels[3 * (y * dim0 + x) + 0] = Li[0];
            // dst.faces[f].pixels[3 * (y * dim0 + x) + 1] = Li[1];
            // dst.faces[f].pixels[3 * (y * dim0 + x) + 2] = Li[2];
          }
        }
      }));
    }

    for (auto& t : workers) {
      t.join();
    }
  }

  auto endT = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = endT - startT;
  std::cout << "Processing time : " << ms.count() << " [ms]" << std::endl;
}

void roughnessFilterLonglat(const std::vector<Image>& levels,
                            float linearRoughness, size_t maxNumSamples,
                            const size_t output_height, Image* dst) {
  dst->width = output_height * 2;
  dst->height = output_height;
  dst->channels = 3;
  dst->pixels.resize(dst->width * dst->height * 3);

  const size_t maxLevel = levels.size() - 1;

  const float numSamples = maxNumSamples;
  const float inumSamples = 1.0f / numSamples;
  const size_t dim0 = dst->height;
  const float omegaP =
      float((4 * kPI) / (2 * dim0 * dim0));  // FIXME(LTE): Validate

  if (linearRoughness < std::numeric_limits<float>::epsilon()) {
    // No roughness filtering.

    for (size_t y = 0; y < dst->height; y++) {
      float v = (y + 0.5f) / float(dst->height);
      for (size_t x = 0; x < dst->width; ++x) {
        float u = (x + 0.5f) / float(dst->width);

        float rgba[4];
        SampleImage(rgba, u, v, levels[0].width, levels[0].height, levels[0].channels, levels[0].pixels.data());

        dst->pixels[3 * (y * dst->width + x) + 0] = rgba[0];
        dst->pixels[3 * (y * dst->width + x) + 1] = rgba[1];
        dst->pixels[3 * (y * dst->width + x) + 2] = rgba[2];

      }
    }

  } else {
    // be careful w/ the size of this structure, the smaller the better
    struct CacheEntry {
      float3 L;
      float brdf_NoL;
      float lerp;
      uint8_t l0;
      uint8_t l1;
    };

    std::vector<CacheEntry> cache;
    cache.reserve(maxNumSamples);

    // precompute everything that only depends on the sample #
    double weight = 0;
    // index of the sample to use
    // our goal is to use maxNumSamples for which NoL is > 0
    // to achieve this, we might have to try more samples than
    // maxNumSamples
    for (size_t sampleIndex = 0; sampleIndex < maxNumSamples; sampleIndex++) {
      /*
       *       (sampling)
       *            L         H (never calculated below)
       *            .\       /.
       *            . \     / .
       *            .  \   /  .
       *            .   \ /   .
       *         ---|----o----|-------> n
       *    cos(2*theta)    cos(theta)
       *       = n.L           = n.H
       *
       * Note: NoH == LoH
       * (H is the half-angle between L and V, and V == N)
       *
       */

      // get Hammersley distribution for the half-sphere
      float u[2];
      hammersley(uint32_t(sampleIndex), inumSamples, u);

      // Importance sampling GGX - Trowbridge-Reitz
      // This corresponds to integrating Dggx()*cos(theta) over the hemisphere
      float H[3];
      hemisphereImportanceSampleDggx(u, linearRoughness, H);

#if 0
        // This produces the same result that the code below using the the non-simplified
        // equation. This let's us see that N == V and that L = -reflect(V, H)
        // Keep this for reference.
        const double3 N = {0, 0, 1};
        const double3 V = N;
        const double3 L = 2 * dot(H, V) * H - V;
        const double NoL = dot(N, L);
        const double NoH = dot(N, H);
        const double NoH2 = NoH * NoH;
        const double NoV = dot(N, V);
        const double LoH = dot(L, H);
#else
      const float NoV = 1;
      const float NoH = H[2];
      const float NoH2 = H[2] * H[2];
      const float NoL = 2 * NoH2 - 1;
      float L[3];
      L[0] = 2 * NoH * H[0];
      L[1] = 2 * NoH * H[1];
      L[2] = NoL;
      const float LoH = dot(L, H);
#endif

      if (NoL > 0) {
        // pre-filtered importance sampling
        // see: "Real-time Shading with Filtered Importance Sampling", Jaroslav
        // Krivanek see: "GPU-Based Importance Sampling, GPU Gems 3", Mark
        // Colbert

        // K is a LOD bias that allows a bit of overlapping between samples
        // log4(K)=1 empirically works well with box-filtered mipmaps
        constexpr float K = 4;

        // OmegaS is is the solid-angle of an important sample (i.e. how much
        // surface of the sphere it represents). It obviously is function of the
        // PDF.
        const float pdf = DistributionGGX(NoH, linearRoughness) / 4;
        const float omegaS = 1 / (numSamples * pdf);

        // The LOD is given by: max[ log4(Os/Op) + K, 0 ]
        const float l = float(log4(omegaS) - log4(omegaP) + log4(K));
        const float mipLevel = std::max(0.0f, std::min(float(l), float(maxLevel)));

        const float V = Visibility(NoV, NoL, linearRoughness);
        const float F = Fresnel(1, 0, LoH);
        const float brdf_NoL = float(F * V * NoL);

        weight += brdf_NoL;

        uint8_t l0 = uint8_t(mipLevel);
        uint8_t l1 = uint8_t(std::min(maxLevel, size_t(l0 + 1)));
        float lerp = mipLevel - l0;

        CacheEntry entry;
        entry.L[0] = L[0];
        entry.L[1] = L[1];
        entry.L[2] = L[2];
        entry.brdf_NoL = brdf_NoL;
        entry.lerp = lerp;
        entry.l0 = l0;
        entry.l1 = l1;
        cache.push_back(entry);
      }
    }

    std::for_each(cache.begin(), cache.end(),
                  [weight](CacheEntry& entry) { entry.brdf_NoL /= weight; });

    // we can sample the cubemap in any order, sort by the weight, it could
    // improve fp precision
    std::sort(cache.begin(), cache.end(),
              [](CacheEntry const& lhs, CacheEntry const& rhs) {
                return lhs.brdf_NoL < rhs.brdf_NoL;
              });

    std::vector<std::thread> workers;
    std::atomic<size_t> i(0);

    int num_threads = std::thread::hardware_concurrency();

    num_threads = std::max(1, num_threads);

    if (dst->height < num_threads) {
      num_threads = dst->height; 
    }

    size_t ndiv = dst->height / num_threads;

    std::cout << "ndiv = " << ndiv << std::endl;
    std::cout << "num_threads = " << num_threads << std::endl;

    for (uint32_t t = 0; t < num_threads; t++) {
      workers.emplace_back(std::thread([&, t]() {

        size_t sy = t * ndiv;
        size_t ey = (t == (num_threads - 1)) ? dst->height : std::min(size_t(dst->height), (t + 1) * ndiv);

        std::cout << "sy = " << sy << ", ey = " << ey << std::endl;

        for (size_t y = sy; y < ey; ++y) {
          float3 R[3];
          const size_t numSamples = cache.size();
          for (size_t x = 0; x < dst->width; ++x) {
            float p[2] = {(x + 0.5f) / float(dst->width), (y + 0.5f) / float(dst->height)};
            // CacheEntry uses mirrored vector direction?
            // so take an negate for X here
            float3 N = uv_to_dir(-p[0], p[1], 0.0f);

            // center the cone around the normal (handle case of normal close to up)
            float3 up;
            if (std::abs(N[2] < 0.999f)) {
              up = float3(0, 0, 1);
            } else {
              up = float3(1, 0, 0);
            }

            R[0] = vnormalize(vcross(up, N));
            R[1] = vcross(N, R[0]);
            R[2] = N;

            float3 Li = 0;
            for (size_t sample = 0; sample < numSamples; sample++) {
              const CacheEntry& e = cache[sample];
              float3 L;
              L[0] = R[0][0] * e.L[0] + R[1][0] * e.L[1] + R[2][0] * e.L[2];
              L[1] = R[0][1] * e.L[0] + R[1][1] * e.L[1] + R[2][1] * e.L[2];
              L[2] = R[0][2] * e.L[0] + R[1][2] * e.L[1] + R[2][2] * e.L[2];
              //const Image& base = levels[e.l0];

              //L = vnormalize(L);
              

              // HACK
              const Image& base = levels[e.l0];
              const Image& next = levels[e.l1];
              //const float3 c0 = trilinearFilterAt(cmBase, next, e.lerp, L);
              // TODO trilinear filtering.

              float uv[2];
              dir_to_uv(&uv[0], &uv[1], L);

              float rgba[4];
              assert(!std::isnan(uv[0]));
              assert(!std::isnan(uv[1]));
              SampleImage(rgba, uv[0], uv[1], base.width, base.height, base.channels, base.pixels.data());

              const float3 c0(rgba[0], rgba[1], rgba[2]);
              //std::cout << "col = " << rgba[0] << std::endl;

              //Li += c0 * e.brdf_NoL;
              Li += c0 * e.brdf_NoL;
            }
            dst->pixels[3 * (y * dst->width + x) + 0] = Li[0];
            dst->pixels[3 * (y * dst->width + x) + 1] = Li[1];
            dst->pixels[3 * (y * dst->width + x) + 2] = Li[2];
          }
        }
      }));
    }

    for (auto &t : workers) {
      t.join();
    }
  }
}

// ----------------------------------------

static float3 uv_to_dir(float u, float v, float phi_offset)
{
  float cos_theta = std::cos(v * kPI);
  float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

  float phi = u * 2.0f * kPI;

  phi += phi_offset;

  // Y-up
  float3 n(sin_theta * std::cos(phi), cos_theta,
           -sin_theta * std::sin(phi));

  return n;
}

static void dir_to_uv(float *uu, float *vv, const float3 n) {
#if 1
  // C version

  float v[3];
  // Z up, right-handed
  v[0] = n[0];
  //v[1] = -n[2];
  v[1] = n[2];
  v[2] = n[1];

  // atan2(y, x) =
  //
  //           y                                  y
  //       pi/2|^                            pi/2 |^
  //           |                                  |
  //   pi      |       0                 pi       |        0
  //  ---------o---------> x       =>>   ---------o--------> x
  //  -pi      |      -0                 pi       |      2 pi
  //           |                                  |
  //           |                                  |
  //      -pi/2|                             3/2pi|
  //

  float phi = std::atan2(v[1], v[0]);

  if (phi < 0.0f) {
    phi += 2.0f * kPI;  
  }

  // -> now phi in > 0.0

  // wrap around in [0, 2 pi] range.
  phi = std::fmod(phi, 2.0f * kPI);

  // for safety
  if (phi < 0.0f) phi = 0.0f;
  if (phi > 2.0f * kPI) phi = 2.0f * kPI;


  float z = v[2];
  if (z < -1.0f) z = -1.0f;
  if (z > 1.0f) z = 1.0f;
  float theta = std::acos(z);

  (*uu) = phi / (2.0f * kPI);
  (*vv) = theta / kPI;
#else
  // GLSL version
  // v to (theta, phi). Y up to Z up.
  float theta = acosf(n[1]);
  float phi = 0.0f;
  if (n[2] == 0.0f) {
  } else {
    phi = atan2f(n[0], -n[2]);
  }

  (*uu) = phi / (2.0 * pi);
  (*vv) = 1.0f - theta / pi;
  if ((*vv) < 0.0f) (*v) = 0.0f;
  if ((*vv) > 0.99999f)
    (*vv) = 0.9999f;  // 0.99999 = prevent texture wrap around.
#endif
}


void CubemapToEquirectangular(const Cubemap& cubemap, const float phi_offset,
                              size_t output_height, Image* output) {
  size_t width = output_height * 2;
  size_t height = output_height;

  output->width = int(width);
  output->height = int(height);
  output->channels = 3;
  output->pixels.resize(width * height * 3);

  for (size_t y = 0; y < height; y++) {
    float v = (y + 0.5f) / float(height);

    float cos_theta = std::cos(v * kPI);
    float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

    for (size_t x = 0; x < width; x++) {
      float u = (x + 0.5f) / float(width);
      float phi = u * 2.0f * kPI;

      phi += phi_offset;

      // Y-up
      float3 n(sin_theta * std::cos(phi), cos_theta,
               -sin_theta * std::sin(phi));

      float3 L = filterAt(cubemap, n);

      output->pixels[3 * (y * width + x) + 0] = L[0];
      output->pixels[3 * (y * width + x) + 1] = L[1];
      output->pixels[3 * (y * width + x) + 2] = L[2];
    }
  }
}

void SaveImage(const Image& image, const std::string& filename) {
  int ret = SaveEXR(image.pixels.data(), image.width, image.height,
                    image.channels, /* fp16 */ 0, filename.c_str());
  if (TINYEXR_SUCCESS != ret) {
    std::cerr << "Save EXR error. code = " << ret << std::endl;
  }
}

static void SaveCubemap(const Cubemap& cubemap, const std::string& basename) {
  for (size_t f = 0; f < 6; f++) {
    std::string filename = basename + "_face" + std::to_string(f) + ".exr";
    SaveImage(cubemap.faces[f], filename);
  }
}

void BuildPrefilteredRoughnessMap(const Cubemap& cubemap, int num_samples,
                                  const size_t output_base_height,
                                  std::vector<Image>* output_levels) {
  int e = int(std::log2(float(cubemap.dim()))) - 1;
  size_t num_levels = 1 + e;

  size_t output_height = output_base_height;

  std::vector<Cubemap> levels;
  levels.push_back(cubemap);
  generateMipmaps(levels);

  std::cout << "num e = " << e << std::endl;

  for (size_t i = e; i >= 0; --i) {
    int level = e - i;
    if (level >= 2) {
      // increase the number of samples per level
      num_samples *= 2;
    }

    float roughness = saturate(float(level) / (num_levels - 1.0f));
    float linear_roughness = roughness * roughness;

    std::cout << "level " << level << ", roughness = " << roughness
              << ", num_samples = " << num_samples << std::endl;

    Cubemap dst(int(std::pow(2, i)));
    std::cout << "dst size " << dst.dim() << std::endl;
    roughnessFilter(dst, levels, linear_roughness, num_samples);
    makeSeamless(dst);

    Image longlat;

    // TODO(LTE): phi offset
    CubemapToEquirectangular(
        dst, 0.0f, (output_base_height < 1) ? 1 : output_base_height, &longlat);

#if 1  // debug
    SaveImage(longlat, "longlat_" + std::to_string(level) + ".exr");
#endif

    output_levels->push_back(std::move(longlat));

    output_height /= 2;
  }
}

bool BuildPrefilteredRoughnessMap(const Image& longlat, int num_samples,
                                  const size_t output_base_height,
                                  std::vector<Image>* output_levels) {
  int e = std::max(1, int(std::log2(float(output_base_height))) - 1);
  size_t num_levels = 1 + e;

  size_t output_height = output_base_height;

  std::vector<Image> levels;
  Image base;
  if (longlat.channels < 3) {
    std::cerr << "Only RGB or RGBA format image are supported.";
    return false;
  }

  // create RGB image.
  base.pixels.resize(longlat.width * longlat.height * 3);
  for (size_t i = 0; i < longlat.width * longlat.height; i++) {
    base.pixels[3 * i + 0] = longlat.pixels[longlat.channels * i + 0];
    base.pixels[3 * i + 1] = longlat.pixels[longlat.channels * i + 1];
    base.pixels[3 * i + 2] = longlat.pixels[longlat.channels * i + 2];
  }
  base.width = longlat.width;
  base.height = longlat.height;
  base.channels = 3;

  levels.push_back(base); // base layer
  generateMipmaps(levels);

  std::cout << "num e = " << e << std::endl;

  for (size_t i = e; i >= 0; --i) {
    int level = e - i;
    if (level >= 2) {
      // increase the number of samples per level
      num_samples *= 2;
    }

    float roughness = saturate(float(level) / (num_levels - 1.0f));
    float linear_roughness = roughness * roughness;

    std::cout << "level " << level << ", roughness = " << roughness
              << ", num_samples = " << num_samples << std::endl;

    size_t dst_dim = size_t(std::pow(2, i));

    if (dst_dim < 2) {
      break;
    }

    std::cout << "dst size " << dst_dim << std::endl;
    Image longlat;
    roughnessFilterLonglat(levels, linear_roughness, num_samples, dst_dim, &longlat);


#if 1  // debug
    SaveImage(longlat, "longlat_" + std::to_string(level) + ".exr");
#endif

    output_levels->push_back(std::move(longlat));

    output_height /= 2;
  }

  return true;
}

}  // namespace example
