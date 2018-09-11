/*
The MIT License (MIT)

Copyright (c) 2015 - 2018 Light Transport Entertainment, Inc.

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

#ifdef _MSC_VER
#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4189)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4477)
#endif

#include "render.h"

#include <array>
#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#include "../../nanort.h"
#include "material.h"
#include "matrix.h"
#include "mesh.h"
#include "render-layer.h"
#include "trackball.h"

#ifdef WIN32
#undef min
#undef max
#endif

namespace example {

// PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// http://www.pcg-random.org/
typedef struct {
  unsigned long long state;
  unsigned long long inc;  // not used?
} pcg32_state_t;

#define PCG32_INITIALIZER \
  { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

float pcg32_random(pcg32_state_t* rng) {
  unsigned long long oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  unsigned int rot = oldstate >> 59u;
  unsigned int ret =
      (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));

  return (float)((double)ret / (double)4294967296.0);
}

void pcg32_srandom(pcg32_state_t* rng, uint64_t initstate, uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1U) | 1U;
  pcg32_random(rng);
  rng->state += initstate;
  pcg32_random(rng);
}

const float kPI = 3.141592f;

typedef nanort::real3<float> float3;

inline float3 Lerp3(float3 v0, float3 v1, float3 v2, float u, float v) {
  return (1.0f - u - v) * v0 + u * v1 + v * v2;
}

inline float3 faceforward(const float3 n, const float3 I) {
  const float ndotI = vdot(n, I);
  if (ndotI > 0.0f) {
    return -n;
  }
  return n;
}

inline float3 reflect(float3 I, float3 N) { return I - 2.0f * vdot(I, N) * N; }

inline void CalcNormal(float3& N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}

void BuildCameraFrame(float3* origin, float3* corner, float3* u, float3* v,
                      float quat[4], float eye[3], float lookat[3], float up[3],
                      float fov, int width, int height) {
  float e[4][4];

  Matrix::LookAt(e, eye, lookat, up);

  float r[4][4];
  build_rotmatrix(r, quat);

  float3 lo;
  lo[0] = lookat[0] - eye[0];
  lo[1] = lookat[1] - eye[1];
  lo[2] = lookat[2] - eye[2];
  float dist = vlength(lo);

  float dir[3];
  dir[0] = 0.0;
  dir[1] = 0.0;
  dir[2] = dist;

  Matrix::Inverse(r);

  float rr[4][4];
  float re[4][4];
  float zero[3] = {0.0f, 0.0f, 0.0f};
  float localUp[3] = {0.0f, 1.0f, 0.0f};
  Matrix::LookAt(re, dir, zero, localUp);

  // translate
  re[3][0] += eye[0];  // 0.0; //lo[0];
  re[3][1] += eye[1];  // 0.0; //lo[1];
  re[3][2] += (eye[2] - dist);

  // rot -> trans
  Matrix::Mult(rr, r, re);

  float m[4][4];
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      m[j][i] = rr[j][i];
    }
  }

  float vzero[3] = {0.0f, 0.0f, 0.0f};
  float eye1[3];
  Matrix::MultV(eye1, m, vzero);

  float lookat1d[3];
  dir[2] = -dir[2];
  Matrix::MultV(lookat1d, m, dir);
  float3 lookat1(lookat1d[0], lookat1d[1], lookat1d[2]);

  float up1d[3];
  Matrix::MultV(up1d, m, up);

  float3 up1(up1d[0], up1d[1], up1d[2]);

  // absolute -> relative
  up1[0] -= eye1[0];
  up1[1] -= eye1[1];
  up1[2] -= eye1[2];
  // printf("up1(after) = %f, %f, %f\n", up1[0], up1[1], up1[2]);

  // Use original up vector
  // up1[0] = up[0];
  // up1[1] = up[1];
  // up1[2] = up[2];

  {
    float flen =
        (0.5f * (float)height / tanf(0.5f * (float)(fov * kPI / 180.0f)));
    float3 look1;
    look1[0] = lookat1[0] - eye1[0];
    look1[1] = lookat1[1] - eye1[1];
    look1[2] = lookat1[2] - eye1[2];
    // vcross(u, up1, look1);
    // flip
    (*u) = nanort::vcross(look1, up1);
    (*u) = vnormalize((*u));

    (*v) = vcross(look1, (*u));
    (*v) = vnormalize((*v));

    look1 = vnormalize(look1);
    look1[0] = flen * look1[0] + eye1[0];
    look1[1] = flen * look1[1] + eye1[1];
    look1[2] = flen * look1[2] + eye1[2];
    (*corner)[0] = look1[0] - 0.5f * (width * (*u)[0] + height * (*v)[0]);
    (*corner)[1] = look1[1] - 0.5f * (width * (*u)[1] + height * (*v)[1]);
    (*corner)[2] = look1[2] - 0.5f * (width * (*u)[2] + height * (*v)[2]);

    (*origin)[0] = eye1[0];
    (*origin)[1] = eye1[1];
    (*origin)[2] = eye1[2];
  }
}

#if 0  // TODO(LTE): Not used method. Delete.
nanort::Ray<float> GenerateRay(const float3& origin, const float3& corner,
                               const float3& du, const float3& dv, float u,
                               float v) {
  float3 dir;

  dir[0] = (corner[0] + u * du[0] + v * dv[0]) - origin[0];
  dir[1] = (corner[1] + u * du[1] + v * dv[1]) - origin[1];
  dir[2] = (corner[2] + u * du[2] + v * dv[2]) - origin[2];
  dir = vnormalize(dir);

  float3 org;

  nanort::Ray<float> ray;
  ray.org[0] = origin[0];
  ray.org[1] = origin[1];
  ray.org[2] = origin[2];
  ray.dir[0] = dir[0];
  ray.dir[1] = dir[1];
  ray.dir[2] = dir[2];

  return ray;
}
#endif

float3 irradianceSH(const float sh[9][3], const float3 n) {
  float rgb[3];

  for (int i = 0; i < 3; i++) {
    rgb[i] = sh[0][i] + sh[1][i] * (n[1]) + sh[2][i] * (n[2]) +
             sh[3][i] * (n[0]) + sh[4][i] * (n[1] * n[0]) +
             sh[5][i] * (n[1] * n[2]) + sh[6][i] * (3.0f * n[2] * n[2] - 1.0f) +
             sh[7][i] * (n[2] * n[0]) + sh[8][i] * (n[0] * n[0] - n[1] * n[1]);

    // clamp negative calue.
    rgb[i] = std::max(rgb[i], 0.0f);
  }

  return float3(rgb);
}

void convert_xyz_to_cube_uv(float x, float y, float z, int* index, float* u,
                            float* v) {
  float absX = fabs(x);
  float absY = fabs(y);
  float absZ = fabs(z);

  int isXPositive = x > 0.0f ? 1 : 0;
  int isYPositive = y > 0.0f ? 1 : 0;
  int isZPositive = z > 0.0f ? 1 : 0;

  float maxAxis, uc, vc;

  // POSITIVE X
  if (isXPositive && absX >= absY && absX >= absZ) {
    // u (0 to 1) goes from +z to -z
    // v (0 to 1) goes from -y to +y
    maxAxis = absX;
    uc = -z;
    vc = y;
    *index = 0;
  }
  // NEGATIVE X
  if (!isXPositive && absX >= absY && absX >= absZ) {
    // u (0 to 1) goes from -z to +z
    // v (0 to 1) goes from -y to +y
    maxAxis = absX;
    uc = z;
    vc = y;
    *index = 1;
  }
  // POSITIVE Y
  if (isYPositive && absY >= absX && absY >= absZ) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from +z to -z
    maxAxis = absY;
    uc = x;
    vc = -z;
    *index = 2;
  }
  // NEGATIVE Y
  if (!isYPositive && absY >= absX && absY >= absZ) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -z to +z
    maxAxis = absY;
    uc = x;
    vc = z;
    *index = 3;
  }
  // POSITIVE Z
  if (isZPositive && (absZ >= absX) && (absZ >= absY)) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -y to +y
    maxAxis = absZ;
    uc = x;
    vc = y;
    *index = 4;
  }
  // NEGATIVE Z
  if (!isZPositive && (absZ >= absX) && (absZ >= absY)) {
    // u (0 to 1) goes from +x to -x
    // v (0 to 1) goes from -y to +y
    maxAxis = absZ;
    uc = -x;
    vc = y;
    *index = 5;
  }

  // Convert range from -1 to 1 to 0 to 1
  *u = 0.5f * (uc / maxAxis + 1.0f);
  *v = 0.5f * (vc / maxAxis + 1.0f);
}

//
// Simple bilinear texture filtering.
//
static void SampleTexture(float* rgba, float u, float v, int width, int height,
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

  for (int i = 0; i < channels; i++) {
    rgba[i] = w[0] * texels[i00 + i] + w[1] * texels[i10 + i] +
              w[2] * texels[i01 + i] + w[3] * texels[i11 + i];
  }
}

static inline void sRGBToLinear(const unsigned char sRGB[3], float linear[3]) {
  const float a = 0.055f;
  const float a1 = 1.055f;
  const float p = 2.4f;

  for (size_t i = 0; i < 3; i++) {
    float s = sRGB[i] / 255.0f;
    if (s <= 0.04045f) {
      linear[i] = s * (1.0f / 12.92f);
    } else {
      linear[i] = std::pow((s + a) / a1, p);  // TODO(LTE): Use fast pow
    }
  }
}

//
// Simple bilinear texture filtering.
//
static void SampleTexture(float* rgba, float u, float v, int width, int height,
                          int channels, const unsigned char* texels) {
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

  // Assume pixels are in sRGB space.
  float t[4][3];

  sRGBToLinear(texels + i00, t[0]);
  sRGBToLinear(texels + i10, t[1]);
  sRGBToLinear(texels + i01, t[2]);
  sRGBToLinear(texels + i11, t[3]);

  for (int i = 0; i < channels; i++) {
    rgba[i] = w[0] * t[0][i] + w[1] * t[1][i] + w[2] * t[2][i] + w[3] * t[3][i];
  }
}

static void FetchTexture(const Texture& texture, float u, float v, float* col) {
  SampleTexture(col, u, v, texture.width, texture.height, texture.components,
                texture.image);
}

static void prefilteredDFG(float NoV, float roughness, float uv[2]) {
  // Karis' approximation based on Lazarov's
  const float c0[4] = {-1.0f, -0.0275f, -0.572f, 0.022f};
  const float c1[4] = {1.0f, 0.0425f, 1.040f, -0.040f};
  float r[4];
  r[0] = roughness * c0[0] + c1[0];
  r[1] = roughness * c0[1] + c1[1];
  r[2] = roughness * c0[2] + c1[2];
  r[3] = roughness * c0[3] + c1[3];

  float a004 = std::min(r[0] * r[0], std::exp2(-9.28f * NoV)) * r[0] + r[1];
  uv[0] = -1.04f * a004 + r[3];
  uv[1] = 1.04f * a004 + r[2];
  // Zioma's approximation based on Karis
  // return vec2(1.0, pow(1.0 - max(roughness, NoV), 3.0));
}

static float roughness_to_lod(const float roughness) {
  // Assume 256x256 cubemap
  return 8.0f * roughness;
}

static void SampleEnvmap(const example::Asset& asset, const float3 n, float lod,
                         float* col) {
  if (asset.cubemap_ibl.size() == 0) {
    // no envmap
    col[0] = 0.0f;
    col[1] = 0.0f;
    col[2] = 0.0f;
  }

  int face_idx;
  float u, v;
  convert_xyz_to_cube_uv(n[0], n[1], n[2], &face_idx, &u, &v);

  // flipY
  v = 1.0f - v;

  int ilod0 = std::min(int(asset.cubemap_ibl.size() - 1),
                       std::max(0, int(std::trunc(lod))));
  int ilod1 =
      std::min(int(asset.cubemap_ibl.size() - 1), std::max(0, ilod0 + 1));
  float frac = lod - float(ilod0);

  const std::array<example::Image, 6>& cubemap0 = asset.cubemap_ibl[ilod0];
  const std::array<example::Image, 6>& cubemap1 = asset.cubemap_ibl[ilod1];

  const example::Image& face0 = cubemap0[face_idx];
  const example::Image& face1 = cubemap1[face_idx];

  float rgba0[4];
  float rgba1[4];
  SampleTexture(rgba0, u, v, face0.width, face0.height, face0.channels,
                face0.pixels.data());
  SampleTexture(rgba1, u, v, face1.width, face1.height, face1.channels,
                face1.pixels.data());

  col[0] = (1.0f - frac) * rgba0[0] + frac * rgba1[0];
  col[1] = (1.0f - frac) * rgba0[1] + frac * rgba1[1];
  col[2] = (1.0f - frac) * rgba0[2] + frac * rgba1[2];
}

// Based on Filament's GLSL shader --------------------------------------

#if defined(TARGET_MOBILE)
// min roughness such that (MIN_ROUGHNESS^4) > 0 in fp16 (i.e. 2^(-14/4),
// slightly rounded up)
#define MIN_ROUGHNESS 0.089f
#define MIN_LINEAR_ROUGHNESS 0.007921f
#else
#define MIN_ROUGHNESS 0.045f
#define MIN_LINEAR_ROUGHNESS 0.002025f
#endif

#define MAX_CLEAR_COAT_ROUGHNESS 0.6

static inline float pow5(float x) { return (x * x) * (x * x) * x; }

static inline float3 saturate(const float3 v) {
  return float3(std::max(0.0f, std::min(v[0], 1.0f)),
                std::max(0.0f, std::min(v[1], 1.0f)),
                std::max(0.0f, std::min(v[2], 1.0f)));
}

static float3 f0ClearCoatToSurface(const float3 f0) {
  // Approximation of iorTof0(f0ToIor(f0), 1.5)
  // This assumes that the clear coat layer has an IOR of 1.5
#if defined(TARGET_MOBILE)
  return saturate(f0 * (f0 * 0.526868f + 0.529324f) - 0.0482256f);
#else
  return saturate(f0 * (f0 * (0.941892f - 0.263008f * f0) + 0.346479f) -
                  0.0285998f);
#endif
}

static float3 prefilteredDFG(const Image& lut, const float NoV,
                             const float roughness) {
  float rgba[4];
  SampleTexture(rgba, NoV, roughness, lut.width, lut.height, lut.channels,
                lut.pixels.data());

  return float3(rgba[0], rgba[1], 0.0f);  // float2
}

float computeSpecularAO(float NoV, float ao, float roughness) {
#if defined(IBL_SPECULAR_OCCLUSION) && defined(MATERIAL_HAS_AMBIENT_OCCLUSION)
  return saturate(pow(NoV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
#else
  return 1.0;
#endif
}

float3 specularDFG(const float3 dfg, const float3 f0) {
#if defined(SHADING_MODEL_CLOTH) || \
    !defined(USE_MULTIPLE_SCATTERING_COMPENSATION)
  return f0 * dfg[0] + dfg[1];
#else
  return dfg[0] * (1.0f - f0) + dfg[1] * f0;
#endif
}

static float3 specularIrradiance(const example::Asset& asset, const float3 r,
                                 float roughness) {
  // lod = nb_mips * sqrt(linear_roughness)
  // where linear_roughness = roughness^2
  // using all the mip levels requires seamless cubemap sampling
  const float IBL_MAX_MIP_LEVEL =
      8.0f;  // FIXME(LTE): Set max mip level from cubemap size.
  float lod = IBL_MAX_MIP_LEVEL * roughness;
  float rgba[4];
  SampleEnvmap(asset, r, lod, rgba);

  return float3(rgba[0], rgba[1], rgba[2]);
}

static float F_Schlick(float f0, float f90, float VoH) {
  return f0 + (f90 - f0) * pow5(1.0f - VoH);
}

void evaluateClearCoatIBL(const example::Asset& asset, const float NoV,
                          const float3& R, const float clearCoat,
                          const float clearCoatRoughness, float specularAO,
                          /* input */ float3* Fd, /* input */ float3* Fr) {
//#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
#if 0
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV = abs(dot(shading_clearCoatNormal, shading_view)) + FLT_EPS;
    vec3 clearCoatR = reflect(-shading_view, shading_clearCoatNormal);
#else
  float clearCoatNoV = NoV;
  float3 clearCoatR = R;
#endif

  // The clear coat layer assumes an IOR of 1.5 (4% reflectance)
  float Fc = F_Schlick(0.04f, 1.0f, clearCoatNoV) * clearCoat;
  float attenuation = 1.0f - Fc;
  (*Fr) = (*Fr) * attenuation * attenuation;
  (*Fr) = (*Fr) + specularIrradiance(asset, clearCoatR, clearCoatRoughness) *
                      (specularAO * Fc);
  (*Fd) = (*Fd) * attenuation;
}

static float3 Shade(const example::Asset& asset,
                    const example::Material& material,
                    const float3 inputBaseColor, const float3 N,
                    const float3 V) {
  float3 baseColor = inputBaseColor;
  float metallic = std::max(0.0f, std::min(1.0f, material.metallic));
  float3 diffuseColor = (1.0f - metallic) * float3(baseColor);
  float reflectance = material.reflectance;
  // Assumes an interface from air to an IOR of 1.5 for dielectrics
  float3 f0 = float3(0.16f * reflectance * reflectance * (1.0f - metallic)) +
              baseColor * metallic;

  // Clamp the roughness to a minimum value to avoid divisions by 0 in the
  // lighting code
  float roughness = material.roughness;
  roughness = std::min(1.0f, std::max(roughness, MIN_ROUGHNESS));

  float linearRoughness = roughness * roughness;

  // Clear coat ----------------------------------------
  float clearCoat = material.clearcoat;
  // Clamp the clear coat roughness to avoid divisions by 0
  float clearCoatRoughness = material.clearcoat_roughness;
  clearCoatRoughness = MIN_ROUGHNESS * (1.0f - clearCoatRoughness) +
                       MAX_CLEAR_COAT_ROUGHNESS * clearCoatRoughness;

  // Remap the roughness to perceptually linear roughness
  float clearCoatLinearRoughness = clearCoatRoughness * clearCoatRoughness;
  // The base layer's f0 is computed assuming an interface from air to an IOR
  // of 1.5, but the clear coat layer forms an interface from IOR 1.5 to IOR
  // 1.5. We recompute f0 by first computing its IOR, then reconverting to f0
  // by using the correct interface
  f0 = f0 * (1.0f - clearCoat) + f0ClearCoatToSurface(f0) * clearCoat;
  // TODO: the base layer should be rougher

  // ----------------------------------------------------

  // Pre-filtered DFG term used for image-based lighting
  float NoV = std::fabs(vdot(N, V));

  // LUT texture version
  float3 dfg = prefilteredDFG(asset.dfg_lut, NoV, roughness);
  //float dfg_uv[2];
  //prefilteredDFG(NoV, roughness, dfg_uv);
  //float3 dfg = float3(dfg_uv[0], dfg_uv[1], 0.0f);

  // Evaluate Light
  // TODO(LTE): light loop
  float3 color(0.0f);

  float3 R = reflect(V, N);

  float ao = 1.0f;  // TODO(LTE): material.ao
  float specularAO = computeSpecularAO(NoV, ao, roughness);

  float3 diffuseBRDF = ao;
  float3 diffuseIrradiance = irradianceSH(asset.sh, N);
  float3 Fd = diffuseColor * diffuseIrradiance * diffuseBRDF;

  // specular indirect
  float3 Fr_DFG = specularDFG(dfg, f0);
  float3 Fr = Fr_DFG * specularIrradiance(asset, R, roughness) * specularAO;

  float energyCompensation = 1.0f;  // FIXME(LTE)
  Fr = Fr * energyCompensation;

  evaluateClearCoatIBL(asset, NoV, R, clearCoat, clearCoatRoughness, specularAO,
                       &Fd, &Fr);

  // TODO
  // evaluateSubsurfaceIBL

  return (Fd + Fr);  // TODO(LTE): multiply iblLuminance here;
}

// -------------------------------------------------------------------------

bool Renderer::Render(RenderLayer* layer, float quat[4],
                      const nanosg::Scene<float, example::Mesh<float> >& scene,
                      const example::Asset& asset, const RenderConfig& config,
                      std::atomic<bool>& cancelFlag, int& _showBufferMode) {
  int width = config.width;
  int height = config.height;

  // camera
  float eye[3] = {config.eye[0], config.eye[1], config.eye[2]};
  float look_at[3] = {config.look_at[0], config.look_at[1], config.look_at[2]};
  float up[3] = {config.up[0], config.up[1], config.up[2]};
  float fov = config.fov;
  float3 origin, corner, u, v;
  BuildCameraFrame(&origin, &corner, &u, &v, quat, eye, look_at, up, fov, width,
                   height);

  auto kCancelFlagCheckMilliSeconds = 300;

  std::vector<std::thread> workers;
  std::atomic<int> i(0);

  uint32_t num_threads = std::max(1U, std::thread::hardware_concurrency());

  auto startT = std::chrono::system_clock::now();

  // TODO(LTE): Path tracing.

  for (auto t = 0; t < num_threads; t++) {
    workers.emplace_back(std::thread([&, t]() {
      pcg32_state_t rng;
      pcg32_srandom(&rng, config.pass,
                    t);  // seed = combination of render pass + thread no.

      int y = 0;
      while ((y = i++) < config.height) {
        auto currT = std::chrono::system_clock::now();

        std::chrono::duration<double, std::milli> ms = currT - startT;
        // Check cancel flag
        if (ms.count() > kCancelFlagCheckMilliSeconds) {
          if (cancelFlag) {
            break;
          }
        }

        // draw dash line to aux buffer for progress.
        // for (int x = 0; x < config.width; x++) {
        //  float c = (x / 8) % 2;
        //  aux_rgba[4*(y*config.width+x)+0] = c;
        //  aux_rgba[4*(y*config.width+x)+1] = c;
        //  aux_rgba[4*(y*config.width+x)+2] = c;
        //  aux_rgba[4*(y*config.width+x)+3] = 0.0f;
        //}

        for (int x = 0; x < config.width; x++) {
          nanort::Ray<float> ray;
          ray.org[0] = origin[0];
          ray.org[1] = origin[1];
          ray.org[2] = origin[2];

          float u0 = pcg32_random(&rng);
          float u1 = pcg32_random(&rng);

          float3 dir;

          // for modes not a "color"
          if (_showBufferMode != SHOW_BUFFER_COLOR) {
            // only one pass
            if (config.pass > 0) continue;

            // to the center of pixel
            u0 = 0.5f;
            u1 = 0.5f;
          }

          dir = corner + (float(x) + u0) * u +
                (float(config.height - y - 1) + u1) * v;
          dir = vnormalize(dir);
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          nanosg::Intersection<float> isect;
          bool hit = scene.Traverse(ray, &isect, /* cull_back_face */ false);

          if (hit) {
            const std::vector<Material>& materials = asset.materials;
            const std::vector<Texture>& textures = asset.textures;
            const Mesh<float>& mesh = asset.meshes[isect.node_id];

            // tigra: add default material
            const Material& default_material = asset.default_material;

            float3 p;
            p[0] = ray.org[0] + isect.t * ray.dir[0];
            p[1] = ray.org[1] + isect.t * ray.dir[1];
            p[2] = ray.org[2] + isect.t * ray.dir[2];

            layer->positionRGBA[4 * (y * config.width + x) + 0] = p.x();
            layer->positionRGBA[4 * (y * config.width + x) + 1] = p.y();
            layer->positionRGBA[4 * (y * config.width + x) + 2] = p.z();
            layer->positionRGBA[4 * (y * config.width + x) + 3] = 1.0f;

            layer->varyCoordRGBA[4 * (y * config.width + x) + 0] = isect.u;
            layer->varyCoordRGBA[4 * (y * config.width + x) + 1] = isect.v;
            layer->varyCoordRGBA[4 * (y * config.width + x) + 2] = 0.0f;
            layer->varyCoordRGBA[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = isect.prim_id;

            float3 N;
            if (mesh.facevarying_normals.size() > 0) {
              float3 n0, n1, n2;
              n0[0] = mesh.facevarying_normals[9 * prim_id + 0];
              n0[1] = mesh.facevarying_normals[9 * prim_id + 1];
              n0[2] = mesh.facevarying_normals[9 * prim_id + 2];
              n1[0] = mesh.facevarying_normals[9 * prim_id + 3];
              n1[1] = mesh.facevarying_normals[9 * prim_id + 4];
              n1[2] = mesh.facevarying_normals[9 * prim_id + 5];
              n2[0] = mesh.facevarying_normals[9 * prim_id + 6];
              n2[1] = mesh.facevarying_normals[9 * prim_id + 7];
              n2[2] = mesh.facevarying_normals[9 * prim_id + 8];
              N = Lerp3(n0, n1, n2, isect.u, isect.v);
              N = vnormalize(N);
            } else {
              unsigned int f0, f1, f2;
              f0 = mesh.faces[3 * prim_id + 0];
              f1 = mesh.faces[3 * prim_id + 1];
              f2 = mesh.faces[3 * prim_id + 2];

              float3 v0, v1, v2;
              v0[0] = mesh.vertices[3 * f0 + 0];
              v0[1] = mesh.vertices[3 * f0 + 1];
              v0[2] = mesh.vertices[3 * f0 + 2];
              v1[0] = mesh.vertices[3 * f1 + 0];
              v1[1] = mesh.vertices[3 * f1 + 1];
              v1[2] = mesh.vertices[3 * f1 + 2];
              v2[0] = mesh.vertices[3 * f2 + 0];
              v2[1] = mesh.vertices[3 * f2 + 1];
              v2[2] = mesh.vertices[3 * f2 + 2];
              CalcNormal(N, v0, v1, v2);
            }

            layer->normalRGBA[4 * (y * config.width + x) + 0] =
                0.5f * N[0] + 0.5f;
            layer->normalRGBA[4 * (y * config.width + x) + 1] =
                0.5f * N[1] + 0.5f;
            layer->normalRGBA[4 * (y * config.width + x) + 2] =
                0.5f * N[2] + 0.5f;
            layer->normalRGBA[4 * (y * config.width + x) + 3] = 1.0f;

            layer->depthRGBA[4 * (y * config.width + x) + 0] = isect.t;
            layer->depthRGBA[4 * (y * config.width + x) + 1] = isect.t;
            layer->depthRGBA[4 * (y * config.width + x) + 2] = isect.t;
            layer->depthRGBA[4 * (y * config.width + x) + 3] = 1.0f;

            float3 UV;
            if (mesh.facevarying_uvs.size() > 0) {
              float3 uv0, uv1, uv2;
              uv0[0] = mesh.facevarying_uvs[6 * prim_id + 0];
              uv0[1] = mesh.facevarying_uvs[6 * prim_id + 1];
              uv1[0] = mesh.facevarying_uvs[6 * prim_id + 2];
              uv1[1] = mesh.facevarying_uvs[6 * prim_id + 3];
              uv2[0] = mesh.facevarying_uvs[6 * prim_id + 4];
              uv2[1] = mesh.facevarying_uvs[6 * prim_id + 5];

              UV = Lerp3(uv0, uv1, uv2, isect.u, isect.v);

              layer->texCoordRGBA[4 * (y * config.width + x) + 0] = UV[0];
              layer->texCoordRGBA[4 * (y * config.width + x) + 1] = UV[1];
            }

            // Fetch texture
            unsigned int material_id = mesh.material_ids[isect.prim_id];
            const Material& material = (static_cast<int>(material_id) >= 0 &&
                                        material_id < materials.size())
                                           ? materials[material_id]
                                           : default_material;

            float base_col[3];

            int basecol_texid = material.diffuse_texid;
            if (basecol_texid >= 0) {
              FetchTexture(textures[basecol_texid], UV[0], UV[1], base_col);
            } else {
              base_col[0] = material.baseColor[0];
              base_col[1] = material.baseColor[1];
              base_col[2] = material.baseColor[2];
            }

            float3 Nf = faceforward(N, dir);

            float3 col = Shade(asset, material, float3(base_col), Nf, dir);

#if 0
            float lod = roughness_to_lod(material.roughness); 
            if (lod >= 7) {
              // Use SH
              float irrad[3];
              irradianceSH(asset.sh, Nf, irrad);

              diffuse_rgb[0] = irrad[0] * config.intensity * diffuseColor[0];
              diffuse_rgb[1] = irrad[1] * config.intensity * diffuseColor[1];
              diffuse_rgb[2] = irrad[2] * config.intensity * diffuseColor[2];
            } else {
              float3 R = reflect(dir, Nf);

              float ibl[3];
              SampleEnvmap(asset, R, lod, ibl);
              diffuse_rgb[0] = ibl[0] * config.intensity * diffuseColor[0];
              diffuse_rgb[1] = ibl[1] * config.intensity * diffuseColor[1];
              diffuse_rgb[2] = ibl[2] * config.intensity * diffuseColor[2];
            }
#endif

            if (config.pass == 0) {
              layer->rgba[4 * (y * config.width + x) + 0] =
                  config.intensity * col[0];
              layer->rgba[4 * (y * config.width + x) + 1] =
                  config.intensity * col[1];
              layer->rgba[4 * (y * config.width + x) + 2] =
                  config.intensity * col[2];
              layer->rgba[4 * (y * config.width + x) + 3] = 1.0f;
              layer->sampleCounts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              layer->rgba[4 * (y * config.width + x) + 0] +=
                  config.intensity * col[0];
              layer->rgba[4 * (y * config.width + x) + 1] +=
                  config.intensity * col[1];
              layer->rgba[4 * (y * config.width + x) + 2] +=
                  config.intensity * col[2];
              layer->rgba[4 * (y * config.width + x) + 3] += 1.0f;
              layer->sampleCounts[y * config.width + x]++;
            }

          } else {
            {
              if (config.pass == 0) {
                // clear pixel
                layer->rgba[4 * (y * config.width + x) + 0] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 1] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 2] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 3] = 0.0f;
                layer->auxRGBA[4 * (y * config.width + x) + 0] = 0.0f;
                layer->auxRGBA[4 * (y * config.width + x) + 1] = 0.0f;
                layer->auxRGBA[4 * (y * config.width + x) + 2] = 0.0f;
                layer->auxRGBA[4 * (y * config.width + x) + 3] = 0.0f;
                layer->sampleCounts[y * config.width + x] =
                    1;  // Set 1 for the first pass
              } else {
                layer->sampleCounts[y * config.width + x]++;
              }

              // See background(IBL)
              float bg_col[3];
              SampleEnvmap(asset, dir, 0.0f, bg_col);
              layer->rgba[4 * (y * config.width + x) + 0] +=
                  asset.bg_intensity * bg_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] +=
                  asset.bg_intensity * bg_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] +=
                  asset.bg_intensity * bg_col[2];
              layer->rgba[4 * (y * config.width + x) + 3] += 1.0f;

              // No super sampling
              layer->normalRGBA[4 * (y * config.width + x) + 0] = 0.0f;
              layer->normalRGBA[4 * (y * config.width + x) + 1] = 0.0f;
              layer->normalRGBA[4 * (y * config.width + x) + 2] = 0.0f;
              layer->normalRGBA[4 * (y * config.width + x) + 3] = 0.0f;
              layer->positionRGBA[4 * (y * config.width + x) + 0] = 0.0f;
              layer->positionRGBA[4 * (y * config.width + x) + 1] = 0.0f;
              layer->positionRGBA[4 * (y * config.width + x) + 2] = 0.0f;
              layer->positionRGBA[4 * (y * config.width + x) + 3] = 0.0f;
              layer->depthRGBA[4 * (y * config.width + x) + 0] = 0.0f;
              layer->depthRGBA[4 * (y * config.width + x) + 1] = 0.0f;
              layer->depthRGBA[4 * (y * config.width + x) + 2] = 0.0f;
              layer->depthRGBA[4 * (y * config.width + x) + 3] = 0.0f;
              layer->texCoordRGBA[4 * (y * config.width + x) + 0] = 0.0f;
              layer->texCoordRGBA[4 * (y * config.width + x) + 1] = 0.0f;
              layer->texCoordRGBA[4 * (y * config.width + x) + 2] = 0.0f;
              layer->texCoordRGBA[4 * (y * config.width + x) + 3] = 0.0f;
              layer->varyCoordRGBA[4 * (y * config.width + x) + 0] = 0.0f;
              layer->varyCoordRGBA[4 * (y * config.width + x) + 1] = 0.0f;
              layer->varyCoordRGBA[4 * (y * config.width + x) + 2] = 0.0f;
              layer->varyCoordRGBA[4 * (y * config.width + x) + 3] = 0.0f;
            }
          }
        }

        for (int x = 0; x < config.width; x++) {
          layer->auxRGBA[4 * (y * config.width + x) + 0] = 0.0f;
          layer->auxRGBA[4 * (y * config.width + x) + 1] = 0.0f;
          layer->auxRGBA[4 * (y * config.width + x) + 2] = 0.0f;
          layer->auxRGBA[4 * (y * config.width + x) + 3] = 0.0f;
        }
      }
    }));
  }

  for (auto& t : workers) {
    t.join();
  }

  return (!cancelFlag);
};

}  // namespace example
