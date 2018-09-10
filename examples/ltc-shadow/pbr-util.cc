#include "pbr-util.h"

#include <cstdint>
#include <cmath>

namespace example {

constexpr float kPI = 3.141592f;

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
void DFG(const bool multiscatter, const size_t width, Image *output) {
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

// ----------------------------------------

} // namespace example
