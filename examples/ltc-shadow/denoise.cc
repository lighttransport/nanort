#include "denoise.h"

#include <cstdlib>
#include <cmath>
#include <cstdint>

struct Image
{
  std::vector<float> pixels;
  int width;
  int height;
  int channels;
};

const float kPI = 3.141592f;

// Bilateral filter parameters
#define DEPTH_WEIGHT (1.0f)
#define NORMAL_WEIGHT (1.0f)
#define PLANE_WEIGHT (1.0f)
#define ANALYTIC_WEIGHT (1.0f)

// "Max integer denoising radius"
#define DENOISE_RADIUS  (3) 

// Infrastructure parameters for using the same shader for H and V passes
//#define NUM_OUTPUTS "Number of attachments to the framebuffer"
//#define PASS_THROUGH "1 = no denoising, 0 = apply denoising"
//#define FINAL_PASS "If true, compute the quotient (source0/source1) into result3, which is mapped to RT0 for that pass"

inline float square(const float x) { return x * x; }

inline float saturate(float x) {
  return std::max(0.0f, std::min(1.0f, x));
}

inline float dot(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline float length(const float v[3]) {
  const float d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (d2 > 1.0e-6f) {
    return std::sqrt(d2);
  }
  return 0.0f;
}

inline float texelFetch(const Image &image, const int px,
                        const int py, const int c) {
  int x = std::max(0, std::min(image.width - 1, px));
  int y = std::max(0, std::min(image.height - 1, py));
  int z = std::max(0, std::min(image.channels - 1, c));

  size_t idx = size_t(image.channels * (y * image.width + x) + z);

  return image.pixels[idx];
}

// Just for reference.
// convert depth buffer value to a camera-space Z value.
// We don't use reconstructCSZ since depth buffer from ray casting exactly stores its depth from a camera.
//
// clipInfo = (z_f == -inf) ? (z_n, -1.0, 1.0) : (z_n * z_f, z_n - z_f, z_f)
float reconstructCSZ(float d, float clipInfo[3]) {
  return clipInfo[0] / (clipInfo[1] * d + clipInfo[2]);
}

struct TapKey {
    float csZ;  // camera-space Z.
    float position[3]; // world space position. TODO(LTE): use camera space?
    float normal[3];
    float analytic;
};


// Assume depth, position and normal has same image resolution.
TapKey getTapKey(
  const Image& depth,
  const Image& position,
  const Image& normal,
  const Image& analytic,
  int px, int py)
{
  TapKey key;

  key.csZ = texelFetch(depth, px, py, 0);
  //  if ((DEPTH_WEIGHT != 0.0) || (PLANE_WEIGHT != 0.0)) {
  //      key.csZ = depth;
  //      //key.csZ = reconstructCSZ(texelFetch(gbuffer_DEPTH_buffer, C, 0).r, gbuffer_camera_clipInfo);
  //  }

  key.position[0] = texelFetch(position, px, py, 0);
  key.position[1] = texelFetch(position, px, py, 1);
  key.position[2] = texelFetch(position, px, py, 2);
  //  if (PLANE_WEIGHT != 0.0) {
  //      key.position = position;
  //      //key.csPosition = reconstructCSPosition(fragCoord.xy, key.csZ, gbuffer_camera_projInfo);
  //  }

  // TODO(LTE): multiply and add offset.
  key.normal[0] = texelFetch(normal, px, py, 0);
  key.normal[1] = texelFetch(normal, px, py, 1);
  key.normal[2] = texelFetch(normal, px, py, 2);
  //  if ((NORMAL_WEIGHT != 0.0) || (PLANE_WEIGHT != 0.0)) {
  //      key.normal = texelFetch(gbuffer_CS_NORMAL_buffer, C, 0).xyz * gbuffer_CS_NORMAL_readMultiplyFirst.xyz + gbuffer_CS_NORMAL_readAddSecond.xyz;
  //  }

  float a[3];
  a[0] = texelFetch(analytic, px, py, 0);
  a[1] = texelFetch(analytic, px, py, 1);
  a[2] = texelFetch(analytic, px, py, 2);

  key.analytic = (a[0] + a[1] + a[2]) / 3.0f;
  //  if (ANALYTIC_WEIGHT != 0.0) {
  //      key.analytic = mean(texelFetch(analytic, C, 0).rgb);
  //  }

  return key;
}

// Sample the signal used for the noise estimation
// source0 : shadowed result
// source1 : unshadowed result
void tap(const Image& source0, const Image &source1, const int px, const int py, float ox, float oy, float result[3]) {
  float n[3];

  n[0] = texelFetch(source0, px + ox, py + oy,
                    0);
  n[1] = texelFetch(source0, px + ox, py + oy,
                    1);
  n[2] = texelFetch(source0, px + ox, py + oy,
                    2);

  float d[3];

  d[0] = texelFetch(source1, px + ox, py + oy, 0);
  d[1] = texelFetch(source1, px + ox, py + oy, 1);
  d[2] = texelFetch(source1, px + ox, py + oy, 2);

  for (int i = 0; i < 3; i++) {
    result[i] = (d[i] < 1.0e-6f) ? 1.0f : (n[i] / d[i]);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Estimate desired radius from the second derivative of the signal itself in source0 relative to baseline,
// which is the noisier image because it contains shadows
float estimateNoise(
  const Image &source0,
  const Image &source1,
  int loc[2],
  float axis[2]) {
    const int NOISE_ESTIMATION_RADIUS = std::min(10, DENOISE_RADIUS);
    float v2[3], v1[3];

    tap(source0, source1, loc[0], loc[1], -NOISE_ESTIMATION_RADIUS * axis[0], -NOISE_ESTIMATION_RADIUS * axis[1], v2);
    tap(source0, source1, loc[0], loc[1], (1 - NOISE_ESTIMATION_RADIUS) * axis[0], (1 - NOISE_ESTIMATION_RADIUS) * axis[1], v1);

    float d2mag = 0.0f;
    // The first two points are accounted for above
    for (int r = -NOISE_ESTIMATION_RADIUS + 2; r <= NOISE_ESTIMATION_RADIUS; ++r) {
        float v0[3];
        tap(source0, source1, loc[0], loc[1], axis[0] * r, axis[1] * r, v0);

        // Second derivative
        float d2[3];
        d2[0] = v2[0] - v1[0] * 2.0f + v0[0];
        d2[1] = v2[1] - v1[1] * 2.0f + v0[1];
        d2[2] = v2[2] - v1[2] * 2.0f + v0[2];

        d2mag += length(d2);

        // Shift weights in the window
        v2[0] = v1[0];
        v2[1] = v1[1];
        v2[2] = v1[2];

        v1[0] = v0[0];
        v1[1] = v0[1];
        v1[2] = v0[2];
    }

    // Scaled value by 1.5 *before* clamping to visualize going out of range
    // It is clamped again when applied.
    return std::max(0.0f, std::min(1.0f, std::sqrt(d2mag * (1.0f / float(DENOISE_RADIUS))) * (1.0f / 1.5f)));
}

inline void hammersley(uint32_t i, float iN, float result[2]) {
    constexpr float tof = 0.5f / 0x80000000U;
    uint32_t bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);

  result[0] = i * iN;
  result[1] = bits * tof;
}

static float hash(float n)
{
  float iptr; // not used
  
  return std::modf(std::sin(n) * 1000.0f, &iptr);
}

// source0 : shadowed
// source1 : unshadowed
static void estimateNoiseOfImage(
  const Image& source0,
  const Image& source1,
  Image* result)
{

  result->width = source0.width;
  result->height = source0.height;
  result->channels = 1;

  result->pixels.resize(result->width * result->height);

  for (size_t y = 0; y < source0.height; y++) {
    for (size_t x = 0; x < source0.width; x++) {

      float result = 0;
      const int N = 4;

      // hammersly
      float angle = hash(y * source0.width + x);

      for (float t = 0; t < N; ++t, angle += kPI/float(N)) {
          float c = cos(angle), s = sin(angle);
          float axis[2] = {c, s};
          int loc[2] = {int(x), int(y)};
          result = std::max(estimateNoise(source0, source1, loc, axis), result);
      }
    }
  }
}

//
// Simple 3x3 box filtering.
// TODO(LTE): Separable box filtering or use integral image for faster filtering
void DenoiseEstimate(const std::vector<float> &image, int width, int height,
                     int channels, std::vector<float> *output) {
  output->resize(size_t(width * height * channels));

  const int R = 1;

  const float weight = (1.0f / square(2.0f * float(R) + 1.0f));

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        float value = 0.0f;

        for (int dy = -R; dy <= R; dy++) {
          int py = std::max(0, std::min(height - 1, y + dy));
          for (int dx = -R; dx <= R; dx++) {
            int px = std::max(0, std::min(height - 1, x + dx));

            size_t idx = size_t(channels * (py * width + px) + c);

            value += image[idx];
          }
        }

        size_t idx = size_t(channels * (y * width + x) + c);

        (*output)[idx] = value * weight;
      }
    }
  }
}

float calculateBilateralWeight(TapKey center, TapKey tap) {

    float depthWeight   = 1.0;
    float normalWeight  = 1.0;
    float planeWeight   = 1.0;
    float analyticWeight  = 1.0;

    if (DEPTH_WEIGHT != 0.0) {
        depthWeight = std::max(0.0, 1.0 - abs(tap.csZ - center.csZ) * DEPTH_WEIGHT);
    }

    if (NORMAL_WEIGHT != 0.0) {
        float normalCloseness = dot(tap.normal, center.normal);
        normalCloseness = normalCloseness*normalCloseness;
        normalCloseness = normalCloseness*normalCloseness;

        float normalError = (1.0 - normalCloseness);
        normalWeight = std::max((1.0 - normalError * NORMAL_WEIGHT), 0.00);
    }


    if (PLANE_WEIGHT != 0.0) {
        float lowDistanceThreshold2 = 0.001;

        // Change in position in camera space
        float dq[3];
        dq[0] = center.position[0] - tap.position[0];
        dq[1] = center.position[1] - tap.position[1];
        dq[2] = center.position[2] - tap.position[2];

        // How far away is this point from the original sample
        // in camera space? (Max value is unbounded)
        float distance2 = dot(dq, dq);

        // How far off the expected plane (on the perpendicular) is this point?  Max value is unbounded.
        float planeError = std::max(std::fabs(dot(dq, tap.normal)), std::fabs(dot(dq, center.normal)));

        planeWeight = (distance2 < lowDistanceThreshold2) ? 1.0f :
            std::pow(std::max(0.0, 1.0 - 2.0 * PLANE_WEIGHT * planeError / std::sqrt(distance2)), 2.0);
    }

    if (ANALYTIC_WEIGHT != 0.0) {
        float aDiff = abs(tap.analytic - center.analytic) * 10.0;
        analyticWeight = std::max(0.0, 1.0 - (aDiff * ANALYTIC_WEIGHT));
    }

    return depthWeight * normalWeight * planeWeight * analyticWeight;


}


//
// Assume all image has same resolution.
//
void applyDenoiser(
  const Image& source0,
  const Image& source1,
  const Image& noiseEstimate,
  const Image& depth,
  const Image& position,
  const Image& normal,
  const Image& analytic,
  const float axis[2],
  const int radius) // denoise radius
{

  int width = noiseEstimate.width;
  int height = noiseEstimate.height;

  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {

      // 3* is because the estimator produces larger values than we want to use, for visualization purposes
      float gaussianRadius = saturate(texelFetch(noiseEstimate, x, y, 0) * 1.5f) * radius;

      float result0[3] = {0, 0, 0};
      float result1[3] = {0, 0, 0};

      float sum0[3] = {0, 0, 0};
      float sum1[3] = {0, 0, 0};

      float totalWeight = 0.0;

      TapKey key = getTapKey(depth, position, normal, analytic, x, y);

      // Detect sky and noiseless pixels and reject them
      // if ((PASS_THROUGH != 1) && (reconstructCSZ(texelFetch(gbuffer_DEPTH_buffer, ssC, 0).r, gbuffer_camera_clipInfo) >= -100) && (gaussianRadius > 0.5)) {
      {

        for (int r = -radius; r <= radius; ++r) {

            int tapOffset[2] = {int(axis[0] * r), int(axis[1] * r)};

            int tapLoc[2];
            tapLoc[0] = x + tapOffset[0];
            tapLoc[1] = y + tapOffset[1];

            float gaussian = std::exp(-square(float(r) / gaussianRadius));
            float weight = gaussian * ((r == 0) ? 1.0 : calculateBilateralWeight(key, getTapKey(depth, position, normal, analytic, tapLoc[0], tapLoc[1])));

            sum0[0] += texelFetch(source0, tapLoc[0], tapLoc[1], 0) * weight;
            sum0[1] += texelFetch(source0, tapLoc[0], tapLoc[1], 1) * weight;
            sum0[2] += texelFetch(source0, tapLoc[0], tapLoc[1], 2) * weight;

            sum1[0] += texelFetch(source1, tapLoc[0], tapLoc[1], 0) * weight;
            sum1[1] += texelFetch(source1, tapLoc[0], tapLoc[1], 1) * weight;
            sum1[2] += texelFetch(source1, tapLoc[0], tapLoc[1], 2) * weight;

            totalWeight += weight;
        }

        // totalWeight >= gaussian[0], so no division by zero here
        result0[0] = sum0[0] / totalWeight;
        result0[1] = sum0[1] / totalWeight;
        result0[2] = sum0[2] / totalWeight;

        result1[0] = sum1[0] / totalWeight;
        result1[1] = sum1[1] / totalWeight;
        result1[2] = sum1[2] / totalWeight;


      }

      

    }
  }

  // IF FINAL PASS
  {
    for (int i = 0; i < 3; ++i) {
      // Threshold degenerate values
      resultRatio[i] = (result1[i] < 0.0001f) ? 1.0f : (result0[i] / result1[i]);
    }
  }

}

