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

#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>
#include <array>

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

void irradianceSH(const float sh[9][3], const float3 n, float rgb[3]) {

  for (int i = 0; i < 3; i++) {
    rgb[i] = 
          sh[0][i]
        + sh[1][i] * (n[1])
        + sh[2][i] * (n[2])
        + sh[3][i] * (n[0])
        + sh[4][i] * (n[1] * n[0])
        + sh[5][i] * (n[1] * n[2])
        + sh[6][i] * (3.0f  * n[2] * n[2] - 1.0f)
        + sh[7][i] * (n[2] * n[0])
        + sh[8][i] * (n[0] * n[0] - n[1] * n[1]);
  }
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

void FetchTexture(const Texture& texture, float u, float v, float* col) {
  SampleTexture(col, u, v, texture.width, texture.height, texture.components,
                texture.image);
}

void SampleEnvmap(const example::Asset& asset, const float n[3], int lod, float *col)
{
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

  lod = std::min(int(asset.cubemap_ibl.size()-1), std::max(0, lod));

  const std::array<example::Image, 6> &cubemap = asset.cubemap_ibl[lod];

  const example::Image &face = cubemap[face_idx];

  float rgba[4];
  SampleTexture(rgba, u, v, face.width, face.height, face.channels, face.pixels.data());

  col[0] = rgba[0];
  col[1] = rgba[1];
  col[2] = rgba[2];
}

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

            // printf("material_id=%d materials=%lld\n", material_id,
            // materials.size());

            float diffuse_col[3];

            float specular_col[3];

            if (static_cast<int>(material_id) >= 0 &&
                material_id < materials.size()) {
              // printf("ok mat\n");

              int diffuse_texid = materials[material_id].diffuse_texid;
              if (diffuse_texid >= 0) {
                FetchTexture(textures[diffuse_texid], UV[0], UV[1],
                             diffuse_col);
              } else {
                diffuse_col[0] = materials[material_id].diffuse[0];
                diffuse_col[1] = materials[material_id].diffuse[1];
                diffuse_col[2] = materials[material_id].diffuse[2];
              }

              int specular_texid = materials[material_id].specular_texid;
              if (specular_texid >= 0) {
                FetchTexture(textures[specular_texid], UV[0], UV[1],
                             specular_col);
              } else {
                specular_col[0] = materials[material_id].specular[0];
                specular_col[1] = materials[material_id].specular[1];
                specular_col[2] = materials[material_id].specular[2];
              }
            } else
            // tigra: wrong material_id, use default_material
            {
              // printf("default_material\n");

              diffuse_col[0] = default_material.diffuse[0];
              diffuse_col[1] = default_material.diffuse[1];
              diffuse_col[2] = default_material.diffuse[2];
              specular_col[0] = default_material.specular[0];
              specular_col[1] = default_material.specular[1];
              specular_col[2] = default_material.specular[2];
            }

            // SH shading
            
            //float NdotV = fabsf(vdot(N, dir));
            float irrad[3];
            irradianceSH(asset.sh, N, irrad);

            irrad[0] *= config.intensity;
            irrad[1] *= config.intensity;
            irrad[2] *= config.intensity;

            if (config.pass == 0) {
              layer->rgba[4 * (y * config.width + x) + 0] =
                  irrad[0] * diffuse_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] =
                  irrad[1] * diffuse_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] =
                  irrad[2] * diffuse_col[2];
              layer->rgba[4 * (y * config.width + x) + 3] = 1.0f;
              layer->sampleCounts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              layer->rgba[4 * (y * config.width + x) + 0] +=
                  irrad[0] * diffuse_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] +=
                  irrad[1] * diffuse_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] +=
                  irrad[2] * diffuse_col[2];
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
              int lod = 0; // TODO(LTE)
              float bg_col[3];
              SampleEnvmap(asset, ray.dir, lod, bg_col);
              layer->rgba[4 * (y * config.width + x) + 0] += asset.bg_intensity * bg_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] += asset.bg_intensity * bg_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] += asset.bg_intensity * bg_col[2];
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
