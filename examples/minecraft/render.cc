/*
The MIT License (MIT)

Copyright (c) 2015 - 2017 Light Transport Entertainment, Inc.

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

#include "render.h"

#include <algorithm>
#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#include "../../nanort.h"
#include "matrix.h"

#include "trackball.h"

#include "enkimi.h"

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
  unsigned int ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

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

typedef struct {
  std::vector<float> vertices;
  std::vector<uint8_t> color_ids;  // color index
  std::vector<float> widths;
} Cubes;

typedef nanort::real3<float> float3;

// Predefined SAH predicator for cube.
class CubePred {
 public:
  CubePred(const float* vertices) : axis_(0), pos_(0.0f), vertices_(vertices) {}

  void Set(int axis, float pos) const {
    axis_ = axis;
    pos_ = pos;
  }

  bool operator()(unsigned int i) const {
    int axis = axis_;
    float pos = pos_;

    float3 p0(&vertices_[3 * i]);

    float center = p0[axis];

    return (center < pos);
  }

 private:
  mutable int axis_;
  mutable float pos_;
  const float* vertices_;
};

class CubeGeometry {
 public:
  CubeGeometry(const float* vertices, const float* widths)
      : vertices_(vertices), widths_(widths) {}

  /// Compute bounding box for `prim_index`th cube.
  /// This function is called for each primitive in BVH build.
  void BoundingBox(float3* bmin, float3* bmax, unsigned int prim_index) const {
    (*bmin)[0] = vertices_[3 * prim_index + 0] - widths_[prim_index];
    (*bmin)[1] = vertices_[3 * prim_index + 1] - widths_[prim_index];
    (*bmin)[2] = vertices_[3 * prim_index + 2] - widths_[prim_index];
    (*bmax)[0] = vertices_[3 * prim_index + 0] + widths_[prim_index];
    (*bmax)[1] = vertices_[3 * prim_index + 1] + widths_[prim_index];
    (*bmax)[2] = vertices_[3 * prim_index + 2] + widths_[prim_index];
  }

  const float* vertices_;
  const float* widths_;
  mutable float3 ray_org_;
  mutable float3 ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;
};

class CubeIntersection {
 public:
  CubeIntersection() {}

  float normal[3];

  // Required member variables.
  float t;
  unsigned int prim_id;
};

template <class I>
class CubeIntersector {
 public:
  CubeIntersector(const float* vertices, const float* widths)
      : vertices_(vertices), widths_(widths) {}

  /// Do ray interesection stuff for `prim_index` th primitive and return hit
  /// distance `t`,
  /// Returns true if there's intersection.
  bool Intersect(float* t_inout, unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    const float3 center(&vertices_[3 * prim_index]);
    const float width = widths_[prim_index];

    const float3 bmin = center - float3(width);
    const float3 bmax = center + float3(width);

    float tmin, tmax;

    const float min_x = ray_dir_sign_[0] ? bmax[0] : bmin[0];
    const float min_y = ray_dir_sign_[1] ? bmax[1] : bmin[1];
    const float min_z = ray_dir_sign_[2] ? bmax[2] : bmin[2];
    const float max_x = ray_dir_sign_[0] ? bmin[0] : bmax[0];
    const float max_y = ray_dir_sign_[1] ? bmin[1] : bmax[1];
    const float max_z = ray_dir_sign_[2] ? bmin[2] : bmax[2];

    // X
    const float tmin_x = (min_x - ray_org_[0]) * ray_inv_dir_[0];
    const float tmax_x = (max_x - ray_org_[0]) * ray_inv_dir_[0];

    // Y
    const float tmin_y = (min_y - ray_org_[1]) * ray_inv_dir_[1];
    const float tmax_y = (max_y - ray_org_[1]) * ray_inv_dir_[1];

    // Z
    const float tmin_z = (min_z - ray_org_[2]) * ray_inv_dir_[2];
    const float tmax_z = (max_z - ray_org_[2]) * ray_inv_dir_[2];

    tmin = nanort::safemax(tmin_z, nanort::safemax(tmin_y, tmin_x));
    tmax = nanort::safemin(tmax_z, nanort::safemin(tmax_y, tmax_x));

    if (tmin > tmax) {
      return false;
    }

    const float t = tmin;

    if (t > (*t_inout)) {
      return false;
    }

    (*t_inout) = t;

    return true;
  }

  /// Returns the nearest hit distance.
  float GetT() const { return t_; }

  /// Update is called when a nearest hit is found.
  void Update(float t, unsigned int prim_idx) const {
    t_ = t;
    prim_id_ = prim_idx;
  }

  /// Prepare BVH traversal(e.g. compute inverse ray direction)
  /// This function is called only once in BVH traversal.
  void PrepareTraversal(const nanort::Ray<float>& ray,
                        const nanort::BVHTraceOptions& trace_options) const {
    ray_org_[0] = ray.org[0];
    ray_org_[1] = ray.org[1];
    ray_org_[2] = ray.org[2];

    ray_dir_[0] = ray.dir[0];
    ray_dir_[1] = ray.dir[1];
    ray_dir_[2] = ray.dir[2];

    // FIXME(syoyo): Consider zero div case.
    ray_inv_dir_[0] = 1.0f / ray.dir[0];
    ray_inv_dir_[1] = 1.0f / ray.dir[1];
    ray_inv_dir_[2] = 1.0f / ray.dir[2];

    ray_dir_sign_[0] = ray.dir[0] < 0.0f ? 1 : 0;
    ray_dir_sign_[1] = ray.dir[1] < 0.0f ? 1 : 0;
    ray_dir_sign_[2] = ray.dir[2] < 0.0f ? 1 : 0;

    trace_options_ = trace_options;
  }

  /// Post BVH traversal stuff(e.g. compute intersection point information)
  /// This function is called only once in BVH traversal.
  /// `hit` = true if there is something hit.
  void PostTraversal(const nanort::Ray<float>& ray, bool hit, CubeIntersection *isect) const {
    if (hit) {
      // compute normal. there should be valid intersection point.
      const float3 center(&vertices_[3 * prim_id_]);
      const float width = widths_[prim_id_];

      const float3 bmin = center - float3(width);
      const float3 bmax = center + float3(width);

      float tmin, tmax;

      const float min_x = ray_dir_sign_[0] ? bmax[0] : bmin[0];
      const float min_y = ray_dir_sign_[1] ? bmax[1] : bmin[1];
      const float min_z = ray_dir_sign_[2] ? bmax[2] : bmin[2];
      const float max_x = ray_dir_sign_[0] ? bmin[0] : bmax[0];
      const float max_y = ray_dir_sign_[1] ? bmin[1] : bmax[1];
      const float max_z = ray_dir_sign_[2] ? bmin[2] : bmax[2];

      // X
      const float tmin_x = (min_x - ray_org_[0]) * ray_inv_dir_[0];

      // Y
      const float tmin_y = (min_y - ray_org_[1]) * ray_inv_dir_[1];

      // Z
      const float tmin_z = (min_z - ray_org_[2]) * ray_inv_dir_[2];

      int axis = 0;
      tmin = tmin_x;
      if (tmin < tmin_y) {
        axis = 1;
        tmin = tmin_y;
      }
      if (tmin < tmin_z) {
        axis = 2;
        tmin = tmin_z;
      }

      isect->t = t_;
      isect->prim_id = prim_id_;

      isect->normal[0] = 0.0f;
      isect->normal[1] = 0.0f;
      isect->normal[2] = 0.0f;

      isect->normal[axis] = ray_dir_sign_[axis] ? 1.0f : -1.0f;
    }
  }

  const float* vertices_;
  const float* widths_;
  mutable float3 ray_org_;
  mutable float3 ray_dir_;
  mutable float3 ray_inv_dir_;
  mutable int ray_dir_sign_[3];
  mutable nanort::BVHTraceOptions trace_options_;

  mutable float t_;
  mutable unsigned int prim_id_;
};

// @fixme { Do not defined as global variable }
Cubes gCubes;
nanort::BVHAccel<float>
    gAccel;

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

  return ray;
}

static std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}

bool LoadMIData(Cubes* cubes, const char* filename, float scale) {
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    printf("Failed to open file: %s\n", filename);
    return false;
  }

  double bmin[3], bmax[3];
  bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<double>::max();
  bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<double>::max();

  cubes->vertices.clear();
  cubes->color_ids.clear();
  cubes->widths.clear();

  enkiRegionFile regionFile = enkiRegionFileLoad(fp);

  for (int i = 0; i < ENKI_MI_REGION_CHUNKS_NUMBER; i++) {
    enkiNBTDataStream stream;
    enkiInitNBTDataStreamForChunk(regionFile, i, &stream);
    if (stream.dataLength) {
      enkiChunkBlockData aChunk = enkiNBTReadChunk(&stream);
      enkiMICoordinate chunkOriginPos =
          enkiGetChunkOrigin(&aChunk);  // y always 0
      printf("Chunk at xyz{ %d, %d, %d }  Number of sections: %d \n",
             chunkOriginPos.x, chunkOriginPos.y, chunkOriginPos.z,
             aChunk.countOfSections);

      // iterate through chunk and count non 0 voxels as a demo
      int64_t numVoxels = 0;
      for (int section = 0; section < ENKI_MI_NUM_SECTIONS_PER_CHUNK;
           ++section) {
        if (aChunk.sections[section]) {
          enkiMICoordinate sectionOrigin =
              enkiGetChunkSectionOrigin(&aChunk, section);
          printf("    Non empty section at xyz{ %d, %d, %d } \n",
                 sectionOrigin.x, sectionOrigin.y, sectionOrigin.z);
          enkiMICoordinate sPos;
          // note order x then z then y iteration for cache efficiency
          for (sPos.y = 0; sPos.y < ENKI_MI_NUM_SECTIONS_PER_CHUNK; ++sPos.y) {
            for (sPos.z = 0; sPos.z < ENKI_MI_NUM_SECTIONS_PER_CHUNK;
                 ++sPos.z) {
              for (sPos.x = 0; sPos.x < ENKI_MI_NUM_SECTIONS_PER_CHUNK;
                   ++sPos.x) {
                uint8_t voxel =
                    enkiGetChunkSectionVoxel(&aChunk, section, sPos);
                if (voxel) {
                  ++numVoxels;

                  cubes->vertices.push_back(sectionOrigin.x + sPos.x);
                  cubes->vertices.push_back(sectionOrigin.y + sPos.y);
                  cubes->vertices.push_back(sectionOrigin.z + sPos.z);
                  cubes->color_ids.push_back(
                      voxel);  // voxel value = color index.

                  bmin[0] = std::min(bmin[0], double(sectionOrigin.x + sPos.x));
                  bmin[1] = std::min(bmin[1], double(sectionOrigin.y + sPos.y));
                  bmin[2] = std::min(bmin[2], double(sectionOrigin.z + sPos.z));

                  bmax[0] = std::max(bmax[0], double(sectionOrigin.x + sPos.x));
                  bmax[1] = std::max(bmax[1], double(sectionOrigin.x + sPos.x));
                  bmax[2] = std::max(bmax[2], double(sectionOrigin.x + sPos.x));
                }
              }
            }
          }
        }
      }
      printf("   Chunk has %g non zero voxels\n", (float)numVoxels);

      enkiNBTRewind(&stream);
    }
    enkiNBTFreeAllocations(&stream);
  }

  enkiRegionFileFreeAllocations(&regionFile);

  fclose(fp);

  printf("bmin = %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("bmax = %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  float bsize[3];
  bsize[0] = bmax[0] - bmin[0];
  bsize[1] = bmax[1] - bmin[1];
  bsize[2] = bmax[2] - bmin[2];

  float bcenter[3];
  bcenter[0] = bmin[0] + bsize[0] * 0.5f;
  bcenter[1] = bmin[1] + bsize[1] * 0.5f;
  bcenter[2] = bmin[2] + bsize[2] * 0.5f;

  float invsize = bsize[0];
  if (bsize[1] > invsize) {
    invsize = bsize[1];
  }
  if (bsize[2] > invsize) {
    invsize = bsize[2];
  }

  invsize = 16.0f / invsize;  // FIXME(syoyo): Choose better invscale.
  printf("invsize = %f\n", invsize);

  // Centerize & scaling
  for (size_t i = 0; i < cubes->vertices.size() / 3; i++) {
    cubes->vertices[3 * i + 0] =
        (cubes->vertices[3 * i + 0] - bcenter[0]) * invsize;
    cubes->vertices[3 * i + 1] =
        (cubes->vertices[3 * i + 1] - bcenter[1]) * invsize;
    cubes->vertices[3 * i + 2] =
        (cubes->vertices[3 * i + 2] - bcenter[2]) * invsize;

    // Set approximate cube width.
    cubes->widths.push_back(0.5f * invsize);
  }

  return true;
}

bool Renderer::LoadMI(const char* filename, float scene_scale) {
  return LoadMIData(&gCubes, filename, scene_scale);
}

bool Renderer::BuildBVH() {
  if (gCubes.widths.size() < 1) {
    std::cout << "num_points == 0" << std::endl;
    return false;
  }

  std::cout << "[Build BVH] " << std::endl;

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  auto t_start = std::chrono::system_clock::now();

  CubeGeometry cube_geom(&gCubes.vertices.at(0), &gCubes.widths.at(0));
  CubePred cube_pred(&gCubes.vertices.at(0));
  bool ret =
      gAccel.Build(gCubes.widths.size(), cube_geom, cube_pred, build_options);
  assert(ret);

  auto t_end = std::chrono::system_clock::now();

  std::chrono::duration<double, std::milli> ms = t_end - t_start;
  std::cout << "BVH build time: " << ms.count() << " [ms]\n";

  nanort::BVHBuildStatistics stats = gAccel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  gAccel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  return true;
}

bool Renderer::Render(RenderLayer* layer, float quat[4],
                      const RenderConfig& config,
                      std::atomic<bool>& cancelFlag) {
  if (!gAccel.IsValid()) {
    return false;
  }

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

  uint32_t* color_palette = enkiGetMineCraftPalette();  // returns a 256 array
                                                        // of uint32_t's in
                                                        // uint8_t rgba order.

  auto startT = std::chrono::system_clock::now();

  // Multi-threaded rendering using C++11 thread.
  for (auto t = 0; t < num_threads; t++) {
    workers.emplace_back(std::thread([&, t]() {
      // Initialize RNG.
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

        for (int x = 0; x < config.width; x++) {
          nanort::Ray<float> ray;
          ray.org[0] = origin[0];
          ray.org[1] = origin[1];
          ray.org[2] = origin[2];

          float u0 = pcg32_random(&rng);
          float u1 = pcg32_random(&rng);

          float3 dir;
          dir = corner + (float(x) + u0) * u +
                (float(config.height - y - 1) + u1) * v;
          dir = vnormalize(dir);
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          CubeIntersector<CubeIntersection> cube_intersector(
              reinterpret_cast<const float*>(gCubes.vertices.data()),
              gCubes.widths.data());
          CubeIntersection isect;
          bool hit = gAccel.Traverse(ray, cube_intersector, &isect);
          if (hit) {
            float3 p;
            p[0] = ray.org[0] + isect.t * ray.dir[0];
            p[1] = ray.org[1] + isect.t * ray.dir[1];
            p[2] = ray.org[2] + isect.t * ray.dir[2];

            layer->position[4 * (y * config.width + x) + 0] = p.x();
            layer->position[4 * (y * config.width + x) + 1] = p.y();
            layer->position[4 * (y * config.width + x) + 2] = p.z();
            layer->position[4 * (y * config.width + x) + 3] = 1.0f;

            layer->varycoord[4 * (y * config.width + x) + 0] = 0.0f;
            layer->varycoord[4 * (y * config.width + x) + 1] = 0.0f;
            layer->varycoord[4 * (y * config.width + x) + 2] = 0.0f;
            layer->varycoord[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = isect.prim_id;

            float3 N;
            N[0] = isect.normal[0];
            N[1] = isect.normal[1];
            N[2] = isect.normal[2];

            layer->normal[4 * (y * config.width + x) + 0] = 0.5 * N[0] + 0.5;
            layer->normal[4 * (y * config.width + x) + 1] = 0.5 * N[1] + 0.5;
            layer->normal[4 * (y * config.width + x) + 2] = 0.5 * N[2] + 0.5;
            layer->normal[4 * (y * config.width + x) + 3] = 1.0f;

            layer->depth[4 * (y * config.width + x) + 0] =
                isect.t;
            layer->depth[4 * (y * config.width + x) + 1] =
                isect.t;
            layer->depth[4 * (y * config.width + x) + 2] =
                isect.t;
            layer->depth[4 * (y * config.width + x) + 3] = 1.0f;

            float diffuse_col[3] = {0.0f, 0.0f, 0.0f};

            uint8_t color_id =
                gCubes.color_ids[isect.prim_id];
            uint32_t col = color_palette[color_id];

            {
              // FIXME(LTE): Support big endian
              diffuse_col[0] = float(((col)&0xFF)) / 255.0f;
              diffuse_col[1] = float(((col >> 8) & 0xFF)) / 255.0f;
              diffuse_col[2] = float(((col >> 16) & 0xFF)) / 255.0f;
              // TODO(LTE): Support alpha
            }

            // TODO(LTE): Support specular color
            float specular_col[3] = {0.0f, 0.0f, 0.0f};

            // Simple shading
            float NdotV = fabsf(vdot(N, dir));

            if (config.pass == 0) {
              layer->rgba[4 * (y * config.width + x) + 0] =
                  NdotV * diffuse_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] =
                  NdotV * diffuse_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] =
                  NdotV * diffuse_col[2];
              layer->rgba[4 * (y * config.width + x) + 3] = 1.0f;
              layer->sample_counts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              layer->rgba[4 * (y * config.width + x) + 0] +=
                  NdotV * diffuse_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] +=
                  NdotV * diffuse_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] +=
                  NdotV * diffuse_col[2];
              layer->rgba[4 * (y * config.width + x) + 3] += 1.0f;
              layer->sample_counts[y * config.width + x]++;
            }

          } else {
            {
              if (config.pass == 0) {
                // clear pixel
                layer->rgba[4 * (y * config.width + x) + 0] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 1] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 2] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 3] = 0.0f;
                layer->sample_counts[y * config.width + x] =
                    1;  // Set 1 for the first pass
              } else {
                layer->sample_counts[y * config.width + x]++;
              }

              // No super sampling
              layer->normal[4 * (y * config.width + x) + 0] = 0.0f;
              layer->normal[4 * (y * config.width + x) + 1] = 0.0f;
              layer->normal[4 * (y * config.width + x) + 2] = 0.0f;
              layer->normal[4 * (y * config.width + x) + 3] = 0.0f;
              layer->position[4 * (y * config.width + x) + 0] = 0.0f;
              layer->position[4 * (y * config.width + x) + 1] = 0.0f;
              layer->position[4 * (y * config.width + x) + 2] = 0.0f;
              layer->position[4 * (y * config.width + x) + 3] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 0] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 1] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 2] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 3] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 0] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 1] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 2] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 3] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 0] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 1] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 2] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 3] = 0.0f;
            }
          }
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
