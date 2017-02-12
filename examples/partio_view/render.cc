/*
The MIT License (MIT)

Copyright (c) 2015 - 2016 Light Transport Entertainment, Inc.

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

#include "Partio.h"

#include "render.h"

#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#include "../../nanort.h"
#include "matrix.h"

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

// --------------------------------------------------------------

// Predefined SAH predicator for sphere.
class SpherePred {
 public:
  SpherePred(const float* vertices)
      : axis_(0), pos_(0.0f), vertices_(vertices) {}

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

class SphereGeometry {
 public:
  SphereGeometry(const float* vertices, const float* radiuss)
      : vertices_(vertices), radiuss_(radiuss) {}

  /// Compute bounding box for `prim_index`th sphere.
  /// This function is called for each primitive in BVH build.
  void BoundingBox(float3* bmin, float3* bmax, unsigned int prim_index) const {
    (*bmin)[0] = vertices_[3 * prim_index + 0] - radiuss_[prim_index];
    (*bmin)[1] = vertices_[3 * prim_index + 1] - radiuss_[prim_index];
    (*bmin)[2] = vertices_[3 * prim_index + 2] - radiuss_[prim_index];
    (*bmax)[0] = vertices_[3 * prim_index + 0] + radiuss_[prim_index];
    (*bmax)[1] = vertices_[3 * prim_index + 1] + radiuss_[prim_index];
    (*bmax)[2] = vertices_[3 * prim_index + 2] + radiuss_[prim_index];
  }

  const float* vertices_;
  const float* radiuss_;
  mutable float3 ray_org_;
  mutable float3 ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;
};

class SphereIntersection {
 public:
  SphereIntersection() {}

  float u;
  float v;

  float3 normal;

  // Required member variables.
  float t;
  unsigned int prim_id;
};

template <class I>
class SphereIntersector {
 public:
  SphereIntersector(const float* vertices, const float* radiuss)
      : vertices_(vertices), radiuss_(radiuss) {}

  /// Do ray interesection stuff for `prim_index` th primitive and return hit
  /// distance `t`,
  /// varycentric coordinate `u` and `v`.
  /// Returns true if there's intersection.
  bool Intersect(float* t_inout, unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    // http://wiki.cgsociety.org/index.php/Ray_Sphere_Intersection

    const float3 center(&vertices_[3 * prim_index]);
    const float radius = radiuss_[prim_index];

    float3 oc = ray_org_ - center;

    float a = vdot(ray_dir_, ray_dir_);
    float b = 2.0 * vdot(ray_dir_, oc);
    float c = vdot(oc, oc) - radius * radius;

    float disc = b * b - 4.0 * a * c;

    float t0, t1;

    if (disc < 0.0) {  // no roots
      return false;
    } else if (disc == 0.0) {
      t0 = t1 = -0.5 * (b / a);
    } else {
      // compute q as described above
      float distSqrt = sqrt(disc);
      float q;
      if (b < 0)
        q = (-b - distSqrt) / 2.0;
      else
        q = (-b + distSqrt) / 2.0;

      // compute t0 and t1
      t0 = q / a;
      t1 = c / q;
    }

    // make sure t0 is smaller than t1
    if (t0 > t1) {
      // if t0 is bigger than t1 swap them around
      float temp = t0;
      t0 = t1;
      t1 = temp;
    }

    // if t1 is less than zero, the object is in the ray's negative direction
    // and consequently the ray misses the sphere
    if (t1 < 0) {
      return false;
    }

    float t;
    if (t0 < 0) {
      t = t1;
    } else {
      t = t0;
    }

    if (t > (*t_inout)) {
      return false;
    }

    (*t_inout) = t;

    return true;
  }

  /// Returns the nearest hit distance.
  float GetT() const { return intersection.t; }

  /// Update is called when a nearest hit is found.
  void Update(float t, unsigned int prim_idx) const {
    intersection.t = t;
    intersection.prim_id = prim_idx;
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

    trace_options_ = trace_options;
  }

  /// Post BVH traversal stuff(e.g. compute intersection point information)
  /// This function is called only once in BVH traversal.
  /// `hit` = true if there is something hit.
  void PostTraversal(const nanort::Ray<float>& ray, bool hit) const {
    if (hit) {
      float3 hitP = ray_org_ + intersection.t * ray_dir_;
      float3 center = float3(&vertices_[3 * intersection.prim_id]);
      float3 n = vnormalize(hitP - center);
      intersection.normal = n;
      intersection.u = (atan2(n[0], n[2]) + M_PI) * 0.5 * (1.0 / M_PI);
      intersection.v = acos(n[1]) / M_PI;
    }
  }

  const float* vertices_;
  const float* radiuss_;
  mutable float3 ray_org_;
  mutable float3 ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;

  mutable I intersection;
};

// -----------------------------------------------------

nanort::BVHAccel<float, SphereGeometry, SpherePred,
                 SphereIntersector<SphereIntersection> >
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

static std::string GetBaseDir(const std::string& filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

bool Renderer::LoadPartio(const char* filename, const float scene_scale,
                          const float constant_radius) {
  Partio::ParticlesDataMutable* p = Partio::read(filename);
  if (!p) {
    std::cerr << "Failed to read particle data : " << filename << std::endl;
    return false;
  }

  Partio::ParticleAttribute posAttr;
  if (!p->attributeInfo("position", posAttr)) {
    std::cerr << "\"position\" attribute does not exist in the particle data."
              << std::endl;
    return false;
  }

  bool has_radius = false;
  Partio::ParticleAttribute radiusAttr;
  if (!p->attributeInfo("radius", radiusAttr)) {
    std::cout << "\"radius\" attribute not found. Use constant radius."
              << std::endl;
  } else {
    has_radius = true;
  }

  int num_particles = p->numParticles();
  // TODO(LTE): Ensure position is VECTOR type.
  const float* pos = p->data<float>(posAttr, 0);
  vertices_.clear();
  for (size_t i = 0; i < num_particles * 3; i++) {
    vertices_.push_back(pos[i]);
  }

  // TODO(LTE): Support loading custom particle attributes.
  if (has_radius) {
  } else {
    radiuss_.clear();

    for (size_t i = 0; i < vertices_.size() / 3; i++) {
      radiuss_.push_back(constant_radius);
    }
  }

  return true;
}

bool Renderer::BuildBVH() {
  std::cout << "[Build BVH] " << std::endl;

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  auto t_start = std::chrono::system_clock::now();

  SphereGeometry sphere_geom(&vertices_.at(0), &radiuss_.at(0));
  SpherePred sphere_pred(&vertices_.at(0));

  bool ret =
      gAccel.Build(radiuss_.size(), build_options, sphere_geom, sphere_pred);
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

bool Renderer::Render(float* rgba, float* aux_rgba, int* sample_counts,
                      float quat[4], const RenderConfig& config,
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

  auto startT = std::chrono::system_clock::now();

  // Initialize RNG.

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
          dir = corner + (float(x) + u0) * u +
                (float(config.height - y - 1) + u1) * v;
          dir = vnormalize(dir);
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          nanort::BVHTraceOptions trace_options;
          SphereIntersector<SphereIntersection> isector(&vertices_.at(0),
                                                        &radiuss_.at(0));
          bool hit = gAccel.Traverse(ray, trace_options, isector);
          if (hit) {
            float3 p;
            p[0] = ray.org[0] + isector.intersection.t * ray.dir[0];
            p[1] = ray.org[1] + isector.intersection.t * ray.dir[1];
            p[2] = ray.org[2] + isector.intersection.t * ray.dir[2];

            config.positionImage[4 * (y * config.width + x) + 0] = p.x();
            config.positionImage[4 * (y * config.width + x) + 1] = p.y();
            config.positionImage[4 * (y * config.width + x) + 2] = p.z();
            config.positionImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.varycoordImage[4 * (y * config.width + x) + 0] =
                isector.intersection.u;
            config.varycoordImage[4 * (y * config.width + x) + 1] =
                isector.intersection.v;
            config.varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
            config.varycoordImage[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = isector.intersection.prim_id;

            float3 N = isector.intersection.normal;

            config.normalImage[4 * (y * config.width + x) + 0] =
                0.5f * N[0] + 0.5f;
            config.normalImage[4 * (y * config.width + x) + 1] =
                0.5f * N[1] + 0.5f;
            config.normalImage[4 * (y * config.width + x) + 2] =
                0.5f * N[2] + 0.5f;
            config.normalImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.depthImage[4 * (y * config.width + x) + 0] =
                isector.intersection.t;
            config.depthImage[4 * (y * config.width + x) + 1] =
                isector.intersection.t;
            config.depthImage[4 * (y * config.width + x) + 2] =
                isector.intersection.t;
            config.depthImage[4 * (y * config.width + x) + 3] = 1.0f;

            // Simple shading
            float NdotV = fabsf(vdot(N, dir));

            if (config.pass == 0) {
              rgba[4 * (y * config.width + x) + 0] = NdotV;
              rgba[4 * (y * config.width + x) + 1] = NdotV;
              rgba[4 * (y * config.width + x) + 2] = NdotV;
              rgba[4 * (y * config.width + x) + 3] = 1.0f;
              sample_counts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              rgba[4 * (y * config.width + x) + 0] += NdotV;
              rgba[4 * (y * config.width + x) + 1] += NdotV;
              rgba[4 * (y * config.width + x) + 2] += NdotV;
              rgba[4 * (y * config.width + x) + 3] += 1.0f;
              sample_counts[y * config.width + x]++;
            }

          } else {
            {
              if (config.pass == 0) {
                // clear pixel
                rgba[4 * (y * config.width + x) + 0] = 0.0f;
                rgba[4 * (y * config.width + x) + 1] = 0.0f;
                rgba[4 * (y * config.width + x) + 2] = 0.0f;
                rgba[4 * (y * config.width + x) + 3] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 0] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 1] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 2] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 3] = 0.0f;
                sample_counts[y * config.width + x] =
                    1;  // Set 1 for the first pass
              } else {
                sample_counts[y * config.width + x]++;
              }

              // No super sampling
              config.normalImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.normalImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.normalImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.normalImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 3] = 0.0f;
            }
          }
        }

        for (int x = 0; x < config.width; x++) {
          aux_rgba[4 * (y * config.width + x) + 0] = 0.0f;
          aux_rgba[4 * (y * config.width + x) + 1] = 0.0f;
          aux_rgba[4 * (y * config.width + x) + 2] = 0.0f;
          aux_rgba[4 * (y * config.width + x) + 3] = 0.0f;
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
