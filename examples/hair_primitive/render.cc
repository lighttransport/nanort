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

#include "render.h"

#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#include "../../nanort.h"
#include "matrix.h"
#include "trackball.h"

#include "cyhair_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

float pcg32_random(pcg32_state_t *rng) {
  unsigned long long oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  unsigned int rot = oldstate >> 59u;
  unsigned int ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

  return (float)((double)ret / (double)4294967296.0);
}

void pcg32_srandom(pcg32_state_t *rng, uint64_t initstate, uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1U) | 1U;
  pcg32_random(rng);
  rng->state += initstate;
  pcg32_random(rng);
}

const float kPI = 3.141592f;

typedef struct {
  std::vector<float> vertices;  /// [xyz] * 4(cubic) * num_curves
  std::vector<float> radiuss;   /// 4(cucic) * num_curves
} CubicCurves;

struct Material {
  // float ambient[3];
  float diffuse[3];
  float specular[3];
  // float reflection[3];
  // float refraction[3];
  int id;
  int diffuse_texid;
  int specular_texid;
  // int reflection_texid;
  // int transparency_texid;
  // int bump_texid;
  // int normal_texid;  // normal map
  // int alpha_texid;  // alpha map

  Material() {
    // ambient[0] = 0.0;
    // ambient[1] = 0.0;
    // ambient[2] = 0.0;
    diffuse[0] = 0.5;
    diffuse[1] = 0.5;
    diffuse[2] = 0.5;
    specular[0] = 0.5;
    specular[1] = 0.5;
    specular[2] = 0.5;
    // reflection[0] = 0.0;
    // reflection[1] = 0.0;
    // reflection[2] = 0.0;
    // refraction[0] = 0.0;
    // refraction[1] = 0.0;
    // refraction[2] = 0.0;
    id = -1;
    diffuse_texid = -1;
    specular_texid = -1;
    // reflection_texid = -1;
    // transparency_texid = -1;
    // bump_texid = -1;
    // normal_texid = -1;
    // alpha_texid = -1;
  }
};

struct Texture {
  int width;
  int height;
  int components;
  unsigned char *image;

  Texture() {
    width = -1;
    height = -1;
    components = -1;
    image = NULL;
  }
};

CubicCurves gCurves;
std::vector<Material> gMaterials;
std::vector<Texture> gTextures;

typedef nanort::real3<float> float3;

inline float3 Lerp3(float3 v0, float3 v1, float3 v2, float u, float v) {
  return (1.0f - u - v) * v0 + u * v1 + v * v2;
}

inline void CalcNormal(float3 &N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}

// -------------------------------------------------------------------------------
// Curve intersector functions

void GetZAlign(const float3 &o, const float3 &l, float matrix[3][3],
               float translate[3]) {
  float dxz, lxdxz, lydxz, lzdxz;

  dxz = sqrtf(l.x() * l.x() + l.z() * l.z());
  if (dxz > 0) {
    lxdxz = l.x() / dxz;
    lydxz = l.y() / dxz;
    lzdxz = l.z() / dxz;
    matrix[0][0] = lzdxz;
    matrix[0][1] = -lxdxz * l.y();
    matrix[0][2] = l.x();
    matrix[1][0] = 0;
    matrix[1][1] = dxz;
    matrix[1][2] = l.y();
    matrix[2][0] = -lxdxz;
    matrix[2][1] = -lydxz * l.z();
    matrix[2][2] = l.z();
  } else {
    matrix[0][0] = 1;
    matrix[0][1] = 0;
    matrix[0][2] = 0;
    matrix[1][0] = 0;
    matrix[1][1] = 0;
    matrix[1][2] = (l.y() > 0) ? -1 : 1;
    matrix[2][0] = 0;
    matrix[2][1] = (l.y() > 0) ? 1 : -1;
    matrix[2][2] = 0;
  }
  translate[0] =
      -(o.x() * matrix[0][0] + o.y() * matrix[1][0] + o.z() * matrix[2][0]);
  translate[1] =
      -(o.x() * matrix[0][1] + o.y() * matrix[1][1] + o.z() * matrix[2][1]);
  translate[2] =
      -(o.x() * matrix[0][2] + o.y() * matrix[1][2] + o.z() * matrix[2][2]);
}

inline float3 Xform(const float3 &p0, float matrix[3][3], float translate[3]) {
  float3 p;

  p[0] = p0.x() * matrix[0][0] + p0.y() * matrix[1][0] + p0.z() * matrix[2][0] +
         translate[0];
  p[1] = p0.x() * matrix[0][1] + p0.y() * matrix[1][1] + p0.z() * matrix[2][1] +
         translate[1];
  p[2] = p0.x() * matrix[0][2] + p0.y() * matrix[1][2] + p0.z() * matrix[2][2] +
         translate[2];

  return p;
}

void EvaluateBezier(const float3 *v, float t, float3 *p) {
  float3 v1[3], v2[2], v3[1];
  float u;

  u = 1 - t;

  v1[0] = float3(v[0].x() * u + v[1].x() * t, v[0].y() * u + v[1].y() * t,
                 v[0].z() * u + v[1].z() * t);
  v1[1] = float3(v[1].x() * u + v[2].x() * t, v[1].y() * u + v[2].y() * t,
                 v[1].z() * u + v[2].z() * t);
  v1[2] = float3(v[2].x() * u + v[3].x() * t, v[2].y() * u + v[3].y() * t,
                 v[2].z() * u + v[3].z() * t);

  v2[0] = float3(v1[0].x() * u + v1[1].x() * t, v1[0].y() * u + v1[1].y() * t,
                 v1[0].z() * u + v1[1].z() * t);
  v2[1] = float3(v1[1].x() * u + v1[2].x() * t, v1[1].y() * u + v1[2].y() * t,
                 v1[1].z() * u + v1[2].z() * t);

  v3[0] = float3(v2[0].x() * u + v2[1].x() * t, v2[0].y() * u + v2[1].y() * t,
                 v2[0].z() * u + v2[1].z() * t);

  (*p) = float3(v3[0].x(), v3[0].y(), v3[0].z());
}

void EvaluateBezierTangent(const float3 *v, float t, float3 *dv) {
  float3 C1 = v[3] - (3.0f * v[2]) + (3.0f * v[1]) - v[0];
  float3 C2 = (3.0f * v[2]) - (6.0f * v[1]) + (3.0f * v[0]);
  float3 C3 = (3.0f * v[1]) - (3.0f * v[0]);

  (*dv) = (3.0f * C1 * t * t) + (2.0f * C2 * t) + C3;
}

class CurvePred {
 public:
  CurvePred(const float *vertices)
      : axis_(0), pos_(0.0f), vertices_(vertices) {}

  void Set(int axis, float pos) const {
    axis_ = axis;
    pos_ = pos;
  }

  bool operator()(unsigned int i) const {
    int axis = axis_;
    float pos = pos_;

    float3 p0(&vertices_[3 * (4 * i + 0)]);
    float3 p1(&vertices_[3 * (4 * i + 1)]);
    float3 p2(&vertices_[3 * (4 * i + 2)]);
    float3 p3(&vertices_[3 * (4 * i + 3)]);

    float center = (p0[axis] + p1[axis] + p2[axis] + p3[axis]) / 4.0f;

    return (center < pos);
  }

 private:
  mutable int axis_;
  mutable float pos_;
  const float *vertices_;
};

class CurveGeometry {
 public:
  CurveGeometry(const float *vertices, const float *radiuss)
      : vertices_(vertices), radiuss_(radiuss) {}

  /// Compute bounding box for `prim_index`th bezier curve.
  /// Since Bezier curve has convex property, we can simply compute bounding box
  /// from control points
  /// This function is called for each primitive in BVH build.
  void BoundingBox(float3 *bmin, float3 *bmax, unsigned int prim_index) const {
    (*bmin)[0] =
        vertices_[3 * (4 * prim_index + 0) + 0] - radiuss_[4 * prim_index];
    (*bmin)[1] =
        vertices_[3 * (4 * prim_index + 0) + 1] - radiuss_[4 * prim_index];
    (*bmin)[2] =
        vertices_[3 * (4 * prim_index + 0) + 2] - radiuss_[4 * prim_index];
    (*bmax)[0] =
        vertices_[3 * (4 * prim_index + 0) + 0] + radiuss_[4 * prim_index];
    (*bmax)[1] =
        vertices_[3 * (4 * prim_index + 0) + 1] + radiuss_[4 * prim_index];
    (*bmax)[2] =
        vertices_[3 * (4 * prim_index + 0) + 2] + radiuss_[4 * prim_index];
    for (int i = 1; i < 4; i++) {
      (*bmin)[0] = std::min(vertices_[3 * (4 * prim_index + i) + 0] -
                                radiuss_[4 * prim_index + i],
                            (*bmin)[0]);
      (*bmin)[1] = std::min(vertices_[3 * (4 * prim_index + i) + 1] -
                                radiuss_[4 * prim_index + i],
                            (*bmin)[1]);
      (*bmin)[2] = std::min(vertices_[3 * (4 * prim_index + i) + 2] -
                                radiuss_[4 * prim_index + i],
                            (*bmin)[2]);
      (*bmax)[0] = std::max(vertices_[3 * (4 * prim_index + i) + 0] +
                                radiuss_[4 * prim_index + i],
                            (*bmax)[0]);
      (*bmax)[1] = std::max(vertices_[3 * (4 * prim_index + i) + 1] +
                                radiuss_[4 * prim_index + i],
                            (*bmax)[1]);
      (*bmax)[2] = std::max(vertices_[3 * (4 * prim_index + i) + 2] +
                                radiuss_[4 * prim_index + i],
                            (*bmax)[2]);
    }
  }

  const float *vertices_;
  const float *radiuss_;
  mutable float3 ray_org_;
  mutable float3 ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;
};

class CurveIntersection {
 public:
  CurveIntersection() {}

  // Required member variables.
  float t;
  unsigned int prim_id;

  // Additional custom intersection properties
  float u;
  float v;
  float3 tangent;  // curve direction
  float3 normal;   // perpendicular to the curve
};

template <class I>
class CurveIntersector {
  // Evaluate bezier curve as N segmnet lines.
  // See "Exploiting Local Orientation Similarity for Efficient Ray Traversal of
  // Hair and Fur" http://www.sven-woop.de/
  // And Embree implementation for more details.

 public:
  CurveIntersector(const float *vertices, const float *radiuss,
                   const int num_subdivisions = 4)
      : vertices_(vertices),
        radiuss_(radiuss),
        num_subdivisions_(num_subdivisions) {}

  /// Do ray interesection stuff for `prim_index` th primitive and return hit
  /// distance `t`,
  bool Intersect(float *t_inout, unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    float R[3][3];  // rot
    float T[3];     // trans
    GetZAlign(ray_org_, ray_dir_, R, T);

    float3 cps[4];   // projected control points.
    float3 ocps[4];  // original control points.

    float radius[2];  // Radius at begin point and end point. Do not consider
                      // intermediate radius currently.
    radius[0] = radiuss_[4 * prim_index + 0];
    radius[1] = radiuss_[4 * prim_index + 3];

    ocps[0][0] = vertices_[12 * prim_index + 0];
    ocps[0][1] = vertices_[12 * prim_index + 1];
    ocps[0][2] = vertices_[12 * prim_index + 2];

    ocps[1][0] = vertices_[12 * prim_index + 3];
    ocps[1][1] = vertices_[12 * prim_index + 4];
    ocps[1][2] = vertices_[12 * prim_index + 5];

    ocps[2][0] = vertices_[12 * prim_index + 6];
    ocps[2][1] = vertices_[12 * prim_index + 7];
    ocps[2][2] = vertices_[12 * prim_index + 8];

    ocps[3][0] = vertices_[12 * prim_index + 9];
    ocps[3][1] = vertices_[12 * prim_index + 10];
    ocps[3][2] = vertices_[12 * prim_index + 11];

    float t_z = 0.0f;
    for (int i = 0; i < 4; i++) {
      cps[i] = Xform(ocps[i], R, T);
      if (t_z < cps[i].z()) {
        t_z = cps[i].z();
      }
    }

    float uw = std::max(radius[0], radius[1]) / 2.0f;
    if (t_z < 4.0f * uw) {
      return false;
    }

    bool has_hit = false;

    int n = num_subdivisions_;
    const float inv_n = 1.0f / static_cast<float>(n);
    for (int s = 0; s < n; s++) {
      float3 p[2];

      float t0 = s / static_cast<float>(n);
      float t1 = (s + 1) / static_cast<float>(n);

      // Evaluate bezier in projected space.
      EvaluateBezier(cps, t0, &p[0]);
      EvaluateBezier(cps, t1, &p[1]);

      float P0x, P0y, P0z, P0w;  // w = radius
      float P1x, P1y, P1z, P1w;

      P0x = p[0].x();
      P0y = p[0].y();
      P0z = p[0].z();
      P0w = 0.5 * radius[0];

      P1x = p[1].x();
      P1y = p[1].y();
      P1z = p[1].z();
      P1w = 0.5 * radius[1];

      // Project ray origin onto the line segment;
      float Ax, Ay, Az, Bx, By, Bz, Bw;
      Ax = 0.0f - P0x;
      Ay = 0.0f - P0y;
      Az = 0.0f - P0z;

      Bx = P1x - P0x;
      By = P1y - P0y;
      Bz = P1z - P0z;
      Bw = P1w - P0w;

      float d0 = (Ax * Bx) + (Ay * By);
      float d1 = (Bx * Bx) + (By * By);

      // Caculaute closest points P on line segments
      float u = d0 / d1;
      // clamp(x, 0.0, 1.0);
      u = std::max(0.0f, std::min(1.0f, u));

      float Px, Py, Pz, Pw;
      Px = P0x + (u * Bx);
      Py = P0y + (u * By);
      Pz = P0z + (u * Bz);
      Pw = P0w + (u * Bw);

      // The Z component holds hit distance
      float t = Pz;

      // The w-component interpolates the curve radius
      float r = Pw;

      // if distance to nearest point P <= curve radius ...
      float r2 = r * r;
      float d2 = (Px * Px) + (Py * Py);
      // d2 <= r2 & ray.tnear < t & t < ray.tfar;
      if ((d2 <= r2) && (t < (*t_inout))) {
        // Store u and v parameters which is used in `Update' function.
        u_param = (u + static_cast<float>(s)) * inv_n;  // Adjust `u' so that it
                                                        // spans [0, 1] for the
                                                        // original cubic bezier
                                                        // curve.
        v_param = sqrtf(d2);

        (*t_inout) = t;
        has_hit = true;
      }
    }
    return has_hit;
  }

  /// Returns the nearest hit distance.
  float GetT() const { return intersection.t; }

  /// Update is called when a nearest hit is found.
  void Update(float t, unsigned int prim_idx) const {
    intersection.t = t;
    intersection.prim_id = prim_idx;

    intersection.u = u_param;
    intersection.v = v_param;
  }

  /// Prepare BVH traversal(e.g. compute inverse ray direction)
  /// This function is called only once in BVH traversal.
  void PrepareTraversal(const nanort::Ray<float> &ray,
                        const nanort::BVHTraceOptions &trace_options) const {
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
  void PostTraversal(const nanort::Ray<float> &ray, bool hit) const {
    if (hit) {
      float3 cps[4];

      unsigned int prim_index = intersection.prim_id;

      cps[0][0] = vertices_[12 * prim_index + 0];
      cps[0][1] = vertices_[12 * prim_index + 1];
      cps[0][2] = vertices_[12 * prim_index + 2];

      cps[1][0] = vertices_[12 * prim_index + 3];
      cps[1][1] = vertices_[12 * prim_index + 4];
      cps[1][2] = vertices_[12 * prim_index + 5];

      cps[2][0] = vertices_[12 * prim_index + 6];
      cps[2][1] = vertices_[12 * prim_index + 7];
      cps[2][2] = vertices_[12 * prim_index + 8];

      cps[3][0] = vertices_[12 * prim_index + 9];
      cps[3][1] = vertices_[12 * prim_index + 10];
      cps[3][2] = vertices_[12 * prim_index + 11];

      // Compute tangent
      float3 Dv;
      EvaluateBezierTangent(cps, intersection.u, &Dv);
      intersection.tangent = vnormalize(Dv);
      // printf("Dv = %f, %f, %f\n", intersection.tangent[0],
      // intersection.tangent[1], intersection.tangent[2]);

      intersection.normal = vnormalize(
          vcross(vcross(ray_dir_, intersection.tangent), intersection.tangent));
    }
  }

  const float *vertices_;
  const float *radiuss_;
  const int num_subdivisions_;
  mutable float3 ray_org_;
  mutable float3 ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;

  mutable I intersection;
  mutable float u_param;
  mutable float v_param;
};

// -----------------------------------------------------

nanort::BVHAccel<float, CurveGeometry, CurvePred,
                 CurveIntersector<CurveIntersection> >
    gAccel;

void BuildCameraFrame(float3 *origin, float3 *corner, float3 *u, float3 *v,
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

nanort::Ray<float> GenerateRay(const float3 &origin, const float3 &corner,
                               const float3 &du, const float3 &dv, float u,
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

void FetchTexture(int tex_idx, float u, float v, float *col) {
  assert(tex_idx >= 0);
  Texture &texture = gTextures[tex_idx];
  int tx = u * texture.width;
  int ty = (1.0f - v) * texture.height;
  int idx_offset = (ty * texture.width + tx) * texture.components;
  col[0] = texture.image[idx_offset + 0] / 255.f;
  col[1] = texture.image[idx_offset + 1] / 255.f;
  col[2] = texture.image[idx_offset + 2] / 255.f;
}

static std::string GetBaseDir(const std::string &filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

int LoadTexture(const std::string &filename) {
  if (filename.empty()) return -1;

  printf("  Loading texture : %s\n", filename.c_str());
  Texture texture;

  int w, h, n;
  unsigned char *data = stbi_load(filename.c_str(), &w, &h, &n, 0);
  if (data) {
    texture.width = w;
    texture.height = h;
    texture.components = n;

    size_t n_elem = w * h * n;
    texture.image = new unsigned char[n_elem];
    for (int i = 0; i < n_elem; i++) {
      texture.image[i] = data[i];
    }

    gTextures.push_back(texture);
    return gTextures.size() - 1;
  }

  printf("  Failed to load : %s\n", filename.c_str());
  return -1;
}

bool Renderer::LoadCyHair(const char *cyhair_filename,
                          const float scene_scale[3],
                          const float scene_translate[3],
                          const int max_strands) {
  CyHair cyhair;
  bool ret = cyhair.Load(cyhair_filename);
  if (!ret) {
    return false;
  }

  std::cout << "[Hair] Loaded CyHair data " << std::endl;
  std::cout << "  # of strands : " << cyhair.num_strands_ << std::endl;
  std::cout << "  # of points  : " << cyhair.total_points_ << std::endl;

  ret = cyhair.ToCubicBezierCurves(&gCurves.vertices, &gCurves.radiuss,
                                   scene_scale, scene_translate, max_strands);

  return ret;
}

bool Renderer::BuildBVH() {
  if (gCurves.vertices.empty()) {
    return false;
  }

  std::cout << "[Build BVH] " << std::endl;

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  auto t_start = std::chrono::system_clock::now();

  CurveGeometry curves_geom(&gCurves.vertices.at(0), &gCurves.radiuss.at(0));
  CurvePred curves_pred(&gCurves.vertices.at(0));

  unsigned int num_curves = gCurves.radiuss.size() / 4;

  bool ret = gAccel.Build(num_curves, build_options, curves_geom, curves_pred);
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

bool Renderer::Render(float *rgba, float *aux_rgba, int *sample_counts,
                      float quat[4], const RenderConfig &config,
                      std::atomic<bool> &cancelFlag) {
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
          CurveIntersector<CurveIntersection> isector(&gCurves.vertices.at(0),
                                                      &gCurves.radiuss.at(0));
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

            config.uparamImage[4 * (y * config.width + x) + 0] =
                isector.intersection.u;
            config.uparamImage[4 * (y * config.width + x) + 1] =
                isector.intersection.u;
            config.uparamImage[4 * (y * config.width + x) + 2] =
                isector.intersection.u;
            config.uparamImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.vparamImage[4 * (y * config.width + x) + 0] =
                isector.intersection.v;
            config.vparamImage[4 * (y * config.width + x) + 1] =
                isector.intersection.v;
            config.vparamImage[4 * (y * config.width + x) + 2] =
                isector.intersection.v;
            config.vparamImage[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = isector.intersection.prim_id;

            float3 N;
            N[0] = isector.intersection.normal[0];
            N[1] = isector.intersection.normal[1];
            N[2] = isector.intersection.normal[2];

            config.normalImage[4 * (y * config.width + x) + 0] =
                0.5 * N[0] + 0.5;
            config.normalImage[4 * (y * config.width + x) + 1] =
                0.5 * N[1] + 0.5;
            config.normalImage[4 * (y * config.width + x) + 2] =
                0.5 * N[2] + 0.5;
            config.normalImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.tangentImage[4 * (y * config.width + x) + 0] =
                0.5 * isector.intersection.tangent[0] + 0.5;
            config.tangentImage[4 * (y * config.width + x) + 1] =
                0.5 * isector.intersection.tangent[1] + 0.5;
            config.tangentImage[4 * (y * config.width + x) + 2] =
                0.5 * isector.intersection.tangent[2] + 0.5;
            config.tangentImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.depthImage[4 * (y * config.width + x) + 0] =
                isector.intersection.t;
            config.depthImage[4 * (y * config.width + x) + 1] =
                isector.intersection.t;
            config.depthImage[4 * (y * config.width + x) + 2] =
                isector.intersection.t;
            config.depthImage[4 * (y * config.width + x) + 3] = 1.0f;

            // Simple shading
            float NdotV = fabsf(vdot(N, dir));

            float diffuse_col[3] = {0.5, 0.5, 0.5};

            if (config.pass == 0) {
              rgba[4 * (y * config.width + x) + 0] = NdotV * diffuse_col[0];
              rgba[4 * (y * config.width + x) + 1] = NdotV * diffuse_col[1];
              rgba[4 * (y * config.width + x) + 2] = NdotV * diffuse_col[2];
              rgba[4 * (y * config.width + x) + 3] = 1.0f;
              sample_counts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              rgba[4 * (y * config.width + x) + 0] += NdotV * diffuse_col[0];
              rgba[4 * (y * config.width + x) + 1] += NdotV * diffuse_col[1];
              rgba[4 * (y * config.width + x) + 2] += NdotV * diffuse_col[2];
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
              config.tangentImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.tangentImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.tangentImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.tangentImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.uparamImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.uparamImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.uparamImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.uparamImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.vparamImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.vparamImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.vparamImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.vparamImage[4 * (y * config.width + x) + 3] = 0.0f;
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

  for (auto &t : workers) {
    t.join();
  }

  return (!cancelFlag);
};

}  // namespace example
