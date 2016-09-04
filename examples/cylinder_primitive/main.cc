#include "nanort.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdint.h>

namespace {

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

unsigned char fclamp(float x) {
  int i = (int)(powf(x, 1.0 / 2.2) * 256.0f);
  if (i > 255) i = 255;
  if (i < 0) i = 0;

  return (unsigned char)i;
}

void SaveImagePNG(const char *filename, const float *rgb, int width,
                  int height) {
  std::vector<unsigned char> ldr(width * height * 3);
  for (size_t i = 0; i < (size_t)(width * height * 3); i++) {
    ldr[i] = fclamp(rgb[i]);
  }

  int len = stbi_write_png(filename, width, height, 3, &ldr.at(0), width * 3);
  if (len < 1) {
    printf("Failed to save image\n");
    exit(-1);
  }
}

int solve2e(float root[], float A, float B, float C) {
  if (fabsf(A) <= 1.0e-6f) {
    float x = -C / B;
    root[0] = x;
    return 1;
  } else {
    float D = B * B - A * C;
    if (D < 0) {
      return 0;
    } else if (D == 0) {
      float x = -B / A;
      root[0] = x;
      return 1;
    } else {
      float x1 = (fabsf(B) + sqrtf(D)) / A;
      if (B >= 0.0) {
        x1 = -x1;
      }
      float x2 = C / (A * x1);
      if (x1 > x2) {
        float tmp = x1;
        x1 = x2;
        x2 = tmp;
      }

      root[0] = x1;
      root[1] = x2;
      return 2;
    }
  }
}

// Predefined SAH predicator for cylinder.
class CylinderPred {
 public:
  CylinderPred(const float *vertices)
      : axis_(0), pos_(0.0f), vertices_(vertices) {}

  void Set(int axis, float pos) const {
    axis_ = axis;
    pos_ = pos;
  }

  bool operator()(unsigned int i) const {
    int axis = axis_;
    float pos = pos_;

    nanort::real3<float> p0(&vertices_[3 * (2 * i + 0)]);
    nanort::real3<float> p1(&vertices_[3 * (2 * i + 1)]);

    float center = (p0[axis] + p1[axis]) / 2.0f;

    return (center < pos);
  }

 private:
  mutable int axis_;
  mutable float pos_;
  const float *vertices_;
};

// -----------------------------------------------------

class CylinderGeometry {
 public:
  CylinderGeometry(const float *vertices, const float *radiuss)
      : vertices_(vertices), radiuss_(radiuss) {}

  /// Compute bounding box for `prim_index`th cylinder.
  /// This function is called for each primitive in BVH build.
  void BoundingBox(nanort::real3<float> *bmin, nanort::real3<float> *bmax,
                   unsigned int prim_index) const {
    (*bmin)[0] =
        vertices_[3 * (2 * prim_index + 0) + 0] - radiuss_[2 * prim_index + 0];
    (*bmin)[1] =
        vertices_[3 * (2 * prim_index + 0) + 1] - radiuss_[2 * prim_index + 0];
    (*bmin)[2] =
        vertices_[3 * (2 * prim_index + 0) + 2] - radiuss_[2 * prim_index + 0];
    (*bmax)[0] =
        vertices_[3 * (2 * prim_index + 0) + 0] + radiuss_[2 * prim_index + 0];
    (*bmax)[1] =
        vertices_[3 * (2 * prim_index + 0) + 1] + radiuss_[2 * prim_index + 0];
    (*bmax)[2] =
        vertices_[3 * (2 * prim_index + 0) + 2] + radiuss_[2 * prim_index + 0];

    (*bmin)[0] = std::min(
        vertices_[3 * (2 * prim_index + 1) + 0] - radiuss_[2 * prim_index + 1],
        (*bmin)[0]);
    (*bmin)[1] = std::min(
        vertices_[3 * (2 * prim_index + 1) + 1] - radiuss_[2 * prim_index + 1],
        (*bmin)[1]);
    (*bmin)[2] = std::min(
        vertices_[3 * (2 * prim_index + 1) + 2] - radiuss_[2 * prim_index + 1],
        (*bmin)[2]);
    (*bmax)[0] = std::max(
        vertices_[3 * (2 * prim_index + 1) + 0] + radiuss_[2 * prim_index + 1],
        (*bmax)[0]);
    (*bmax)[1] = std::max(
        vertices_[3 * (2 * prim_index + 1) + 1] + radiuss_[2 * prim_index + 1],
        (*bmax)[1]);
    (*bmax)[2] = std::max(
        vertices_[3 * (2 * prim_index + 1) + 2] + radiuss_[2 * prim_index + 1],
        (*bmax)[2]);
  }

  const float *vertices_;
  const float *radiuss_;
  mutable nanort::real3<float> ray_org_;
  mutable nanort::real3<float> ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;
};

class CylinderIntersection {
 public:
  CylinderIntersection() {}

  float u;
  float v;
  nanort::real3<float> normal;

  // Required member variables.
  float t;
  unsigned int prim_id;
};

template <class I>
class CylinderIntersector {
 public:
  CylinderIntersector(const float *vertices, const float *radiuss,
                      const bool test_cap = true)
      : vertices_(vertices), radiuss_(radiuss), test_cap_(test_cap) {}

  /// Do ray interesection stuff for `prim_index` th primitive and return hit
  /// distance `t`,
  /// varycentric coordinate `u` and `v`.
  /// Returns true if there's intersection.
  bool Intersect(float *t_inout, unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }
    const float kEPS = 1.0e-6f;

    nanort::real3<float> p0, p1;
    p0[0] = vertices_[3 * (2 * prim_index + 0) + 0];
    p0[1] = vertices_[3 * (2 * prim_index + 0) + 1];
    p0[2] = vertices_[3 * (2 * prim_index + 0) + 2];
    p1[0] = vertices_[3 * (2 * prim_index + 1) + 0];
    p1[1] = vertices_[3 * (2 * prim_index + 1) + 1];
    p1[2] = vertices_[3 * (2 * prim_index + 1) + 2];

    float r0 = radiuss_[2 * prim_index + 0];
    float r1 = radiuss_[2 * prim_index + 1];

    float tmax = (*t_inout);
    float rr = std::max<float>(r0, r1);
    nanort::real3<float> ORG = ray_org_;
    nanort::real3<float> n = ray_dir_;
    nanort::real3<float> d = p1 - p0;
    nanort::real3<float> m = ORG - p0;

    float md = vdot(m, d);
    float nd = vdot(n, d);
    float dd = vdot(d, d);

    bool hitCap = false;
    float capT = std::numeric_limits<float>::max();  // far

    if (test_cap_) {
      nanort::real3<float> dN0 = vnormalize(p0 - p1);
      nanort::real3<float> dN1 = vneg(dN0);
      nanort::real3<float> rd = vnormalize(ray_dir_);

      if (fabs(vdot(ray_dir_, dN0)) > kEPS) {
        // test with 2 planes
        float p0D = -vdot(p0, dN0);  // plane D
        float p1D = -vdot(p1, dN1);  // plane D

        float p0T = -(vdot(ray_org_, dN0) + p0D) / vdot(rd, dN0);
        float p1T = -(vdot(ray_org_, dN1) + p1D) / vdot(rd, dN1);

        nanort::real3<float> q0 = ray_org_ + p0T * rd;
        nanort::real3<float> q1 = ray_org_ + p1T * rd;

        float qp0Sqr = vdot(q0 - p0, q0 - p0);
        float qp1Sqr = vdot(q1 - p1, q1 - p1);
        // printf("p0T = %f, p1T = %f, q0Sqr = %f, rr^2 = %f\n", p0T, p1T,
        // q0Sqr,
        // rr*rr);

        if (p0T > 0.0 && p0T < tmax && (qp0Sqr < rr * rr)) {
          // hit p0's plane
          hit_cap_ = hitCap = true;
          capT = p0T;
          (*t_inout) = capT;
          u_param_ = sqrt(qp0Sqr);
          v_param_ = 0;
        }

        if (p1T > 0.0 && p1T < tmax && p1T < capT && (qp1Sqr < rr * rr)) {
          hit_cap_ = hitCap = true;
          capT = p1T;
          (*t_inout) = capT;
          u_param_ = sqrt(qp1Sqr);
          v_param_ = 1.0;
        }
      }
    }

    if (md <= 0.0 && nd <= 0.0) return hitCap;
    if (md >= dd && nd >= 0.0) return hitCap;

    float nn = vdot(n, n);
    float mn = vdot(m, n);
    float A = dd * nn - nd * nd;
    float k = vdot(m, m) - rr * rr;
    float C = dd * k - md * md;
    float B = dd * mn - nd * md;

    float root[2] = {};
    int nRet = solve2e(root, A, B, C);
    if (nRet) {
      float t = root[0];
      if (0 <= t && t <= tmax && t <= capT) {
        float s = md + t * nd;
        s /= dd;
        if (0 <= s && s <= 1) {
          hit_cap_ = hitCap = false;
          (*t_inout) = t;
          u_param_ = 0;
          v_param_ = s;

          return true;
        }
      }
    }
    return hitCap;
  }

  /// Returns the nearest hit distance.
  float GetT() const { return intersection.t; }

  /// Update is called when a nearest hit is found.
  void Update(float t, unsigned int prim_idx) const {
    intersection.t = t;
    intersection.prim_id = prim_idx;
    intersection.u = u_param_;
    intersection.v = v_param_;
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
      float v = intersection.v;
      unsigned int index = intersection.prim_id;
      nanort::real3<float> p0, p1;

      p0[0] = vertices_[3 * (2 * index + 0) + 0];
      p0[1] = vertices_[3 * (2 * index + 0) + 1];
      p0[2] = vertices_[3 * (2 * index + 0) + 2];
      p1[0] = vertices_[3 * (2 * index + 1) + 0];
      p1[1] = vertices_[3 * (2 * index + 1) + 1];
      p1[2] = vertices_[3 * (2 * index + 1) + 2];

      nanort::real3<float> center =
          p0 + nanort::real3<float>(v, v, v) * (p1 - p0);
      nanort::real3<float> position = ray_org_ + intersection.t * ray_dir_;

      nanort::real3<float> n;
      if (hit_cap_) {
        nanort::real3<float> c = 0.5f * (p1 - p0) + p0;
        n = vnormalize(p1 - p0);

        if (vdot((position - c), n) > 0.0) {
          // hit p1's plane
        } else {
          // hit p0's plane
          n = vneg(n);
        }
      } else {
        n = position - center;
        n = vnormalize(n);
      }

      intersection.normal[0] = n[0];
      intersection.normal[1] = n[1];
      intersection.normal[2] = n[2];
    }
  }

  const float *vertices_;
  const float *radiuss_;
  const bool test_cap_;
  mutable nanort::real3<float> ray_org_;
  mutable nanort::real3<float> ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;

  mutable I intersection;
  mutable bool hit_cap_;
  mutable float u_param_;
  mutable float v_param_;
};

// -----------------------------------------------------

void GenerateRandomCylinders(float *vertices, float *radiuss, size_t n,
                             const float bmin[3], const float bmax[3]) {
  pcg32_state_t rng;
  pcg32_srandom(&rng, 0, 1);

  float bsize = bmax[0] - bmin[0];
  if (bsize < bmax[1] - bmin[1]) {
    bsize = bmax[1] - bmin[1];
  }
  if (bsize < bmax[2] - bmin[2]) {
    bsize = bmax[2] - bmin[2];
  }

  for (size_t i = 0; i < n; i++) {
    // [0, 1)
    float x0 = pcg32_random(&rng);
    float y0 = pcg32_random(&rng);
    float z0 = pcg32_random(&rng);
    float x1 = pcg32_random(&rng);
    float y1 = pcg32_random(&rng);
    float z1 = pcg32_random(&rng);

    vertices[3 * (2 * i + 0) + 0] = x0 * (bmax[0] - bmin[0]) + bmin[0];
    vertices[3 * (2 * i + 0) + 1] = y0 * (bmax[1] - bmin[1]) + bmin[1];
    vertices[3 * (2 * i + 0) + 2] = z0 * (bmax[2] - bmin[2]) + bmin[2];
    vertices[3 * (2 * i + 1) + 0] = x1 * (bmax[0] - bmin[0]) + bmin[0];
    vertices[3 * (2 * i + 1) + 1] = y1 * (bmax[1] - bmin[1]) + bmin[1];
    vertices[3 * (2 * i + 1) + 2] = z1 * (bmax[2] - bmin[2]) + bmin[2];

    // Adjust radius according to # of primitives.
    radiuss[2 * i + 0] = (0.25 * bsize) / sqrt((double)n);
    radiuss[2 * i + 1] = (0.25 * bsize) / sqrt((double)n);
  }
}

}  // namespace

int main(int argc, char **argv) {
  int width = 512;
  int height = 513;

  if (argc < 2) {
    printf("Needs # of cylinders\n");
    return 0;
  }

  size_t n = atoi(argv[1]);

  nanort::BVHBuildOptions<float> options;  // Use default option
  options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", options.bin_size);

  std::vector<float> vertices(3 * 2 * n);
  std::vector<float> radiuss(2 * n);

  float rbmin[3] = {-1, -1, -1};
  float rbmax[3] = {1, 1, 1};
  GenerateRandomCylinders(&vertices.at(0), &radiuss.at(0), n, rbmin, rbmax);

  CylinderGeometry cylinder_geom(&vertices.at(0), &radiuss.at(0));
  CylinderPred cylinder_pred(&vertices.at(0));

  nanort::BVHAccel<CylinderGeometry, CylinderPred,
                   CylinderIntersector<CylinderIntersection> >
      accel;
  bool ret = accel.Build(n, options, cylinder_geom, cylinder_pred);
  assert(ret);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  std::vector<float> rgb(width * height * 3, 0.0f);

  // Shoot rays.
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Simple camera. change eye pos and direction fit to your scene.

      nanort::Ray<float> ray;
      ray.org[0] = 0.0f;
      ray.org[1] = 0.0f;
      ray.org[2] = 4.0f;

      nanort::real3<float> dir;
      dir[0] = (x / (float)width) - 0.5f;
      dir[1] = (y / (float)height) - 0.5f;
      dir[2] = -1.0f;
      dir = vnormalize(dir);
      ray.dir[0] = dir[0];
      ray.dir[1] = dir[1];
      ray.dir[2] = dir[2];

      float kFar = 1.0e+30f;
      ray.min_t = 0.0f;
      ray.max_t = kFar;

      nanort::BVHTraceOptions trace_options;
      CylinderIntersector<CylinderIntersection> isector(&vertices.at(0),
                                                        &radiuss.at(0));
      bool hit = accel.Traverse(ray, trace_options, isector);
      if (hit) {
        // Flip Y
        rgb[3 * ((height - y - 1) * width + x) + 0] =
            fabsf(isector.intersection.normal[0]);
        rgb[3 * ((height - y - 1) * width + x) + 1] =
            fabsf(isector.intersection.normal[1]);
        rgb[3 * ((height - y - 1) * width + x) + 2] =
            fabsf(isector.intersection.normal[2]);
      }
    }
  }

  SaveImagePNG("render.png", &rgb.at(0), width, height);

  return 0;
}
