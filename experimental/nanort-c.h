/* NanoRT in pure C89
   Limitation: No parallel BVH build.
*/
#ifndef NANORT_C_H_
#define NANORT_C_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _nanort_bvh_node_t {
  float bmin[3];
  int flag; /* 1 = leaf node, 0 = branch node. */
  float bmax[3];
  int axis;

  /*
    leaf
      data[0] = npoints
      data[1] = index

    branch
      data[0] = child[0]
      data[1] = child[1]
  */
  unsigned int data[2];
} nanort_bvh_node_t;

typedef struct _nanort_bvh_build_options_t {
  float cost_t_aabb;
  unsigned int min_leaf_primitives;
  unsigned int max_tree_depth;
  unsigned int bin_size;
} nanort_bvh_build_options_t;

typedef struct _nanort_bvh_build_statistics_t {
  unsigned int max_tree_depth;
  unsigned int num_leaf_nodes;
  unsigned int num_branch_nodes;
  float build_secs;
} nanort_bvh_build_statistics_t;

typedef struct _nanort_bvh_trace_options_t {
  unsigned int prim_ids_range[2];
  int cull_back_face;
} nanort_bvh_trace_options_t;

typedef struct _nanort_bvh_accel_t {
  nanort_bvh_node_t *nodes;
  unsigned int num_nodes;
  int pad0;

  unsigned int *indices;
  unsigned int num_indices;
  int pad1;

  nanort_bvh_build_options_t build_options;
  nanort_bvh_build_statistics_t build_statistics;
} nanort_bvh_accel_t;

typedef struct _nanort_intersection_t {
  float t;
  float u, v;
} nanort_intersection_t;

typedef struct _nanort_ray_t {
  float org[3];     /* must set */
  float dir[3];     /* must set */
  float min_t;      /* minium ray hit distance. must set. */
  float max_t;      /* maximum ray hit distance. must set. */
  float inv_dir[3]; /* filled internally */
  int dir_sign[3];  /* filled internally */
} nanort_ray_t;

/* Currenly NanoRT-C only supports triangle primitive. */

#define NANORT_SUCCESS (0)
#define NANORT_INVALID_PARAMETER (-100)

/** Initialize nanort_bvh_build_options_t with default values. */
extern void nanort_bvh_build_options_init(nanort_bvh_build_options_t *options);

/** Initialize nanort_bvh_trace_options_t with default values. */
extern void nanort_bvh_trace_options_init(nanort_bvh_trace_options_t *options);

extern void nanort_bvh_accel_init(nanort_bvh_accel_t *accel);
extern int nanort_bvh_accel_build(nanort_bvh_accel_t *accel,
                                  nanort_bvh_build_statistics_t *stats,
                                  const float *vertices,
                                  unsigned int num_indices,
                                  const unsigned int *indices,
                                  const nanort_bvh_build_options_t *options);
extern int nanort_bvh_accel_traverse(nanort_intersection_t *isect_out,
                                     const nanort_bvh_accel_t *accel,
                                     const float *vertices,
                                     const unsigned int *faces,
                                     const nanort_ray_t *ray,
                                     const nanort_bvh_trace_options_t *options);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef NANORT_C_IMPLEMENTATION

#include <assert.h>
#include <float.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>

#define NANORT_MAX_BIN_SIZE (2 * 3 * 512)
#define NANORT_LEAF_NODE_FLAG (1)
#define NANORT_MAX_STACK_DEPTH (512)

#define NANORT_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define NANORT_MAX(a, b) (((a) > (b)) ? (a) : (b))

/* For Watertight Ray/Triangle Intersection. */
typedef struct {
  float Sx;
  float Sy;
  float Sz;
  int kx;
  int ky;
  int kz;
} nanort_raycoeff_t;

static int nanort_intersect_aabb_ray(float *tmin_out, /* out */
                                     float *tmax_out, /* out */
                                     float min_t, float max_t,
                                     const float bmin[3], const float bmax[3],
                                     const float ray_org[3],
                                     const float ray_inv_dir[3],
                                     const int ray_dir_sign[3]) {
  float tmin, tmax;

  const float min_x = ray_dir_sign[0] ? bmax[0] : bmin[0];
  const float min_y = ray_dir_sign[1] ? bmax[1] : bmin[1];
  const float min_z = ray_dir_sign[2] ? bmax[2] : bmin[2];
  const float max_x = ray_dir_sign[0] ? bmin[0] : bmax[0];
  const float max_y = ray_dir_sign[1] ? bmin[1] : bmax[1];
  const float max_z = ray_dir_sign[2] ? bmin[2] : bmax[2];

  /* X */
  const float tmin_x = (min_x - ray_org[0]) * ray_inv_dir[0];
  /* MaxMult robust BVH traversal(up to 4 ulp). 1.0000000000000004 for double
   * precision. */
  const float tmax_x = (max_x - ray_org[0]) * ray_inv_dir[0] * 1.00000024f;

  /* Y */
  const float tmin_y = (min_y - ray_org[1]) * ray_inv_dir[1];
  const float tmax_y = (max_y - ray_org[1]) * ray_inv_dir[1] * 1.00000024f;

  /* Z */
  const float tmin_z = (min_z - ray_org[2]) * ray_inv_dir[2];
  const float tmax_z = (max_z - ray_org[2]) * ray_inv_dir[2] * 1.00000024f;

  tmin = NANORT_MAX(tmin_z, NANORT_MAX(tmin_y, NANORT_MAX(tmin_x, min_t)));
  tmax = NANORT_MIN(tmax_z, NANORT_MIN(tmax_y, NANORT_MIN(tmax_x, max_t)));

  if (tmin <= tmax) {
    (*tmin_out) = tmin;
    (*tmax_out) = tmax;

    return 1;
  }

  return 0; /* no hit */
}

static int nanort_ray_triangle_intersection(float *t_inout, float *u_out,
                                            float *v_out, const float *vertices,
                                            const unsigned int *faces,
                                            const unsigned int prim_index,
                                            const nanort_ray_t *ray,
                                            const nanort_raycoeff_t *ray_coeff,
                                            const int cull_back_face) {
  const unsigned int f0 = faces[3 * prim_index + 0];
  const unsigned int f1 = faces[3 * prim_index + 1];
  const unsigned int f2 = faces[3 * prim_index + 2];

  float p0[3];
  float p1[3];
  float p2[3];

  float A[3];
  float B[3];
  float C[3];

  float Ax;
  float Ay;
  float Bx;
  float By;
  float Cx;
  float Cy;

  float U;
  float V;
  float W;

  float det;

  float Az;
  float Bz;
  float Cz;
  float T;

  float rcp_det;
  float tt;

  p0[0] = vertices[3 * f0 + 0];
  p0[1] = vertices[3 * f0 + 1];
  p0[2] = vertices[3 * f0 + 2];
  p1[0] = vertices[3 * f1 + 0];
  p1[1] = vertices[3 * f1 + 1];
  p1[2] = vertices[3 * f1 + 2];
  p2[0] = vertices[3 * f2 + 0];
  p2[1] = vertices[3 * f2 + 1];
  p2[2] = vertices[3 * f2 + 2];

  A[0] = p0[0] - ray->org[0];
  A[1] = p0[1] - ray->org[1];
  A[2] = p0[2] - ray->org[2];
  B[0] = p1[0] - ray->org[0];
  B[1] = p1[1] - ray->org[1];
  B[2] = p1[2] - ray->org[2];
  C[0] = p2[0] - ray->org[0];
  C[1] = p2[1] - ray->org[1];
  C[2] = p2[2] - ray->org[2];

  Ax = A[ray_coeff->kx] - ray_coeff->Sx * A[ray_coeff->kz];
  Ay = A[ray_coeff->ky] - ray_coeff->Sy * A[ray_coeff->kz];
  Bx = B[ray_coeff->kx] - ray_coeff->Sx * B[ray_coeff->kz];
  By = B[ray_coeff->ky] - ray_coeff->Sy * B[ray_coeff->kz];
  Cx = C[ray_coeff->kx] - ray_coeff->Sx * C[ray_coeff->kz];
  Cy = C[ray_coeff->ky] - ray_coeff->Sy * C[ray_coeff->kz];

  U = Cx * By - Cy * Bx;
  V = Ax * Cy - Ay * Cx;
  W = Bx * Ay - By * Ax;

  /* Fall back to test against edges using double precision. */
  if (U == 0.0f || V == 0.0f || W == 0.0f) {
    double CxBy = (double)(Cx) * (double)(By);
    double CyBx = (double)(Cy) * (double)(Bx);
    double AxCy = (double)(Ax) * (double)(Cy);
    double AyCx = (double)(Ay) * (double)(Cx);
    double BxAy = (double)(Bx) * (double)(Ay);
    double ByAx = (double)(By) * (double)(Ax);

    U = (float)(CxBy - CyBx);
    V = (float)(AxCy - AyCx);
    W = (float)(BxAy - ByAx);
  }

  if (cull_back_face) {
    if (U < 0.0f || V < 0.0f || W < 0.0f) return 0;
  } else {
    if ((U < 0.0f || V < 0.0f || W < 0.0f) &&
        (U > 0.0f || V > 0.0f || W > 0.0f)) {
      return 0;
    }
  }

  det = U + V + W;
  if (det == 0.0f) return 0;

  Az = ray_coeff->Sz * A[ray_coeff->kz];
  Bz = ray_coeff->Sz * B[ray_coeff->kz];
  Cz = ray_coeff->Sz * C[ray_coeff->kz];
  T = U * Az + V * Bz + W * Cz;

  rcp_det = 1.0f / det;
  tt = T * rcp_det;

  if (tt > (*t_inout)) {
    return 0;
  }

  (*t_inout) = tt;
  /*  Use Thomas-Mueller style barycentric coord.
   *  U + V + W = 1.0 and interp(p) = U * p0 + V * p1 + W * p2
   *  We want interp(p) = (1 - u - v) * p0 + u * v1 + v * p2;
   *  => u = V, v = W.
   */
  (*u_out) = V * rcp_det;
  (*v_out) = W * rcp_det;

  return 1;
}

static int nanort_test_leaf_node(
    float *t_inout, float *u_out, float *v_out, const nanort_bvh_node_t *node,
    const nanort_ray_t *ray, const nanort_raycoeff_t *ray_coeff,
    const float *vertices, const unsigned int *faces,
    const unsigned int *indices, const nanort_bvh_trace_options_t *options) {
  unsigned int i = 0;
  int hit = 0;

  unsigned int num_primitives = node->data[0];
  unsigned int offset = node->data[1];

  float t = (*t_inout); /* current hit distance */

  for (i = 0; i < num_primitives; i++) {
    unsigned int prim_idx = indices[i + offset];
    float local_t = t;
    float u, v;

    if ((prim_idx < options->prim_ids_range[0]) ||
        (prim_idx >= options->prim_ids_range[1])) {
      return 0;
    }

    if (nanort_ray_triangle_intersection(&local_t, &u, &v, vertices, faces,
                                         prim_idx, ray, ray_coeff,
                                         options->cull_back_face)) {
      if (local_t > ray->min_t) {
        /* Update isect state */
        t = local_t;

        (*u_out) = u;
        (*v_out) = v;

        hit = 1;
      }
    }
  }

  return hit;
}

static float nanort_surface_area(const float bmin[3], const float bmax[3]) {
  float box[3];
  box[0] = bmax[0] - bmin[0];
  box[1] = bmax[1] - bmin[1];
  box[2] = bmax[2] - bmin[2];
  return 2.0f * (box[0] * box[1] + box[1] * box[2] + box[2] * box[0]);
}

static float nanort_sah(size_t ns1, float left_area, size_t ns2,
                        float right_area, float inv_s, float t_aabb,
                        float t_tri) {
  float t = 2.0f * t_aabb + (left_area * inv_s) * (float)(ns1)*t_tri +
            (right_area * inv_s) * (float)(ns2)*t_tri;

  return t;
}

static void nanort_bounding_box(float bmin[3], float bmax[3],
                                const float *vertices,
                                const unsigned int *faces, unsigned int index) {
  int i;

  unsigned int f0 = faces[3 * index + 0];
  unsigned int f1 = faces[3 * index + 1];
  unsigned int f2 = faces[3 * index + 2];

  const float *p[3];

  p[0] = &vertices[3 * f0];
  p[1] = &vertices[3 * f1];
  p[2] = &vertices[3 * f2];

  bmin[0] = p[0][0];
  bmin[1] = p[0][1];
  bmin[2] = p[0][2];
  bmax[0] = p[0][0];
  bmax[1] = p[0][1];
  bmax[2] = p[0][2];

  for (i = 1; i < 3; i++) {
    bmin[0] = NANORT_MIN(bmin[0], p[i][0]);
    bmin[1] = NANORT_MIN(bmin[1], p[i][1]);
    bmin[2] = NANORT_MIN(bmin[2], p[i][2]);

    bmax[0] = NANORT_MIN(bmax[0], p[i][0]);
    bmax[1] = NANORT_MIN(bmax[1], p[i][1]);
    bmax[2] = NANORT_MIN(bmax[2], p[i][2]);
  }
}

static void nanort_bounding_boxes(float bmin[3], float bmax[3],
                                  const float *vertices,
                                  const unsigned int *faces,
                                  const unsigned int *indices,
                                  unsigned int left_index,
                                  unsigned int right_index) {
  {
    unsigned int idx = indices[left_index];
    nanort_bounding_box(bmin, bmax, vertices, faces, idx);
  }

  {
    unsigned int i;
    int k;
    for (i = left_index + 1; i < right_index; i++) {
      unsigned int idx = indices[i];
      float bbox_min[3], bbox_max[3];
      nanort_bounding_box(bbox_min, bbox_max, vertices, faces, idx);
      for (k = 0; k < 3; k++) { /* xyz */
        if (bmin[k] > bbox_min[k]) bmin[k] = bbox_min[k];
        if (bmax[k] < bbox_max[k]) bmax[k] = bbox_max[k];
      }
    }
  }
}

static void nanort_contribute_bin_buffer(
    unsigned int *bins, unsigned int bin_size, const float scene_min[3],
    const float scene_max[3], const float *vertices, const unsigned int *faces,
    unsigned int *indices, unsigned int left_idx, unsigned int right_idx) {
  unsigned int i, j;
  unsigned int idx_bmin[3];
  unsigned int idx_bmax[3];

  /* calculate extent */
  float scene_size[3], scene_inv_size[3];
  scene_size[0] = scene_max[0] - scene_min[0];
  scene_size[1] = scene_max[1] - scene_min[1];
  scene_size[2] = scene_max[2] - scene_min[2];
  for (i = 0; i < 3; ++i) {
    assert(scene_size[i] >= 0.0f);

    if (scene_size[i] > 0.0f) {
      scene_inv_size[i] = (float)bin_size / scene_size[i];
    } else {
      scene_inv_size[i] = 0.0;
    }
  }

  /* clear bin data */
  memset(bins, 0, sizeof(2 * 3 * bin_size)); /* 2 * 3 = (bmin, bmax) * xyz */

  for (i = left_idx; i < right_idx; i++) {
    /*
     * Quantize the position into [0, BIN_SIZE)
     *
     * q[i] = (int)(p[i] - scene_bmin) / scene_size
     */
    float bmin[3];
    float bmax[3];
    float quantized_bmin[3], quantized_bmax[3];

    nanort_bounding_box(bmin, bmax, vertices, faces, indices[i]);

    quantized_bmin[0] = (bmin[0] - scene_min[0]) * scene_inv_size[0];
    quantized_bmin[1] = (bmin[1] - scene_min[1]) * scene_inv_size[1];
    quantized_bmin[2] = (bmin[2] - scene_min[2]) * scene_inv_size[2];

    quantized_bmax[0] = (bmax[0] - scene_min[0]) * scene_inv_size[0];
    quantized_bmax[1] = (bmax[1] - scene_min[1]) * scene_inv_size[1];
    quantized_bmax[2] = (bmax[2] - scene_min[2]) * scene_inv_size[2];

    /* idx is now in [0, BIN_SIZE) */
    for (j = 0; j < 3; ++j) {
      int q0 = (int)(quantized_bmin[j]);
      int q1 = (int)(quantized_bmax[j]);
      if (q0 < 0) q0 = 0;
      if (q1 < 0) q1 = 0;

      idx_bmin[j] = (unsigned int)(q0);
      idx_bmax[j] = (unsigned int)(q1);

      if (idx_bmin[j] >= bin_size) idx_bmin[j] = (unsigned int)(bin_size)-1;
      if (idx_bmax[j] >= bin_size) idx_bmax[j] = (unsigned int)(bin_size)-1;

      assert(idx_bmin[j] < bin_size);
      assert(idx_bmax[j] < bin_size);

      /* increment bin counter */
      bins[0 * (bin_size * 3) + j * bin_size + idx_bmin[j]] += 1;
      bins[1 * (bin_size * 3) + j * bin_size + idx_bmax[j]] += 1;
    }
  }
}

static void nanort_find_cut_from_bin_buffer(
    float *cut_pos,     /* [out] xyz */
    int *min_cost_axis, /* [out] */
    const unsigned int *bins, const unsigned int bin_size, const float bmin[3],
    const float bmax[3], size_t num_primitives,
    float cost_t_aabb) { /* should be in [0.0, 1.0] */
  const float kEPS = FLT_EPSILON;

  unsigned int i, j;
  size_t left, right;
  float bsize[3], bstep[3];
  float bmin_left[3], bmax_left[3];
  float bmin_right[3], bmax_right[3];
  float sa_left, sa_right, sa_total;
  float pos;
  float min_cost[3];
  float inv_sa_total = 0.0f;
  float cost_t_tri = 1.0f - cost_t_aabb;

  (*min_cost_axis) = 0;

  bsize[0] = bmax[0] - bmin[0];
  bsize[1] = bmax[1] - bmin[1];
  bsize[2] = bmax[2] - bmin[2];
  bstep[0] = bsize[0] * (1.0f / (float)bin_size);
  bstep[1] = bsize[1] * (1.0f / (float)bin_size);
  bstep[2] = bsize[2] * (1.0f / (float)bin_size);
  sa_total = nanort_surface_area(bmin, bmax);

  if (sa_total > kEPS) {
    inv_sa_total = 1.0f / sa_total;
  }

  for (j = 0; j < 3; ++j) {
    /*
     * Compute SAH cost for right side of each cell of the bbox.
     * Exclude both extreme side of the bbox.
     *
     *  i:      0    1    2    3
     *     +----+----+----+----+----+
     *     |    |    |    |    |    |
     *     +----+----+----+----+----+
     */

    float min_cost_pos = bmin[j] + 0.5f * bstep[j];
    min_cost[j] = FLT_MAX;

    left = 0;
    right = num_primitives;
    bmin_left[0] = bmin_right[0] = bmin[0];
    bmin_left[1] = bmin_right[1] = bmin[1];
    bmin_left[2] = bmin_right[2] = bmin[2];
    bmax_left[0] = bmax_right[0] = bmax[0];
    bmax_left[1] = bmax_right[1] = bmax[1];
    bmax_left[2] = bmax_right[2] = bmax[2];

    for (i = 0; i < bin_size - 1; ++i) {
      float cost;

      left += bins[0 * (3 * bin_size) + j * bin_size + i];
      right -= bins[1 * (3 * bin_size) + j * bin_size + i];

      assert(left <= num_primitives);
      assert(right <= num_primitives);

      /*
       * Split pos bmin + (i + 1) * (bsize / BIN_SIZE)
       * +1 for i since we want a position on right side of the cell.
       */

      pos = bmin[j] + (i + 0.5f) * bstep[j];
      bmax_left[j] = pos;
      bmin_right[j] = pos;

      sa_left = nanort_surface_area(bmin_left, bmax_left);
      sa_right = nanort_surface_area(bmin_right, bmax_right);

      cost = nanort_sah(left, sa_left, right, sa_right, inv_sa_total,
                        cost_t_aabb, cost_t_tri);
      if (cost < min_cost[j]) {
        /* Update the min cost */
        min_cost[j] = cost;
        min_cost_pos = pos;
      }
    }

    cut_pos[j] = min_cost_pos;
  }

  /* Find min cost axis */
  {
    float cost = min_cost[0];
    (*min_cost_axis) = 0;
    if (cost > min_cost[1]) {
      (*min_cost_axis) = 1;
      cost = min_cost[1];
    }
    if (cost > min_cost[2]) {
      (*min_cost_axis) = 2;
      cost = min_cost[2];
    }
  }
}

static unsigned int nanort_build_tree(
    nanort_bvh_node_t *nodes_out, unsigned int nodes_out_size,
    nanort_bvh_build_statistics_t *stats_out, const float *vertices,
    const unsigned int *faces, unsigned int *indices /* inout */,
    const unsigned int left_idx, const unsigned int right_idx,
    const unsigned int depth, const nanort_bvh_build_options_t *option) {
  unsigned int offset = nodes_out_size;

  float bmin[3], bmax[3];
  unsigned int n = right_idx - left_idx;

  if (stats_out->max_tree_depth < depth) {
    stats_out->max_tree_depth = depth;
  }

  assert(left_idx <= right_idx);
  nanort_bounding_boxes(bmin, bmax, vertices, faces, indices, left_idx,
                        right_idx);

  if ((n < option->min_leaf_primitives) || (depth >= option->max_tree_depth)) {
    /* Create leaf node. */
    nanort_bvh_node_t leaf;

    leaf.bmin[0] = bmin[0];
    leaf.bmin[1] = bmin[1];
    leaf.bmin[2] = bmin[2];

    leaf.bmax[0] = bmax[0];
    leaf.bmax[1] = bmax[1];
    leaf.bmax[2] = bmax[2];

    leaf.flag = NANORT_LEAF_NODE_FLAG;
    leaf.data[0] = n;
    leaf.data[1] = left_idx;

    nodes_out[offset] = leaf;

    /* out_stat->num_leaf_nodes++; */

    return offset;
  }

  /* Create branch node. */
  {
    int min_cut_axis = 0;
    float cut_pos[3] = {0.0, 0.0, 0.0};

    /* bins will be cleared inside of nanort_contribute_bin_buffer() */
    unsigned int bins[NANORT_MAX_BIN_SIZE];
    nanort_contribute_bin_buffer(bins, option->bin_size, bmin, bmax, vertices,
                                 faces, indices, left_idx, right_idx);
    nanort_find_cut_from_bin_buffer(cut_pos, &min_cut_axis, bins,
                                    option->bin_size, bmin, bmax, n,
                                    option->cost_t_aabb);
  }

  return offset;
}

void nanort_bvh_accel_init(nanort_bvh_accel_t *accel) {
  if (accel) {
    memset(accel, 0, sizeof(nanort_bvh_accel_t));
  }
  return;
}

int nanort_bvh_accel_build(nanort_bvh_accel_t *accel,            /* out */
                           nanort_bvh_build_statistics_t *stats, /* out */
                           const float *vertices, unsigned int num_faces,
                           const unsigned int *faces,
                           const nanort_bvh_build_options_t *options) {
  unsigned int i;
  unsigned int n;

  accel->num_indices = 0;

  if (!vertices) {
    return NANORT_INVALID_PARAMETER;
  }

  if (!faces) {
    return NANORT_INVALID_PARAMETER;
  }

  if (num_faces == 0) {
    return NANORT_INVALID_PARAMETER;
  }

  if (!stats) {
    return NANORT_INVALID_PARAMETER;
  }

  assert(options->max_tree_depth <= NANORT_MAX_STACK_DEPTH);

  /* Allocate internal buffer for primitive indices */
  n = num_faces / 3; /* Assume all faces are composed of triangle. */

  accel->num_indices = n;
  accel->indices = (unsigned int *)malloc(sizeof(unsigned int) * n);

  for (i = 0; i < n; i++) {
    accel->indices[i] = i;
  }

  accel->build_options = (*options);

  /* Assume no duplicated indices, thus its safe to allocate `n` nodes
   * firstly(no dynamically adjust array). */
  accel->nodes = (nanort_bvh_node_t *)malloc(sizeof(nanort_bvh_node_t) * n);
  nanort_build_tree(accel->nodes, 0, stats, vertices, faces, accel->indices, 0,
                    n,
                    /* depth */ 0, &accel->build_options);

  return NANORT_SUCCESS;
}

int nanort_bvh_accel_traverse(nanort_intersection_t *isect_out,
                              const nanort_bvh_accel_t *accel,
                              const float *vertices, const unsigned int *faces,
                              const nanort_ray_t *ray,
                              const nanort_bvh_trace_options_t *options) {
  int node_stack_index = 0;
  unsigned int node_stack[NANORT_MAX_STACK_DEPTH];
  float t = ray->max_t;
  float u = 0.0f, v = 0.0f;
  float hit_t = ray->max_t;
  float min_t = FLT_MAX;
  float max_t = -FLT_MAX;
  int dir_sign[3];
  float ray_inv_dir[3];
  float ray_org[3];
  nanort_raycoeff_t ray_coeff;

  if (!isect_out) {
    return 0;
  }

  node_stack[0] = 0;

  dir_sign[0] = ray->dir[0] < 0.0f ? 1 : 0;
  dir_sign[1] = ray->dir[1] < 0.0f ? 1 : 0;
  dir_sign[2] = ray->dir[2] < 0.0f ? 1 : 0;

  /* @fixme { Check edge case; i.e., 1/0 } */
  ray_inv_dir[0] = 1.0f / ray->dir[0];
  ray_inv_dir[1] = 1.0f / ray->dir[1];
  ray_inv_dir[2] = 1.0f / ray->dir[2];

  ray_org[0] = ray->org[0];
  ray_org[1] = ray->org[1];
  ray_org[2] = ray->org[2];

  {
    /* Calculate dimension where the ray direction is maximal. */
    float abs_dir = (float)fabs((double)ray->dir[0]);
    ray_coeff.kz = 0;
    if (abs_dir < (float)fabs((double)ray->dir[1])) {
      ray_coeff.kz = 1;
      abs_dir = (float)fabs((double)ray->dir[1]);
    }
    if (abs_dir < (float)fabs((double)ray->dir[2])) {
      ray_coeff.kz = 2;
      abs_dir = (float)fabs((double)ray->dir[2]);
    }

    ray_coeff.kx = ray_coeff.kz + 1;
    if (ray_coeff.kx == 3) ray_coeff.kx = 0;
    ray_coeff.ky = ray_coeff.kx + 1;
    if (ray_coeff.ky == 3) ray_coeff.ky = 0;

    /* Swap kx and ky dimention to preserve widing direction of triangles. */
    if (ray->dir[ray_coeff.kz] < 0.0f) {
      int tmp = ray_coeff.ky;
      ray_coeff.ky = ray_coeff.kx;
      ray_coeff.kx = tmp;
    }

    /* Claculate shear constants. */
    ray_coeff.Sx = ray->dir[ray_coeff.kx] / ray->dir[ray_coeff.kz];
    ray_coeff.Sy = ray->dir[ray_coeff.ky] / ray->dir[ray_coeff.kz];
    ray_coeff.Sz = 1.0f / ray->dir[ray_coeff.kz];
  }

  while (node_stack_index >= 0) {
    unsigned int index = node_stack[node_stack_index];
    const nanort_bvh_node_t *node = &(accel->nodes[index]);
    int hit = 0;

    node_stack_index--;

    hit =
        nanort_intersect_aabb_ray(&min_t, &max_t, ray->min_t, hit_t, node->bmin,
                                  node->bmax, ray_org, ray_inv_dir, dir_sign);

    if (node->flag == 0) { /* branch node */
      if (hit) {
        int order_near = dir_sign[node->axis];
        int order_far = 1 - order_near;

        /* Traverse near first. */
        node_stack[++node_stack_index] = node->data[order_far];
        node_stack[++node_stack_index] = node->data[order_near];
      }
    } else { /* leaf node */
      if (hit) {
        if (nanort_test_leaf_node(&t, &u, &v, node, ray, &ray_coeff, vertices,
                                  faces, accel->indices, options)) {
          hit_t = t;
        }
      }
    }
  }

  assert(node_stack_index < NANORT_MAX_STACK_DEPTH);

  {
    int hit = t < ray->max_t;
    if (hit) {
      isect_out->t = t;
      isect_out->u = u;
      isect_out->v = v;
    }

    return hit;
  }
}

void nanort_bvh_build_options_init(nanort_bvh_build_options_t *options) {
  options->cost_t_aabb = 0.2f;
  options->min_leaf_primitives = 4;
  options->max_tree_depth = 256;
  options->bin_size = 64;
}

void nanort_bvh_trace_options_init(nanort_bvh_trace_options_t *options) {
  options->prim_ids_range[0] = 0;
  options->prim_ids_range[1] = 0x7FFFFFFF; /* Up to 2G face IDs. */
  options->cull_back_face = 0;
}

#endif /* NANORT_C_IMPLEMENTATION */

#endif /* NANORT_C_H_ */
