/* NanoRT in pure C89
   Limitation: No parallel BVH build.
*/
#ifndef NANORT_C_H_
#define NANORT_C_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _nanort_bvh_node_t
{
  float bmin[3];
  int   flag; /* 1 = leaf node, 0 = branch node. */
  float bmax[3];
  int   axis; 

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

typedef struct _nanort_bvh_build_options_t
{
  float cost_t_aabb;
  unsigned int min_leaf_primitives;
  unsigned int max_tree_depth;
  unsigned int bin_size;
} nanort_bvh_build_options_t;

typedef struct _nanort_bvh_build_statistics_t
{
  unsigned int max_tree_depth;
  unsigned int num_leaf_nodes;
  unsigned int num_branch_nodes;
  float build_secs;
} nanort_bvh_build_statistics_t;

typedef struct _nanort_bvh_trace_options_t
{
  unsigned int prim_ids_range[2];
  int          cull_back_face;
} nanort_bvh_trace_options_t;

typedef struct _nanort_bvh_accel_t
{
  nanort_bvh_node_t *nodes;
  unsigned int num_nodes;
  int          pad0;

  unsigned int *indices;
  unsigned int num_indices;
  int          pad1;

  nanort_bvh_build_options_t    build_options;
  nanort_bvh_build_statistics_t build_statistics;
} nanort_bvh_accel_t;

typedef struct _nanort_intersection_t
{
  float t;
  float u, v;
} nanort_intersection_t;

typedef struct _nanort_ray_t
{
  float org[3];      /* must set */
  float dir[3];      /* must set */
  float min_t;       /* minium ray hit distance. must set. */
  float max_t;       /* maximum ray hit distance. must set. */
  float inv_dir[3];  /* filled internally */
  int dir_sign[3];   /* filled internally */
} nanort_ray_t;

extern nanort_bvh_accel_t *nanort_bvh_accel_new(void);
extern void nanort_bvh_accel_init(nanort_bvh_accel_t *accel);
extern int nanort_bvh_accel_build(nanort_bvh_accel_t *accel);
extern int nanort_bvh_accel_traverse(const nanort_bvh_accel_t *accel);

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef NANORT_C_IMPLEMENTATION

#include <string.h>
#include <stdlib.h>
#include <memory.h>

static int nanort_triangle_interect(float *t_inout, unsigned int prim_index)
{
  (void)t_inout;
  (void)prim_index;
  return 0;
}

nanort_bvh_accel_t *nanort_bvh_accel_new(void)
{
	nanort_bvh_accel_t *bvh = (nanort_bvh_accel_t *)malloc(sizeof(nanort_bvh_accel_t));
	nanort_bvh_accel_init(bvh);
	return bvh;
}

void nanort_bvh_accel_init(nanort_bvh_accel_t *accel)
{
  memset(accel, 0, sizeof(nanort_bvh_accel_t));
}

int nanort_bvh_accel_build(nanort_bvh_accel_t *accel)
{
  /* @todo */
  accel->num_indices = 0;
  return 0;
}

int nanort_bvh_accel_traverse(const nanort_bvh_accel_t *accel)
{
  /* @todo */
  float t;
  unsigned int prim_index = 0;
  int ret;

  (void)accel;
  ret = nanort_triangle_interect(&t, prim_index);
  (void)ret;
  return 0;
}

#endif  /* NANORT_C_IMPLEMENTATION */

#endif  /* NANORT_C_H_ */
