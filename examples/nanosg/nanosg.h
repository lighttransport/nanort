/*
The MIT License (MIT)

Copyright (c) 2017 Light Transport Entertainment, Inc.

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

#ifndef NANOSG_H_
#define NANOSG_H_

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <limits>
#include <vector>
#include <iostream>

#include "nanort.h"

namespace nanosg {

template<class T>
class PrimitiveInterface;

template<class T>
class PrimitiveInterface{
public:
   void print(){ static_cast<T &>(this)->print(); }
};

class SpherePrimitive : PrimitiveInterface<SpherePrimitive> {
public:
    void print(){ std::cout << "Sphere" << std::endl; }
};

// 4x4 matrix
template <typename T> class Matrix {
public:
  Matrix();
  ~Matrix();

  static void Print(T m[4][4]) {
    for (int i = 0; i < 4; i++) {
      printf("m[%d] = %f, %f, %f, %f\n", i, m[i][0], m[i][1], m[i][2], m[i][3]);
    }
  }

  static void Identity(T m[4][4]) {
    m[0][0] = 1.0;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[0][3] = 0.0;
    m[1][0] = 0.0;
    m[1][1] = 1.0;
    m[1][2] = 0.0;
    m[1][3] = 0.0;
    m[2][0] = 0.0;
    m[2][1] = 0.0;
    m[2][2] = 1.0;
    m[2][3] = 0.0;
    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;
  }

  static void Copy(T dst[4][4], const T src[4][4]) {
    memcpy(dst, src, sizeof(T) * 16);
  }

  static void Inverse(T m[4][4]) {
    /*
     * codes from intel web
     * cramer's rule version
     */
    int i, j;
    T tmp[12];  /* tmp array for pairs */
    T tsrc[16]; /* array of transpose source matrix */
    T det;      /* determinant */

    /* transpose matrix */
    for (i = 0; i < 4; i++) {
      tsrc[i] = m[i][0];
      tsrc[i + 4] = m[i][1];
      tsrc[i + 8] = m[i][2];
      tsrc[i + 12] = m[i][3];
    }

    /* calculate pair for first 8 elements(cofactors) */
    tmp[0] = tsrc[10] * tsrc[15];
    tmp[1] = tsrc[11] * tsrc[14];
    tmp[2] = tsrc[9] * tsrc[15];
    tmp[3] = tsrc[11] * tsrc[13];
    tmp[4] = tsrc[9] * tsrc[14];
    tmp[5] = tsrc[10] * tsrc[13];
    tmp[6] = tsrc[8] * tsrc[15];
    tmp[7] = tsrc[11] * tsrc[12];
    tmp[8] = tsrc[8] * tsrc[14];
    tmp[9] = tsrc[10] * tsrc[12];
    tmp[10] = tsrc[8] * tsrc[13];
    tmp[11] = tsrc[9] * tsrc[12];

    /* calculate first 8 elements(cofactors) */
    m[0][0] = tmp[0] * tsrc[5] + tmp[3] * tsrc[6] + tmp[4] * tsrc[7];
    m[0][0] -= tmp[1] * tsrc[5] + tmp[2] * tsrc[6] + tmp[5] * tsrc[7];
    m[0][1] = tmp[1] * tsrc[4] + tmp[6] * tsrc[6] + tmp[9] * tsrc[7];
    m[0][1] -= tmp[0] * tsrc[4] + tmp[7] * tsrc[6] + tmp[8] * tsrc[7];
    m[0][2] = tmp[2] * tsrc[4] + tmp[7] * tsrc[5] + tmp[10] * tsrc[7];
    m[0][2] -= tmp[3] * tsrc[4] + tmp[6] * tsrc[5] + tmp[11] * tsrc[7];
    m[0][3] = tmp[5] * tsrc[4] + tmp[8] * tsrc[5] + tmp[11] * tsrc[6];
    m[0][3] -= tmp[4] * tsrc[4] + tmp[9] * tsrc[5] + tmp[10] * tsrc[6];
    m[1][0] = tmp[1] * tsrc[1] + tmp[2] * tsrc[2] + tmp[5] * tsrc[3];
    m[1][0] -= tmp[0] * tsrc[1] + tmp[3] * tsrc[2] + tmp[4] * tsrc[3];
    m[1][1] = tmp[0] * tsrc[0] + tmp[7] * tsrc[2] + tmp[8] * tsrc[3];
    m[1][1] -= tmp[1] * tsrc[0] + tmp[6] * tsrc[2] + tmp[9] * tsrc[3];
    m[1][2] = tmp[3] * tsrc[0] + tmp[6] * tsrc[1] + tmp[11] * tsrc[3];
    m[1][2] -= tmp[2] * tsrc[0] + tmp[7] * tsrc[1] + tmp[10] * tsrc[3];
    m[1][3] = tmp[4] * tsrc[0] + tmp[9] * tsrc[1] + tmp[10] * tsrc[2];
    m[1][3] -= tmp[5] * tsrc[0] + tmp[8] * tsrc[1] + tmp[11] * tsrc[2];

    /* calculate pairs for second 8 elements(cofactors) */
    tmp[0] = tsrc[2] * tsrc[7];
    tmp[1] = tsrc[3] * tsrc[6];
    tmp[2] = tsrc[1] * tsrc[7];
    tmp[3] = tsrc[3] * tsrc[5];
    tmp[4] = tsrc[1] * tsrc[6];
    tmp[5] = tsrc[2] * tsrc[5];
    tmp[6] = tsrc[0] * tsrc[7];
    tmp[7] = tsrc[3] * tsrc[4];
    tmp[8] = tsrc[0] * tsrc[6];
    tmp[9] = tsrc[2] * tsrc[4];
    tmp[10] = tsrc[0] * tsrc[5];
    tmp[11] = tsrc[1] * tsrc[4];

    /* calculate second 8 elements(cofactors) */
    m[2][0] = tmp[0] * tsrc[13] + tmp[3] * tsrc[14] + tmp[4] * tsrc[15];
    m[2][0] -= tmp[1] * tsrc[13] + tmp[2] * tsrc[14] + tmp[5] * tsrc[15];
    m[2][1] = tmp[1] * tsrc[12] + tmp[6] * tsrc[14] + tmp[9] * tsrc[15];
    m[2][1] -= tmp[0] * tsrc[12] + tmp[7] * tsrc[14] + tmp[8] * tsrc[15];
    m[2][2] = tmp[2] * tsrc[12] + tmp[7] * tsrc[13] + tmp[10] * tsrc[15];
    m[2][2] -= tmp[3] * tsrc[12] + tmp[6] * tsrc[13] + tmp[11] * tsrc[15];
    m[2][3] = tmp[5] * tsrc[12] + tmp[8] * tsrc[13] + tmp[11] * tsrc[14];
    m[2][3] -= tmp[4] * tsrc[12] + tmp[9] * tsrc[13] + tmp[10] * tsrc[14];
    m[3][0] = tmp[2] * tsrc[10] + tmp[5] * tsrc[11] + tmp[1] * tsrc[9];
    m[3][0] -= tmp[4] * tsrc[11] + tmp[0] * tsrc[9] + tmp[3] * tsrc[10];
    m[3][1] = tmp[8] * tsrc[11] + tmp[0] * tsrc[8] + tmp[7] * tsrc[10];
    m[3][1] -= tmp[6] * tsrc[10] + tmp[9] * tsrc[11] + tmp[1] * tsrc[8];
    m[3][2] = tmp[6] * tsrc[9] + tmp[11] * tsrc[11] + tmp[3] * tsrc[8];
    m[3][2] -= tmp[10] * tsrc[11] + tmp[2] * tsrc[8] + tmp[7] * tsrc[9];
    m[3][3] = tmp[10] * tsrc[10] + tmp[4] * tsrc[8] + tmp[9] * tsrc[9];
    m[3][3] -= tmp[8] * tsrc[9] + tmp[11] * tsrc[0] + tmp[5] * tsrc[8];

    /* calculate determinant */
    det = tsrc[0] * m[0][0] + tsrc[1] * m[0][1] + tsrc[2] * m[0][2] +
          tsrc[3] * m[0][3];

    /* calculate matrix inverse */
    det = 1.0 / det;

    for (j = 0; j < 4; j++) {
      for (i = 0; i < 4; i++) {
        m[j][i] *= det;
      }
    }
  }

  static void Transpose(T m[4][4]) {
    T t[4][4];

    // Transpose
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        t[j][i] = m[i][j];
      }
    }

    // Copy
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        m[j][i] = t[j][i];
      }
    }
  }

  static void Mult(T dst[4][4], const T m0[4][4], const T m1[4][4]) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        dst[i][j] = 0;
        for (int k = 0; k < 4; ++k) {
          dst[i][j] += m0[k][j] * m1[i][k];
        }
      }
    }
  }

  static void MultV(T dst[3], const T m[4][4], const T v[3]) {
    T tmp[3];
    tmp[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
    tmp[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
    tmp[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
    dst[0] = tmp[0];
    dst[1] = tmp[1];
    dst[2] = tmp[2];
  }

};

typedef Matrix<float> Matrixf;
typedef Matrix<double> Matrixd;

template<typename T>
static void XformBoundingBox(T xbmin[3], // out
                                 T xbmax[3], // out
                                 T bmin[3], T bmax[3],
                                 T m[4][4]) {

  // create bounding vertex from (bmin, bmax)
  T b[8][3];

  b[0][0] = bmin[0];
  b[0][1] = bmin[1];
  b[0][2] = bmin[2];
  b[1][0] = bmax[0];
  b[1][1] = bmin[1];
  b[1][2] = bmin[2];
  b[2][0] = bmin[0];
  b[2][1] = bmax[1];
  b[2][2] = bmin[2];
  b[3][0] = bmax[0];
  b[3][1] = bmax[1];
  b[3][2] = bmin[2];

  b[4][0] = bmin[0];
  b[4][1] = bmin[1];
  b[4][2] = bmax[2];
  b[5][0] = bmax[0];
  b[5][1] = bmin[1];
  b[5][2] = bmax[2];
  b[6][0] = bmin[0];
  b[6][1] = bmax[1];
  b[6][2] = bmax[2];
  b[7][0] = bmax[0];
  b[7][1] = bmax[1];
  b[7][2] = bmax[2];

  T xb[8][3];
  for (int i = 0; i < 8; i++) {
    Matrix<T>::MultV(xb[i], m, b[i]);
  }

  xbmin[0] = xb[0][0];
  xbmin[1] = xb[0][1];
  xbmin[2] = xb[0][2];
  xbmax[0] = xb[0][0];
  xbmax[1] = xb[0][1];
  xbmax[2] = xb[0][2];

  for (int i = 1; i < 8; i++) {

    xbmin[0] = std::min(xb[i][0], xbmin[0]);
    xbmin[1] = std::min(xb[i][1], xbmin[1]);
    xbmin[2] = std::min(xb[i][2], xbmin[2]);

    xbmax[0] = std::max(xb[i][0], xbmax[0]);
    xbmax[1] = std::max(xb[i][1], xbmax[1]);
    xbmax[2] = std::max(xb[i][2], xbmax[2]);
  }
}

///
/// Renderable node
///
template<typename T>
class Node
{
 public:
	explicit Node(const std::vector<nanort::BVHNode<T> > &bvh_nodes, const std::vector<unsigned int > &bvh_indices)
    : bvh_nodes_(bvh_nodes)
    , bvh_indices_(bvh_indices)
	{ 
		xbmin_[0] = xbmin_[1] = xbmin_[2] = std::numeric_limits<T>::max();
		xbmax_[0] = xbmax_[1] = xbmax_[2] = -std::numeric_limits<T>::max();

		lbmin_[0] = lbmin_[1] = lbmin_[2] = std::numeric_limits<T>::max();
		lbmax_[0] = lbmax_[1] = lbmax_[2] = -std::numeric_limits<T>::max();

    Matrix<T>::Identity(xform_);
    Matrix<T>::Identity(inv_xform_);
    Matrix<T>::Identity(inv_xform33_); inv_xform33_[3][3] = static_cast<T>(0.0);
    Matrix<T>::Identity(inv_transpose_xform33_); inv_transpose_xform33_[3][3] = static_cast<T>(0.0);

    if (!bvh_nodes_.empty()) {
      lbmin_[0] = bvh_nodes_[0].bmin[0];
      lbmin_[1] = bvh_nodes_[0].bmin[1];
      lbmin_[2] = bvh_nodes_[0].bmin[2];
      lbmax_[0] = bvh_nodes_[0].bmax[0];
      lbmax_[1] = bvh_nodes_[0].bmax[1];
      lbmax_[2] = bvh_nodes_[0].bmax[2];
    }
	}

	~Node() {}

	///
	/// Update internal state.
	///
	void Update() {

    if (bvh_nodes_.empty()) {
      return;
    }

    // Compute the bounding box in world coordinate.
    XformBoundingBox(xbmin_, xbmax_, lbmin_, lbmax_, xform_);

    // Inverse(xform)
    Matrix<T>::Copy(inv_xform_, xform_);
    Matrix<T>::Inverse(inv_xform_);

    // Clear translation, then inverse(xform) 
    Matrix<T>::Copy(inv_xform33_, xform_);
    inv_xform33_[3][0] = static_cast<T>(0.0);
    inv_xform33_[3][1] = static_cast<T>(0.0);
    inv_xform33_[3][2] = static_cast<T>(0.0);
    Matrix<T>::Inverse(inv_xform33_);

    // Inverse transpose of xform33
    Matrix<T>::Copy(inv_transpose_xform33_, inv_xform33_);
    Matrix<T>::Transpose(inv_transpose_xform33_);
	}

	inline void GetWorldBoundingBox(T bmin[3], T bmax[3]) const {
		bmin[0] = xbmin_[0];		
		bmin[1] = xbmin_[1];		
		bmin[2] = xbmin_[2];		

		bmax[0] = xbmax_[0];		
		bmax[1] = xbmax_[1];		
		bmax[2] = xbmax_[2];		
	}

	inline void GetLocalBoundingBox(T bmin[3], T bmax[3]) const {
		bmin[0] = lbmin_[0];		
		bmin[1] = lbmin_[1];		
		bmin[2] = lbmin_[2];		

		bmax[0] = lbmax_[0];		
		bmax[1] = lbmax_[1];		
		bmax[2] = lbmax_[2];		
	}

 private:

	// bounding box(local space)
	T lbmin_[3];
	T lbmax_[3];

	// bounding box after xform(world space)
	T xbmin_[3];
	T xbmax_[3];
	
	T xform_[4][4];				// Transformation matrix. local -> world.
	T inv_xform_[4][4];		// inverse(xform). world -> local
	T inv_xform33_[4][4];	// inverse(xform0 with upper-left 3x3 elemets only(for transforming direction vector)
	T inv_transpose_xform33_[4][4]; // inverse(transpose(xform)) with upper-left 3x3 elements only(for transforming normal vector)	
	
  //nanort::BVHAccel<T, P, Pred, I> accel_;

  const std::vector<nanort::BVHNode<T> > &bvh_nodes_;
  const std::vector<unsigned int > &bvh_indices_;
	
};

// -------------------------------------------------

// Predefined SAH predicator for cube.
template<typename T = float>
class NodeBBoxPred {
 public:
  NodeBBoxPred(const std::vector<Node<T> >* nodes) : axis_(0), pos_(0.0f), nodes_(nodes) {}

  void Set(int axis, float pos) const {
    axis_ = axis;
    pos_ = pos;
  }

  bool operator()(unsigned int i) const {
    int axis = axis_;
    float pos = pos_;

    T bmin[3], bmax[3];

	  (*nodes_)[i].GetWorldBoundingBox(bmin, bmax);
    
    T center = bmax[axis] - bmin[axis];

    return (center < pos);
  }

 private:
  mutable int axis_;
  mutable float pos_;
  const std::vector<Node<T> > *nodes_;
};

template<typename T = float>
class NodeBBoxGeometry {
 public:
  NodeBBoxGeometry(const std::vector<Node<T> >* nodes)
      : nodes_(nodes) {}

  /// Compute bounding box for `prim_index`th cube.
  /// This function is called for each primitive in BVH build.
  void BoundingBox(nanort::real3<T>* bmin, nanort::real3<T>* bmax, unsigned int prim_index) const {
    T a[3], b[3];
    (*nodes_)[prim_index].GetWorldBoundingBox(a, b);
    (*bmin)[0] = a[0];
    (*bmin)[1] = a[1];
    (*bmin)[2] = a[2];
    (*bmax)[0] = b[0];
    (*bmax)[1] = b[1];
    (*bmax)[2] = b[2];
  }

  const std::vector<Node<T> >* nodes_;
  mutable nanort::real3<T> ray_org_;
  mutable nanort::real3<T> ray_dir_;
  mutable nanort::BVHTraceOptions trace_options_;
};

class NodeBBoxIntersection {
 public:
  NodeBBoxIntersection() {}

  // Required member variables.
  float t;
  unsigned int prim_id;
};

template <class I, typename T = float>
class NodeBBoxIntersector {
 public:
  NodeBBoxIntersector(const Node<T> *nodes)
      : nodes_(nodes) {}

  /// Do ray interesection stuff for `prim_index` th primitive and return hit
  /// distance `t`,
  /// Returns true if there's intersection.
  bool Intersect(T* t_inout, unsigned int prim_index) const {
    if ((prim_index < trace_options_.prim_ids_range[0]) ||
        (prim_index >= trace_options_.prim_ids_range[1])) {
      return false;
    }

    T bmin[3], bmax[3];
    nodes_[prim_index].GetWorldBoundingBox(bmin, bmax);

    T tmin, tmax;

    const T min_x = ray_dir_sign_[0] ? bmax[0] : bmin[0];
    const T min_y = ray_dir_sign_[1] ? bmax[1] : bmin[1];
    const T min_z = ray_dir_sign_[2] ? bmax[2] : bmin[2];
    const T max_x = ray_dir_sign_[0] ? bmin[0] : bmax[0];
    const T max_y = ray_dir_sign_[1] ? bmin[1] : bmax[1];
    const T max_z = ray_dir_sign_[2] ? bmin[2] : bmax[2];

    // X
    const T tmin_x = (min_x - ray_org_[0]) * ray_inv_dir_[0];
    const T tmax_x = (max_x - ray_org_[0]) * ray_inv_dir_[0];

    // Y
    const T tmin_y = (min_y - ray_org_[1]) * ray_inv_dir_[1];
    const T tmax_y = (max_y - ray_org_[1]) * ray_inv_dir_[1];

    // Z
    const T tmin_z = (min_z - ray_org_[2]) * ray_inv_dir_[2];
    const T tmax_z = (max_z - ray_org_[2]) * ray_inv_dir_[2];

    tmin = nanort::safemax(tmin_z, nanort::safemax(tmin_y, tmin_x));
    tmax = nanort::safemin(tmax_z, nanort::safemin(tmax_y, tmax_x));

    if (tmin > tmax) {
      return false;
    }

    const T t = tmin;

    if (t > (*t_inout)) {
      return false;
    }

    (*t_inout) = t;

    return true;
  }

  /// Returns the nearest hit distance.
  T GetT() const { return intersection.t; }

  /// Update is called when a nearest hit is found.
  void Update(T t, unsigned int prim_idx) const {
    intersection.t = t;
    intersection.prim_id = prim_idx;
  }

  /// Prepare BVH traversal(e.g. compute inverse ray direction)
  /// This function is called only once in BVH traversal.
  void PrepareTraversal(const nanort::Ray<T>& ray,
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
  void PostTraversal(const nanort::Ray<T>& ray, bool hit) const {
    (void)ray;
    (void)hit;
  }

  const Node<T>* nodes_;
  mutable nanort::real3<T> ray_org_;
  mutable nanort::real3<T> ray_dir_;
  mutable nanort::real3<T> ray_inv_dir_;
  mutable int ray_dir_sign_[3];
  mutable nanort::BVHTraceOptions trace_options_;

  mutable I intersection;
};

template<typename T = float>
class Scene
{
 public:
	Scene() {
    bmin_[0] = bmin_[1] = bmin_[2] = std::numeric_limits<T>::max();
    bmax_[0] = bmax_[1] = bmax_[2] = -std::numeric_limits<T>::max();
  }

	~Scene() {};

  ///
  /// Add intersectable node to the scene.
  ///
	bool AddNode(const Node<T> &node) {
    nodes_.push_back(node);
  }

  ///
  /// Commit the scene. Must be called before tracing rays into the scene.
  ///
  bool Commit() {

    // Update nodes.
    for (size_t i = 0; i < nodes_.size(); i++) {
      nodes_[i].Update();
    }

    // Build toplevel BVH.
    NodeBBoxGeometry<T> geom(&nodes_);
    NodeBBoxPred<T> pred(&nodes_);
    bool ret = toplevel_accel_.Build(static_cast<unsigned int>(nodes_.size()), geom, pred);

    if (ret) {
      toplevel_accel_.BoundingBox(bmin_, bmax_); 
    } else {
      // Set invalid bbox value.
      bmin_[0] = std::numeric_limits<T>::max();
      bmin_[1] = std::numeric_limits<T>::max();
      bmin_[2] = std::numeric_limits<T>::max();

      bmax_[0] = -std::numeric_limits<T>::max();
      bmax_[1] = -std::numeric_limits<T>::max();
      bmax_[2] = -std::numeric_limits<T>::max();
    }

    return ret;
  }

  ///
  /// Get the scene bounding box.
  ///
  void GetBoundingBox(T bmin[3], T bmax[3]) const {
    bmin[0] = bmin_[0];
    bmin[1] = bmin_[1];
    bmin[2] = bmin_[2];

    bmax[0] = bmax_[0];
    bmax[1] = bmax_[1];
    bmax[2] = bmax_[2];
  }

  ///
  /// Trace the ray into the scene.
  /// First find the intersection of nodes' bounding box using toplevel BVH.
  /// Then, trace into the hit node to find the intersection of the primitive.
  ///
  bool Traverse(nanort::Ray<T> &ray) {
    
    if (!toplevel_accel_.IsValid()) {
      return false;
    }

    NodeBBoxIntersector<NodeBBoxIntersection, T> node_bbox_intersector;
    bool hit = toplevel_accel_.Travese(ray, node_bbox_intersector);

    return hit;

  }

 private:
  // Scene bounding box.
  // Valid after calling `Commit()`.
  T bmin_[3];
  T bmax_[3];
  
	std::vector<Node<T> > nodes_;
  
  // Toplevel BVH accel.
  nanort::BVHAccel<T, NodeBBoxGeometry<T>, NodeBBoxPred<T>, NodeBBoxIntersector<NodeBBoxIntersection, T> > toplevel_accel_;
};

} // namespace nanosg

#endif // NANOSG_H_
