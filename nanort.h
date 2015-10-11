//
// NanoRT, single header only modern ray tracing kernel.
//

/*
The MIT License (MIT)

Copyright (c) 2015 Light Transport Entertainment, Inc.

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

#ifndef __NANORT_H__
#define __NANORT_H__

#include <vector>
#include <cmath>
#include <limits>

namespace nanort {

// Parallelized BVH build is not yet fully tested,
// thus turn off if you face a problem when building BVH.
#define NANORT_ENABLE_PARALLEL_BUILD (0)

namespace {

struct float3 {
  float3() {}
  float3(float xx, float yy, float zz) {
    x = xx;
    y = yy;
    z = zz;
  }
  float3(const float *p) {
    x = p[0];
    y = p[1];
    z = p[2];
  }

  float3 operator*(float f) const { return float3(x * f, y * f, z * f); }
  float3 operator-(const float3 &f2) const {
    return float3(x - f2.x, y - f2.y, z - f2.z);
  }
  float3 operator*(const float3 &f2) const {
    return float3(x * f2.x, y * f2.y, z * f2.z);
  }
  float3 operator+(const float3 &f2) const {
    return float3(x + f2.x, y + f2.y, z + f2.z);
  }
  float3 &operator+=(const float3 &f2) {
    x += f2.x;
    y += f2.y;
    z += f2.z;
    return (*this);
  }
  float3 operator/(const float3 &f2) const {
    return float3(x / f2.x, y / f2.y, z / f2.z);
  }
  float operator[](int i) const { return (&x)[i]; }
  float &operator[](int i) { return (&x)[i]; }

  float3 neg() { return float3(-x, -y, -z); }

  float length() { return sqrtf(x * x + y * y + z * z); }

  void normalize() {
    float len = length();
    if (fabs(len) > 1.0e-6f) {
      float inv_len = 1.0 / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
  }

  float x, y, z;
  // float pad;  // for alignment
};

inline float3 operator*(float f, const float3 &v) {
  return float3(v.x * f, v.y * f, v.z * f);
}

inline float3 vcross(float3 a, float3 b) {
  float3 c;
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

inline float vdot(float3 a, float3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

} // namespace

typedef struct {
  float t;
  float u;
  float v;
  unsigned int faceID;
} Intersection;

typedef struct {
  float org[3];    // must set
  float dir[3];    // must set
  float invDir[3]; // filled internally
  int dirSign[3];  // filled internally
} Ray;

class BVHNode {
public:
  BVHNode(){};
  ~BVHNode(){};

  float bmin[3];
  float bmax[3];

  int flag; // 1 = leaf node, 0 = branch node
  int axis;

  // leaf
  //   data[0] = npoints
  //   data[1] = index
  //
  // branch
  //   data[0] = child[0]
  //   data[1] = child[1]
  unsigned int data[2];
};

///< BVH build option.
struct BVHBuildOptions {
  float costTaabb;
  int minLeafPrimitives;
  int maxTreeDepth;
  int binSize;
  int shallowDepth;
  size_t minPrimitivesForParallelBuild;

  // Cache bounding box computation.
  // Requires more memory, but BVHbuild can be faster.
  bool cacheBBox;

  // Set default value: Taabb = 0.2
  BVHBuildOptions()
      : costTaabb(0.2), minLeafPrimitives(4), maxTreeDepth(256), binSize(64),
        shallowDepth(3), minPrimitivesForParallelBuild(1024 * 128),
        cacheBBox(false) {}
};

///< BVH build statistics.
class BVHBuildStatistics {
public:
  int maxTreeDepth;
  int numLeafNodes;
  int numBranchNodes;
  float epsScale;
  double buildSecs;

  // Set default value: Taabb = 0.2
  BVHBuildStatistics()
      : maxTreeDepth(0), numLeafNodes(0), numBranchNodes(0), epsScale(1.0f),
        buildSecs(0.0) {}
};

class BBox {
public:
  float bmin[3];
  float bmax[3];

  BBox() {
    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();
  }
};

class BVHAccel {
public:
  BVHAccel() : epsScale_(1.0f){};
  ~BVHAccel(){};

  ///< Build BVH for input mesh.
  bool Build(const float *vertices, const unsigned int *faces,
             const unsigned int numFaces, const BVHBuildOptions &options);

  ///< Get statistics of built BVH tree. Valid after Build()
  BVHBuildStatistics GetStatistics() const { return stats_; }

  ///< Dump built BVH to the file.
  bool Dump(const char *filename);

  /// Load BVH binary
  bool Load(const char *filename);

  ///< Traverse into BVH along ray and find closest hit point if found
  bool Traverse(Intersection &isect, const float *vertices,
                const unsigned int *faces, Ray &ray);

  const std::vector<BVHNode> &GetNodes() const { return nodes_; }
  const std::vector<unsigned int> &GetIndices() const { return indices_; }

  void BoundingBox(float bmin[3], float bmax[3]) const {
    if (nodes_.empty()) {
      bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
      bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();
    } else {
      bmin[0] = nodes_[0].bmin[0];
      bmin[1] = nodes_[0].bmin[1];
      bmin[2] = nodes_[0].bmin[2];
      bmax[0] = nodes_[0].bmax[0];
      bmax[1] = nodes_[0].bmax[1];
      bmax[2] = nodes_[0].bmax[2];
    }
  }

private:
#if NANORT_ENABLE_PARALLEL_BUILD
  typedef struct {
    unsigned int leftIdx;
    unsigned int rightIdx;
    unsigned int offset;
  } ShallowNodeInfo;

  // Used only during BVH construction
  std::vector<ShallowNodeInfo> shallowNodeInfos_;

  ///< Builds shallow BVH tree recursively.
  unsigned int BuildShallowTree(std::vector<BVHNode> &outNodes,
                                const float *vertices,
                                const unsigned int *faces, unsigned int leftIdx,
                                unsigned int rightIdx, int depth,
                                int maxShallowDepth, float epsScale);
#endif

  ///< Builds BVH tree recursively.
  size_t BuildTree(BVHBuildStatistics &outStat, std::vector<BVHNode> &outNodes,
                   const float *vertices, const unsigned int *faces,
                   unsigned int leftIdx, unsigned int rightIdx, int depth,
                   float epsScale);

  BVHBuildOptions options_;
  std::vector<BVHNode> nodes_;
  std::vector<unsigned int> indices_; // max 4G triangles.
  BVHBuildStatistics stats_;
  float epsScale_;
  std::vector<BBox> bboxes_;
};

} // namespace nanort

#ifdef NANORT_IMPLEMENTATION

#include <limits>
#include <cassert>
#include <algorithm>
#include <functional>

//
// SAH functions
//
namespace nanort {

struct BinBuffer {

  BinBuffer(int size) {
    binSize = size;
    bin.resize(2 * 3 * size);
    clear();
  }

  void clear() { memset(&bin[0], 0, sizeof(size_t) * 2 * 3 * binSize); }

  std::vector<size_t> bin; // (min, max) * xyz * binsize
  int binSize;
};

inline float CalculateSurfaceArea(const float3 &min, const float3 &max) {
  float3 box = max - min;
  return 2.0 * (box[0] * box[1] + box[1] * box[2] + box[2] * box[0]);
}

inline void GetBoundingBoxOfTriangle(float3 &bmin, float3 &bmax,
                                     const float *vertices,
                                     const unsigned int *faces,
                                     unsigned int index) {
  unsigned int f0 = faces[3 * index + 0];
  unsigned int f1 = faces[3 * index + 1];
  unsigned int f2 = faces[3 * index + 2];

  float3 p[3];

  p[0] = float3(&vertices[3 * f0]);
  p[1] = float3(&vertices[3 * f1]);
  p[2] = float3(&vertices[3 * f2]);

  bmin = p[0];
  bmax = p[0];

  for (int i = 1; i < 3; i++) {
    bmin[0] = std::min(bmin[0], p[i][0]);
    bmin[1] = std::min(bmin[1], p[i][1]);
    bmin[2] = std::min(bmin[2], p[i][2]);

    bmax[0] = std::max(bmax[0], p[i][0]);
    bmax[1] = std::max(bmax[1], p[i][1]);
    bmax[2] = std::max(bmax[2], p[i][2]);
  }
}

void ContributeBinBuffer(BinBuffer *bins, // [out]
                         const float3 &sceneMin, const float3 &sceneMax,
                         const float *vertices, const unsigned int *faces,
                         unsigned int *indices, unsigned int leftIdx,
                         unsigned int rightIdx, float epsScale) {
  const float kEPS = std::numeric_limits<float>::epsilon() * epsScale;

  float binSize = (float)bins->binSize;

  // Calculate extent
  float3 sceneSize, sceneInvSize;
  sceneSize = sceneMax - sceneMin;
  for (int i = 0; i < 3; ++i) {
    assert(sceneSize[i] >= 0.0);

    if (sceneSize[i] > kEPS) {
      sceneInvSize[i] = binSize / sceneSize[i];
    } else {
      sceneInvSize[i] = 0.0;
    }
  }

  // Clear bin data
  std::fill(bins->bin.begin(), bins->bin.end(), 0);
  // memset(&bins->bin[0], 0, sizeof(2 * 3 * bins->binSize));

  size_t idxBMin[3];
  size_t idxBMax[3];

  for (size_t i = leftIdx; i < rightIdx; i++) {

    //
    // Quantize the position into [0, BIN_SIZE)
    //
    // q[i] = (int)(p[i] - scene_bmin) / scene_size
    //
    float3 bmin;
    float3 bmax;

    GetBoundingBoxOfTriangle(bmin, bmax, vertices, faces, indices[i]);

    float3 quantizedBMin = (bmin - sceneMin) * sceneInvSize;
    float3 quantizedBMax = (bmax - sceneMin) * sceneInvSize;

    // idx is now in [0, BIN_SIZE)
    for (size_t j = 0; j < 3; ++j) {
      int q0 = (int)quantizedBMin[j];
      if (q0 < 0)
        q0 = 0;
      int q1 = (int)quantizedBMax[j];
      if (q1 < 0)
        q1 = 0;

      idxBMin[j] = (unsigned int)q0;
      idxBMax[j] = (unsigned int)q1;

      if (idxBMin[j] >= binSize)
        idxBMin[j] = binSize - 1;
      if (idxBMax[j] >= binSize)
        idxBMax[j] = binSize - 1;

      assert(idxBMin[j] < binSize);
      assert(idxBMax[j] < binSize);

      // Increment bin counter
      bins->bin[0 * (bins->binSize * 3) + j * bins->binSize + idxBMin[j]] += 1;
      bins->bin[1 * (bins->binSize * 3) + j * bins->binSize + idxBMax[j]] += 1;
    }
  }
}

inline float SAH(size_t ns1, float leftArea, size_t ns2, float rightArea,
                 float invS, float Taabb, float Ttri) {
  // const float Taabb = 0.2f;
  // const float Ttri = 0.8f;
  float T;

  T = 2.0f * Taabb + (leftArea * invS) * (float)(ns1)*Ttri +
      (rightArea * invS) * (float)(ns2)*Ttri;

  return T;
}

bool FindCutFromBinBuffer(float *cutPos,    // [out] xyz
                          int &minCostAxis, // [out]
                          const BinBuffer *bins, const float3 &bmin,
                          const float3 &bmax, size_t numTriangles,
                          float costTaabb, // should be in [0.0, 1.0]
                          float epsScale) {
  const float eps = std::numeric_limits<float>::epsilon() * epsScale;

  size_t left, right;
  float3 bsize, bstep;
  float3 bminLeft, bmaxLeft;
  float3 bminRight, bmaxRight;
  float saLeft, saRight, saTotal;
  float pos;
  float minCost[3];

  float costTtri = 1.0 - costTaabb;

  minCostAxis = 0;

  bsize = bmax - bmin;
  bstep = bsize * (1.0 / bins->binSize);
  saTotal = CalculateSurfaceArea(bmin, bmax);

  float invSaTotal = 0.0;
  if (saTotal > eps) {
    invSaTotal = 1.0 / saTotal;
  }

  for (int j = 0; j < 3; ++j) {

    //
    // Compute SAH cost for right side of each cell of the bbox.
    // Exclude both extreme side of the bbox.
    //
    //  i:      0    1    2    3
    //     +----+----+----+----+----+
    //     |    |    |    |    |    |
    //     +----+----+----+----+----+
    //

    float minCostPos = bmin[j] + 0.5 * bstep[j];
    minCost[j] = std::numeric_limits<float>::max();

    left = 0;
    right = numTriangles;
    bminLeft = bminRight = bmin;
    bmaxLeft = bmaxRight = bmax;

    for (int i = 0; i < bins->binSize - 1; ++i) {
      left += bins->bin[0 * (3 * bins->binSize) + j * bins->binSize + i];
      right -= bins->bin[1 * (3 * bins->binSize) + j * bins->binSize + i];

      assert(left <= numTriangles);
      assert(right <= numTriangles);

      //
      // Split pos bmin + (i + 1) * (bsize / BIN_SIZE)
      // +1 for i since we want a position on right side of the cell.
      //

      pos = bmin[j] + (i + 0.5) * bstep[j];
      bmaxLeft[j] = pos;
      bminRight[j] = pos;

      saLeft = CalculateSurfaceArea(bminLeft, bmaxLeft);
      saRight = CalculateSurfaceArea(bminRight, bmaxRight);

      float cost =
          SAH(left, saLeft, right, saRight, invSaTotal, costTaabb, costTtri);
      if (cost < minCost[j]) {
        //
        // Update the min cost
        //
        minCost[j] = cost;
        minCostPos = pos;
        // minCostAxis = j;
      }
    }

    cutPos[j] = minCostPos;
  }

  // cutAxis = minCostAxis;
  // cutPos = minCostPos;

  // Find min cost axis
  float cost = minCost[0];
  minCostAxis = 0;
  if (cost > minCost[1]) {
    minCostAxis = 1;
    cost = minCost[1];
  }
  if (cost > minCost[2]) {
    minCostAxis = 2;
    cost = minCost[2];
  }

  return true;
}

class SAHPred : public std::unary_function<unsigned int, bool> {
public:
  SAHPred(int axis, float pos, const float *vertices, const unsigned int *faces)
      : axis_(axis), pos_(pos), vertices_(vertices), faces_(faces) {}

  bool operator()(unsigned int i) const {
    int axis = axis_;
    float pos = pos_;

    unsigned int i0 = faces_[3 * i + 0];
    unsigned int i1 = faces_[3 * i + 1];
    unsigned int i2 = faces_[3 * i + 2];

    float3 p0(&vertices_[3 * i0]);
    float3 p1(&vertices_[3 * i1]);
    float3 p2(&vertices_[3 * i2]);

    float center = p0[axis] + p1[axis] + p2[axis];

    return (center < pos * 3.0f);
  }

private:
  int axis_;
  float pos_;
  const float *vertices_;
  const unsigned int *faces_;
};

#ifdef _OPENMP
void ComputeBoundingBoxOMP(float3 &bmin, float3 &bmax, const float *vertices,
                           const unsigned int *faces, unsigned int *indices,
                           unsigned int leftIndex, unsigned int rightIndex,
                           float epsScale) {
  const float kEPS = std::numeric_limits<float>::epsilon() * epsScale;

  long long i = leftIndex;
  long long idx = indices[i];
  long long n = rightIndex - leftIndex;
  bmin[0] = vertices[3 * faces[3 * idx + 0] + 0] - kEPS;
  bmin[1] = vertices[3 * faces[3 * idx + 0] + 1] - kEPS;
  bmin[2] = vertices[3 * faces[3 * idx + 0] + 2] - kEPS;
  bmax[0] = vertices[3 * faces[3 * idx + 0] + 0] + kEPS;
  bmax[1] = vertices[3 * faces[3 * idx + 0] + 1] + kEPS;
  bmax[2] = vertices[3 * faces[3 * idx + 0] + 2] + kEPS;

  float local_bmin[3] = {bmin[0], bmin[1], bmin[2]};
  float local_bmax[3] = {bmax[0], bmax[1], bmax[2]};

#pragma omp parallel firstprivate(local_bmin, local_bmax) if (n > (1024 * 128))
  {

#pragma omp for
    for (i = leftIndex; i < rightIndex; i++) { // for each faces
      size_t idx = indices[i];
      for (int j = 0; j < 3; j++) { // for each face vertex
        size_t fid = faces[3 * idx + j];
        for (int k = 0; k < 3; k++) { // xyz
          float minval = vertices[3 * fid + k] - kEPS;
          float maxval = vertices[3 * fid + k] + kEPS;
          if (local_bmin[k] > minval)
            local_bmin[k] = minval;
          if (local_bmax[k] < maxval)
            local_bmax[k] = maxval;
        }
      }
    }

#pragma omp critical
    {
      for (int k = 0; k < 3; k++) {

        if (local_bmin[k] < bmin[k]) {
          {
            if (local_bmin[k] < bmin[k])
              bmin[k] = local_bmin[k];
          }
        }

        if (local_bmax[k] > bmax[k]) {
          {
            if (local_bmax[k] > bmax[k])
              bmax[k] = local_bmax[k];
          }
        }
      }
    }
  }
}
#endif

void ComputeBoundingBox(float3 &bmin, float3 &bmax, const float *vertices,
                        const unsigned int *faces, unsigned int *indices,
                        unsigned int leftIndex, unsigned int rightIndex,
                        float epsScale) {
  const float kEPS = std::numeric_limits<float>::epsilon() * epsScale;

  long long i = leftIndex;
  long long idx = indices[i];
  bmin[0] = vertices[3 * faces[3 * idx + 0] + 0] - kEPS;
  bmin[1] = vertices[3 * faces[3 * idx + 0] + 1] - kEPS;
  bmin[2] = vertices[3 * faces[3 * idx + 0] + 2] - kEPS;
  bmax[0] = vertices[3 * faces[3 * idx + 0] + 0] + kEPS;
  bmax[1] = vertices[3 * faces[3 * idx + 0] + 1] + kEPS;
  bmax[2] = vertices[3 * faces[3 * idx + 0] + 2] + kEPS;

  float local_bmin[3] = {bmin[0], bmin[1], bmin[2]};
  float local_bmax[3] = {bmax[0], bmax[1], bmax[2]};

  {

    for (i = leftIndex; i < rightIndex; i++) { // for each faces
      size_t idx = indices[i];
      for (int j = 0; j < 3; j++) { // for each face vertex
        size_t fid = faces[3 * idx + j];
        for (int k = 0; k < 3; k++) { // xyz
          float minval = vertices[3 * fid + k] - kEPS;
          float maxval = vertices[3 * fid + k] + kEPS;
          if (local_bmin[k] > minval)
            local_bmin[k] = minval;
          if (local_bmax[k] < maxval)
            local_bmax[k] = maxval;
        }
      }
    }

    for (int k = 0; k < 3; k++) {
      bmin[k] = local_bmin[k];
      bmax[k] = local_bmax[k];
    }
  }
}

void GetBoundingBox(float3 &bmin, float3 &bmax, std::vector<BBox> &bboxes,
                    unsigned int *indices, unsigned int leftIndex,
                    unsigned int rightIndex, float epsScale) {
  const float kEPS = std::numeric_limits<float>::epsilon() * epsScale;

  long long i = leftIndex;
  long long idx = indices[i];
  bmin[0] = bboxes[idx].bmin[0] - kEPS;
  bmin[1] = bboxes[idx].bmin[1] - kEPS;
  bmin[2] = bboxes[idx].bmin[2] - kEPS;
  bmax[0] = bboxes[idx].bmax[0] + kEPS;
  bmax[1] = bboxes[idx].bmax[1] + kEPS;
  bmax[2] = bboxes[idx].bmax[2] + kEPS;

  float local_bmin[3] = {bmin[0], bmin[1], bmin[2]};
  float local_bmax[3] = {bmax[0], bmax[1], bmax[2]};

  {

    for (i = leftIndex; i < rightIndex; i++) { // for each faces
      size_t idx = indices[i];

      for (int k = 0; k < 3; k++) { // xyz
        float minval = bboxes[idx].bmin[k] - kEPS;
        float maxval = bboxes[idx].bmax[k] + kEPS;
        if (local_bmin[k] > minval)
          local_bmin[k] = minval;
        if (local_bmax[k] < maxval)
          local_bmax[k] = maxval;
      }
    }

    for (int k = 0; k < 3; k++) {
      bmin[k] = local_bmin[k];
      bmax[k] = local_bmax[k];
    }
  }
}

//
// --
//

#if NANORT_ENABLE_PARALLEL_BUILD
unsigned int BVHAccel::BuildShallowTree(std::vector<BVHNode> &outNodes,
                                        const float *vertices,
                                        const unsigned int *faces,
                                        unsigned int leftIdx,
                                        unsigned int rightIdx, int depth,
                                        int maxShallowDepth, float epsScale) {
  assert(leftIdx <= rightIdx);

  unsigned int offset = outNodes.size();

  if (stats_.maxTreeDepth < depth) {
    stats_.maxTreeDepth = depth;
  }

  float3 bmin, bmax;
  ComputeBoundingBox(bmin, bmax, vertices, faces, &indices_.at(0), leftIdx,
                     rightIdx, epsScale);

  long long n = rightIdx - leftIdx;
  if ((n < options_.minLeafPrimitives) || (depth >= options_.maxTreeDepth)) {
    // Create leaf node.
    BVHNode leaf;

    leaf.bmin[0] = bmin[0];
    leaf.bmin[1] = bmin[1];
    leaf.bmin[2] = bmin[2];

    leaf.bmax[0] = bmax[0];
    leaf.bmax[1] = bmax[1];
    leaf.bmax[2] = bmax[2];

    assert(leftIdx < std::numeric_limits<unsigned int>::max());

    leaf.flag = 1; // leaf
    leaf.data[0] = n;
    leaf.data[1] = (unsigned int)leftIdx;

    outNodes.push_back(leaf); // atomic update

    stats_.numLeafNodes++;

    return offset;
  }

  //
  // Create branch node.
  //
  if (depth >= maxShallowDepth) {

    // Delay to build tree
    ShallowNodeInfo info;
    info.leftIdx = leftIdx;
    info.rightIdx = rightIdx;
    info.offset = offset;
    shallowNodeInfos_.push_back(info);

    // Add dummy node.
    BVHNode node;
    node.axis = -1;
    node.flag = -1;
    outNodes.push_back(node);

    return offset;

  } else {

    //
    // Compute SAH and find best split axis and position
    //
    int minCutAxis = 0;
    float cutPos[3] = {0.0, 0.0, 0.0};

    BinBuffer bins(options_.binSize);
    ContributeBinBuffer(&bins, bmin, bmax, vertices, faces, &indices_.at(0),
                        leftIdx, rightIdx, epsScale);
    FindCutFromBinBuffer(cutPos, minCutAxis, &bins, bmin, bmax, n,
                         options_.costTaabb, epsScale);

    // Try all 3 axis until good cut position avaiable.
    unsigned int midIdx;
    int cutAxis = minCutAxis;
    for (int axisTry = 0; axisTry < 1; axisTry++) {

      unsigned int *begin = &indices_[leftIdx];
      unsigned int *end = &indices_[rightIdx - 1] + 1; // mimics end() iterator.
      unsigned int *mid = 0;

      // try minCutAxis first.
      cutAxis = (minCutAxis + axisTry) % 3;

      //
      // Split at (cutAxis, cutPos)
      // indices_ will be modified.
      //
      mid = std::partition(begin, end,
                           SAHPred(cutAxis, cutPos[cutAxis], vertices, faces));

      midIdx = leftIdx + (mid - begin);
      if ((midIdx == leftIdx) || (midIdx == rightIdx)) {

        // Can't split well.
        // Switch to object median(which may create unoptimized tree, but
        // stable)
        midIdx = leftIdx + (n >> 1);

        // Try another axis if there's axis to try.

      } else {

        // Found good cut. exit loop.
        break;
      }
    }

    BVHNode node;
    node.axis = cutAxis;
    node.flag = 0; // 0 = branch

    outNodes.push_back(node);

    unsigned int leftChildIndex = 0;
    unsigned int rightChildIndex = 0;

    leftChildIndex =
        BuildShallowTree(outNodes, vertices, faces, leftIdx, midIdx, depth + 1,
                         maxShallowDepth, epsScale);

    rightChildIndex =
        BuildShallowTree(outNodes, vertices, faces, midIdx, rightIdx, depth + 1,
                         maxShallowDepth, epsScale);

    if ((leftChildIndex != (unsigned int)(-1)) &&
        (rightChildIndex != (unsigned int)(-1))) {
      outNodes[offset].data[0] = leftChildIndex;
      outNodes[offset].data[1] = rightChildIndex;

      outNodes[offset].bmin[0] = bmin[0];
      outNodes[offset].bmin[1] = bmin[1];
      outNodes[offset].bmin[2] = bmin[2];

      outNodes[offset].bmax[0] = bmax[0];
      outNodes[offset].bmax[1] = bmax[1];
      outNodes[offset].bmax[2] = bmax[2];
    } else {
      if ((leftChildIndex == (unsigned int)(-1)) &&
          (rightChildIndex != (unsigned int)(-1))) {
        fprintf(stderr, "??? : %u, %u\n", leftChildIndex, rightChildIndex);
        exit(-1);
      } else if ((leftChildIndex != (unsigned int)(-1)) &&
                 (rightChildIndex == (unsigned int)(-1))) {
        fprintf(stderr, "??? : %u, %u\n", leftChildIndex, rightChildIndex);
        exit(-1);
      }
    }
  }

  stats_.numBranchNodes++;

  return offset;
}
#endif

size_t BVHAccel::BuildTree(BVHBuildStatistics &outStat,
                           std::vector<BVHNode> &outNodes,
                           const float *vertices, const unsigned int *faces,
                           unsigned int leftIdx, unsigned int rightIdx,
                           int depth, float epsScale) {
  assert(leftIdx <= rightIdx);

  size_t offset = outNodes.size();

  if (outStat.maxTreeDepth < depth) {
    outStat.maxTreeDepth = depth;
  }

  float3 bmin, bmax;
  if (!bboxes_.empty()) {
    GetBoundingBox(bmin, bmax, bboxes_, &indices_.at(0), leftIdx, rightIdx,
                   epsScale);
  } else {
    ComputeBoundingBox(bmin, bmax, vertices, faces, &indices_.at(0), leftIdx,
                       rightIdx, epsScale);
  }

  long long n = rightIdx - leftIdx;
  if ((n < options_.minLeafPrimitives) || (depth >= options_.maxTreeDepth)) {
    // Create leaf node.
    BVHNode leaf;

    leaf.bmin[0] = bmin[0];
    leaf.bmin[1] = bmin[1];
    leaf.bmin[2] = bmin[2];

    leaf.bmax[0] = bmax[0];
    leaf.bmax[1] = bmax[1];
    leaf.bmax[2] = bmax[2];

    assert(leftIdx < std::numeric_limits<unsigned int>::max());

    leaf.flag = 1; // leaf
    leaf.data[0] = n;
    leaf.data[1] = (unsigned int)leftIdx;

    outNodes.push_back(leaf); // atomic update

    outStat.numLeafNodes++;

    return offset;
  }

  //
  // Create branch node.
  //

  //
  // Compute SAH and find best split axis and position
  //
  int minCutAxis = 0;
  float cutPos[3] = {0.0, 0.0, 0.0};

  BinBuffer bins(options_.binSize);
  ContributeBinBuffer(&bins, bmin, bmax, vertices, faces, &indices_.at(0),
                      leftIdx, rightIdx, epsScale);
  FindCutFromBinBuffer(cutPos, minCutAxis, &bins, bmin, bmax, n,
                       options_.costTaabb, epsScale);

  // Try all 3 axis until good cut position avaiable.
  unsigned int midIdx;
  int cutAxis = minCutAxis;
  for (int axisTry = 0; axisTry < 1; axisTry++) {

    unsigned int *begin = &indices_[leftIdx];
    unsigned int *end = &indices_[rightIdx - 1] + 1; // mimics end() iterator.
    unsigned int *mid = 0;

    // try minCutAxis first.
    cutAxis = (minCutAxis + axisTry) % 3;

    //
    // Split at (cutAxis, cutPos)
    // indices_ will be modified.
    //
    mid = std::partition(begin, end,
                         SAHPred(cutAxis, cutPos[cutAxis], vertices, faces));

    midIdx = leftIdx + (mid - begin);
    if ((midIdx == leftIdx) || (midIdx == rightIdx)) {

      // Can't split well.
      // Switch to object median(which may create unoptimized tree, but
      // stable)
      midIdx = leftIdx + (n >> 1);

      // Try another axis if there's axis to try.

    } else {

      // Found good cut. exit loop.
      break;
    }
  }

  BVHNode node;
  node.axis = cutAxis;
  node.flag = 0; // 0 = branch

  outNodes.push_back(node); // atomic update

  unsigned int leftChildIndex = 0;
  unsigned int rightChildIndex = 0;

  leftChildIndex = BuildTree(outStat, outNodes, vertices, faces, leftIdx,
                             midIdx, depth + 1, epsScale);

  rightChildIndex = BuildTree(outStat, outNodes, vertices, faces, midIdx,
                              rightIdx, depth + 1, epsScale);

  {
    outNodes[offset].data[0] = leftChildIndex;
    outNodes[offset].data[1] = rightChildIndex;

    outNodes[offset].bmin[0] = bmin[0];
    outNodes[offset].bmin[1] = bmin[1];
    outNodes[offset].bmin[2] = bmin[2];

    outNodes[offset].bmax[0] = bmax[0];
    outNodes[offset].bmax[1] = bmax[1];
    outNodes[offset].bmax[2] = bmax[2];
  }

  outStat.numBranchNodes++;

  return offset;
}

bool BVHAccel::Build(const float *vertices, const unsigned int *faces,
                     unsigned int numFaces, const BVHBuildOptions &options) {
  options_ = options;
  stats_ = BVHBuildStatistics();

  assert(options_.binSize > 1);

  size_t n = numFaces;

  //
  // 1. Create triangle indices(this will be permutated in BuildTree)
  //
  indices_.resize(n);

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (long long i = 0; i < (long long)n; i++) {
    indices_[i] = i;
  }

  //
  // 2. Compute bounding box to find scene scale.
  //
  float epsScale = 1.0f;
  float3 bmin, bmax;
  if (options.cacheBBox) {

    bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
    bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

    bboxes_.resize(n);
    for (size_t i = 0; i < n; i++) { // for each faces
      size_t idx = indices_[i];

      BBox bbox;
      for (int j = 0; j < 3; j++) { // for each face vertex
        size_t fid = faces[3 * idx + j];
        for (int k = 0; k < 3; k++) { // xyz
          float minval = vertices[3 * fid + k];
          float maxval = vertices[3 * fid + k];
          if (bbox.bmin[k] > minval) {
            bbox.bmin[k] = minval;
          }
          if (bbox.bmax[k] < maxval) {
            bbox.bmax[k] = maxval;
          }
        }
      }

      bboxes_[idx] = bbox;

      for (int k = 0; k < 3; k++) { // xyz
        if (bmin[k] > bbox.bmin[k]) {
          bmin[k] = bbox.bmin[k];
        }
        if (bmax[k] < bbox.bmax[k]) {
          bmax[k] = bbox.bmax[k];
        }
      }
    }

  } else {

#ifdef _OPENMP
    ComputeBoundingBoxOMP(bmin, bmax, vertices, faces, &indices_.at(0), 0, n,
                          epsScale);
#else
    ComputeBoundingBox(bmin, bmax, vertices, faces, &indices_.at(0), 0, n,
                       epsScale);
#endif
  }

  // Find max
  float3 bsize = bmax - bmin;
  epsScale = std::abs(bsize[0]);
  if (epsScale < std::abs(bsize[1])) {
    epsScale = std::abs(bsize[1]);
  }
  if (epsScale < std::abs(bsize[2])) {
    epsScale = std::abs(bsize[2]);
  }

//
// 3. Build tree
//
#ifdef _OPENMP
#if NANORT_ENABLE_PARALLEL_BUILD

  // Do parallel build for enoughly large dataset.
  if (n > options.minPrimitivesForParallelBuild) {

    BuildShallowTree(nodes_, vertices, faces, 0, n, /* root depth */ 0,
                     options.shallowDepth, epsScale); // [0, n)

    assert(shallowNodeInfos_.size() > 0);

    // Build deeper tree in parallel
    std::vector<std::vector<BVHNode> > local_nodes(shallowNodeInfos_.size());
    std::vector<BVHBuildStatistics> local_stats(shallowNodeInfos_.size());

#pragma omp parallel for
    for (int i = 0; i < (int)shallowNodeInfos_.size(); i++) {
      unsigned int leftIdx = shallowNodeInfos_[i].leftIdx;
      unsigned int rightIdx = shallowNodeInfos_[i].rightIdx;
      BuildTree(local_stats[i], local_nodes[i], vertices, faces, leftIdx,
                rightIdx, options.shallowDepth, epsScale);
    }

    // Join local nodes
    for (int i = 0; i < (int)local_nodes.size(); i++) {

      assert(!local_nodes[i].empty());
      size_t offset = nodes_.size();

      // Add offset to child index(for branch node).
      for (size_t j = 0; j < local_nodes[i].size(); j++) {
        if (local_nodes[i][j].flag == 0) { // branch
          local_nodes[i][j].data[0] += offset - 1;
          local_nodes[i][j].data[1] += offset - 1;
        }
      }

      // replace
      nodes_[shallowNodeInfos_[i].offset] = local_nodes[i][0];

      // Skip root element of the local node.
      nodes_.insert(nodes_.end(), local_nodes[i].begin() + 1,
                    local_nodes[i].end());
    }

    // Join statistics
    for (int i = 0; i < (int)local_nodes.size(); i++) {
      stats_.maxTreeDepth =
          std::max(stats_.maxTreeDepth, local_stats[i].maxTreeDepth);
      stats_.numLeafNodes += local_stats[i].numLeafNodes;
      stats_.numBranchNodes += local_stats[i].numBranchNodes;
    }

  } else {
    BuildTree(stats_, nodes_, vertices, faces, 0, n, /* root depth */ 0,
              epsScale); // [0, n)
  }

#else // !NANORT_ENABLE_PARALLEL_BUILD
  {
    BuildTree(stats_, nodes_, vertices, faces, 0, n, /* root depth */ 0,
              epsScale); // [0, n)
  }
#endif
#else // !_OPENMP
  {
    BuildTree(stats_, nodes_, vertices, faces, 0, n, /* root depth */ 0,
              epsScale); // [0, n)
  }
#endif

  stats_.epsScale = epsScale;
  epsScale_ = epsScale;

  return true;
}

bool BVHAccel::Dump(const char *filename) {
  FILE *fp = fopen(filename, "wb");
  if (!fp) {
    fprintf(stderr, "[BVHAccel] Cannot write a file: %s\n", filename);
    return false;
  }

  unsigned long long numNodes = nodes_.size();
  assert(nodes_.size() > 0);

  unsigned long long numIndices = indices_.size();

  size_t r = 0;
  r = fwrite(&numNodes, sizeof(unsigned long long), 1, fp);
  assert(r == 1);

  r = fwrite(&nodes_.at(0), sizeof(BVHNode), numNodes, fp);
  assert(r == numNodes);

  r = fwrite(&numIndices, sizeof(unsigned long long), 1, fp);
  assert(r == 1);

  r = fwrite(&indices_.at(0), sizeof(unsigned int), numIndices, fp);
  assert(r == numIndices);

  fclose(fp);

  return true;
}

bool BVHAccel::Load(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Cannot open file: %s\n", filename);
    return false;
  }

  unsigned long long numNodes;
  unsigned long long numIndices;

  size_t r = 0;
  r = fread(&numNodes, sizeof(unsigned long long), 1, fp);
  assert(r == 1);
  assert(numNodes > 0);

  nodes_.resize(numNodes);
  r = fread(&nodes_.at(0), sizeof(BVHNode), numNodes, fp);
  assert(r == numNodes);

  r = fread(&numIndices, sizeof(unsigned long long), 1, fp);
  assert(r == 1);

  indices_.resize(numIndices);

  r = fread(&indices_.at(0), sizeof(unsigned int), numIndices, fp);
  assert(r == numIndices);

  fclose(fp);

  return true;
}

namespace {

const int kMaxStackDepth = 512;

inline bool IntersectRayAABB(float &tminOut, // [out]
                             float &tmaxOut, // [out]
                             float maxT, float bmin[3], float bmax[3],
                             float3 rayOrg, float3 rayInvDir,
                             int rayDirSign[3]) {
  float tmin, tmax;

  const float min_x = rayDirSign[0] ? bmax[0] : bmin[0];
  const float min_y = rayDirSign[1] ? bmax[1] : bmin[1];
  const float min_z = rayDirSign[2] ? bmax[2] : bmin[2];
  const float max_x = rayDirSign[0] ? bmin[0] : bmax[0];
  const float max_y = rayDirSign[1] ? bmin[1] : bmax[1];
  const float max_z = rayDirSign[2] ? bmin[2] : bmax[2];

  // X
  const float tmin_x = (min_x - rayOrg[0]) * rayInvDir[0];
  const float tmax_x = (max_x - rayOrg[0]) * rayInvDir[0];

  // Y
  const float tmin_y = (min_y - rayOrg[1]) * rayInvDir[1];
  const float tmax_y = (max_y - rayOrg[1]) * rayInvDir[1];

  tmin = (tmin_x > tmin_y) ? tmin_x : tmin_y;
  tmax = (tmax_x < tmax_y) ? tmax_x : tmax_y;

  // Z
  const float tmin_z = (min_z - rayOrg[2]) * rayInvDir[2];
  const float tmax_z = (max_z - rayOrg[2]) * rayInvDir[2];

  tmin = (tmin > tmin_z) ? tmin : tmin_z;
  tmax = (tmax < tmax_z) ? tmax : tmax_z;

  //
  // Hit include (tmin == tmax) edge case(hit 2D plane).
  //
  if ((tmax > 0.0) && (tmin <= tmax) && (tmin <= maxT)) {

    tminOut = tmin;
    tmaxOut = tmax;

    return true;
  }

  return false; // no hit
}

inline bool TriangleIsect(float &tInOut, float &uOut, float &vOut,
                          const float3 &v0, const float3 &v1, const float3 &v2,
                          const float3 &rayOrg, const float3 &rayDir,
                          float epsScale) {
  const float kEPS = std::numeric_limits<float>::epsilon() * epsScale;

  float3 p0(v0[0], v0[1], v0[2]);
  float3 p1(v1[0], v1[1], v1[2]);
  float3 p2(v2[0], v2[1], v2[2]);
  float3 e1, e2;
  float3 p, s, q;

  e1 = p1 - p0;
  e2 = p2 - p0;

  p = vcross(rayDir, e2);

  float invDet;
  float det = vdot(e1, p);
  if (std::abs(det) < kEPS) { // no-cull
    return false;
  }

  invDet = 1.0 / det;

  s = rayOrg - p0;
  q = vcross(s, e1);

  float u = vdot(s, p) * invDet;
  float v = vdot(q, rayDir) * invDet;
  float t = vdot(e2, q) * invDet;

  if (u < 0.0 || u > 1.0)
    return false;
  if (v < 0.0 || u + v > 1.0)
    return false;
  if (t < 0.0 || t > tInOut)
    return false;

  tInOut = t;
  uOut = u;
  vOut = v;

  return true;
}

bool TestLeafNode(Intersection &isect, // [inout]
                  const BVHNode &node, const std::vector<unsigned int> &indices,
                  const float *vertices, const unsigned int *faces,
                  const Ray &ray, float epsScale) {
  bool hit = false;

  unsigned int numTriangles = node.data[0];
  unsigned int offset = node.data[1];

  float t = isect.t; // current hit distance

  float3 rayOrg;
  rayOrg[0] = ray.org[0];
  rayOrg[1] = ray.org[1];
  rayOrg[2] = ray.org[2];

  float3 rayDir;
  rayDir[0] = ray.dir[0];
  rayDir[1] = ray.dir[1];
  rayDir[2] = ray.dir[2];

  for (unsigned int i = 0; i < numTriangles; i++) {
    int faceIdx = indices[i + offset];

    int f0 = faces[3 * faceIdx + 0];
    int f1 = faces[3 * faceIdx + 1];
    int f2 = faces[3 * faceIdx + 2];

    float3 v0, v1, v2;
    v0[0] = vertices[3 * f0 + 0];
    v0[1] = vertices[3 * f0 + 1];
    v0[2] = vertices[3 * f0 + 2];

    v1[0] = vertices[3 * f1 + 0];
    v1[1] = vertices[3 * f1 + 1];
    v1[2] = vertices[3 * f1 + 2];

    v2[0] = vertices[3 * f2 + 0];
    v2[1] = vertices[3 * f2 + 1];
    v2[2] = vertices[3 * f2 + 2];

    float u, v;
    if (TriangleIsect(t, u, v, v0, v1, v2, rayOrg, rayDir, epsScale)) {
      // Update isect state
      isect.t = t;
      isect.u = u;
      isect.v = v;
      isect.faceID = faceIdx;
      hit = true;
    }
  }

  return hit;
}

#if 0
void BuildIntersection(NanoRTIntersection &isect, const Mesh *mesh, Ray &ray) {
  // face index
  const unsigned int *faces = mesh->faces;
  const float *vertices = mesh->vertices;
  isect.f0 = faces[3 * isect.faceID + 0];
  isect.f1 = faces[3 * isect.faceID + 1];
  isect.f2 = faces[3 * isect.faceID + 2];

  float3 p0, p1, p2;
  p0[0] = vertices[3 * isect.f0 + 0];
  p0[1] = vertices[3 * isect.f0 + 1];
  p0[2] = vertices[3 * isect.f0 + 2];
  p1[0] = vertices[3 * isect.f1 + 0];
  p1[1] = vertices[3 * isect.f1 + 1];
  p1[2] = vertices[3 * isect.f1 + 2];
  p2[0] = vertices[3 * isect.f2 + 0];
  p2[1] = vertices[3 * isect.f2 + 1];
  p2[2] = vertices[3 * isect.f2 + 2];

  // calc shading point.
  isect.position[0] = ray.org[0] + isect.t * ray.dir[0];
  isect.position[1] = ray.org[1] + isect.t * ray.dir[1];
  isect.position[2] = ray.org[2] + isect.t * ray.dir[2];

  // calc geometric normal.
  float3 p10 = p1 - p0;
  float3 p20 = p2 - p0;
  float3 n = vcross(p10, p20);
  n.normalize();

  isect.geometricNormal = n;

  if (mesh->facevarying_normals) {
    const float* normals = mesh->facevarying_normals;
    float3 n0, n1, n2;

    n0[0] = normals[9 * isect.faceID + 0];
    n0[1] = normals[9 * isect.faceID + 1];
    n0[2] = normals[9 * isect.faceID + 2];
    n1[0] = normals[9 * isect.faceID + 3];
    n1[1] = normals[9 * isect.faceID + 4];
    n1[2] = normals[9 * isect.faceID + 5];
    n2[0] = normals[9 * isect.faceID + 6];
    n2[1] = normals[9 * isect.faceID + 7];
    n2[2] = normals[9 * isect.faceID + 8];

    // lerp
    isect.normal[0] = (1.0 - isect.u - isect.v) * n0[0] + isect.u * n1[0] + isect.v * n2[0];
    isect.normal[1] = (1.0 - isect.u - isect.v) * n0[1] + isect.u * n1[1] + isect.v * n2[1];
    isect.normal[2] = (1.0 - isect.u - isect.v) * n0[2] + isect.u * n1[2] + isect.v * n2[2];
  } else {
    isect.normal = n;
  }

  if (mesh->facevarying_uvs) {
    const float* uvs = mesh->facevarying_uvs;
    float3 st0, st1, st2;

    st0[0] = uvs[6 * isect.faceID + 0];
    st0[1] = uvs[6 * isect.faceID + 1];
    st1[0] = uvs[6 * isect.faceID + 2];
    st1[1] = uvs[6 * isect.faceID + 3];
    st2[0] = uvs[6 * isect.faceID + 4];
    st2[1] = uvs[6 * isect.faceID + 5];

    // lerp
    isect.texcoord[0] = (1.0 - isect.u - isect.v) * st0[0] + isect.u * st1[0] + isect.v * st2[0];
    isect.texcoord[1] = (1.0 - isect.u - isect.v) * st0[1] + isect.u * st1[1] + isect.v * st2[1];
  }

}
#endif

} // namespace

bool BVHAccel::Traverse(Intersection &isect, const float *vertices,
                        const unsigned int *faces, Ray &ray) {
  float hitT = std::numeric_limits<float>::max(); // far = no hit.

  int nodeStackIndex = 0;
  int nodeStack[512];
  nodeStack[0] = 0;

  // Init isect info as no hit
  isect.t = hitT;
  isect.u = 0.0;
  isect.v = 0.0;
  isect.faceID = -1;

  int dirSign[3];
  dirSign[0] = ray.dir[0] < 0.0 ? 1 : 0;
  dirSign[1] = ray.dir[1] < 0.0 ? 1 : 0;
  dirSign[2] = ray.dir[2] < 0.0 ? 1 : 0;

  // @fixme { Check edge case; i.e., 1/0 }
  float3 rayInvDir;
  rayInvDir[0] = 1.0 / ray.dir[0];
  rayInvDir[1] = 1.0 / ray.dir[1];
  rayInvDir[2] = 1.0 / ray.dir[2];

  float3 rayOrg;
  rayOrg[0] = ray.org[0];
  rayOrg[1] = ray.org[1];
  rayOrg[2] = ray.org[2];

  float minT, maxT;
  while (nodeStackIndex >= 0) {
    int index = nodeStack[nodeStackIndex];
    BVHNode &node = nodes_[index];

    nodeStackIndex--;

    bool hit = IntersectRayAABB(minT, maxT, hitT, node.bmin, node.bmax, rayOrg,
                                rayInvDir, dirSign);

    if (node.flag == 0) { // branch node

      if (hit) {

        int orderNear = dirSign[node.axis];
        int orderFar = 1 - orderNear;

        // Traverse near first.
        nodeStack[++nodeStackIndex] = node.data[orderFar];
        nodeStack[++nodeStackIndex] = node.data[orderNear];
      }

    } else { // leaf node
      if (hit) {
        if (TestLeafNode(isect, node, indices_, vertices, faces, ray,
                         epsScale_)) {
          hitT = isect.t;
        }
      }
    }
  }

  assert(nodeStackIndex < kMaxStackDepth);

  if (isect.t < std::numeric_limits<float>::max()) {
    // BuildIntersection(isect, mesh, ray);
    return true;
  }

  return false;
}

} // namespace

#endif

#endif // __NANORT_H__
