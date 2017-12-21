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

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wvariadic-macros"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#if __has_warning("-Wcast-qual")
#pragma clang diagnostic ignored "-Wcast-qual"
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#endif

#ifdef _WIN32
#  define RTCORE_API extern "C" __declspec(dllexport)
#else
#  define RTCORE_API extern "C" __attribute__ ((visibility ("default")))
#endif

#include "embree2/rtcore.h"
#include "embree2/rtcore_ray.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cassert>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nanosg.h"

#include <stdint.h>  // Use cstint for C++11 compiler.

namespace nanort_embree2 {

template <typename T>
inline void lerp(T dst[3], const T v0[3], const T v1[3], const T v2[3], float u,
                 float v) {
  dst[0] = (static_cast<T>(1.0) - u - v) * v0[0] + u * v1[0] + v * v2[0];
  dst[1] = (static_cast<T>(1.0) - u - v) * v0[1] + u * v1[1] + v * v2[1];
  dst[2] = (static_cast<T>(1.0) - u - v) * v0[2] + u * v1[2] + v * v2[2];
}

template <typename T>
inline T vlength(const T v[3]) {
  const T d = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (std::fabs(d) > std::numeric_limits<T>::epsilon()) {
    return std::sqrt(d);
  } else {
    return static_cast<T>(0.0);
  }
}

template <typename T>
inline void vnormalize(T dst[3], const T v[3]) {
  dst[0] = v[0];
  dst[1] = v[1];
  dst[2] = v[2];
  const T len = vlength(v);
  if (std::fabs(len) > std::numeric_limits<T>::epsilon()) {
    const T inv_len = static_cast<T>(1.0) / len;
    dst[0] *= inv_len;
    dst[1] *= inv_len;
    dst[2] *= inv_len;
  }
}

template <typename T>
inline void vcross(T dst[3], const T a[3], const T b[3]) {
  dst[0] = a[1] * b[2] - a[2] * b[1];
  dst[1] = a[2] * b[0] - a[0] * b[2];
  dst[2] = a[0] * b[1] - a[1] * b[0];
}

template <typename T>
inline void vsub(T dst[3], const T a[3], const T b[3]) {
  dst[0] = a[0] - b[0];
  dst[1] = a[1] - b[1];
  dst[2] = a[2] - b[2];
}

template <typename T>
inline void calculate_normal(T Nn[3], const T v0[3], const T v1[3],
                             const T v2[3]) {
  T v10[3];
  T v20[3];

  vsub(v10, v1, v0);
  vsub(v20, v2, v0);

  T N[3];
  vcross(N, v10, v20);  // CCW
  // vcross(N, v20, v10); // CC
  vnormalize(Nn, N);
}

template <typename T = float>
class TriMesh {
 public:
  explicit TriMesh(const size_t num_triangles, const size_t num_vertices) {
    // Embree uses 16 bytes stride
    stride = sizeof(float) * 4;
    vertices.resize(num_vertices * 4);
    faces.resize(num_triangles * 3);
  }

  ~TriMesh() {}

  std::string name;

  size_t stride;
  std::vector<T> vertices;          /// [xyz] * num_vertices
  std::vector<unsigned int> faces;  /// triangle x num_faces

  T pivot_xform[4][4];

  // --- Required methods in Scene::Traversal. ---

  ///
  /// Get the geometric normal and the shading normal at `face_idx' th face.
  ///
  void GetNormal(T Ng[3], T Ns[3], const unsigned int face_idx, const T u,
                 const T v) const {
    (void)u;
    (void)v;

    // Compute geometric normal.
    unsigned int f0, f1, f2;
    T v0[3], v1[3], v2[3];

    f0 = faces[3 * face_idx + 0];
    f1 = faces[3 * face_idx + 1];
    f2 = faces[3 * face_idx + 2];

    v0[0] = vertices[3 * f0 + 0];
    v0[1] = vertices[3 * f0 + 1];
    v0[2] = vertices[3 * f0 + 2];

    v1[0] = vertices[3 * f1 + 0];
    v1[1] = vertices[3 * f1 + 1];
    v1[2] = vertices[3 * f1 + 2];

    v2[0] = vertices[3 * f2 + 0];
    v2[1] = vertices[3 * f2 + 1];
    v2[2] = vertices[3 * f2 + 2];

    calculate_normal(Ng, v0, v1, v2);

    // Use geometric normal.
    Ns[0] = Ng[0];
    Ns[1] = Ng[1];
    Ns[2] = Ng[2];
  }

  // --- end of required methods in Scene::Traversal. ---
};

///
/// Simple handle resource management.
///
class HandleAllocator {
 public:
  // id = 0 is reserved.
  HandleAllocator() : counter_(1) { (void)_pad_; }
  ~HandleAllocator() {}

  ///
  /// Allocates handle object.
  ///
  uint32_t Allocate() {
    uint32_t handle = 0;

    if (!freeList_.empty()) {
      // Reuse previously issued handle.
      handle = freeList_.back();
      freeList_.pop_back();
      return handle;
    }

    handle = counter_;
    assert(handle >= 1);
    assert(handle < 0xFFFFFFFF);

    counter_++;

    return handle;
  }

  /// Release handle object.
  void Release(uint32_t handle) {
    if (handle == counter_ - 1) {
      if (counter_ > 1) {
        counter_--;
      }
    } else {
      assert(handle >= 1);
      freeList_.push_back(handle);
    }
  }

 private:
  std::vector<uint32_t> freeList_;
  uint32_t counter_;
  uint32_t _pad_;
};

class Scene {
 public:
  Scene(RTCSceneFlags sflags, RTCAlgorithmFlags aflags)
      : scene_flags_(sflags), algorithm_flags_(aflags) {
    (void)scene_flags_;
    (void)algorithm_flags_;
  }

  ~Scene() {
    std::map<uint32_t, TriMesh<float> *>::iterator it(trimesh_map_.begin());
    std::map<uint32_t, TriMesh<float> *>::iterator itEnd(trimesh_map_.end());

    for (; it != itEnd; it++) {
      delete it->second;
    }
  }

  ///
  /// Get scene bounding box.
  ///
  void GetBounds(RTCBounds &bounds) {
    float bmin[3], bmax[3];
    trimesh_scene_.GetBoundingBox(bmin, bmax);
    bounds.lower_x = bmin[0];
    bounds.lower_y = bmin[1];
    bounds.lower_z = bmin[2];

    bounds.upper_x = bmax[0];
    bounds.upper_y = bmax[1];
    bounds.upper_z = bmax[2];
  }

  ///
  ///
  ///
  uint32_t NewTriMesh(size_t num_triangles, size_t num_vertices) {
    uint32_t geom_id = geom_ids_.Allocate();

    TriMesh<float> *trimesh = new TriMesh<float>(num_triangles, num_vertices);

    trimesh_map_[geom_id] = trimesh;

    return geom_id;
  }

  TriMesh<float> *GetTriMesh(const uint32_t geom_id) {
    if (trimesh_map_.find(geom_id) != trimesh_map_.end()) {
      return trimesh_map_[geom_id];
    }
    return NULL;
  }

  size_t NumShapes() { return trimesh_map_.size(); }

  void Build() {
    std::map<uint32_t, TriMesh<float> *>::iterator it(trimesh_map_.begin());
    std::map<uint32_t, TriMesh<float> *>::iterator itEnd(trimesh_map_.end());

    for (; it != itEnd; it++) {
      nanosg::Node<float, TriMesh<float> > node(it->second);

      trimesh_scene_.AddNode(node);
    }

    trimesh_scene_.Commit();
  }

  bool Intersect(nanort::Ray<float> &ray,
                 nanosg::Intersection<float> *isect_out,
                 const bool cull_back_face) {
    return trimesh_scene_.Traverse(ray, isect_out, cull_back_face);
  }

 private:
  RTCSceneFlags scene_flags_;
  RTCAlgorithmFlags algorithm_flags_;
  HandleAllocator geom_ids_;

  nanosg::Scene<float, TriMesh<float> > trimesh_scene_;
  std::vector<nanosg::Node<float, TriMesh<float> > > trimesh_nodes_;

  // Records triangle mesh for geom_id
  std::map<uint32_t, TriMesh<float> *> trimesh_map_;
};

class Device {
 public:
  Device(const std::string &config)
      : config_(config), error_func_(NULL), user_ptr_(NULL) {}

  ~Device() {}

  void SetErrorFunction(RTCErrorFunc2 func, void *user_ptr) {
    error_func_ = func;
    user_ptr_ = user_ptr;
  }

  void AddScene(Scene *scene) { scene_map_[scene] = scene; }

  bool DeleteScene(Scene *scene) {
    if (scene_map_.find(scene) != scene_map_.end()) {
      std::map<const Scene *, Scene *>::iterator it = scene_map_.find(scene);

      scene_map_.erase(it);

      delete scene;
      return true;
    }

    return false;
  }

 private:
  std::string config_;

  std::map<const Scene *, Scene *> scene_map_;

  // Callbacks
  RTCErrorFunc2 error_func_;
  void *user_ptr_;
};

class Context {
 public:
  Context() {}
  ~Context() {
    std::map<const Device *, Device *>::iterator it(device_map_.begin());
    std::map<const Device *, Device *>::iterator itEnd(device_map_.end());

    for (; it != itEnd; it++) {
      delete it->second;
      it->second = NULL;
    }
  }

  Device *NewDevice(const char *config) {
    std::string cfg;
    if (config) {
      cfg = std::string(config);
    }

    Device *device = new Device(cfg);

    device_map_[device] = device;

    return device;
  }

  bool DeleteDevice(Device *device) {
    if (device_map_.find(device) != device_map_.end()) {
      std::map<const Device *, Device *>::iterator it =
          device_map_.find(device);
      device_map_.erase(it);

      delete device;
      return true;
    }
    return false;
  }

  bool DeleteScene(Scene *scene) {
    // Assume scene is assigned to the device uniquely
    std::map<const Device *, Device *>::iterator it(device_map_.begin());
    std::map<const Device *, Device *>::iterator itEnd(device_map_.end());

    for (; it != itEnd; it++) {
      if (it->second->DeleteScene(scene)) {
        return true;
      }
    }

    return false;
  }

  void SetError(const std::string &err) { error_ = err; }

 private:
  std::string error_;
  std::map<const Device *, Device *> device_map_;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#endif

static Context &GetContext() {
  static Context s_ctx;

  return s_ctx;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

// TODO(LTE): Lock to avoid thread-racing.

RTCORE_API RTCDevice rtcNewDevice(const char *cfg = NULL) {
  Device *device = GetContext().NewDevice(cfg);

  return reinterpret_cast<RTCDevice>(device);
}

RTCORE_API void rtcDeleteScene(RTCScene scene) {
  Scene *s = reinterpret_cast<Scene *>(scene);

  bool ret = GetContext().DeleteScene(s);

  if (!ret) {
    std::stringstream ss;
    ss << "Invalid scene : " << scene << std::endl;
    GetContext().SetError(ss.str());
  }
}

RTCORE_API void rtcDeleteDevice(RTCDevice device) {
#if 0
  (void)device;
  std::cout << "TODO: Implement rtcDeleteScene()" << std::endl;
#else
  Device *dev = reinterpret_cast<Device *>(device);

  bool ret = GetContext().DeleteDevice(dev);

  if (!ret) {
    std::stringstream ss;
    ss << "Invalid device : " << device << std::endl;
    GetContext().SetError(ss.str());
  }
#endif
}

RTCORE_API void rtcDeviceSetErrorFunction2(RTCDevice device, RTCErrorFunc2 func,
                                           void *userPtr) {
  Device *ptr = reinterpret_cast<Device *>(device);
  ptr->SetErrorFunction(func, userPtr);
}

RTCORE_API RTCScene rtcDeviceNewScene(RTCDevice device, RTCSceneFlags flags,
                                      RTCAlgorithmFlags aflags) {
  Scene *scene = new Scene(flags, aflags);

  Device *d = reinterpret_cast<Device *>(device);
  d->AddScene(scene);

  return reinterpret_cast<RTCScene>(scene);
}

RTCORE_API void rtcGetBounds(RTCScene scene, RTCBounds &bounds_o) {
  Scene *s = reinterpret_cast<Scene *>(scene);
  s->GetBounds(bounds_o);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif

RTCORE_API void rtcIntersect(RTCScene scene, RTCRay &rtc_ray) {
  Scene *s = reinterpret_cast<Scene *>(scene);

  nanort::Ray<float> ray;

  ray.org[0] = rtc_ray.org[0];
  ray.org[1] = rtc_ray.org[1];
  ray.org[2] = rtc_ray.org[2];

  ray.dir[0] = rtc_ray.dir[0];
  ray.dir[1] = rtc_ray.dir[1];
  ray.dir[2] = rtc_ray.dir[2];

  // TODO(LTE): .time, .mask

  ray.min_t = rtc_ray.tnear;
  ray.max_t = rtc_ray.tfar;

  nanosg::Intersection<float> isect;
  // FIXME(LTE): Read RTC_CONFIG_BACKFACE_CULLING from Embree configuration
  const bool cull_back_face = false;
  const bool hit = s->Intersect(ray, &isect, cull_back_face);

  // Overwrite members.
  if (hit) {
    rtc_ray.tfar = isect.t;
    rtc_ray.u = isect.u;
    rtc_ray.v = isect.v;
    rtc_ray.geomID = isect.node_id;
    rtc_ray.primID = isect.prim_id;
    rtc_ray.instID =
        RTC_INVALID_GEOMETRY_ID;  // Instancing is not yet supported.
  } else {
    rtc_ray.geomID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray.primID = RTC_INVALID_GEOMETRY_ID;
    rtc_ray.instID = RTC_INVALID_GEOMETRY_ID;
  }

  (void)ray;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

RTCORE_API unsigned rtcNewTriangleMesh(
    RTCScene scene,          //!< the scene the mesh belongs to
    RTCGeometryFlags flags,  //!< geometry flags
    size_t numTriangles,     //!< number of triangles
    size_t numVertices,      //!< number of vertices
    size_t numTimeSteps = 1  //!< number of motion blur time steps
    ) {
  if (numTimeSteps != 1) {
    std::stringstream ss;
    ss << "[rtcNewTriMesh] Motion blur is not supported. numTimeSteps : "
       << numTimeSteps << std::endl;
    GetContext().SetError(ss.str());
    return 0;
  }

  if (numTriangles < 1) {
    std::stringstream ss;
    ss << "[rtcNewTriMesh] Invalid numTriangles : " << numTriangles
       << std::endl;
    GetContext().SetError(ss.str());
    return 0;
  }

  if (numVertices < 1) {
    std::stringstream ss;
    ss << "[rtcNewTriMesh] Invalid numVertices : " << numVertices << std::endl;
    GetContext().SetError(ss.str());
    return 0;
  }

  // TODO(LTE): Acquire lock?
  Scene *s = reinterpret_cast<Scene *>(scene);
  assert(s);
  const uint32_t geom_id = s->NewTriMesh(numTriangles, numVertices);

  // TODO(LTE): Support flags.
  (void)flags;

  return geom_id;
}

RTCORE_API void *rtcMapBuffer(RTCScene scene, unsigned geomID,
                              RTCBufferType type) {
  if (type == RTC_VERTEX_BUFFER) {
  } else if (type == RTC_INDEX_BUFFER) {
  } else {
    std::stringstream ss;
    ss << "[rtcMapBuffer] Unsupported type : " << type << std::endl;
    GetContext().SetError(ss.str());
    return NULL;
  }

  // TODO(LTE): Acquire lock?
  Scene *s = reinterpret_cast<Scene *>(scene);
  assert(s);
  TriMesh<float> *trimesh = s->GetTriMesh(geomID);
  if (trimesh) {
    if (type == RTC_VERTEX_BUFFER) {
      return reinterpret_cast<void *>(trimesh->vertices.data());
    } else if (type == RTC_INDEX_BUFFER) {
      return reinterpret_cast<void *>(trimesh->faces.data());
    }
  } else {
    std::stringstream ss;
    ss << "[rtcMapBuffer] geomID : " << geomID << " not found in the scene."
       << std::endl;
    GetContext().SetError(ss.str());
    return NULL;
  }

  return NULL;  // never reach here.
}

RTCORE_API void rtcUnmapBuffer(RTCScene scene, unsigned geomID,
                               RTCBufferType type) {
  if (type == RTC_VERTEX_BUFFER) {
  } else if (type == RTC_INDEX_BUFFER) {
  } else {
    std::stringstream ss;
    ss << "[rtcUnmapBuffer] Unsupported type : " << type << std::endl;
    GetContext().SetError(ss.str());
    return;
  }
  // TODO(LTE): Release lock?
  (void)scene;
  (void)geomID;
}

RTCORE_API unsigned rtcNewInstance2(
    RTCScene target,  //!< the scene the instance belongs to
    RTCScene source,  //!< the scene to instantiate
    size_t numTimeSteps =
        1) {  //!< number of timesteps, one matrix per timestep
  if (numTimeSteps != 1) {
    std::stringstream ss;
    ss << "[rtcNewInstance2] numTimeSteps must be 1" << std::endl;
    GetContext().SetError(ss.str());
    return 0;
  }

  // TODO(LTE): Implement
  (void)target;
  (void)source;

  return 0;
}

RTCORE_API void rtcSetTransform2(
    RTCScene scene,        //!< scene handle
    unsigned int geomID,   //!< ID of geometry
    RTCMatrixType layout,  //!< layout of transformation matrix
    const float *xfm,      //!< pointer to transformation matrix
    size_t timeStep = 0    //!< timestep to set the matrix for
    ) {
  // TODO(LTE): Implement
  (void)scene;
  (void)geomID;
  (void)layout;
  (void)xfm;
  (void)timeStep;
}

RTCORE_API void rtcUpdate(RTCScene scene, unsigned geomID) {
  // TODO(LTE): Implement
  (void)scene;
  (void)geomID;
}

RTCORE_API void rtcCommit(RTCScene scene) {
  Scene *s = reinterpret_cast<Scene *>(scene);
  assert(s);

  s->Build();
}

}  // namespace nanort_embree2
