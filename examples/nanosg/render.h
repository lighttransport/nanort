#ifndef EXAMPLE_RENDER_H_
#define EXAMPLE_RENDER_H_

#include <atomic>  // C++11

#include "render-config.h"
#include "nanosg.h"

namespace example {

template<typename T>
struct Mesh {
  std::vector<T> vertices;               /// [xyz] * num_vertices
  std::vector<T> facevarying_normals;    /// [xyz] * 3(triangle) * num_faces
  std::vector<T> facevarying_tangents;   /// [xyz] * 3(triangle) * num_faces
  std::vector<T> facevarying_binormals;  /// [xyz] * 3(triangle) * num_faces
  std::vector<T> facevarying_uvs;        /// [xy]  * 3(triangle) * num_faces
  std::vector<T>
      facevarying_vertex_colors;           /// [xyz] * 3(triangle) * num_faces
  std::vector<unsigned int> faces;         /// triangle x num_faces
  std::vector<unsigned int> material_ids;  /// index x num_faces
};

class Renderer {
 public:
  Renderer() {}
  ~Renderer() {}

  /// Loads wavefront .obj mesh.
  bool LoadObjMesh(const char* obj_filename, float scene_scale);

  /// Builds bvh.
  bool BuildBVH();

  /// Returns false when the rendering was canceled.
  bool Render(float* rgba, float* aux_rgba, int *sample_counts, float quat[4],
              const nanosg::Scene<float, Mesh<float>> &scene, const RenderConfig& config, std::atomic<bool>& cancel_flag);
};
};

#endif  // EXAMPLE_RENDER_H_
