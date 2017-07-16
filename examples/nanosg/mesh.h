#ifndef EXAMPLE_MESH_H_
#define EXAMPLE_MESH_H_

#include <vector>

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

}  // namespace example

#endif // EXAMPLE_MESH_H_
