#pragma once

#include <vector>
#include <cstdint>

namespace objlab {

struct Mesh {
  std::vector<float> vertices; // XYZXYZXYZ...
  std::vector<uint32_t> indices;
  std::vector<float> normals; // XYZXYZXYZ...
  std::vector<float> vcolors; // vertex colors. XYZXYZXYZ...
  std::vector<float> texcoords; // UVUVUV..
  std::vector<uint8_t> num_verts_per_faces; // 3 = triangle, 4 = quad

  // sorted indices for a given view config(this is for saving .obj)
  std::vector<uint32_t> sorted_indices;

  int32_t draw_object_id; // Index to corresponding DrawObject

  bool triangles_only = false;
};


} // namespace objlab
