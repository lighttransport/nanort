#include "obj-writer.hh"

#include <iostream>
#include <fstream>
#include <cassert>

namespace objlab {

bool SaveMeshAsObj(int32_t shape_id, const std::vector<Mesh> &meshes, const std::string &base_filename) {
  // TODO(LTE): Write .mtl

  // TODO(LTE): Support multiple meshes.
  if (meshes.empty()) {
    std::cerr << "meshes is empty " << meshes.size() << "\n";
    return false;
  }

  if (shape_id >= meshes.size()) {
    std::cerr << "shape_id " << shape_id << " out-of-range. meshes.size() = " << meshes.size() << "\n";
    return false;
  }


  std::string obj_filename = base_filename + ".obj";

  std::ofstream ofs(obj_filename);

  if (!ofs) {
    std::cerr << "Failed to open file to write: " << obj_filename << "\n";
    return false;
  }

  const Mesh &mesh = meshes[size_t(shape_id)];

  bool has_sorted_indices = (mesh.sorted_indices.size() == mesh.indices.size());

  bool has_normals = (mesh.normals.size() > 0) ? true : false;
  bool has_uvs = (mesh.texcoords.size() > 0) ? true : false;

  std::cout << "Use sorted indices? " << has_sorted_indices << "\n";
  std::cout << "# of vertices = " << mesh.vertices.size() / 3 << "\n";
  std::cout << "# of normals = " << mesh.normals.size() / 3 << "\n";
  std::cout << "# of texcoords = " << mesh.texcoords.size() / 2 << "\n";

  for (size_t i = 0; i < mesh.vertices.size() / 3; i++) {
    ofs << "v " << mesh.vertices[3 * i + 0] << ", " << mesh.vertices[3 * i + 1] << ", " << mesh.vertices[3 * i + 2] << "\n";
  }

  for (size_t i = 0; i < mesh.normals.size() / 3; i++) {
    ofs << "vn " << mesh.normals[3 * i + 0] << ", " << mesh.normals[3 * i + 1] << ", " << mesh.normals[3 * i + 2] << "\n";
  }

  for (size_t i = 0; i < mesh.texcoords.size() / 2; i++) {
    // invert y texcoord
    ofs << "vt " << mesh.texcoords[2 * i + 0] << ", " << (1.f - mesh.texcoords[2 * i + 1]) << "\n";
  }

  const std::vector<uint32_t> &indices = has_sorted_indices ? mesh.sorted_indices : mesh.indices;

  size_t idx_offset = 0;
  for (size_t i = 0; i < mesh.num_verts_per_faces.size(); i++) {
    ofs << "f";
    for (size_t v = 0; v < mesh.num_verts_per_faces[i]; v++) {

      // obj face start with 1.
      size_t idx = indices[idx_offset + v] + 1;

      // assume same index number is used for vertex/normal/uv

      if (has_uvs) {
        if (has_normals) {
          ofs << " " << idx << "/" << idx << "/" << idx ;
        } else {
          ofs << " " << idx << "/" << idx;
        }
      } else {
        if (has_normals) {
          ofs << " " << idx << "//" << idx;
        } else {
          ofs << " " << idx;
        }
      }

    }

    ofs << "\n";

    idx_offset += mesh.num_verts_per_faces[i];

  }

  return true;
}


} // namespace objlab
