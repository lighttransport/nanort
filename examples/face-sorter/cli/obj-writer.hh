#pragma once

#include "mesh.hh"

#include <string>
#include <vector>

namespace objlab {

///
/// Save Mesh as wavefront .obj
///
bool SaveMeshAsObj(int32_t shape_id, const std::vector<Mesh> &meshes, const std::string &base_filename);


} // namespace objlab
