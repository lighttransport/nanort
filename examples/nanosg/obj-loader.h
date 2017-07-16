#ifndef EXAMPLE_OBJ_LOADER_H_
#define EXAMPLE_OBJ_LOADER_H_

#include <vector>

#include "mesh.h"
#include "material.h"

namespace example {

///
/// Loads wavefront .obj mesh
///
bool LoadObj(const std::string &filename, float scale, Mesh<float> *mesh, std::vector<Material> *materials, std::vector<Texture> *textures);

}

#endif // EXAMPLE_OBJ_LOADER_H_
