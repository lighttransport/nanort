#ifndef EXAMPLE_SCENE_H_
#define EXAMPLE_SCENE_H_

#define NANORT_USE_CPP11_FEATURE
#define NANORT_ENABLE_PARALLEL_BUILD
#include "nanort.h"

#include <vector>
#include <string>


namespace example {

typedef struct {
  std::vector<float> vertices;               /// [xyz] * num_vertices
  std::vector<float> facevarying_normals;    /// [xyz] * 3(triangle) * num_faces
  std::vector<float> facevarying_tangents;   /// [xyz] * 3(triangle) * num_faces
  std::vector<float> facevarying_binormals;  /// [xyz] * 3(triangle) * num_faces
  std::vector<float> facevarying_uvs;        /// [xy]  * 3(triangle) * num_faces
  std::vector<float> vertex_colors;          /// [rgb] * num_vertices
  std::vector<unsigned int> faces;           /// triangle x num_faces
  std::vector<unsigned int> material_ids;    /// index x num_faces

} Mesh;

struct Material {
  std::string name;

  float diffuse[3];
  //float specular[3];
  // float reflection[3];
  // float refraction[3];
  float ior;
  float dissolve;
  int id;
  int diffuse_texid;
  //int specular_texid;
  // int reflection_texid;
  // int transparency_texid;
  // int bump_texid;
  // int normal_texid;  // normal map
  int alpha_texid;  // alpha map

  Material() {
    diffuse[0] = diffuse[1] = diffuse[2] = 0.5f;

    ior = 1.55f;
    dissolve = 0;
    id = -1;
    diffuse_texid = -1;
    alpha_texid = -1;
  }
};

struct Texture {
  int width;
  int height;
  int components;
  std::vector<float> image;

  Texture() {
    width = -1;
    height = -1;
    components = -1;
  }
};

struct Envmap {
  size_t width;
  size_t height;
  std::vector<float> image;
  float horiz_offset;

  Envmap() {
    width = 0;
    height = 0;
  }
};

struct Sphere {
  float vertex[3];
  float radius;

  Sphere() {
    vertex[0] = 0;
    vertex[1] = 0;
    vertex[2] = 0;

    radius = 0;
  }
};

///
/// Simple scene
///
struct Scene {
  Mesh mesh;
  nanort::BVHAccel<float> accel;

  std::vector<Material> materials;
  std::vector<Texture> textures;
  Envmap envmap;

  Sphere sphere;
  float sphereVertice[3];
  float sphereRadius;
  nanort::BVHAccel<float> accelSphere;
};


} // namespace example

#endif // EXAMPLE_SCENE_H_
