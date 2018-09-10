#ifndef EXAMPLE_MATERIAL_H_
#define EXAMPLE_MATERIAL_H_

#include <cstdlib>

#ifdef __clang__
#pragma clang diagnostic push
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#endif

namespace example {

struct Material {
  std::string name;

  float baseColor[3];
  float reflectance;
  float metallic;
  float clearcoat;  // clearcoat_thickness in PBR .mtl
  float clearcoat_roughness;
  float anisotropy;

  int id;
  int diffuse_texid;

  float roughness;

  Material() {
    baseColor[0] = 0.5f;
    baseColor[1] = 0.5f;
    baseColor[2] = 0.5f;

    reflectance = 0.0f;
    roughness = 0.0f;
    metallic = 0.0f;
    anisotropy = 0.0f;
    clearcoat = 0.0f;
    clearcoat_roughness = 0.0f;

    id = -1;
    diffuse_texid = -1;
  }
};

struct Texture {
  int width;
  int height;
  int components;
  int _pad_;
  unsigned char *image;

  Texture() {
    width = -1;
    height = -1;
    components = -1;
    image = nullptr;
  }
};

struct HDRTexture {
  int width;
  int height;
  int components;
  std::vector<float> image;

  HDRTexture() {
    width = -1;
    height = -1;
    components = -1;
  }
};


}  // namespace example


#endif  // EXAMPLE_MATERIAL_H_
