#pragma once

#include <glm.hpp>

#pragma pack(push, 1)

template <typename T>
struct Color {
  T r, g, b, a;
};

struct pixel : public Color<uint8_t> {};
#pragma pack(pop)

struct PointLight {
  PointLight(const glm::vec3& p, const Color<float>& c = {1, 1, 1})
      : position{p}, color{c} {}
  glm::vec3 position;
  Color<float> color;
};

// TODO add default values
// TODO add texture pointers
template <typename T>
struct PbrMaterial {
  T metalness;
  T roughness;
  T emissiveness;
  Color<T> albedo;
  T ILBContrib;
};


using Material = PbrMaterial<float>;
