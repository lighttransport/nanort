#pragma once

#include <array>
#include <glm.hpp>

template <typename T>
glm::vec3 toVec3(T arr) {
  return {arr[0], arr[1], arr[2]};
}

template <typename T>
std::array<T, 3> toArr(const glm::vec3& v) {
  return {v.x, v.y, v.z};
}
