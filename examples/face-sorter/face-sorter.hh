#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
//#include <iostream> // dbg

namespace face_sorter {

template<typename T>
class TriangleFaceCenterAccessor
{
 public:
  TriangleFaceCenterAccessor(const T *vertices, const uint32_t *indices, const size_t num_faces)
    : _vertices(vertices), _indices(indices), _num_faces(num_faces) {
  }

  TriangleFaceCenterAccessor(const TriangleFaceCenterAccessor<T> &rhs) = default;

  std::array<T, 3> operator()(uint32_t idx) const {

    assert(idx < 3 * _num_faces);

    uint32_t i0 = _indices[3 * idx + 0];
    uint32_t i1 = _indices[3 * idx + 1];
    uint32_t i2 = _indices[3 * idx + 2];

    std::array<T, 3> center;

    T p0[3] = {_vertices[3*i0+0], _vertices[3 * i0 + 1], _vertices[3 * i0 + 2]};
    T p1[3] = {_vertices[3*i1+0], _vertices[3 * i1 + 1], _vertices[3 * i1 + 2]};
    T p2[3] = {_vertices[3*i2+0], _vertices[3 * i2 + 1], _vertices[3 * i2 + 2]};

    center[0] = (p0[0] + p1[0] + p2[0]) / static_cast<T>(3.0);
    center[1] = (p0[1] + p1[1] + p2[1]) / static_cast<T>(3.0);
    center[2] = (p0[2] + p1[2] + p2[2]) / static_cast<T>(3.0);

    return center;
  }

 private:
  const T *_vertices;
  const uint32_t *_indices;
  const size_t _num_faces;
};

// Sort polygon face by its barycentric z value.
//
// @tparam T Value type for vertex data(float or double)
// @tparam Face accessor
//
// @param[in] num_faces The number of facesRay origin
// @param[in] ray_origin Ray origin
// @param[in] ray_direction Ray direction
// @param[out] sorted_indices Sorted polygon face indices.

template<typename T, class FA>
void SortByBarycentricZ(
  const size_t num_faces,
  const float ray_org[3],
  const float ray_dir[3],
  const FA &fa,
  std::vector<uint32_t> *sorted_indices)
{
  // fill indices.
  sorted_indices->resize(num_faces);

  // [0...num_faces-1]
  std::iota(sorted_indices->begin(), sorted_indices->end(), 0);

  std::sort(sorted_indices->begin(), sorted_indices->end(), [&fa, &ray_org, &ray_dir](const uint32_t a, const uint32_t b) {

    std::array<T, 3> ac = fa(a);
    std::array<T, 3> bc = fa(b);

    // Take a simple dot to get a distance from a ray.
    T ap[3];
    ap[0] = ac[0] - static_cast<T>(ray_org[0]);
    ap[1] = ac[1] - static_cast<T>(ray_org[1]);
    ap[2] = ac[2] - static_cast<T>(ray_org[2]);

    T da = ap[0] * static_cast<T>(ray_dir[0])
         + ap[1] * static_cast<T>(ray_dir[1])
         + ap[2] * static_cast<T>(ray_dir[2]);

    T bp[3];
    bp[0] = bc[0] - static_cast<T>(ray_org[0]);
    bp[1] = bc[1] - static_cast<T>(ray_org[1]);
    bp[2] = bc[2] - static_cast<T>(ray_org[2]);

    T db = bp[0] * static_cast<T>(ray_dir[0])
         + bp[1] * static_cast<T>(ray_dir[1])
         + bp[2] * static_cast<T>(ray_dir[2]);

    return (da < db);
  });
}


}  // namespace face_sorter
