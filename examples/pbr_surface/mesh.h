#pragma once

#include <vector>

template <typename T>
struct Mesh {
  size_t num_vertices;
  size_t num_faces;
  std::vector<T> vertices;
  std::vector<T> facevarying_normals;
  std::vector<unsigned int> faces;
  std::vector<T> facevarying_uvs;
  std::string name;
  void lerp(T dst[3], const T v0[3], const T v1[3], const T v2[3],
                   float u, float v) {
    dst[0] = (static_cast<T>(1.0) - u - v) * v0[0] + u * v1[0] + v * v2[0];
    dst[1] = (static_cast<T>(1.0) - u - v) * v0[1] + u * v1[1] + v * v2[1];
    dst[2] = (static_cast<T>(1.0) - u - v) * v0[2] + u * v1[2] + v * v2[2];
  }

  bool has_uvs() const
  {
    return facevarying_uvs.size() > 0;
  }
    glm::vec2 getTextureCoord(unsigned int face, T u, T v) {
    T t0[3], t1[3], t2[3];
    t0[0] = facevarying_uvs[6 * face + 0];
    t0[1] = facevarying_uvs[6 * face + 1];
    t0[2] = T(0);

    t1[0] = facevarying_uvs[6 * face + 2];
    t1[1] = facevarying_uvs[6 * face + 3];
    t1[2] = T(0);

    t2[0] = facevarying_uvs[6 * face + 4];
    t2[1] = facevarying_uvs[6 * face + 5];
    t2[2] = T(0);

    T tcoord[3];
    lerp(tcoord, t0, t1, t2, u, v);

    return {tcoord[0], tcoord[1]};
  }


};