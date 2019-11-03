#include <random>
#include <ctime>
#include <cassert>
#include <iostream>

#include "../face-sorter.hh"

int main(int argc, char **argv)
{
  int num = 100;

  if (argc > 1) {
    num = std::atoi(argv[1]);
  }

  std::srand(std::time(nullptr));

  std::vector<float> vertices;
  std::vector<uint32_t> indices;

  for (size_t i = 0; i < num; i++) {
    float z = float(std::rand());
    //std::cout << "z = " << z << "\n";

    vertices.push_back(0.0f);
    vertices.push_back(0.0f);
    vertices.push_back(z);

    vertices.push_back(1.0f);
    vertices.push_back(0.0f);
    vertices.push_back(z);

    vertices.push_back(1.0f);
    vertices.push_back(1.0f);
    vertices.push_back(z);

    indices.push_back(3 * i + 0);
    indices.push_back(3 * i + 1);
    indices.push_back(3 * i + 2);
  }

#if 0 // dbg
  for (size_t i = 0; i < num; i++) {
    uint32_t face_idx = indices[3 * i + 0];
    float v0[3];

    v0[0] = vertices[3 * face_idx + 0];
    v0[1] = vertices[3 * face_idx + 1];
    v0[2] = vertices[3 * face_idx + 2];

    std::cout << "before [" << i << "] v = " << v0[0] << ", " << v0[1] << ", " << v0[2] << "\n";
  }
#endif

  float ray_org[3];
  float ray_dir[3];

  ray_org[0] = 0.0f;
  ray_org[1] = 0.0f;
  ray_org[2] = 1000.0f;

  ray_dir[0] = 0.0f;
  ray_dir[1] = 0.0f;
  ray_dir[2] = -1.0f;

  face_sorter::TriangleFaceCenterAccessor<float> fa(vertices.data(), indices.data(), num);

  std::vector<uint32_t> sorted_face_indices;
  face_sorter::SortByBarycentricZ<float>(num,
    ray_org, ray_dir, fa, &sorted_face_indices);

  assert(sorted_face_indices.size() == num);

  for (size_t i = 0; i < sorted_face_indices.size(); i++) {
    uint32_t face_idx = sorted_face_indices[i];

    uint32_t i0 = indices[3 * face_idx + 0];
    float v0[3];

    v0[0] = vertices[3 * i0 + 0];
    v0[1] = vertices[3 * i0 + 1];
    v0[2] = vertices[3 * i0 + 2];

    std::cout << "[" << i << "] = " << sorted_face_indices[i] << ", v = " << v0[0] << ", " << v0[1] << ", " << v0[2] << "\n";
  }

  return EXIT_SUCCESS;

};
