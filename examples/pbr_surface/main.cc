#include <array>
#include <glm.hpp>
#include <iostream>
#include <vector>

#include "nanort.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef _MSC_VER
#define STBI_MSC_SECURE_CRT
#endif
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "pbr_maths.hh"

template <typename T>
glm::vec3 toVec3(T arr) {
  return {arr[0], arr[1], arr[2]};
}

template <typename T>
std::array<T, 3> toArr(const glm::vec3& v) {
  return {v.x, v.y, v.z};
}

template <typename T>
struct Mesh {
  size_t num_vertices;
  size_t num_faces;
  std::vector<T> vertices;
  std::vector<T> facevarying_normals;
  std::vector<unsigned int> faces;
  std::vector<T> facevarying_uvs;

  inline void lerp(T dst[3], const T v0[3], const T v1[3], const T v2[3],
                   float u, float v) {
    dst[0] = (static_cast<T>(1.0) - u - v) * v0[0] + u * v1[0] + v * v2[0];
    dst[1] = (static_cast<T>(1.0) - u - v) * v0[1] + u * v1[1] + v * v2[1];
    dst[2] = (static_cast<T>(1.0) - u - v) * v0[2] + u * v1[2] + v * v2[2];
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

#pragma pack(push, 1)
template <typename T>
struct Color {
  T r, g, b, a;
};

struct PointLight {
  PointLight(const glm::vec3& p, const Color<float>& c = {1, 1, 1})
      : position{p}, color{c} {}
  glm::vec3 position;
  Color<float> color;
};

struct pixel : public Color<uint8_t> {};
#pragma pack(pop)

template <typename T>
struct PbrMaterial {
  T metalness;
  T roughness;
  Color<T> albedo;
};

void loadSampler(pbr_maths::sampler2D& sampler, const stbi_uc* data, int w,
                 int h, int c) {
  sampler.width = w;
  sampler.height = h;
  sampler.pixels = new pbr_maths::sampler2D::pixel[w * h];

  for (size_t i = 0; i < w * h; ++i) {
    sampler.pixels[i].r = data[i * c + 0];
    sampler.pixels[i].g = data[i * c + 1];
    sampler.pixels[i].b = data[i * c + 2];
    if (c == 4) sampler.pixels[i].a = data[i * c + 3];
  }
}

void loadCombineMetalRoughSampler(pbr_maths::sampler2D& sampler,
                                  const stbi_uc* metalData, int mw, int mh,
                                  int mc, const stbi_uc* roughnessData, int rw,
                                  int rh, int rc) {
  assert(mw == rw);
  assert(mh == rh);

  sampler.pixels = new pbr_maths::sampler2D::pixel[mw * mh];

  for (size_t i = 0; i < mw * mh; ++i) {
    // We don't really care about these ones
    sampler.pixels[i].r = 0;
    sampler.pixels[i].a = 255;

    sampler.pixels[i].g = roughnessData[i * rc];
    sampler.pixels[i].b = metalData[i * mc];
  }
}

int main() {
  pbr_maths::sampler2D normalMap, baseColorMap;
  int w, h, c;
  auto normdata = stbi_load("./MetalPlates02_nrm.jpg", &w, &h, &c, 0);
  if (normdata) loadSampler(normalMap, normdata, w, h, c);

  auto colData = stbi_load("./MetalPlates02_col.jpg", &w, &h, &c, 0);
  if (colData) loadSampler(baseColorMap, colData, w, h, c);

  pbr_maths::sampler2D metalRoughMap;
  int rw, rh, rc, mw, mh, mc;
  auto roughData = stbi_load("./MetalPlates02_rgh.jpg", &rw, &rh, &rc, 0);
  auto metalData = stbi_load("./MetalPlates02_met.jpg", &mw, &mh, &mc, 0);
  if (roughData && metalData)
    loadCombineMetalRoughSampler(metalRoughMap, metalData, mw, mh, mc,
                                 roughData, rw, rh, rc);

  stbi_image_free(normdata);
  stbi_image_free(colData);
  stbi_image_free(roughData);
  stbi_image_free(metalData);

  // Here the metal and roughness data are mixed together

  PbrMaterial<float> material;
  material.metalness = 1;
  material.roughness = 1;
  material.albedo.r = 1;
  material.albedo.g = 1;
  material.albedo.b = 1;

  std::vector<PointLight> lights;

  lights.emplace_back(glm::vec3{0, 0.5, 2}, Color<float>{1, 1, 1});

  Mesh<float> mesh;
  mesh.num_faces = 2;
  mesh.num_vertices = mesh.num_faces * 3;

  // Poly1
  mesh.vertices.push_back(-1);
  mesh.vertices.push_back(+1);
  mesh.vertices.push_back(-2);

  mesh.vertices.push_back(+1);
  mesh.vertices.push_back(+1);
  mesh.vertices.push_back(-2);

  mesh.vertices.push_back(-1);
  mesh.vertices.push_back(-1);
  mesh.vertices.push_back(-2);

  // Poly2
  mesh.vertices.push_back(+1);
  mesh.vertices.push_back(+1);
  mesh.vertices.push_back(-2);

  mesh.vertices.push_back(+1);
  mesh.vertices.push_back(-1);
  mesh.vertices.push_back(-2);

  mesh.vertices.push_back(-1);
  mesh.vertices.push_back(-1);
  mesh.vertices.push_back(-2);

  // All normals towards the camera
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(+1);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(+1);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(+1);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(+1);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(+1);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(0);
  mesh.facevarying_normals.push_back(+1);

  // 1
  mesh.facevarying_uvs.push_back(0);
  mesh.facevarying_uvs.push_back(1);
  // 2
  mesh.facevarying_uvs.push_back(1);
  mesh.facevarying_uvs.push_back(1);
  // 3
  mesh.facevarying_uvs.push_back(0);
  mesh.facevarying_uvs.push_back(0);

  // 4
  mesh.facevarying_uvs.push_back(1);
  mesh.facevarying_uvs.push_back(1);
  // 5
  mesh.facevarying_uvs.push_back(1);
  mesh.facevarying_uvs.push_back(0);
  // 6
  mesh.facevarying_uvs.push_back(0);
  mesh.facevarying_uvs.push_back(0);

  mesh.faces.push_back(0);
  mesh.faces.push_back(1);
  mesh.faces.push_back(2);
  mesh.faces.push_back(3);
  mesh.faces.push_back(4);
  mesh.faces.push_back(5);

  nanort::BVHBuildOptions<float> build_options;
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  nanort::TriangleMesh<float> plane(mesh.vertices.data(), mesh.faces.data(),
                                    3 * sizeof(float));
  nanort::TriangleSAHPred<float> plane_pred(
      mesh.vertices.data(), mesh.faces.data(), 3 * sizeof(float));

  nanort::BVHAccel<float> accel;
  accel.Build(mesh.num_faces, plane, plane_pred, build_options);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();
  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  const size_t width = 1024;
  const size_t height = 1024;

  std::vector<pixel> img(width * height);
  // memset(img.data(), 255, img.size() * sizeof(pixel));

  for (size_t y{0}; y < height; ++y)
    for (size_t x{0}; x < width; ++x) {
      // access pixel we are going to calculate
      auto& pixel = img[(y * width) + x];
      pixel.r = pixel.g = pixel.b = 0;
      nanort::BVHTraceOptions trace_options;

      nanort::Ray<float> camRay;
      glm::vec3 org{0, 0, 3};
      camRay.org[0] = org.x;
      camRay.org[1] = org.y;
      camRay.org[2] = org.z;

      glm::vec3 dir;
      dir.x = (x / (float)width) - 0.5f;
      dir.y = (y / (float)height) - 0.5f;
      dir.z = -1;

      glm::normalize(dir);

      camRay.dir[0] = dir.x;
      camRay.dir[1] = dir.y;
      camRay.dir[2] = dir.z;

      float kFar = 1.0e+30f;
      camRay.min_t = 0.0f;
      camRay.max_t = kFar;

      nanort::TriangleIntersector<float, nanort::TriangleIntersection<float> >
          triangle_intersector(mesh.vertices.data(), mesh.faces.data(),
                               sizeof(float) * 3);

      nanort::TriangleIntersection<float> isect;

      if (accel.Traverse(camRay, triangle_intersector, &isect)) {
        glm::vec3 hit = org + isect.t * dir;
        glm::vec2 uv = mesh.getTextureCoord(isect.prim_id, isect.u, isect.v);

        // std::cout << "hit at " << hit.x << ',' << hit.y << ',' << hit.z <<
        // '\n';
        // 1) shoot ray to point light
        for (const auto& light : lights) {
          nanort::Ray<float> lightRay;

          // glm::vec3 org(isect.u, isect.v, isect.t);
          glm::vec3 dir = light.position - hit;
          glm::normalize(dir);
          hit += 0.00001f * dir;  // bias

          memcpy(lightRay.org, toArr<float>(hit).data(), 3 * sizeof(float));
          memcpy(lightRay.dir, toArr<float>(dir).data(), 3 * sizeof(float));

          // 2) if nothing was hit, draw pixel with shader
          if (!accel.Traverse(lightRay, triangle_intersector, &isect)) {
            // This object represet a fragment shader, and is literally a
            // translation in C++ from an official example from khronos
            static pbr_maths::PBRShaderCPU shader;

            // Fill in the uniform/varying variables
            shader.u_Camera.x = camRay.org[0];
            shader.u_Camera.y = camRay.org[1];
            shader.u_Camera.z = camRay.org[2];

            shader.u_MetallicRoughnessValues =
                glm::vec2{material.metalness, material.roughness};
            shader.u_BaseColorFactor = glm::vec4{
                material.albedo.r, material.albedo.g, material.albedo.b, 1};
            shader.v_Position = hit;

            shader.u_LightDirection = dir;
            shader.u_LightColor =
                glm::vec3(light.color.r, light.color.g, light.color.b);

            shader.v_Normal =
                glm::vec3{mesh.facevarying_normals[3 * 3 * isect.prim_id + 0],
                          mesh.facevarying_normals[3 * 3 * isect.prim_id + 1],
                          mesh.facevarying_normals[3 * 3 * isect.prim_id + 2]};
            shader.v_UV = uv;

            if (normalMap.pixels) {
              shader.useNormalMap = true;
              shader.u_NormalSampler = normalMap;
            } else {
              shader.useNormalMap = false;
            }

            if (baseColorMap.pixels) {
              shader.useBaseColorMap = true;
              shader.u_BaseColorSampler = baseColorMap;
            } else {
              shader.useBaseColorMap = false;
            }

            if (metalRoughMap.pixels) {
              shader.useMetalRoughMap = true;
              shader.u_MetallicRoughnessSampler = metalRoughMap;
            } else {
              shader.useMetalRoughMap = false;
            }

            //"Execute shader" on the current pixel, for the current light
            // source
            shader.main();

            // Accumulate the color
            pixel.r += std::min<int>(255, 255 * shader.gl_FragColor.r);
            pixel.g += std::min<int>(255, 255 * shader.gl_FragColor.g);
            pixel.b += std::min<int>(255, 255 * shader.gl_FragColor.b);
          }
        }
      }

      pixel.a = 255;
    }

  stbi_flip_vertically_on_write(true);  // Flip Y
  stbi_write_png("out.png", width, height, 4, (void*)img.data(), 0);

  normalMap.releasePixels();
  baseColorMap.releasePixels();
  metalRoughMap.releasePixels();

  return 0;
}
