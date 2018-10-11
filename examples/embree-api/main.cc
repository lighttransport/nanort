/*
The MIT License (MIT)

Copyright (c) 2015 - 2017 Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifdef _WIN32
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif


#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#pragma warning(disable : 4457)
#pragma warning(disable : 4456)
#endif

#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "obj-loader.h"  // ../nanosg

static inline void vnormalize(float dst[3], const float v[3]) {
  dst[0] = v[0];
  dst[1] = v[1];
  dst[2] = v[2];
  const float len = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  if (std::fabs(len) > std::numeric_limits<float>::epsilon()) {
    const float inv_len = 1.0f / len;
    dst[0] *= inv_len;
    dst[1] *= inv_len;
    dst[2] *= inv_len;
  }
}

static void MultV(float dst[3], const float m[4][4], const float v[3]) {
  float tmp[3];
  tmp[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
  tmp[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
  tmp[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
  dst[0] = tmp[0];
  dst[1] = tmp[1];
  dst[2] = tmp[2];
}

// Create TriangleMesh
static unsigned int CreateTriangleMesh(const example::Mesh<float> &mesh,
                                       RTCScene scene) {
  unsigned int geom_id = rtcNewTriangleMesh(
      scene, RTC_GEOMETRY_STATIC,
      /* numTriangles */ mesh.faces.size() / 3, mesh.vertices.size() / 3, 1);

  float *vertices = reinterpret_cast<float *>(
      rtcMapBuffer(scene, geom_id, RTC_VERTEX_BUFFER));
  int *faces =
      reinterpret_cast<int *>(rtcMapBuffer(scene, geom_id, RTC_INDEX_BUFFER));

  for (size_t i = 0; i < mesh.vertices.size() / 3; i++) {
    // TODO(LTE): Use instanciation for applying xform.
    float v[3];
    MultV(v, mesh.pivot_xform, &mesh.vertices[3 * i + 0]);

    // Embree uses 4 floats(16 bytes stride)
    vertices[4 * i + 0] = v[0];
    vertices[4 * i + 1] = v[1];
    vertices[4 * i + 2] = v[2];
    vertices[4 * i + 3] = 0.0f;
  }

  for (size_t i = 0; i < mesh.faces.size() / 3; i++) {
    faces[3 * i + 0] = static_cast<int>(mesh.faces[3 * i + 0]);
    faces[3 * i + 1] = static_cast<int>(mesh.faces[3 * i + 1]);
    faces[3 * i + 2] = static_cast<int>(mesh.faces[3 * i + 2]);
  }

  rtcUnmapBuffer(scene, geom_id, RTC_VERTEX_BUFFER);
  rtcUnmapBuffer(scene, geom_id, RTC_INDEX_BUFFER);

  return geom_id;
}

static void SaveImagePNG(const char *filename, const float *rgb, int width,
                         int height) {
  unsigned char *bytes = new unsigned char[size_t(width * height * 3)];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const int index = y * width + x;
      bytes[index * 3 + 0] = static_cast<unsigned char>(
          std::max(0.0f, std::min(rgb[index * 3 + 0] * 255.0f, 255.0f)));
      bytes[index * 3 + 1] = static_cast<unsigned char>(
          std::max(0.0f, std::min(rgb[index * 3 + 1] * 255.0f, 255.0f)));
      bytes[index * 3 + 2] = static_cast<unsigned char>(
          std::max(0.0f, std::min(rgb[index * 3 + 2] * 255.0f, 255.0f)));
    }
  }
  stbi_write_png(filename, width, height, 3, bytes, width * 3);
  delete[] bytes;
}

#ifdef __clang__
// suppress RTC_INVALID_GEOMETRY_ID
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  if (argc < 2) {
    std::cout << "render input.obj (obj_scale). " << std::endl;
  }

  std::string obj_filename = "../common/cornellbox_suzanne_lucy.obj";

  if (argc > 1) {
    obj_filename = argv[1];
  }

  float obj_scale = 1.0f;
  if (argc > 2) {
    obj_scale = float(atof(argv[2]));
  }

  RTCDevice device = rtcNewDevice(NULL);
  RTCScene scene = rtcDeviceNewScene(
      device, RTC_SCENE_STATIC | RTC_SCENE_INCOHERENT, RTC_INTERSECT1);

  std::vector<example::Mesh<float> > meshes;
  std::vector<example::Material> materials;
  std::vector<example::Texture> textures;
  {
    // load .obj

    bool ret = LoadObj(obj_filename, obj_scale, &meshes, &materials, &textures);
    if (!ret) {
      std::cerr << "Failed to load .obj [ " << obj_filename << " ]"
                << std::endl;
      return EXIT_FAILURE;
    }

    // Create meshes for ray tracing.
    for (size_t i = 0; i < meshes.size(); i++) {
      unsigned int mesh_id = CreateTriangleMesh(meshes[i], scene);
      (void)mesh_id;
    }

    rtcCommit(scene);

    RTCBounds bounds;
    rtcGetBounds(scene, bounds);

    std::cout << "Scene bounding min: " << bounds.lower_x << ", "
              << bounds.lower_y << ", " << bounds.lower_z << std::endl;
    std::cout << "Scene bounding max: " << bounds.upper_x << ", "
              << bounds.upper_y << ", " << bounds.upper_z << std::endl;
  }

  int width = 512;
  int height = 512;

  std::vector<float> rgb;
  rgb.resize(size_t(width * height * 3));

// Shoot rays.
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int y = 0; y < height; y++) {
    std::cout << y << " / " << height << std::endl;
    for (int x = 0; x < width; x++) {
      // Simple camera. change eye pos and direction fit to .obj model.

      RTCRay ray;
      ray.org[0] = 0.0f;
      ray.org[1] = 5.0f;
      ray.org[2] = 20.0f;

      float dir[3], ndir[3];
      dir[0] = (x / float(width)) - 0.5f;
      dir[1] = (y / float(height)) - 0.5f;
      dir[2] = -1.0f;
      vnormalize(ndir, dir);
      ray.dir[0] = ndir[0];
      ray.dir[1] = ndir[1];
      ray.dir[2] = ndir[2];

      float kFar = 1.0e+30f;
      ray.tnear = 0.0f;
      ray.tfar = kFar;

      rtcIntersect(scene, ray);
      if ((ray.tfar < kFar) && (ray.geomID != RTC_INVALID_GEOMETRY_ID) &&
          (ray.primID != RTC_INVALID_GEOMETRY_ID)) {
        const example::Mesh<float> &mesh = meshes[ray.geomID];
        // std::cout << "tfar " << ray.tfar << std::endl;
        // Write your shader here.
        float normal[3];
        normal[0] = 0.0f;
        normal[1] = 0.0f;
        normal[2] = 0.0f;
        unsigned int fid = ray.primID;
        if (mesh.facevarying_normals.size() > 0) {
          // std::cout << "fid " << fid << std::endl;
          normal[0] = mesh.facevarying_normals[9 * fid + 0];
          normal[1] = mesh.facevarying_normals[9 * fid + 1];
          normal[2] = mesh.facevarying_normals[9 * fid + 2];
        }
        // Flip Y
        rgb[3 * size_t((height - y - 1) * width + x) + 0] =
            0.5f * normal[0] + 0.5f;
        rgb[3 * size_t((height - y - 1) * width + x) + 1] =
            0.5f * normal[1] + 0.5f;
        rgb[3 * size_t((height - y - 1) * width + x) + 2] =
            0.5f * normal[2] + 0.5f;
      }
    }
  }

  // Save image.
  SaveImagePNG("render.png", &rgb.at(0), width, height);

  rtcDeleteScene(scene);
  rtcDeleteDevice(device);

  return EXIT_SUCCESS;
}
