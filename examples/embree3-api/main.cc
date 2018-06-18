/*
The MIT License (MIT)

Copyright (c) 2015 - 2017 Light Transport Entertainment, Inc.
Copyright (c) 2018 Guillaume Jacquenot

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
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wvariadic-macros"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wshadow"
#pragma clang diagnostic ignored "-Wimplicit-fallthrough"
#if __has_warning("-Wcomma")
#pragma clang diagnostic ignored "-Wcomma"
#endif
#if __has_warning("-Wdouble-promotion")
#pragma clang diagnostic ignored "-Wdouble-promotion"
#endif
#if __has_warning("-Wcast-qual")
#pragma clang diagnostic ignored "-Wcast-qual"
#endif
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)
#pragma warning(disable : 4457)
#pragma warning(disable : 4456)
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <iomanip>

#include <embree3/rtcore.h>

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
  dst[0] = m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2] + m[3][0];
  dst[1] = m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2] + m[3][1];
  dst[2] = m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2] + m[3][2];
}

// Create TriangleMesh
static unsigned int CreateTriangleMesh(const example::Mesh<float> &mesh,
                                       RTCDevice& device,
                                       RTCScene& scene) {
  RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryBuildQuality(geom, RTC_BUILD_QUALITY_HIGH);
  rtcSetGeometryTimeStepCount(geom, 1);
  unsigned int geom_id = rtcAttachGeometry(scene, geom);
  rtcReleaseGeometry(geom);

  float *vertices = reinterpret_cast<float *>(
      rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,0,RTC_FORMAT_FLOAT3,4*sizeof(float),mesh.vertices.size() / 3));
  int *faces =
      reinterpret_cast<int *>(rtcSetNewGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,3*sizeof(int),mesh.faces.size() / 3));

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

  for (size_t i = 0; i < mesh.faces.size() / 3; ++i) {
    faces[3 * i + 0] = static_cast<int>(mesh.faces[3 * i + 0]);
    faces[3 * i + 1] = static_cast<int>(mesh.faces[3 * i + 1]);
    faces[3 * i + 2] = static_cast<int>(mesh.faces[3 * i + 2]);
  }

  rtcCommitGeometry(geom);
  return geom_id;
}

static void SaveImagePNG(const char *filename, const float *rgb, int width,
                         int height) {
  unsigned char *bytes = new unsigned char[size_t(width * height * 3)];
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
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

  char* cfg=NULL;
  RTCDevice device = rtcNewDevice(cfg);
  RTCScene scene = rtcNewScene(device);

  rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
  rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

  std::vector<example::Mesh<float> > meshes;
  std::vector<example::Material> materials;
  std::vector<example::Texture> textures;
  {
    // load .obj
    const bool ret = LoadObj(obj_filename, obj_scale, &meshes, &materials, &textures);
    if (!ret) {
      std::cerr << "Failed to load .obj [ " << obj_filename << " ]"
                << std::endl;
      return EXIT_FAILURE;
    }

    // Create meshes for ray tracing.
    for (size_t i = 0; i < meshes.size(); ++i) {
      unsigned int mesh_id = CreateTriangleMesh(meshes[i], device, scene);
      (void)mesh_id;
    }

    rtcCommitScene(scene);

    RTCBounds bounds;
    rtcGetSceneBounds(scene,&bounds);

    std::cout << "Scene bounding min: " << bounds.lower_x << ", "
              << bounds.lower_y << ", " << bounds.lower_z << std::endl;
    std::cout << "Scene bounding max: " << bounds.upper_x << ", "
              << bounds.upper_y << ", " << bounds.upper_z << std::endl;
  }

  int width = 1024;
  int height = 1024;

  std::vector<float> rgb(width * height * 3, 0.0f);

  auto t_start = std::chrono::high_resolution_clock::now();
  // Shoot rays.
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Simple camera. change eye pos and direction fit to .obj model.
      RTCRayHit ray;
      ray.ray.flags = 0;
      ray.ray.org_x = 0.0f;
      ray.ray.org_y = 5.0f;
      ray.ray.org_z = 20.0f;

      float dir[3], ndir[3];
      dir[0] = (x / float(width)) - 0.5f;
      dir[1] = (y / float(height)) - 0.5f;
      dir[2] = -1.0f;
      vnormalize(ndir, dir);
      ray.ray.dir_x = ndir[0];
      ray.ray.dir_y = ndir[1];
      ray.ray.dir_z = ndir[2];

      float kFar = 1.0e+30f;
      ray.ray.tnear = 0.0f;
      ray.ray.tfar = kFar;

      {
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        rtcIntersect1(scene,&context,&ray);
        ray.hit.Ng_x = -ray.hit.Ng_x; // EMBREE_FIXME: only correct for triangles,quads, and subdivision surfaces
        ray.hit.Ng_y = -ray.hit.Ng_y;
        ray.hit.Ng_z = -ray.hit.Ng_z;
      }
      if ((ray.ray.tfar < kFar) && (ray.hit.geomID != RTC_INVALID_GEOMETRY_ID) &&
          (ray.hit.primID != RTC_INVALID_GEOMETRY_ID)) {
        const example::Mesh<float> &mesh = meshes[ray.hit.geomID];
        // std::cout << "tfar " << ray.tfar << std::endl;
        // Write your shader here.
        float normal[3] = {0.0f, 0.0f, 0.0f};
        unsigned int fid = ray.hit.primID;
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
  auto t_end = std::chrono::high_resolution_clock::now();
  double elaspedTimeMs = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  std::cout <<  std::fixed << std::setprecision(2)
            << "Elapsed time " << elaspedTimeMs << " ms" << std::endl << std::flush;

  // Save image.
  SaveImagePNG("render.png", &rgb.at(0), width, height);

  rtcReleaseScene(scene);
  rtcReleaseDevice(device);

  return EXIT_SUCCESS;
}
