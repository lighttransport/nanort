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
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wvariadic-macros"
#pragma clang diagnostic ignored "-Wc++11-extensions"
#endif

#include <embree2/rtcore.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "obj-loader.h" // ../nanosg

// Create TriangleMesh
static unsigned int CreateTriangleMesh(const example::Mesh<float> &mesh, RTCScene scene) {
  unsigned int geom_id = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC,
                                            /* numTriangles */ mesh.faces.size() / 3,
                                            mesh.vertices.size() / 3, 1);

  float *vertices = reinterpret_cast<float *>(
      rtcMapBuffer(scene, geom_id, RTC_VERTEX_BUFFER));
  int *faces =
      reinterpret_cast<int *>(rtcMapBuffer(scene, geom_id, RTC_INDEX_BUFFER));

  for (size_t i = 0; i < mesh.vertices.size() / 3; i++) {
    // Embree uses 4 floats(16 bytes stride)
    vertices[4 * i + 0] = mesh.vertices[3 * i + 0];
    vertices[4 * i + 1] = mesh.vertices[3 * i + 1];
    vertices[4 * i + 2] = mesh.vertices[3 * i + 2];
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



int main(int argc, char **argv)
{
	(void)argc;
	(void)argv;

  if (argc < 2) {
    std::cerr << "Wavefront .obj file required. " << std::endl;
    return EXIT_FAILURE;
  }

  std::string obj_filename = argv[1];

  float obj_scale = 1.0f;
  if (argc > 2) {
    obj_scale = float(atof(argv[2])); 
  }

	RTCDevice device = rtcNewDevice(NULL);
  RTCScene scene = rtcDeviceNewScene(device, RTC_SCENE_STATIC | RTC_SCENE_INCOHERENT, RTC_INTERSECT1);
  {
    // load .obj
    std::vector<example::Mesh<float> > meshes;
    std::vector<example::Material> materials;
    std::vector<example::Texture> textures;

    bool ret = LoadObj(obj_filename, obj_scale, &meshes, &materials, &textures);
    if (!ret) {
      std::cerr << "Failed to load .obj [ " << obj_filename << " ]" << std::endl;
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

    std::cout << "Scene bounding min: " << bounds.lower_x << ", " << bounds.lower_y << ", " << bounds.lower_z << std::endl;
    std::cout << "Scene bounding max: " << bounds.upper_x << ", " << bounds.upper_y << ", " << bounds.upper_z << std::endl;

  }
  rtcDeleteScene(scene);
	rtcDeleteDevice(device);

	return EXIT_SUCCESS;
}
