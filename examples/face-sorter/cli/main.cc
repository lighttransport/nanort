/*
The MIT License (MIT)

Copyright (c) 2015 - 2020 Light Transport Entertainment, Inc.

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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <atomic>  // C++11
#include <chrono>  // C++11
#include <mutex>   // C++11
#include <thread>  // C++11

#include "render-config.h"
//#include "render.h"

#include "face-sorter.hh"
#include "mesh.hh"
#include "obj-writer.hh"

#if 1
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"


// TODO: Specify CCW or CW
static void CalcNormal(float N[3], float v0[3], float v1[3], float v2[3]) {
  float v10[3];
  v10[0] = v1[0] - v0[0];
  v10[1] = v1[1] - v0[1];
  v10[2] = v1[2] - v0[2];

  float v20[3];
  v20[0] = v2[0] - v0[0];
  v20[1] = v2[1] - v0[1];
  v20[2] = v2[2] - v0[2];

  N[0] = v20[1] * v10[2] - v20[2] * v10[1];
  N[1] = v20[2] * v10[0] - v20[0] * v10[2];
  N[2] = v20[0] * v10[1] - v20[1] * v10[0];

  float len2 = N[0] * N[0] + N[1] * N[1] + N[2] * N[2];
  if (len2 > 0.0f) {
    float len = sqrtf(len2);

    N[0] /= len;
    N[1] /= len;
    N[2] /= len;
  }
}

static std::string GetBaseDir(const std::string& filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

static bool FileExists(const std::string& abs_filename) {
  bool ret;
  FILE* fp = fopen(abs_filename.c_str(), "rb");
  if (fp) {
    ret = true;
    fclose(fp);
  } else {
    ret = false;
  }

  return ret;
}

static bool LoadObjAndConvert(
    float bmin[3], float bmax[3],
    std::vector<objlab::Mesh>* meshes,                 // out
    //std::vector<objlab::DrawObject>* drawObjects,      // out
    std::vector<tinyobj::material_t>& materials,       // out
    //std::map<std::string, objlab::Texture>& textures,  // out
    //std::vector<objlab::Image>& images,                // out
    const char* filename) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;

  meshes->clear();

  auto start_time = std::chrono::system_clock::now();

  std::string base_dir = GetBaseDir(filename);
  if (base_dir.empty()) {
    base_dir = ".";
  }
#ifdef _WIN32
  base_dir += "\\";
#else
  base_dir += "/";
#endif

  std::string warn;
  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              filename, base_dir.c_str());
  if (!warn.empty()) {
    std::cout << "WARN: " << warn << std::endl;
  }
  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  auto end_time = std::chrono::system_clock::now();

  if (!ret) {
    std::cerr << "Failed to load " << filename << std::endl;
    return false;
  }

  std::chrono::duration<double, std::milli> ms = end_time - start_time;

  std::cout << "Parsing time: " << ms.count() << " [ms]\n";

  std::cout << "# of vertices  = " << (attrib.vertices.size()) / 3 << "\n";
  std::cout << "# of normals   = " << (attrib.normals.size()) / 3 << "\n";
  std::cout << "# of texcoords = " << (attrib.texcoords.size()) / 2 << "\n";
  std::cout << "# of materials = " << materials.size() << "\n";
  std::cout << "# of shapes    = " << shapes.size() << "\n";

  // Append `default` material
  materials.push_back(tinyobj::material_t());

  for (size_t i = 0; i < materials.size(); i++) {
    printf("material[%d].diffuse_texname = %s\n", int(i),
           materials[i].diffuse_texname.c_str());
    printf("material[%d].alpha_texname = %s\n", int(i),
           materials[i].alpha_texname.c_str());
  }

#if 0
  // Load diffuse textures
  {
    for (size_t m = 0; m < materials.size(); m++) {
      tinyobj::material_t* mp = &materials[m];

      bool has_alpha = false;
      int alpha_w = 0, alpha_h = 0;
      std::vector<uint8_t> alpha_image;

      if (!(mp->alpha_texname.empty())) {
        int comp;

        std::string texture_filename = mp->alpha_texname;
        if (!FileExists(texture_filename)) {
          // Append base dir.
          texture_filename = base_dir + mp->alpha_texname;
          if (!FileExists(texture_filename)) {
            std::cerr << "Unable to find file: " << mp->alpha_texname
                      << std::endl;
            exit(1);
          }
        }

        unsigned char* image_data =
            stbi_load(texture_filename.c_str(), &alpha_w, &alpha_h, &comp,
                      STBI_default);
        if (!image_data) {
          std::cerr << "Unable to load texture: " << texture_filename
                    << std::endl;
          exit(1);
        }

        if (comp != 1) {
          std::cerr << "Alpha texture must be grayscale image: "
                    << texture_filename << std::endl;
          exit(1);
        }

        std::cout << "alpha_w = " << alpha_w << ", alpha_h = " << alpha_h << ", channels = " << comp << "\n";

        alpha_image.resize(size_t(alpha_w * alpha_h));
        memcpy(alpha_image.data(), image_data, size_t(alpha_w * alpha_h));

        stbi_image_free(image_data);

        has_alpha = true;
      }

      if (mp->diffuse_texname.length() > 0) {
        // Only load the texture if it is not already loaded
        if (textures.find(mp->diffuse_texname) == textures.end()) {
          GLuint texture_id;
          int w, h;
          int comp;

          std::string texture_filename = mp->diffuse_texname;
          if (!FileExists(texture_filename)) {
            // Append base dir.
            texture_filename = base_dir + mp->diffuse_texname;
            if (!FileExists(texture_filename)) {
              std::cerr << "Unable to find file: " << mp->diffuse_texname
                        << std::endl;
              exit(1);
            }
          }

          std::vector<uint8_t> pixels;
          {
            unsigned char* image_data =
                stbi_load(texture_filename.c_str(), &w, &h, &comp, STBI_default);
            if (!image_data) {
              std::cerr << "Unable to load texture: " << texture_filename
                        << std::endl;
              exit(1);
            }
            std::cout << "Loaded texture: " << texture_filename << ", w = " << w
                      << ", h = " << h << ", comp = " << comp << std::endl;

            pixels.resize(size_t(w * h * comp));
            if (comp == 4) {
              memcpy(pixels.data(), image_data, size_t(w * h * comp));

              if (has_alpha) {
                // Overwrite alpha channel with separate alpha image.
                if (alpha_w != w) {
                  std::cerr << "alpha image and color image has different image "
                               "width.\n";
                  exit(-1);
                }
                if (alpha_h != h) {
                  std::cerr << "alpha image and color image has different image "
                               "height.\n";
                  exit(-1);
                }

                for (size_t i = 0; i < size_t(w * h); i++) {
                  pixels[4 * i + 3] = alpha_image[i];
                }
              }
            } else {
              memcpy(pixels.data(), image_data, size_t(w * h * comp));
            }

            stbi_image_free(image_data);
          }

          glGenTextures(1, &texture_id);
          glBindTexture(GL_TEXTURE_2D, texture_id);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
          glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
          if (comp == 3) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB,
                         GL_UNSIGNED_BYTE, pixels.data());
          } else if (comp == 4) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, pixels.data());
          } else {
            assert(0);  // TODO
          }
          glBindTexture(GL_TEXTURE_2D, 0);

          objlab::Texture texture;
          texture.gl_tex_id = uint32_t(texture_id);
          texture.image_idx = int(images.size());

          // TODO(LTE): Do not create duplicated image for same filename.
          objlab::Image image;
          image.data = pixels;

          image.width = size_t(w);
          image.height = size_t(h);
          image.channels = comp;

          images.emplace_back(image);

          textures.insert(std::make_pair(mp->diffuse_texname, texture));
        }
      }
    }
  }
#endif

  bmin[0] = bmin[1] = bmin[2] = std::numeric_limits<float>::max();
  bmax[0] = bmax[1] = bmax[2] = -std::numeric_limits<float>::max();

  {
    for (size_t s = 0; s < shapes.size(); s++) {
      //objlab::DrawObject o;
      //std::vector<float> buffer;  // pos(3float), normal(3float), color(3float)

      objlab::Mesh mesh;

      //// Check for smoothing group and compute smoothing normals
      //std::map<int, vec3> smoothVertexNormals;
      //if (HasSmoothingGroup(shapes[s]) > 0) {
      //  std::cout << "Compute smoothingNormal for shape [" << s << "]"
      //            << std::endl;
      //  ComputeSmoothingNormals(attrib, shapes[s], smoothVertexNormals);
      //}

      for (size_t f = 0; f < shapes[s].mesh.indices.size() / 3; f++) {
        tinyobj::index_t idx0 = shapes[s].mesh.indices[3 * f + 0];
        tinyobj::index_t idx1 = shapes[s].mesh.indices[3 * f + 1];
        tinyobj::index_t idx2 = shapes[s].mesh.indices[3 * f + 2];

        int current_material_id = shapes[s].mesh.material_ids[f];

        if ((current_material_id < 0) ||
            (current_material_id >= static_cast<int>(materials.size()))) {
          // Invaid material ID. Use default material.
          current_material_id =
              int(materials.size()) -
              1;  // Default material is added to the last item in `materials`.
        }
        // if (current_material_id >= materials.size()) {
        //    std::cerr << "Invalid material index: " << current_material_id <<
        //    std::endl;
        //}
        //
        float diffuse[3];
        for (size_t i = 0; i < 3; i++) {
          diffuse[i] = materials[size_t(current_material_id)].diffuse[i];
        }
        float tc[3][2];
        if (attrib.texcoords.size() > 0) {
          if ((idx0.texcoord_index < 0) || (idx1.texcoord_index < 0) ||
              (idx2.texcoord_index < 0)) {
            // face does not contain valid uv index.
            tc[0][0] = 0.0f;
            tc[0][1] = 0.0f;
            tc[1][0] = 0.0f;
            tc[1][1] = 0.0f;
            tc[2][0] = 0.0f;
            tc[2][1] = 0.0f;
          } else {
            assert(attrib.texcoords.size() >
                   size_t(2 * idx0.texcoord_index + 1));
            assert(attrib.texcoords.size() >
                   size_t(2 * idx1.texcoord_index + 1));
            assert(attrib.texcoords.size() >
                   size_t(2 * idx2.texcoord_index + 1));

#if 1
            // Flip Y coord.
            tc[0][0] = attrib.texcoords[2 * size_t(idx0.texcoord_index)];
            tc[0][1] =
                1.0f - attrib.texcoords[2 * size_t(idx0.texcoord_index) + 1];
            tc[1][0] = attrib.texcoords[2 * size_t(idx1.texcoord_index)];
            tc[1][1] =
                1.0f - attrib.texcoords[2 * size_t(idx1.texcoord_index) + 1];
            tc[2][0] = attrib.texcoords[2 * size_t(idx2.texcoord_index)];
            tc[2][1] =
                1.0f - attrib.texcoords[2 * size_t(idx2.texcoord_index) + 1];
#else
            tc[0][0] = attrib.texcoords[2 * size_t(idx0.texcoord_index)];
            tc[0][1] = attrib.texcoords[2 * size_t(idx0.texcoord_index) + 1];
            tc[1][0] = attrib.texcoords[2 * size_t(idx1.texcoord_index)];
            tc[1][1] = attrib.texcoords[2 * size_t(idx1.texcoord_index) + 1];
            tc[2][0] = attrib.texcoords[2 * size_t(idx2.texcoord_index)];
            tc[2][1] = attrib.texcoords[2 * size_t(idx2.texcoord_index) + 1];
#endif
          }
        } else {
          tc[0][0] = 0.0f;
          tc[0][1] = 0.0f;
          tc[1][0] = 0.0f;
          tc[1][1] = 0.0f;
          tc[2][0] = 0.0f;
          tc[2][1] = 0.0f;
        }

        float v[3][3];
        for (size_t k = 0; k < 3; k++) {
          int f0 = idx0.vertex_index;
          int f1 = idx1.vertex_index;
          int f2 = idx2.vertex_index;
          assert(f0 >= 0);
          assert(f1 >= 0);
          assert(f2 >= 0);

          v[0][k] = attrib.vertices[3 * size_t(f0) + k];
          v[1][k] = attrib.vertices[3 * size_t(f1) + k];
          v[2][k] = attrib.vertices[3 * size_t(f2) + k];
          bmin[k] = std::min(v[0][k], bmin[k]);
          bmin[k] = std::min(v[1][k], bmin[k]);
          bmin[k] = std::min(v[2][k], bmin[k]);
          bmax[k] = std::max(v[0][k], bmax[k]);
          bmax[k] = std::max(v[1][k], bmax[k]);
          bmax[k] = std::max(v[2][k], bmax[k]);
        }

        float n[3][3];
        {
          bool invalid_normal_index = false;
          if (attrib.normals.size() > 0) {
            int nf0 = idx0.normal_index;
            int nf1 = idx1.normal_index;
            int nf2 = idx2.normal_index;

            if ((nf0 < 0) || (nf1 < 0) || (nf2 < 0)) {
              // normal index is missing from this face.
              invalid_normal_index = true;
            } else {
              for (size_t k = 0; k < 3; k++) {
                assert(size_t(3 * nf0) + k < attrib.normals.size());
                assert(size_t(3 * nf1) + k < attrib.normals.size());
                assert(size_t(3 * nf2) + k < attrib.normals.size());
                n[0][k] = attrib.normals[3 * size_t(nf0) + k];
                n[1][k] = attrib.normals[3 * size_t(nf1) + k];
                n[2][k] = attrib.normals[3 * size_t(nf2) + k];
              }
            }
          } else {
            invalid_normal_index = true;
          }

#if 0
          if (invalid_normal_index && !smoothVertexNormals.empty()) {
            // Use smoothing normals
            int f0 = idx0.vertex_index;
            int f1 = idx1.vertex_index;
            int f2 = idx2.vertex_index;

            if (f0 >= 0 && f1 >= 0 && f2 >= 0) {
              n[0][0] = smoothVertexNormals[f0].v[0];
              n[0][1] = smoothVertexNormals[f0].v[1];
              n[0][2] = smoothVertexNormals[f0].v[2];

              n[1][0] = smoothVertexNormals[f1].v[0];
              n[1][1] = smoothVertexNormals[f1].v[1];
              n[1][2] = smoothVertexNormals[f1].v[2];

              n[2][0] = smoothVertexNormals[f2].v[0];
              n[2][1] = smoothVertexNormals[f2].v[1];
              n[2][2] = smoothVertexNormals[f2].v[2];

              invalid_normal_index = false;
            }
          }
#endif

          if (invalid_normal_index) {
            // compute geometric normal
            CalcNormal(n[0], v[0], v[1], v[2]);
            n[1][0] = n[0][0];
            n[1][1] = n[0][1];
            n[1][2] = n[0][2];
            n[2][0] = n[0][0];
            n[2][1] = n[0][1];
            n[2][2] = n[0][2];
          }
        }

        for (int k = 0; k < 3; k++) {

          mesh.vertices.push_back(v[k][0]);
          mesh.vertices.push_back(v[k][1]);
          mesh.vertices.push_back(v[k][2]);

          // facevarying normals/texcoords
          mesh.normals.push_back(n[k][0]);
          mesh.normals.push_back(n[k][1]);
          mesh.normals.push_back(n[k][2]);

          mesh.texcoords.push_back(tc[k][0]);
          mesh.texcoords.push_back(tc[k][1]);

        }

        mesh.indices.push_back(3 * uint32_t(f) + 0);
        mesh.indices.push_back(3 * uint32_t(f) + 1);
        mesh.indices.push_back(3 * uint32_t(f) + 2);

        mesh.num_verts_per_faces.push_back(3);  // triangle
      }

      meshes->push_back(mesh);
    }
  }

  std::cout << "bmin = " << bmin[0] << ", " << bmin[1] << ", " << bmin[2]
            << "\n";
  std::cout << "bmax = " << bmax[0] << ", " << bmax[1] << ", " << bmax[2]
            << "\n";

  return true;
}
#endif


// Assume all triangle faces
static std::vector<uint32_t> SortIndices(const float ray_org[3], const float ray_dir[3],
  const std::vector<uint32_t> &indices,
  const std::vector<float> &vertices)
{
  size_t num_triangles = indices.size() / 3;

  std::cout << "sort: ray_org = " << ray_org[0] << ", " << ray_org[1] << ", " << ray_org[2] << "\n";
  std::cout << "sort: ray_dir = " << ray_dir[0] << ", " << ray_dir[1] << ", " << ray_dir[2] << "\n";

  face_sorter::TriangleFaceCenterAccessor<float> fa(
      vertices.data(), indices.data(), num_triangles);

  std::vector<uint32_t> sorted_face_indices;
  face_sorter::SortByBarycentricZ<float>(num_triangles, ray_org, ray_dir, fa,
                                         &sorted_face_indices);

  assert(num_triangles == sorted_face_indices.size());

  std::vector<uint32_t> sorted_indices;
  sorted_indices.resize(indices.size());

  for (size_t i = 0; i < num_triangles; i++) {
    size_t face_idx = sorted_face_indices[i];

    sorted_indices[3 * i + 0] = indices[3 * face_idx + 0];
    sorted_indices[3 * i + 1] = indices[3 * face_idx + 1];
    sorted_indices[3 * i + 2] = indices[3 * face_idx + 2];
  }

  return sorted_indices;
}

int main(int argc, char** argv) {
  std::string config_filename;

  if (argc < 3) {
    std::cout << "Usage: input.obj output-sorted.obj <config.json>\n";
    return EXIT_FAILURE;
  }

  if (argc > 3) {
    config_filename = argv[3];
  }

  // obj_filename in config.json is not used.
  std::string input_objfilename = argv[1];
  std::string output_objfilename = argv[2];

  example::RenderConfig render_config;

  if (!config_filename.empty()) {
    bool ret =
        example::LoadRenderConfig(&render_config, config_filename.c_str());
    if (!ret) {
      fprintf(stderr, "Failed to load [ %s ]\n", config_filename.c_str());
      return -1;
    }
  }

#if 1
  float bmin[3], bmax[3];
  std::vector<objlab::Mesh> meshes;
  std::vector<tinyobj::material_t> materials;       // not used

  // NOTE: mesh will be triangulated
  if (!LoadObjAndConvert(bmin, bmax, &meshes, materials, input_objfilename.c_str())) {
    std::cerr << "Failed to load .obj: " << render_config.obj_filename << "\n";
    return EXIT_FAILURE;
  }
#else

#endif

  std::cout << "meshes = " << meshes.size() << "\n";

  if (render_config.shape_id >= meshes.size()) {
    std::cerr << "Invalid shape_id : " << render_config.shape_id << ", must be less than meshes.size() = " << meshes.size() << "\n";
    return EXIT_FAILURE;
  }

  objlab::Mesh &mesh = meshes[size_t(render_config.shape_id)];

  mesh.sorted_indices = SortIndices(render_config.ray_org, render_config.ray_dir,
    mesh.indices, mesh.vertices);

  assert(mesh.sorted_indices.size() == mesh.indices.size());

  if (!objlab::SaveMeshAsObj(render_config.shape_id, meshes, output_objfilename)) {
    std::cerr << "Failed to save .obj: " << output_objfilename + "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
