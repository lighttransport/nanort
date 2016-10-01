/*
The MIT License (MIT)

Copyright (c) 2015 - 2016 Light Transport Entertainment, Inc.

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

#include "render.h"

#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#include "../../nanort.h"
#include "matrix.h"

#include "trackball.h"

#define TINYGLTF_LOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "tiny_gltf_loader.h"

namespace example {

// PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// http://www.pcg-random.org/
typedef struct {
  unsigned long long state;
  unsigned long long inc;  // not used?
} pcg32_state_t;

#define PCG32_INITIALIZER \
  { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

float pcg32_random(pcg32_state_t* rng) {
  unsigned long long oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  unsigned int rot = oldstate >> 59u;
  unsigned int ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

  return (float)((double)ret / (double)4294967296.0);
}

void pcg32_srandom(pcg32_state_t* rng, uint64_t initstate, uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1U) | 1U;
  pcg32_random(rng);
  rng->state += initstate;
  pcg32_random(rng);
}

const float kPI = 3.141592f;

typedef struct {
  size_t num_faces;
  std::vector<unsigned char> vertices;
  std::vector<unsigned int> faces;  /// triangle x num_faces
  size_t vertex_stride_bytes;
  size_t vertex_buffer_offset;

  const float* get_vertex_addr(size_t i) {
    return reinterpret_cast<const float*>(vertices.data() + i * vertex_stride_bytes);
  }
} Mesh;

Mesh gMesh;
nanort::BVHAccel<float, nanort::TriangleMesh<float>, nanort::TriangleSAHPred<float>,
                 nanort::TriangleIntersector<> >
    gAccel;

typedef nanort::real3<float> float3;

inline float3 Lerp3(float3 v0, float3 v1,
                            float3 v2, float u, float v) {
  return (1.0f - u - v) * v0 + u * v1 + v * v2;
}

inline void CalcNormal(float3& N, float3 v0, float3 v1,
                       float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}

void BuildCameraFrame(float3* origin, float3* corner,
                      float3* u, float3* v, float quat[4],
                      float eye[3], float lookat[3], float up[3], float fov,
                      int width, int height) {
  float e[4][4];

  Matrix::LookAt(e, eye, lookat, up);

  float r[4][4];
  build_rotmatrix(r, quat);

  float3 lo;
  lo[0] = lookat[0] - eye[0];
  lo[1] = lookat[1] - eye[1];
  lo[2] = lookat[2] - eye[2];
  float dist = vlength(lo);

  float dir[3];
  dir[0] = 0.0;
  dir[1] = 0.0;
  dir[2] = dist;

  Matrix::Inverse(r);

  float rr[4][4];
  float re[4][4];
  float zero[3] = {0.0f, 0.0f, 0.0f};
  float localUp[3] = {0.0f, 1.0f, 0.0f};
  Matrix::LookAt(re, dir, zero, localUp);

  // translate
  re[3][0] += eye[0];  // 0.0; //lo[0];
  re[3][1] += eye[1];  // 0.0; //lo[1];
  re[3][2] += (eye[2] - dist);

  // rot -> trans
  Matrix::Mult(rr, r, re);

  float m[4][4];
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      m[j][i] = rr[j][i];
    }
  }

  float vzero[3] = {0.0f, 0.0f, 0.0f};
  float eye1[3];
  Matrix::MultV(eye1, m, vzero);

  float lookat1d[3];
  dir[2] = -dir[2];
  Matrix::MultV(lookat1d, m, dir);
  float3 lookat1(lookat1d[0], lookat1d[1], lookat1d[2]);

  float up1d[3];
  Matrix::MultV(up1d, m, up);

  float3 up1(up1d[0], up1d[1], up1d[2]);

  // absolute -> relative
  up1[0] -= eye1[0];
  up1[1] -= eye1[1];
  up1[2] -= eye1[2];
  // printf("up1(after) = %f, %f, %f\n", up1[0], up1[1], up1[2]);

  // Use original up vector
  // up1[0] = up[0];
  // up1[1] = up[1];
  // up1[2] = up[2];

  {
    float flen =
        (0.5f * (float)height / tanf(0.5f * (float)(fov * kPI / 180.0f)));
    float3 look1;
    look1[0] = lookat1[0] - eye1[0];
    look1[1] = lookat1[1] - eye1[1];
    look1[2] = lookat1[2] - eye1[2];
    // vcross(u, up1, look1);
    // flip
    (*u) = nanort::vcross(look1, up1);
    (*u) = vnormalize((*u));

    (*v) = vcross(look1, (*u));
    (*v) = vnormalize((*v));

    look1 = vnormalize(look1);
    look1[0] = flen * look1[0] + eye1[0];
    look1[1] = flen * look1[1] + eye1[1];
    look1[2] = flen * look1[2] + eye1[2];
    (*corner)[0] = look1[0] - 0.5f * (width * (*u)[0] + height * (*v)[0]);
    (*corner)[1] = look1[1] - 0.5f * (width * (*u)[1] + height * (*v)[1]);
    (*corner)[2] = look1[2] - 0.5f * (width * (*u)[2] + height * (*v)[2]);

    (*origin)[0] = eye1[0];
    (*origin)[1] = eye1[1];
    (*origin)[2] = eye1[2];
  }
}

nanort::Ray<float> GenerateRay(const float3& origin,
                        const float3& corner, const float3& du,
                        const float3& dv, float u, float v) {
  float3 dir;

  dir[0] = (corner[0] + u * du[0] + v * dv[0]) - origin[0];
  dir[1] = (corner[1] + u * du[1] + v * dv[1]) - origin[1];
  dir[2] = (corner[2] + u * du[2] + v * dv[2]) - origin[2];
  dir = vnormalize(dir);

  float3 org;

  nanort::Ray<float> ray;
  ray.org[0] = origin[0];
  ray.org[1] = origin[1];
  ray.org[2] = origin[2];
  ray.dir[0] = dir[0];

  return ray;
}

static std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}

bool LoadGLTF(Mesh& mesh, const char* filename, float scale) {
  tinygltf::Scene scene;
  tinygltf::TinyGLTFLoader loader;
  std::string err;
  std::string input_filename(filename);
  std::string ext = GetFilePathExtension(input_filename);

  bool ret = false;
  if (ext.compare("glb") == 0) {
    // assume binary glTF.
    ret = loader.LoadBinaryFromFile(&scene, &err, input_filename.c_str());
  } else {
    // assume ascii glTF.
    ret = loader.LoadASCIIFromFile(&scene, &err, input_filename.c_str());
  }

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!ret) {
    return false;
  }

  size_t vertex_offset = 0;

  mesh.vertices.clear();
  mesh.faces.clear();

  std::map<std::string, tinygltf::Mesh>::const_iterator itMesh(
      scene.meshes.begin());
  std::map<std::string, tinygltf::Mesh>::const_iterator itMeshEnd(
      scene.meshes.end());

  for (; itMesh != itMeshEnd; itMesh++) {
    const tinygltf::Mesh& gltf_mesh = itMesh->second;

    for (size_t i = 0; i < gltf_mesh.primitives.size(); i++) {
      const tinygltf::Primitive& primitive = gltf_mesh.primitives[i];

      if (primitive.indices.empty()) continue;

      // Currently TRIANGLES only.
      assert(primitive.mode == TINYGLTF_MODE_TRIANGLES);

      std::map<std::string, std::string>::const_iterator it(
          primitive.attributes.begin());
      std::map<std::string, std::string>::const_iterator itEnd(
          primitive.attributes.end());

      for (; it != itEnd; it++) {
        const tinygltf::Accessor& accessor = scene.accessors[it->second];

        // @todo { Support multiple primitives. }
        if (it->first.compare("POSITION") == 0) {
          std::cout << accessor.bufferView << std::endl;
          const tinygltf::BufferView& vertexBufferView = scene.bufferViews[accessor.bufferView];
          const tinygltf::Buffer &vertexBuffer = scene.buffers[vertexBufferView.buffer];

          assert(accessor.type == TINYGLTF_TYPE_VEC3);
          assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
          assert(accessor.count > 0);

          const unsigned char* vertexData = vertexBuffer.data.data() + vertexBufferView.byteOffset + accessor.byteOffset;
          mesh.vertices.resize(accessor.count * accessor.byteStride);
          memcpy(mesh.vertices.data(), vertexData, accessor.count * accessor.byteStride);
          mesh.vertex_stride_bytes = accessor.byteStride;
        }
      }

      const tinygltf::Accessor &indexAccessor = scene.accessors[primitive.indices];

     // Convert face indices.
     assert((indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT) ||
            (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) ||
            (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_INT) ||
            (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT));
      
     const tinygltf::BufferView& indexBufferView = scene.bufferViews[indexAccessor.bufferView];
     const tinygltf::Buffer &indexBuffer = scene.buffers[indexBufferView.buffer];
     std::cout << "indexAccessor.bufferView = " << indexAccessor.bufferView << std::endl;
     std::cout << "indexAccessor.byteOffset = " << indexAccessor.byteOffset << std::endl;

     const unsigned char* indexData = indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;
     if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_SHORT) {
       const short* indexPtr = reinterpret_cast<const short*>(indexData);
       for (size_t f = 0; f < indexAccessor.count; f++) {
        mesh.faces.push_back(indexPtr[f] + vertex_offset);
       }
     } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
       const unsigned short* indexPtr = reinterpret_cast<const unsigned short*>(indexData);
       for (size_t f = 0; f < indexAccessor.count; f++) {
        mesh.faces.push_back(indexPtr[f] + vertex_offset);
       }
     } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_INT) {
       const int* indexPtr = reinterpret_cast<const int*>(indexData);
       for (size_t f = 0; f < indexAccessor.count; f++) {
        mesh.faces.push_back(indexPtr[f] + vertex_offset);
       }
     } else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
       const unsigned int* indexPtr = reinterpret_cast<const unsigned int*>(indexData);
       for (size_t f = 0; f < indexAccessor.count; f++) {
        mesh.faces.push_back(indexPtr[f] + vertex_offset);
       }
     }
      
    }
  }

  mesh.num_faces = mesh.faces.size() / 3;

  // @todo { material_ids, facevatying_normals, uvs, tangents, binormals }
  std::cout << "mesh.vertices.size = " << mesh.vertices.size() << std::endl;
  std::cout << "# of faces = " << mesh.num_faces << std::endl;

  return true;
}

bool Renderer::LoadGLTFMesh(const char* gltf_filename, float scene_scale) {
  return LoadGLTF(gMesh, gltf_filename, scene_scale);
}

bool Renderer::BuildBVH() {
  if (gMesh.num_faces < 1) {
    std::cout << "num_faces == 0" << std::endl;
    return false;
  }

  std::cout << "[Build BVH] " << std::endl;

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  auto t_start = std::chrono::system_clock::now();

  nanort::TriangleMesh<float> triangle_mesh(reinterpret_cast<const float*>(gMesh.vertices.data()), gMesh.faces.data(), gMesh.vertex_stride_bytes);
  nanort::TriangleSAHPred<float> triangle_pred(reinterpret_cast<const float*>(gMesh.vertices.data()),
                                        gMesh.faces.data(), gMesh.vertex_stride_bytes);

  printf("num_triangles = %lu\n", gMesh.num_faces);

  bool ret = gAccel.Build(gMesh.num_faces, build_options, triangle_mesh,
                          triangle_pred);
  assert(ret);

  auto t_end = std::chrono::system_clock::now();

  std::chrono::duration<double, std::milli> ms = t_end - t_start;
  std::cout << "BVH build time: " << ms.count() << " [ms]\n";

  nanort::BVHBuildStatistics stats = gAccel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  gAccel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  return true;
}

bool Renderer::Render(RenderLayer* layer, float quat[4],
                      const RenderConfig& config,
                      std::atomic<bool>& cancelFlag) {
  if (!gAccel.IsValid()) {
    return false;
  }

  int width = config.width;
  int height = config.height;

  // camera
  float eye[3] = {config.eye[0], config.eye[1], config.eye[2]};
  float look_at[3] = {config.look_at[0], config.look_at[1], config.look_at[2]};
  float up[3] = {config.up[0], config.up[1], config.up[2]};
  float fov = config.fov;
  float3 origin, corner, u, v;
  BuildCameraFrame(&origin, &corner, &u, &v, quat, eye, look_at, up, fov, width,
                   height);

  auto kCancelFlagCheckMilliSeconds = 300;

  std::vector<std::thread> workers;
  std::atomic<int> i(0);

  uint32_t num_threads = std::max(1U, std::thread::hardware_concurrency());

  auto startT = std::chrono::system_clock::now();

  // Initialize RNG.

  for (auto t = 0; t < num_threads; t++) {
    workers.emplace_back(std::thread([&, t]() {
      pcg32_state_t rng;
      pcg32_srandom(&rng, config.pass,
                    t);  // seed = combination of render pass + thread no.

      int y = 0;
      while ((y = i++) < config.height) {
        auto currT = std::chrono::system_clock::now();

        std::chrono::duration<double, std::milli> ms = currT - startT;
        // Check cancel flag
        if (ms.count() > kCancelFlagCheckMilliSeconds) {
          if (cancelFlag) {
            break;
          }
        }

        for (int x = 0; x < config.width; x++) {
          nanort::Ray<float> ray;
          ray.org[0] = origin[0];
          ray.org[1] = origin[1];
          ray.org[2] = origin[2];

          float u0 = pcg32_random(&rng);
          float u1 = pcg32_random(&rng);

          float3 dir;
          dir = corner + (float(x) + u0) * u +
                (float(config.height - y - 1) + u1) * v;
          dir = vnormalize(dir);
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          nanort::TriangleIntersector<> triangle_intersector(
              reinterpret_cast<const float*>(gMesh.vertices.data()), gMesh.faces.data(), gMesh.vertex_stride_bytes);
          nanort::BVHTraceOptions trace_options;
          bool hit = gAccel.Traverse(ray, trace_options, triangle_intersector);
          if (hit) {
            float3 p;
            p[0] =
                ray.org[0] + triangle_intersector.intersection.t * ray.dir[0];
            p[1] =
                ray.org[1] + triangle_intersector.intersection.t * ray.dir[1];
            p[2] =
                ray.org[2] + triangle_intersector.intersection.t * ray.dir[2];

            layer->position[4 * (y * config.width + x) + 0] = p.x();
            layer->position[4 * (y * config.width + x) + 1] = p.y();
            layer->position[4 * (y * config.width + x) + 2] = p.z();
            layer->position[4 * (y * config.width + x) + 3] = 1.0f;

            layer->varycoord[4 * (y * config.width + x) + 0] =
                triangle_intersector.intersection.u;
            layer->varycoord[4 * (y * config.width + x) + 1] =
                triangle_intersector.intersection.v;
            layer->varycoord[4 * (y * config.width + x) + 2] = 0.0f;
            layer->varycoord[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = triangle_intersector.intersection.prim_id;

            float3 N;
            { 
              unsigned int f0, f1, f2;
              f0 = gMesh.faces[3 * prim_id + 0];
              f1 = gMesh.faces[3 * prim_id + 1];
              f2 = gMesh.faces[3 * prim_id + 2];

              float3 v0, v1, v2;
              v0[0] = gMesh.get_vertex_addr(f0)[0];
              v0[1] = gMesh.get_vertex_addr(f0)[1];
              v0[2] = gMesh.get_vertex_addr(f0)[2];
              v1[0] = gMesh.get_vertex_addr(f1)[0];
              v1[1] = gMesh.get_vertex_addr(f1)[1];
              v1[2] = gMesh.get_vertex_addr(f1)[2];
              v2[0] = gMesh.get_vertex_addr(f2)[0];
              v2[1] = gMesh.get_vertex_addr(f2)[1];
              v2[2] = gMesh.get_vertex_addr(f2)[2];
              CalcNormal(N, v0, v1, v2);
            }

            layer->normal[4 * (y * config.width + x) + 0] = 0.5 * N[0] + 0.5;
            layer->normal[4 * (y * config.width + x) + 1] = 0.5 * N[1] + 0.5;
            layer->normal[4 * (y * config.width + x) + 2] = 0.5 * N[2] + 0.5;
            layer->normal[4 * (y * config.width + x) + 3] = 1.0f;

            layer->depth[4 * (y * config.width + x) + 0] =
                triangle_intersector.intersection.t;
            layer->depth[4 * (y * config.width + x) + 1] =
                triangle_intersector.intersection.t;
            layer->depth[4 * (y * config.width + x) + 2] =
                triangle_intersector.intersection.t;
            layer->depth[4 * (y * config.width + x) + 3] = 1.0f;

            // @todo { material }
            float diffuse_col[3] = {0.5f, 0.5f, 0.5f};
            float specular_col[3] = {0.0f, 0.0f, 0.0f};

            // Simple shading
            float NdotV = fabsf(vdot(N, dir));

            if (config.pass == 0) {
              layer->rgba[4 * (y * config.width + x) + 0] =
                  NdotV * diffuse_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] =
                  NdotV * diffuse_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] =
                  NdotV * diffuse_col[2];
              layer->rgba[4 * (y * config.width + x) + 3] = 1.0f;
              layer->sample_counts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              layer->rgba[4 * (y * config.width + x) + 0] +=
                  NdotV * diffuse_col[0];
              layer->rgba[4 * (y * config.width + x) + 1] +=
                  NdotV * diffuse_col[1];
              layer->rgba[4 * (y * config.width + x) + 2] +=
                  NdotV * diffuse_col[2];
              layer->rgba[4 * (y * config.width + x) + 3] += 1.0f;
              layer->sample_counts[y * config.width + x]++;
            }

          } else {
            {
              if (config.pass == 0) {
                // clear pixel
                layer->rgba[4 * (y * config.width + x) + 0] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 1] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 2] = 0.0f;
                layer->rgba[4 * (y * config.width + x) + 3] = 0.0f;
                layer->sample_counts[y * config.width + x] =
                    1;  // Set 1 for the first pass
              } else {
                layer->sample_counts[y * config.width + x]++;
              }

              // No super sampling
              layer->normal[4 * (y * config.width + x) + 0] = 0.0f;
              layer->normal[4 * (y * config.width + x) + 1] = 0.0f;
              layer->normal[4 * (y * config.width + x) + 2] = 0.0f;
              layer->normal[4 * (y * config.width + x) + 3] = 0.0f;
              layer->position[4 * (y * config.width + x) + 0] = 0.0f;
              layer->position[4 * (y * config.width + x) + 1] = 0.0f;
              layer->position[4 * (y * config.width + x) + 2] = 0.0f;
              layer->position[4 * (y * config.width + x) + 3] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 0] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 1] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 2] = 0.0f;
              layer->depth[4 * (y * config.width + x) + 3] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 0] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 1] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 2] = 0.0f;
              layer->texcoord[4 * (y * config.width + x) + 3] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 0] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 1] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 2] = 0.0f;
              layer->varycoord[4 * (y * config.width + x) + 3] = 0.0f;
            }
          }
        }
      }
    }));
  }

  for (auto& t : workers) {
    t.join();
  }

  return (!cancelFlag);
};

}  // namespace example
