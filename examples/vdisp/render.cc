/*
The MIT License (MIT)

Copyright (c) 2015 - 2018 Light Transport Entertainment, Inc.

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

#ifdef _MSC_VER
#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4189)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4477)
#endif

#include "render.h"
#include "common-util.h"
#include "matrix.h"

#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "trackball.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef WIN32
#undef min
#undef max
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wdouble-promotion"
#endif

namespace example {

static void ClearAOVPixel(size_t px, size_t py, RenderLayer* layer) {
  size_t idx = py * layer->width + px;
  layer->diffuse[4 * idx + 0] = 0.0f;
  layer->diffuse[4 * idx + 1] = 0.0f;
  layer->diffuse[4 * idx + 2] = 0.0f;
  layer->diffuse[4 * idx + 3] = 0.0f;

  layer->normal[4 * idx + 0] = 0.0f;
  layer->normal[4 * idx + 1] = 0.0f;
  layer->normal[4 * idx + 2] = 0.0f;
  layer->normal[4 * idx + 3] = 0.0f;

  layer->position[4 * idx + 0] = 0.0f;
  layer->position[4 * idx + 1] = 0.0f;
  layer->position[4 * idx + 2] = 0.0f;
  layer->position[4 * idx + 3] = 0.0f;

  layer->texcoord[4 * idx + 0] = 0.0f;
  layer->texcoord[4 * idx + 1] = 0.0f;
  layer->texcoord[4 * idx + 2] = 0.0f;
  layer->texcoord[4 * idx + 3] = 0.0f;

  layer->varycoord[4 * idx + 0] = 0.0f;
  layer->varycoord[4 * idx + 1] = 0.0f;
  layer->varycoord[4 * idx + 2] = 0.0f;
  layer->varycoord[4 * idx + 3] = 0.0f;

  layer->scatter_events[idx] = 0;
}

static void BuildCameraFrame(float3* origin, float3* corner, float3* u,
                             float3* v, float quat[4], float eye[3],
                             float lookat[3], float up[3], float fov, int width,
                             int height) {
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
        (0.5f * float(height) / std::tan(0.5f * float(fov * kPI / 180.0f)));
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

#if 0
static nanort::Ray<float> GenerateRay(const float3& origin, const float3& corner,
                               const float3& du, const float3& dv, float u,
                               float v) {
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
#endif

static void FetchTexture(const Scene& scene, int tex_idx, float u, float v,
                         float* col) {
  assert(tex_idx >= 0);
  const Texture& texture = scene.textures[size_t(tex_idx)];
  int tx = int(u * texture.width);
  int ty = int((1.0f - v) * texture.height);
  int idx_offset = (ty * texture.width + tx) * texture.components;
  col[0] = texture.image[size_t(idx_offset + 0)] / 255.f;
  col[1] = texture.image[size_t(idx_offset + 1)] / 255.f;
  col[2] = texture.image[size_t(idx_offset + 2)] / 255.f;
}

static void FetchNormal(const Scene& scene,
                        const nanort::TriangleIntersection<float>& isect,
                        float3& N) {
  const Mesh& mesh = scene.mesh;

  unsigned int prim_id = isect.prim_id;
  if (mesh.facevarying_normals.size() > 0) {
    float3 n0, n1, n2;
    n0[0] = mesh.facevarying_normals[9 * prim_id + 0];
    n0[1] = mesh.facevarying_normals[9 * prim_id + 1];
    n0[2] = mesh.facevarying_normals[9 * prim_id + 2];
    n1[0] = mesh.facevarying_normals[9 * prim_id + 3];
    n1[1] = mesh.facevarying_normals[9 * prim_id + 4];
    n1[2] = mesh.facevarying_normals[9 * prim_id + 5];
    n2[0] = mesh.facevarying_normals[9 * prim_id + 6];
    n2[1] = mesh.facevarying_normals[9 * prim_id + 7];
    n2[2] = mesh.facevarying_normals[9 * prim_id + 8];
    N = Lerp3(n0, n1, n2, isect.u, isect.v);
  } else {
    unsigned int f0, f1, f2;
    f0 = mesh.faces[3 * prim_id + 0];
    f1 = mesh.faces[3 * prim_id + 1];
    f2 = mesh.faces[3 * prim_id + 2];

    float3 v0, v1, v2;
    v0[0] = mesh.vertices[3 * f0 + 0];
    v0[1] = mesh.vertices[3 * f0 + 1];
    v0[2] = mesh.vertices[3 * f0 + 2];
    v1[0] = mesh.vertices[3 * f1 + 0];
    v1[1] = mesh.vertices[3 * f1 + 1];
    v1[2] = mesh.vertices[3 * f1 + 2];
    v2[0] = mesh.vertices[3 * f2 + 0];
    v2[1] = mesh.vertices[3 * f2 + 1];
    v2[2] = mesh.vertices[3 * f2 + 2];
    CalcNormal(N, v0, v1, v2);
  }
}

static bool FetchUV(const Scene& scene,
                    const nanort::TriangleIntersection<float>& isect,
                    float3& UV) {
  const Mesh& mesh = scene.mesh;
  unsigned int prim_id = isect.prim_id;
  if (mesh.facevarying_uvs.size() > 0) {
    float3 uv0, uv1, uv2;
    uv0[0] = mesh.facevarying_uvs[6 * prim_id + 0];
    uv0[1] = mesh.facevarying_uvs[6 * prim_id + 1];
    uv1[0] = mesh.facevarying_uvs[6 * prim_id + 2];
    uv1[1] = mesh.facevarying_uvs[6 * prim_id + 3];
    uv2[0] = mesh.facevarying_uvs[6 * prim_id + 4];
    uv2[1] = mesh.facevarying_uvs[6 * prim_id + 5];

    UV = Lerp3(uv0, uv1, uv2, isect.u, isect.v);
    return true;
  }
  return false;
}

static void FetchMaterialAndTexture(
    const Scene& scene, const nanort::TriangleIntersection<float>& isect,
    float* diffuse_col, float* specular_col, float& ior, float& dissolve) {
  const Mesh& mesh = scene.mesh;

  // Fetch material & texture
  unsigned int material_id = mesh.material_ids[isect.prim_id];

  diffuse_col[0] = 0.5f;
  diffuse_col[1] = 0.5f;
  diffuse_col[2] = 0.5f;

  specular_col[0] = 0.0f;
  specular_col[1] = 0.0f;
  specular_col[2] = 0.0f;

  ior = 1;

  dissolve = 0;

  const std::vector<Material>& materials = scene.materials;

  if (material_id < materials.size()) {
    int diffuse_texid = materials[material_id].diffuse_texid;
    float3 UV;
    bool flag = FetchUV(scene, isect, UV);

    if (diffuse_texid >= 0 && flag) {
      FetchTexture(scene, diffuse_texid, UV[0], UV[1], diffuse_col);
    } else {
      diffuse_col[0] = materials[material_id].diffuse[0];
      diffuse_col[1] = materials[material_id].diffuse[1];
      diffuse_col[2] = materials[material_id].diffuse[2];
    }

    ior = materials[material_id].ior;

    int alpha_texid = materials[material_id].alpha_texid;
    if (alpha_texid >= 0 && flag) {
      float dummy[3];
      FetchTexture(scene, alpha_texid, UV[0], UV[1], dummy);
      dissolve = dummy[0];
    } else {
      dissolve = materials[material_id].dissolve;
    }
  }
}

static void TraceRay(const Scene& scene, nanort::Ray<float> ray, size_t px,
                     size_t py, float* col, const RenderConfig& config,
                     RenderLayer* layer) {
  int depth = 0;

  nanort::TriangleIntersection<> isect;
  nanort::TriangleIntersector<> triangle_intersector(
      scene.mesh.vertices.data(), scene.mesh.faces.data(), sizeof(float) * 3);

  bool hit = scene.accel.Traverse(ray, triangle_intersector, &isect);

  if (!hit) {
    col[0] = 0.0f;
    col[1] = 0.0f;
    col[2] = 0.0f;
    return;
  }

  float3 P;
  P[0] = ray.org[0] + isect.t * ray.dir[0];
  P[1] = ray.org[1] + isect.t * ray.dir[1];
  P[2] = ray.org[2] + isect.t * ray.dir[2];

  float3 N;
  FetchNormal(scene, isect, N);

  // AOV
  if (depth == 0) {
    layer->position[4 * (py * layer->width + px) + 0] = P.x();
    layer->position[4 * (py * layer->width + px) + 1] = P.y();
    layer->position[4 * (py * layer->width + px) + 2] = P.z();
    layer->position[4 * (py * layer->width + px) + 3] = 1.0f;

    layer->varycoord[4 * (py * layer->width + px) + 0] = isect.u;
    layer->varycoord[4 * (py * layer->width + px) + 1] = isect.v;
    layer->varycoord[4 * (py * layer->width + px) + 2] = 0.0f;
    layer->varycoord[4 * (py * layer->width + px) + 3] = 1.0f;

    layer->normal[4 * (py * layer->width + px) + 0] = 0.5f * N[0] + 0.5f;
    layer->normal[4 * (py * layer->width + px) + 1] = 0.5f * N[1] + 0.5f;
    layer->normal[4 * (py * layer->width + px) + 2] = 0.5f * N[2] + 0.5f;
    layer->normal[4 * (py * layer->width + px) + 3] = 1.0f;

    layer->depth[(py * layer->width + px)] = isect.t;

    float diffuse_col[3] = {0.5f, 0.5f, 0.5f};
    float specular_col[3] = {0.f, 0.f, 0.f};

    // Fetch material & texture
    unsigned int material_id = scene.mesh.material_ids[isect.prim_id];

    if (static_cast<int>(material_id) >= 0) {
      float dummy;

      FetchMaterialAndTexture(scene, isect, diffuse_col, specular_col, dummy,
                              dummy);
    }

    if (config.pass == 0) {
      layer->diffuse[4 * (py * layer->width + px) + 0] = diffuse_col[0];
      layer->diffuse[4 * (py * layer->width + px) + 1] = diffuse_col[1];
      layer->diffuse[4 * (py * layer->width + px) + 2] = diffuse_col[2];
      layer->diffuse[4 * (py * layer->width + px) + 3] = 1.0f;
    } else {  // additive.
      layer->diffuse[4 * (py * layer->width + px) + 0] += diffuse_col[0];
      layer->diffuse[4 * (py * layer->width + px) + 1] += diffuse_col[1];
      layer->diffuse[4 * (py * layer->width + px) + 2] += diffuse_col[2];
      layer->diffuse[4 * (py * layer->width + px) + 3] += 1.0f;
    }
  }

  const float3 D = float3(ray.dir[0], ray.dir[1], ray.dir[2]);
  const float DDotN = vdot(D, N);

  col[0] = std::fabs(DDotN);
  col[1] = std::fabs(DDotN);
  col[2] = std::fabs(DDotN);
}

static std::string GetBaseDir(const std::string& filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

static std::string GetFileExtension(const std::string& filename) {
  if (filename.find_last_of(".") != std::string::npos)
    return filename.substr(filename.find_last_of(".") + 1);
  return "";
}

static int LoadTexture(const std::string& filename,
                       std::vector<Texture>* textures_inout) {
  if (filename.empty()) return -1;

  printf("  Loading texture : %s\n", filename.c_str());
  Texture texture;

  std::string ext = GetFileExtension(filename);

  if ((ext.compare("exr") == 0) || 
      (ext.compare("EXR") == 0)) {

  } else {
    // assume LDR image.

    int w, h, n;
    unsigned char* data = stbi_load(filename.c_str(), &w, &h, &n, 0);
    if (data) {
      texture.width = w;
      texture.height = h;
      texture.components = n;

      size_t n_elem = size_t(w * h * n);
      texture.image.resize(n_elem);

      // Load pixel as linear color space.
      for (size_t i = 0; i < n_elem; i++) {
        texture.image[i] = data[i] / 255.0f;
      }

      int id = int(textures_inout->size());
      textures_inout->push_back(texture);

      free(data);

      return id;
    }
  }

  printf("  Failed to load : %s\n", filename.c_str());
  return -1;
}

static bool LoadObj(Mesh& mesh, const char* filename, float scale,
                    Scene& scene) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;

  std::string basedir = GetBaseDir(filename) + "/";
  const char* basepath =
      (basedir.compare("/") == 0) ? nullptr : basedir.c_str();

  auto t_start = std::chrono::system_clock::now();

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename,
                              basepath, /* triangulate */ true);
  if (!ret) {
    std::cerr << "Failed to load .obj : " << filename << std::endl;
    if (!err.empty()) {
      std::cerr << "err : " << err << std::endl;
    }
    return false;
  }

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = t_end - t_start;

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  std::cout << "[LoadOBJ] Parse time : " << ms.count() << " [msecs]"
            << std::endl;

  std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
  std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size()
            << std::endl;

  size_t num_vertices = 0;
  size_t num_faces = 0;

  num_vertices = attrib.vertices.size() / 3;
  printf("  vertices: %ld\n", attrib.vertices.size() / 3);

  for (size_t i = 0; i < shapes.size(); i++) {
    printf("  shape[%ld].name = %s\n", i, shapes[i].name.c_str());
    printf("  shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
    assert((shapes[i].mesh.indices.size() % 3) == 0);

    num_faces += shapes[i].mesh.indices.size() / 3;
  }
  std::cout << "[LoadOBJ] # of faces: " << num_faces << std::endl;
  std::cout << "[LoadOBJ] # of vertices: " << num_vertices << std::endl;

  // Shape -> Mesh
  mesh.vertices.resize(num_vertices * 3, 0.0f);
  mesh.vertex_colors.resize(num_vertices * 3, 1.0f);
  mesh.faces.resize(num_faces * 3, 0);
  mesh.material_ids.resize(num_faces, 0);
  mesh.facevarying_normals.resize(num_faces * 3 * 3, 0.0f);
  mesh.facevarying_uvs.resize(num_faces * 3 * 2, 0.0f);

  // @todo {}
  // mesh.facevarying_tangents = NULL;
  // mesh.facevarying_binormals = NULL;

  // size_t vertexIdxOffset = 0;
  size_t faceIdxOffset = 0;

  for (size_t i = 0; i < attrib.vertices.size(); i++) {
    mesh.vertices[i] = scale * attrib.vertices[i];
  }

  for (size_t i = 0; i < attrib.colors.size(); i++) {
    mesh.vertex_colors[i] = attrib.colors[i];
  }

  for (size_t i = 0; i < shapes.size(); i++) {
    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
      mesh.faces[3 * (faceIdxOffset + f) + 0] =
          uint32_t(shapes[i].mesh.indices[3 * f + 0].vertex_index);
      mesh.faces[3 * (faceIdxOffset + f) + 1] =
          uint32_t(shapes[i].mesh.indices[3 * f + 1].vertex_index);
      mesh.faces[3 * (faceIdxOffset + f) + 2] =
          uint32_t(shapes[i].mesh.indices[3 * f + 2].vertex_index);

      mesh.material_ids[faceIdxOffset + f] =
          uint32_t(shapes[i].mesh.material_ids[f]);
    }

    if (attrib.normals.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0].normal_index;
        f1 = shapes[i].mesh.indices[3 * f + 1].normal_index;
        f2 = shapes[i].mesh.indices[3 * f + 2].normal_index;

        if (f0 > 0 && f1 > 0 && f2 > 0) {
          float3 n0, n1, n2;

          n0[0] = attrib.normals[size_t(3 * f0 + 0)];
          n0[1] = attrib.normals[size_t(3 * f0 + 1)];
          n0[2] = attrib.normals[size_t(3 * f0 + 2)];

          n1[0] = attrib.normals[size_t(3 * f1 + 0)];
          n1[1] = attrib.normals[size_t(3 * f1 + 1)];
          n1[2] = attrib.normals[size_t(3 * f1 + 2)];

          n2[0] = attrib.normals[size_t(3 * f2 + 0)];
          n2[1] = attrib.normals[size_t(3 * f2 + 1)];
          n2[2] = attrib.normals[size_t(3 * f2 + 2)];

          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] =
              n0[0];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] =
              n0[1];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] =
              n0[2];

          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] =
              n1[0];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] =
              n1[1];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] =
              n1[2];

          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] =
              n2[0];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] =
              n2[1];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] =
              n2[2];
        } else {  // face contains invalid normal index. calc geometric normal.
          f0 = shapes[i].mesh.indices[3 * f + 0].vertex_index;
          f1 = shapes[i].mesh.indices[3 * f + 1].vertex_index;
          f2 = shapes[i].mesh.indices[3 * f + 2].vertex_index;

          float3 v0, v1, v2;

          v0[0] = attrib.vertices[size_t(3 * f0 + 0)];
          v0[1] = attrib.vertices[size_t(3 * f0 + 1)];
          v0[2] = attrib.vertices[size_t(3 * f0 + 2)];

          v1[0] = attrib.vertices[size_t(3 * f1 + 0)];
          v1[1] = attrib.vertices[size_t(3 * f1 + 1)];
          v1[2] = attrib.vertices[size_t(3 * f1 + 2)];

          v2[0] = attrib.vertices[size_t(3 * f2 + 0)];
          v2[1] = attrib.vertices[size_t(3 * f2 + 1)];
          v2[2] = attrib.vertices[size_t(3 * f2 + 2)];

          float3 N;
          CalcNormal(N, v0, v1, v2);

          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] =
              N[0];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] =
              N[1];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] =
              N[2];

          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] =
              N[0];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] =
              N[1];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] =
              N[2];

          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] =
              N[0];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] =
              N[1];
          mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] =
              N[2];
        }
      }
    } else {
      // calc geometric normal
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0].vertex_index;
        f1 = shapes[i].mesh.indices[3 * f + 1].vertex_index;
        f2 = shapes[i].mesh.indices[3 * f + 2].vertex_index;

        float3 v0, v1, v2;

        v0[0] = attrib.vertices[size_t(3 * f0 + 0)];
        v0[1] = attrib.vertices[size_t(3 * f0 + 1)];
        v0[2] = attrib.vertices[size_t(3 * f0 + 2)];

        v1[0] = attrib.vertices[size_t(3 * f1 + 0)];
        v1[1] = attrib.vertices[size_t(3 * f1 + 1)];
        v1[2] = attrib.vertices[size_t(3 * f1 + 2)];

        v2[0] = attrib.vertices[size_t(3 * f2 + 0)];
        v2[1] = attrib.vertices[size_t(3 * f2 + 1)];
        v2[2] = attrib.vertices[size_t(3 * f2 + 2)];

        float3 N;
        CalcNormal(N, v0, v1, v2);

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] = N[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] = N[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] = N[2];
      }
    }

    if (attrib.texcoords.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0].texcoord_index;
        f1 = shapes[i].mesh.indices[3 * f + 1].texcoord_index;
        f2 = shapes[i].mesh.indices[3 * f + 2].texcoord_index;

        if (f0 > 0 && f1 > 0 && f2 > 0) {
          float3 n0, n1, n2;

          n0[0] = attrib.texcoords[size_t(2 * f0 + 0)];
          n0[1] = attrib.texcoords[size_t(2 * f0 + 1)];

          n1[0] = attrib.texcoords[size_t(2 * f1 + 0)];
          n1[1] = attrib.texcoords[size_t(2 * f1 + 1)];

          n2[0] = attrib.texcoords[size_t(2 * f2 + 0)];
          n2[1] = attrib.texcoords[size_t(2 * f2 + 1)];

          mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
          mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];

          mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
          mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];

          mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
          mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
        }
      }
    }

    faceIdxOffset += shapes[i].mesh.indices.size() / 3;
  }

  // material_t -> Material and Texture
  scene.materials.resize(materials.size());
  scene.textures.clear();

  for (size_t i = 0; i < materials.size(); i++) {
    scene.materials[i].diffuse[0] = materials[i].diffuse[0];
    scene.materials[i].diffuse[1] = materials[i].diffuse[1];
    scene.materials[i].diffuse[2] = materials[i].diffuse[2];
    scene.materials[i].ior = materials[i].ior;
    scene.materials[i].dissolve = 1.0f - materials[i].dissolve;

    scene.materials[i].id = int(i);

    // map_Kd
    scene.materials[i].diffuse_texid =
        LoadTexture(materials[i].diffuse_texname, &scene.textures);
    // map_Ks
    // scene.materials[i].specular_texid =
    //    LoadTexture(materials[i].specular_texname, &scene.textures);
    // map_d
    scene.materials[i].alpha_texid =
        LoadTexture(materials[i].alpha_texname, &scene.textures);

    // (vector) displacement map
    scene.materials[i].vdisp_texid =
        LoadTexture(materials[i].displacement_texname, &scene.textures);
  }

  return true;
}

bool Renderer::LoadObjMesh(const char* obj_filename, float scene_scale,
                           Scene& scene) {
  return LoadObj(scene.mesh, obj_filename, scene_scale, scene);
}

bool Renderer::Build(Scene& scene, RenderConfig& config) {
  std::cout << "[Build BVH] " << std::endl;

  nanort::BVHBuildOptions<float> options;  // Use default option
  options.cache_bbox = false;
  options.min_leaf_primitives = 16;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", options.bin_size);

  nanort::TriangleMesh<float> triangle_mesh(
      scene.mesh.vertices.data(), scene.mesh.faces.data(), sizeof(float) * 3);
  nanort::TriangleSAHPred<float> triangle_pred(
      scene.mesh.vertices.data(), scene.mesh.faces.data(), sizeof(float) * 3);

  std::cout << "# of vertices = " << scene.mesh.vertices.data() << std::endl;
  std::cout << "# of faces = " << scene.mesh.faces.data() << std::endl;

  bool ret = scene.accel.Build(uint32_t(scene.mesh.faces.size() / 3),
                               triangle_mesh, triangle_pred, options);

  if (!ret) {
    return false;
  }

  nanort::BVHBuildStatistics stats = scene.accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  scene.accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  // Fit camera to mesh's bounding box.
  config.eye[0] = (bmax[0] + bmin[0]) / 2.0f;
  config.eye[1] = (bmax[1] + bmin[1]) / 2.0f;
  config.eye[2] = bmax[2] + ((bmax[0] - bmin[0]) * 0.5f /
                             std::tan(config.fov * 0.5f * kPI / 180.f));

  config.look_at[0] = config.look_at[1] = 0.0f;
  config.look_at[2] = -1.0f;

  config.up[0] = config.up[2] = 0.0f;
  config.up[1] = 1.0f;

  return true;
}

bool Renderer::Render(const Scene& scene, float quat[4],
                      const RenderConfig& config, RenderLayer* layer,
                      std::atomic<bool>& cancelFlag) {
  if (!scene.accel.IsValid()) {
    return false;
  }

  size_t width = layer->width;
  size_t height = layer->height;

  // camera
  float eye[3] = {config.eye[0], config.eye[1], config.eye[2]};
  float look_at[3] = {config.look_at[0], config.look_at[1], config.look_at[2]};
  float up[3] = {config.up[0], config.up[1], config.up[2]};
  float fov = config.fov;
  float3 origin, corner, u, v;
  BuildCameraFrame(&origin, &corner, &u, &v, quat, eye, look_at, up, fov,
                   int(width), int(height));

  auto kCancelFlagCheckMilliSeconds = 300;

  std::vector<std::thread> workers;
  std::atomic<size_t> i(0);

  uint32_t num_threads = std::max(1U, std::thread::hardware_concurrency());

  auto startT = std::chrono::system_clock::now();

  // Initialize RNG.

  for (uint32_t t = 0; t < num_threads; t++) {
    workers.emplace_back(std::thread([&, t]() {
      // seed = combination of render pass + thread no.
      RNG rng(uint64_t(config.pass), uint64_t(t));

      size_t y = 0;
      while ((y = i++) < size_t(config.height)) {
        auto currT = std::chrono::system_clock::now();

        std::chrono::duration<double, std::milli> ms = currT - startT;
        // Check cancel flag
        if (ms.count() > kCancelFlagCheckMilliSeconds) {
          if (cancelFlag) {
            break;
          }
        }

        for (size_t x = 0; x < size_t(layer->width); x++) {
          nanort::Ray<float> ray;
          ray.org[0] = origin[0];
          ray.org[1] = origin[1];
          ray.org[2] = origin[2];

          float u0 = rng.Draw();
          float u1 = rng.Draw();

          float3 dir;
          dir = corner + (float(x) + u0) * u +
                (float(config.height - int(y) - 1) + u1) * v;
          dir = vnormalize(dir);
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          {
            float col[3];
            if (config.pass == 0) {
              ClearAOVPixel(size_t(x), size_t(y), layer);
            }

            TraceRay(scene, ray, size_t(x), size_t(y), col, config, 
                      layer);

            if (config.pass == 0) {
              layer->rgba[4 * (y * layer->width + x) + 0] = col[0];
              layer->rgba[4 * (y * layer->width + x) + 1] = col[1];
              layer->rgba[4 * (y * layer->width + x) + 2] = col[2];
              layer->rgba[4 * (y * layer->width + x) + 3] = 1.0f;
              layer->count[y * layer->width + x] =
                  1;  // Set 1 for the first pass
            } else {
              layer->rgba[4 * (y * layer->width + x) + 0] += col[0];
              layer->rgba[4 * (y * layer->width + x) + 1] += col[1];
              layer->rgba[4 * (y * layer->width + x) + 2] += col[2];
              layer->rgba[4 * (y * layer->width + x) + 3] += 1.0f;
              layer->count[y * layer->width + x]++;
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
