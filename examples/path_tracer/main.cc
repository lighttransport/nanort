#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#if defined(NANORT_USE_CPP11_FEATURE)
#include <thread>
#include <mutex>
#endif

#define NOMINMAX
#include "tiny_obj_loader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "nanort.h"

#define USE_MULTIHIT_RAY_TRAVERSAL (0)

#ifndef M_PI
#define M_PI 3.141592683
#endif

const int uMaxBounces = 10;
// const int SPP = 10000;
const int SPP = 100;

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#include <windows.h>
#ifdef __cplusplus
}
#endif
#pragma comment(lib, "winmm.lib")
#else
#if defined(__unix__) || defined(__APPLE__)
#include <sys/time.h>
#else
#include <ctime>
#endif
#endif

namespace {

// This class is NOT thread-safe timer!
class timerutil {
 public:
#ifdef _WIN32
  typedef DWORD time_t;

  timerutil() { ::timeBeginPeriod(1); }
  ~timerutil() { ::timeEndPeriod(1); }

  void start() { t_[0] = ::timeGetTime(); }
  void end() { t_[1] = ::timeGetTime(); }

  time_t sec() { return (time_t)((t_[1] - t_[0]) / 1000); }
  time_t msec() { return (time_t)((t_[1] - t_[0])); }
  time_t usec() { return (time_t)((t_[1] - t_[0]) * 1000); }
  time_t current() { return ::timeGetTime(); }

#else
#if defined(__unix__) || defined(__APPLE__)
  typedef unsigned long int time_t;

  void start() { gettimeofday(tv + 0, &tz); }
  void end() { gettimeofday(tv + 1, &tz); }

  time_t sec() { return (time_t)(tv[1].tv_sec - tv[0].tv_sec); }
  time_t msec() {
    return this->sec() * 1000 +
           (time_t)((tv[1].tv_usec - tv[0].tv_usec) / 1000);
  }
  time_t usec() {
    return this->sec() * 1000000 + (time_t)(tv[1].tv_usec - tv[0].tv_usec);
  }
  time_t current() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (time_t)(t.tv_sec * 1000 + t.tv_usec);
  }

#else  // C timer
  // using namespace std;
  typedef clock_t time_t;

  void start() { t_[0] = clock(); }
  void end() { t_[1] = clock(); }

  time_t sec() { return (time_t)((t_[1] - t_[0]) / CLOCKS_PER_SEC); }
  time_t msec() { return (time_t)((t_[1] - t_[0]) * 1000 / CLOCKS_PER_SEC); }
  time_t usec() { return (time_t)((t_[1] - t_[0]) * 1000000 / CLOCKS_PER_SEC); }
  time_t current() { return (time_t)clock(); }

#endif
#endif

 private:
#ifdef _WIN32
  DWORD t_[2];
#else
#if defined(__unix__) || defined(__APPLE__)
  struct timeval tv[2];
  struct timezone tz;
#else
  time_t t_[2];
#endif
#endif
};

struct float3 {
  float3() {}
  float3(float xx, float yy, float zz) {
    x = xx;
    y = yy;
    z = zz;
  }
  float3(const float *p) {
    x = p[0];
    y = p[1];
    z = p[2];
  }

  float3 operator*(float f) const { return float3(x * f, y * f, z * f); }
  float3 operator-(const float3 &f2) const {
    return float3(x - f2.x, y - f2.y, z - f2.z);
  }
  float3 operator-() const { return float3(-x, -y, -z); }
  float3 operator*(const float3 &f2) const {
    return float3(x * f2.x, y * f2.y, z * f2.z);
  }
  float3 operator+(const float3 &f2) const {
    return float3(x + f2.x, y + f2.y, z + f2.z);
  }
  float3 &operator+=(const float3 &f2) {
    x += f2.x;
    y += f2.y;
    z += f2.z;
    return (*this);
  }
  float3 &operator*=(const float3 &f2) {
    x *= f2.x;
    y *= f2.y;
    z *= f2.z;
    return (*this);
  }
  float3 &operator*=(const float &f2) {
    x *= f2;
    y *= f2;
    z *= f2;
    return (*this);
  }
  float3 operator/(const float3 &f2) const {
    return float3(x / f2.x, y / f2.y, z / f2.z);
  }
  float3 operator/(const float &f2) const {
    return float3(x / f2, y / f2, z / f2);
  }
  float operator[](int i) const { return (&x)[i]; }
  float &operator[](int i) { return (&x)[i]; }

  float3 neg() { return float3(-x, -y, -z); }

  float length() { return sqrtf(x * x + y * y + z * z); }

  void normalize() {
    float len = length();
    if (fabs(len) > 1.0e-6) {
      float inv_len = 1.0 / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
  }

  float x, y, z;
  // float pad;  // for alignment
};

inline float3 normalize(float3 v) {
  v.normalize();
  return v;
}

inline float3 operator*(float f, const float3 &v) {
  return float3(v.x * f, v.y * f, v.z * f);
}

inline float3 vcross(float3 a, float3 b) {
  float3 c;
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

inline float vdot(float3 a, float3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float uniformFloat(float min, float max) {
  return min + float(rand()) / RAND_MAX * (max - min);
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
void revisedONB(const float3 &n, float3 &b1, float3 &b2) {
  if (n.z < 0.0f) {
    const float a = 1.0f / (1.0f - n.z);
    const float b = n.x * n.y * a;
    b1 = float3(1.0f - n.x * n.x * a, -b, n.x);
    b2 = float3(b, n.y * n.y * a - 1.0f, -n.y);
  } else {
    const float a = 1.0f / (1.0f + n.z);
    const float b = -n.x * n.y * a;
    b1 = float3(1.0f - n.x * n.x * a, b, -n.x);
    b2 = float3(b, 1.0f - n.y * n.y * a, -n.y);
  }
}

float3 directionCosTheta(float3 normal) {
  float u1 = uniformFloat(0, 1);
  float phi = uniformFloat(0, 2 * M_PI);

  float r = sqrt(u1);

  float x = r * cosf(phi);
  float y = r * sinf(phi);
  float z = sqrtf(1.0 - u1);

#if 0  // simpler way
  float3 xDir =
      fabsf(normal.x) < fabsf(normal.y) ? float3(1, 0, 0) : float3(0, 1, 0);
  float3 yDir = normalize(vcross(normal, xDir));
  xDir = vcross(yDir, normal);
#else  // better way
  float3 xDir, yDir;
  revisedONB(normal, xDir, yDir);
#endif
  return xDir * x + yDir * y + z * normal;
}

inline float PdfAtoW(const float aPdfA, const float aDist,
                     const float aCosThere) {
  return aPdfA * (aDist * aDist) / std::abs(aCosThere);
}

typedef struct {
  size_t num_vertices;
  size_t num_faces;
  float *vertices;                   /// [xyz] * num_vertices
  float *facevarying_normals;        /// [xyz] * 3(triangle) * num_faces
  float *facevarying_tangents;       /// [xyz] * 3(triangle) * num_faces
  float *facevarying_binormals;      /// [xyz] * 3(triangle) * num_faces
  float *facevarying_uvs;            /// [xyz] * 3(triangle) * num_faces
  float *facevarying_vertex_colors;  /// [xyz] * 3(triangle) * num_faces
  unsigned int *faces;               /// triangle x num_faces
  unsigned int *material_ids;        /// index x num_faces
} Mesh;

struct Material {
  float ambient[3];
  float diffuse[3];
  float reflection[3];
  float refraction[3];
  int id;
  int diffuse_texid;
  int reflection_texid;
  int transparency_texid;
  int bump_texid;
  int normal_texid;  // normal map
  int alpha_texid;   // alpha map

  Material() {
    ambient[0] = 0.0;
    ambient[1] = 0.0;
    ambient[2] = 0.0;
    diffuse[0] = 0.5;
    diffuse[1] = 0.5;
    diffuse[2] = 0.5;
    reflection[0] = 0.0;
    reflection[1] = 0.0;
    reflection[2] = 0.0;
    refraction[0] = 0.0;
    refraction[1] = 0.0;
    refraction[2] = 0.0;
    id = -1;
    diffuse_texid = -1;
    reflection_texid = -1;
    transparency_texid = -1;
    bump_texid = -1;
    normal_texid = -1;
    alpha_texid = -1;
  }
};

void calcNormal(float3 &N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N.normalize();
}

class EmissiveFace {
 public:
  EmissiveFace(unsigned int f = 0, unsigned int m = 0) : face_(f), mtl_(m) {}
  unsigned int face_;
  unsigned int mtl_;
};

class MeshLight {
 public:
  MeshLight(const Mesh &mesh, const std::vector<tinyobj::material_t> &materials)
      : mesh_(mesh), materials_(materials) {
    for (unsigned int face = 0; face < mesh_.num_faces; face++) {
      unsigned int mtl_id = mesh_.material_ids[face];
      const tinyobj::material_t &faceMtl = materials_[mtl_id];

      if (faceMtl.emission[0] > 0.0f || faceMtl.emission[1] > 0.0f ||
          faceMtl.emission[2] > 0.0f) {
        EmissiveFace ef(face, mtl_id);
        emissive_faces_.push_back(ef);
      }
    }
  }

  void sampleDirect(const float3 &x, float Xi1, float Xi2, float3 &dstDir,
                    float &dstDist, float &dstPdf, float3 &dstRadiance) const {
    unsigned int num_faces = emissive_faces_.size();
    unsigned int face = std::min(
        static_cast<unsigned int>(floor(Xi1 * num_faces)), num_faces - 1);
    float lightPickPdf = 1.0f / float(num_faces);

    // normalize random number
    Xi1 = Xi1 * num_faces - face;

    unsigned int fid = emissive_faces_[face].face_;
    unsigned int mtlid = emissive_faces_[face].mtl_;

    unsigned int f0 = mesh_.faces[3 * fid + 0];
    unsigned int f1 = mesh_.faces[3 * fid + 1];
    unsigned int f2 = mesh_.faces[3 * fid + 2];

    float3 v0, v1, v2;

    v0[0] = mesh_.vertices[3 * f0 + 0];
    v0[1] = mesh_.vertices[3 * f0 + 1];
    v0[2] = mesh_.vertices[3 * f0 + 2];

    v1[0] = mesh_.vertices[3 * f1 + 0];
    v1[1] = mesh_.vertices[3 * f1 + 1];
    v1[2] = mesh_.vertices[3 * f1 + 2];

    v2[0] = mesh_.vertices[3 * f2 + 0];
    v2[1] = mesh_.vertices[3 * f2 + 1];
    v2[2] = mesh_.vertices[3 * f2 + 2];

    float Xi1_ = std::sqrt(Xi1);
    float c0 = 1.0f - Xi1_;
    float c1 = Xi1_ * (1.0 - Xi2);
    float c2 = Xi1_ * Xi2;

    float3 tmp = vcross(v1 - v0, v2 - v0);
    float3 norm = normalize(tmp);
    float area = tmp.length() / 2.0f;

    dstPdf = 0.0;
    float3 lp = c0 * v0 + c1 * v1 + c2 * v2;
    float areaPdf = lightPickPdf * (1.0f / area);
    dstDir = lp - x;
    dstDist = dstDir.length();

    float3 ll = materials_[mtlid].emission;
    if (dstDist > 0.000001f) {
      dstDir.normalize();
      float cosAtLight = std::max(vdot(-dstDir, norm), 0.0f);
      dstRadiance = ll * cosAtLight;  // light has cosine edf

      // convert pdf to solid angle measure
      dstPdf = PdfAtoW(areaPdf, dstDist, cosAtLight);
    }
  }

  std::vector<EmissiveFace> emissive_faces_;
  const Mesh &mesh_;
  const std::vector<tinyobj::material_t> &materials_;
};

// Save in RAW headerless format, for use when exr tools are not available in
// system
void SaveImageRaw(const char *filename, const float *rgb, int width,
                  int height) {
  std::vector<unsigned char> rawbuf;
  rawbuf.resize(3 * width * height);
  unsigned char *raw = &rawbuf.at(0);

  // @note { Apply gamma correction would be nice? }
  for (int i = 0; i < width * height; i++) {
    raw[i * 3] = (char)(rgb[3 * i + 0] * 255.0);
    raw[i * 3 + 1] = (char)(rgb[3 * i + 1] * 255.0);
    raw[i * 3 + 2] = (char)(rgb[3 * i + 2] * 255.0);
  }
  FILE *f = fopen(filename, "wb");
  if (!f) {
    printf("Error: Couldnt open output binary file %s\n", filename);
    return;
  }
  fwrite(raw, 3 * width * height, 1, f);
  fclose(f);
  printf("Info: Saved RAW RGB image of [%dx%d] dimensions to [ %s ]\n", width,
         height, filename);
}

void SaveImagePNG(const char *filename, const float *rgb, int width,
                  int height) {
  unsigned char *bytes = new unsigned char[width * height * 3];
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const int index = y * width + x;
      bytes[index * 3 + 0] = (unsigned char)std::max(
          0.0f, std::min(rgb[index * 3 + 0] * 255.0f, 255.0f));
      bytes[index * 3 + 1] = (unsigned char)std::max(
          0.0f, std::min(rgb[index * 3 + 1] * 255.0f, 255.0f));
      bytes[index * 3 + 2] = (unsigned char)std::max(
          0.0f, std::min(rgb[index * 3 + 2] * 255.0f, 255.0f));
    }
  }
  stbi_write_png(filename, width, height, 3, bytes, width * 3);
  delete[] bytes;
}

void SaveImage(const char *filename, const float *rgb, int width, int height) {
  const char *err = NULL;
  int ret = SaveEXR(rgb, width, height, /* RGB */ 3, /* fp16 */0, filename, &err);
  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      fprintf(stderr, "EXR save error: %s(%d)\n", err, ret);
      FreeEXRErrorMessage(err);
    } else {
      fprintf(stderr, "EXR save error: %d\n", ret);
    }
  } else {
    printf("Saved image to [ %s ]\n", filename);
  }
}

bool LoadObj(Mesh &mesh, std::vector<tinyobj::material_t> &materials,
             const char *filename, float scale, const char *mtl_path) {
  std::vector<tinyobj::shape_t> shapes;

  std::string err = tinyobj::LoadObj(shapes, materials, filename, mtl_path);

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
  std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size()
            << std::endl;

  size_t num_vertices = 0;
  size_t num_faces = 0;
  for (size_t i = 0; i < shapes.size(); i++) {
    printf("  shape[%ld].name = %s\n", i, shapes[i].name.c_str());
    printf("  shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
    assert((shapes[i].mesh.indices.size() % 3) == 0);
    printf("  shape[%ld].vertices: %ld\n", i, shapes[i].mesh.positions.size());
    assert((shapes[i].mesh.positions.size() % 3) == 0);
    printf("  shape[%ld].normals: %ld\n", i, shapes[i].mesh.normals.size());
    assert((shapes[i].mesh.normals.size() % 3) == 0);

    num_vertices += shapes[i].mesh.positions.size() / 3;
    num_faces += shapes[i].mesh.indices.size() / 3;
  }
  std::cout << "[LoadOBJ] # of faces: " << num_faces << std::endl;
  std::cout << "[LoadOBJ] # of vertices: " << num_vertices << std::endl;

  // @todo { material and texture. }

  // Shape -> Mesh
  mesh.num_faces = num_faces;
  mesh.num_vertices = num_vertices;
  mesh.vertices = new float[num_vertices * 3];
  mesh.faces = new unsigned int[num_faces * 3];
  mesh.material_ids = new unsigned int[num_faces];
  memset(mesh.material_ids, 0, sizeof(int) * num_faces);
  mesh.facevarying_normals = new float[num_faces * 3 * 3];
  mesh.facevarying_uvs = new float[num_faces * 3 * 2];
  memset(mesh.facevarying_uvs, 0, sizeof(float) * 2 * 3 * num_faces);

  // @todo {}
  mesh.facevarying_tangents = NULL;
  mesh.facevarying_binormals = NULL;

  size_t vertexIdxOffset = 0;
  size_t faceIdxOffset = 0;
  for (size_t i = 0; i < shapes.size(); i++) {
    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
      mesh.faces[3 * (faceIdxOffset + f) + 0] =
          shapes[i].mesh.indices[3 * f + 0];
      mesh.faces[3 * (faceIdxOffset + f) + 1] =
          shapes[i].mesh.indices[3 * f + 1];
      mesh.faces[3 * (faceIdxOffset + f) + 2] =
          shapes[i].mesh.indices[3 * f + 2];

      mesh.faces[3 * (faceIdxOffset + f) + 0] += vertexIdxOffset;
      mesh.faces[3 * (faceIdxOffset + f) + 1] += vertexIdxOffset;
      mesh.faces[3 * (faceIdxOffset + f) + 2] += vertexIdxOffset;

      mesh.material_ids[faceIdxOffset + f] = shapes[i].mesh.material_ids[f];
    }

    for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
      mesh.vertices[3 * (vertexIdxOffset + v) + 0] =
          scale * shapes[i].mesh.positions[3 * v + 0];
      mesh.vertices[3 * (vertexIdxOffset + v) + 1] =
          scale * shapes[i].mesh.positions[3 * v + 1];
      mesh.vertices[3 * (vertexIdxOffset + v) + 2] =
          scale * shapes[i].mesh.positions[3 * v + 2];
    }

    if (shapes[i].mesh.normals.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0];
        f1 = shapes[i].mesh.indices[3 * f + 1];
        f2 = shapes[i].mesh.indices[3 * f + 2];

        float3 n0, n1, n2;

        n0[0] = shapes[i].mesh.normals[3 * f0 + 0];
        n0[1] = shapes[i].mesh.normals[3 * f0 + 1];
        n0[2] = shapes[i].mesh.normals[3 * f0 + 2];

        n1[0] = shapes[i].mesh.normals[3 * f1 + 0];
        n1[1] = shapes[i].mesh.normals[3 * f1 + 1];
        n1[2] = shapes[i].mesh.normals[3 * f1 + 2];

        n2[0] = shapes[i].mesh.normals[3 * f2 + 0];
        n2[1] = shapes[i].mesh.normals[3 * f2 + 1];
        n2[2] = shapes[i].mesh.normals[3 * f2 + 2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] = n0[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] = n1[2];

        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
        mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] = n2[2];
      }
    } else {
      // calc geometric normal
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0];
        f1 = shapes[i].mesh.indices[3 * f + 1];
        f2 = shapes[i].mesh.indices[3 * f + 2];

        float3 v0, v1, v2;

        v0[0] = shapes[i].mesh.positions[3 * f0 + 0];
        v0[1] = shapes[i].mesh.positions[3 * f0 + 1];
        v0[2] = shapes[i].mesh.positions[3 * f0 + 2];

        v1[0] = shapes[i].mesh.positions[3 * f1 + 0];
        v1[1] = shapes[i].mesh.positions[3 * f1 + 1];
        v1[2] = shapes[i].mesh.positions[3 * f1 + 2];

        v2[0] = shapes[i].mesh.positions[3 * f2 + 0];
        v2[1] = shapes[i].mesh.positions[3 * f2 + 1];
        v2[2] = shapes[i].mesh.positions[3 * f2 + 2];

        float3 N;
        calcNormal(N, v0, v1, v2);

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

    if (shapes[i].mesh.texcoords.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0];
        f1 = shapes[i].mesh.indices[3 * f + 1];
        f2 = shapes[i].mesh.indices[3 * f + 2];

        float3 n0, n1, n2;

        n0[0] = shapes[i].mesh.texcoords[2 * f0 + 0];
        n0[1] = shapes[i].mesh.texcoords[2 * f0 + 1];

        n1[0] = shapes[i].mesh.texcoords[2 * f1 + 0];
        n1[1] = shapes[i].mesh.texcoords[2 * f1 + 1];

        n2[0] = shapes[i].mesh.texcoords[2 * f2 + 0];
        n2[1] = shapes[i].mesh.texcoords[2 * f2 + 1];

        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];

        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];

        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
        mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
      }
    }

    vertexIdxOffset += shapes[i].mesh.positions.size() / 3;
    faceIdxOffset += shapes[i].mesh.indices.size() / 3;
  }

  return true;
}
}  // namespace

inline float sign(float f) { return f < 0 ? -1 : 1; }

inline float3 reflect(float3 I, float3 N) { return I - 2 * vdot(I, N) * N; }

inline float3 refract(float3 I, float3 N, float eta) {
  float NdotI = vdot(N, I);
  float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
  if (k < 0.0f)
    return float3(0, 0, 0);
  else
    return eta * I - (eta * NdotI + sqrtf(k)) * N;
}

inline float pow5(float val) { return val * val * val * val * val; }

inline float fresnel_schlick(float3 H, float3 norm, float n1) {
  float r0 = n1 * n1;
  return r0 + (1 - r0) * pow5(1 - vdot(H, norm));
}

void progressBar(int tick, int total, int width = 50) {
  float ratio = 100.0f * tick / total;
  float count = width * tick / total;
  std::string bar(width, ' ');
  std::fill(bar.begin(), bar.begin() + count, '+');
  printf("[ %6.2f %% ] [ %s ]%c", ratio, bar.c_str(),
         tick == total ? '\n' : '\r');
  std::fflush(stdout);
}

bool CheckForOccluder(float3 p1, float3 p2, const Mesh &mesh,
                      const nanort::BVHAccel<float> &accel) {
  static const float ray_eps = 0.00001f;

  float3 dir = p2 - p1;
  float dist = dir.length();
  dir.normalize();

  nanort::Ray<float> shadow_ray;
  shadow_ray.min_t = ray_eps;
  shadow_ray.max_t = dist - ray_eps;
  shadow_ray.dir[0] = dir[0];
  shadow_ray.dir[1] = dir[1];
  shadow_ray.dir[2] = dir[2];
  shadow_ray.org[0] = p1[0];
  shadow_ray.org[1] = p1[1];
  shadow_ray.org[2] = p1[2];

  nanort::TriangleIntersector<> triangle_intersector(mesh.vertices, mesh.faces,
                                                     sizeof(float) * 3);
  nanort::TriangleIntersection<> isect;
  if (!accel.Traverse(shadow_ray, triangle_intersector, &isect)) {
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  int width = 512;
  int height = 512;

  float scale = 1.0f;

  std::string objFilename = "../common/cornellbox_suzanne_lucy.obj";
  std::string mtlPath = "../common/";

  if (argc > 1) {
    objFilename = std::string(argv[1]);
  }

  if (argc > 2) {
    scale = atof(argv[2]);
  }

  if (argc > 3) {
    mtlPath = std::string(argv[3]);
  }

#ifdef _OPENMP
  printf("Using OpenMP: yes!\n");
#else
  printf("Using OpenMP: no!\n");
#endif

  bool ret = false;

  Mesh mesh;
  std::vector<tinyobj::material_t> materials;
  ret = LoadObj(mesh, materials, objFilename.c_str(), scale, mtlPath.c_str());
  if (!ret) {
    fprintf(stderr, "Failed to load [ %s ]\n", objFilename.c_str());
    return -1;
  }

  MeshLight lights(mesh, materials);

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  timerutil t;
  t.start();

  nanort::TriangleMesh<float> triangle_mesh(mesh.vertices, mesh.faces,
                                            sizeof(float) * 3);
  nanort::TriangleSAHPred<float> triangle_pred(mesh.vertices, mesh.faces,
                                               sizeof(float) * 3);

  printf("num_triangles = %lu\n", mesh.num_faces);
  printf("faces = %p\n", mesh.faces);

  nanort::BVHAccel<float> accel;
  ret =
      accel.Build(mesh.num_faces, triangle_mesh, triangle_pred, build_options);
  assert(ret);

  t.end();
  printf("  BVH build time: %f secs\n", t.msec() / 1000.0);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  std::vector<float> rgb(width * height * 3, 0.0f);

  srand(0);


// Shoot rays.
#if defined(NANORT_USE_CPP11_FEATURE)

  size_t num_threads = std::max(size_t(1), size_t(std::thread::hardware_concurrency()));
  std::vector<std::thread> workers;
  std::atomic<uint32_t> i(0);
  std::atomic<uint32_t> tcount(0); // thread_id counter

  std::cout << "# of threads = " << std::to_string(num_threads) << "\n";

  for (size_t t = 0; t < num_threads; t++) {
    workers.emplace_back(std::thread([&]() { 
      uint32_t y = 0;
      uint32_t tid = tcount++;
      while ((y = i++) < height) {
    
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < height; y++) {
#endif
    for (int x = 0; x < width; x++) {
      float3 finalColor = float3(0, 0, 0);
      for (int i = 0; i < SPP; ++i) {
        float px = x + uniformFloat(-0.5, 0.5);
        float py = y + uniformFloat(-0.5, 0.5);
        // Simple camera. change eye pos and direction fit to .obj model.

        float3 rayDir = float3((px / (float)width) - 0.5f,
                               (py / (float)height) - 0.5f, -1.0f);
        rayDir.normalize();

        float3 rayOrg = float3(0.0f, 5.0f, 20.0f);

        float3 color = float3(0, 0, 0);
        float3 weight = float3(1, 1, 1);

        int b;

        bool do_emmition = true;  // just skit emmition if light sampling was
                                  // done on previous event (No MIS)
        for (b = 0; b < uMaxBounces; ++b) {
          // Russian Roulette
          float rr_fac = 1.0f;
          if (b > 3) {
            float rr_rand = uniformFloat(0, 1);
            float termination_probability = 0.2f;
            if (rr_rand < termination_probability) {
              break;
            }
            rr_fac = 1.0 - termination_probability;
          }
          weight *= 1.0 / rr_fac;

          nanort::Ray<float> ray;
          float kFar = 1.0e+30f;
          ray.min_t = 0.001f;
          ray.max_t = kFar;

          ray.dir[0] = rayDir[0];
          ray.dir[1] = rayDir[1];
          ray.dir[2] = rayDir[2];
          ray.org[0] = rayOrg[0];
          ray.org[1] = rayOrg[1];
          ray.org[2] = rayOrg[2];

          nanort::TriangleIntersector<> triangle_intersector(
              mesh.vertices, mesh.faces, sizeof(float) * 3);
          nanort::TriangleIntersection<> isect;
          bool hit = accel.Traverse(ray, triangle_intersector, &isect);

          if (!hit) {
            break;
          }

          rayOrg += rayDir * isect.t;

          unsigned int fid = isect.prim_id;
          float3 norm(0, 0, 0);
          if (mesh.facevarying_normals) {
            float3 normals[3];
            for (int vId = 0; vId < 3; vId++) {
              normals[vId][0] = mesh.facevarying_normals[9 * fid + 3 * vId + 0];
              normals[vId][1] = mesh.facevarying_normals[9 * fid + 3 * vId + 1];
              normals[vId][2] = mesh.facevarying_normals[9 * fid + 3 * vId + 2];
            }
            float u = isect.u;
            float v = isect.v;
            norm = (1.0 - u - v) * normals[0] + u * normals[1] + v * normals[2];
            norm.normalize();
          }

          // Flip normal torwards incoming ray for backface shading
          float3 originalNorm = norm;
          if (vdot(norm, rayDir) > 0) {
            norm *= -1;
          }

          // Get properties from the material of the hit primitive
          unsigned int matId = mesh.material_ids[fid];
          tinyobj::material_t mat = materials[matId];

          float3 diffuseColor(mat.diffuse);
          float3 emissiveColor(mat.emission);
          float3 specularColor(mat.specular);
          float3 refractionColor(mat.transmittance);
          float ior = mat.ior;

          // Calculate fresnel factor based on ior.
          float inside =
              sign(vdot(rayDir, originalNorm));  // 1 for inside, -1 for outside
          // Assume ior of medium outside of objects = 1.0
          float n1 = inside < 0 ? 1.0 / ior : ior;
          float n2 = 1.0 / n1;

          float fresnel = fresnel_schlick(-rayDir, norm, (n1 - n2) / (n1 + n2));

          // Compute probabilities for each surface interaction.
          // Specular is just regular reflectiveness * fresnel.
          float rhoS = vdot(float3(1, 1, 1) / 3.0f, specularColor) * fresnel;
          // If we don't have a specular reflection, choose either diffuse or
          // transmissive
          // Mix them based on the dissolve value of the material
          float rhoD = vdot(float3(1, 1, 1) / 3.0f, diffuseColor) *
                       (1.0 - fresnel) * (1.0 - mat.dissolve);
          float rhoR = vdot(float3(1, 1, 1) / 3.0f, refractionColor) *
                       (1.0 - fresnel) * mat.dissolve;

          float rhoE = vdot(float3(1, 1, 1) / 3.0f, emissiveColor);

          // Normalize probabilities so they sum to 1.0
          float totalrho = rhoS + rhoD + rhoR + rhoE;
          // No scattering event is likely, just stop here
          if (totalrho < 0.0001) {
            break;
          }

          rhoS /= totalrho;
          rhoD /= totalrho;
          rhoR /= totalrho;
          rhoE /= totalrho;

          // Choose an interaction based on the calculated probabilities
          float rand = uniformFloat(0, 1);
          float3 outDir;
          // REFLECT glossy
          if (rand < rhoS) {
            outDir = reflect(rayDir, norm);
            weight *= specularColor;
            do_emmition = true;
          }
          // REFLECT diffuse
          else if (rand < rhoS + rhoD) {
            float3 brdfEval = (1.0f / M_PI) * diffuseColor;
            float3 dl = float3(0.0, 0.0, 0.0), ldir, ll;
            float lpdf, ldist;
            lights.sampleDirect(rayOrg, uniformFloat(0, 1), uniformFloat(0, 1),
                                ldir, ldist, lpdf, ll);

            if (lpdf > 0.0f) {
              float cosTheta = std::abs(vdot(ldir, norm));
              float3 directLight = (brdfEval * ll * cosTheta) / lpdf;
              bool visible =
                  !CheckForOccluder(rayOrg, rayOrg + ldir * ldist, mesh, accel);

              color += directLight * visible * weight;
            }

            // Sample cosine weighted hemisphere
            outDir = directionCosTheta(norm);
            weight *= diffuseColor;
            do_emmition = false;
          }
          // REFRACT
          else if (rand < rhoD + rhoS + rhoR) {
            outDir = refract(rayDir, -inside * originalNorm, n1);
            weight *= refractionColor;
            do_emmition = true;
          }
          // EMIT
          else {
            // Weight light by cosine factor (surface emits most light in normal
            // direction)
            if (do_emmition) {
              color += std::max(vdot(originalNorm, -rayDir), 0.0f) *
                       emissiveColor * weight;
            }
            break;
          }

          // Calculate new ray start position and set outgoing direction.
          rayDir = outDir;
        }

        finalColor += color;
      }

      finalColor *= 1.0 / SPP;

      // Gamme Correct
      finalColor[0] = pow(finalColor[0], 1.0 / 2.2);
      finalColor[1] = pow(finalColor[1], 1.0 / 2.2);
      finalColor[2] = pow(finalColor[2], 1.0 / 2.2);

      rgb[3 * ((height - y - 1) * width + x) + 0] = finalColor[0];
      rgb[3 * ((height - y - 1) * width + x) + 1] = finalColor[1];
      rgb[3 * ((height - y - 1) * width + x) + 2] = finalColor[2];
    }

#if defined(NANORT_USE_CPP11_FEATURE)
    if (tid == 0) { // master thread
#endif
      progressBar(y + 1, height);
#if defined(NANORT_USE_CPP11_FEATURE)
    }
#endif

#if defined(NANORT_USE_CPP11_FEATURE)
  } // while
  })); // lambda
  } // for num_threads

  for (auto &t : workers) {
    t.join();
  }
#else
  }
#endif

  // Save image.
  SaveImage("render.exr", &rgb.at(0), width, height);
  // Save Raw Image that can be opened by tools like GIMP
  SaveImageRaw("render.data", &rgb.at(0), width, height);
  SaveImagePNG("render.png", &rgb.at(0), width, height);

  return 0;
}
