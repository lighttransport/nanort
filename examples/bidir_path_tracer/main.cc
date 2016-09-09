#include <iostream>
#include <algorithm>

#define NOMINMAX
#include "tiny_obj_loader.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "nanort.h"

#define USE_MULTIHIT_RAY_TRAVERSAL (0)

#ifndef M_PI
#define M_PI 3.141592683
#endif

const int uMaxBounces = 6;
const int SPP = 16;

typedef nanort::BVHAccel<nanort::TriangleMesh, nanort::TriangleSAHPred, nanort::TriangleIntersector<> > Accel;

namespace {

// This class is NOT thread-safe timer!

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
#include <windows.h>
#include <mmsystem.h>
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

#else // C timer
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
  float3 operator-() const {
    return float3(-x, -y, -z);
  }
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
  return min + float(rand())/RAND_MAX * (max-min);
}

float3 directionCosTheta(float3 normal, float *pdfOmega) {
  float u1 = uniformFloat(0, 1);
  float phi = uniformFloat(0, 2 * M_PI);

  float r = sqrt(u1);

  float x = r * cosf(phi);
  float y = r * sinf(phi);
  float z = sqrtf(1.0 - u1);

  *pdfOmega = 1.0f;

  float3 xDir = fabsf(normal.x) < fabsf(normal.y) ? float3(1, 0, 0) : float3(0, 1, 0);
  float3 yDir = normalize(vcross(normal, xDir));
  xDir = vcross(yDir, normal);
  return xDir * x + yDir * y + z * normal;
}


typedef struct {
  size_t num_vertices;
  size_t num_faces;
  float *vertices;              /// [xyz] * num_vertices
  float *facevarying_normals;   /// [xyz] * 3(triangle) * num_faces
  float *facevarying_tangents;  /// [xyz] * 3(triangle) * num_faces
  float *facevarying_binormals; /// [xyz] * 3(triangle) * num_faces
  float *facevarying_uvs;       /// [xyz] * 3(triangle) * num_faces
  float *facevarying_vertex_colors;   /// [xyz] * 3(triangle) * num_faces
  unsigned int *faces;         /// triangle x num_faces
  unsigned int *material_ids;   /// index x num_faces
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
  int normal_texid;     // normal map
  int alpha_texid;      // alpha map

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

void calcNormal(float3& N, float3 v0, float3 v1, float3 v2)
{
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N.normalize();
}

//Save in RAW headerless format, for use when exr tools are not available in system
void SaveImageRaw(const char* filename, const float* rgb, int width, int height) {
  std::vector<unsigned char>rawbuf;
  rawbuf.resize(3*width*height);
  unsigned char* raw = &rawbuf.at(0);

  // @note { Apply gamma correction would be nice? }
  for (int i = 0; i < width * height; i++) {
    raw[i*3] = (char)(rgb[3*i+0] * 255.0);
    raw[i*3+1] = (char)(rgb[3*i+1] * 255.0);
    raw[i*3+2] = (char)(rgb[3*i+2] * 255.0);
  }
  FILE* f=fopen(filename, "wb");
  if(!f){
    printf("Error: Couldnt open output binary file %s\n", filename);
    return;
  }
  fwrite(raw, 3*width*height, 1, f);
  fclose(f);
  printf("Info: Saved RAW RGB image of [%dx%d] dimensions to [ %s ]\n", width, height, filename);
}

void SaveImage(const char* filename, const float* rgb, int width, int height) {

  float* image_ptr[3];
  std::vector<float> images[3];
  images[0].resize(width * height);
  images[1].resize(width * height);
  images[2].resize(width * height);

  for (int i = 0; i < width * height; i++) {
    images[0][i] = rgb[3*i+0];
    images[1][i] = rgb[3*i+1];
    images[2][i] = rgb[3*i+2];
  }

  image_ptr[0] = &(images[2].at(0)); // B
  image_ptr[1] = &(images[1].at(0)); // G
  image_ptr[2] = &(images[0].at(0)); // R

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 3;
  const char* channel_names[] = {"B", "G", "R"}; // must be BGR order.

  image.channel_names = channel_names;
  image.images = (unsigned char**)image_ptr;
  image.width = width;
  image.height = height;

  image.pixel_types = (int *)malloc(sizeof(int) * image.num_channels);
  image.requested_pixel_types = (int *)malloc(sizeof(int) * image.num_channels);
  for (int i = 0; i < image.num_channels; i++) {
    image.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
    image.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
  }

  const char* err;
  int fail = SaveMultiChannelEXRToFile(&image, filename, &err);
  if (fail) {
    fprintf(stderr, "Error: %s\n", err);
  } else {
    printf("Saved image to [ %s ]\n", filename);
  }

  free(image.pixel_types);
  free(image.requested_pixel_types);

}

bool LoadObj(Mesh &mesh, std::vector<tinyobj::material_t>& materials, const char *filename, float scale) {
  std::vector<tinyobj::shape_t> shapes;
  std::string err;
  bool success = tinyobj::LoadObj(shapes, materials, err, filename);

  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!success) {
    return false;
  }

  std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
  std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size() << std::endl;

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

        f0 = shapes[i].mesh.indices[3*f+0];
        f1 = shapes[i].mesh.indices[3*f+1];
        f2 = shapes[i].mesh.indices[3*f+2];

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

        f0 = shapes[i].mesh.indices[3*f+0];
        f1 = shapes[i].mesh.indices[3*f+1];
        f2 = shapes[i].mesh.indices[3*f+2];

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

        f0 = shapes[i].mesh.indices[3*f+0];
        f1 = shapes[i].mesh.indices[3*f+1];
        f2 = shapes[i].mesh.indices[3*f+2];

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

enum VertexType {
    Light,
    Lens,
    Diffuse,
    Specular,
    Glossy
};

struct Vertex {
  float3 position;
  float3 normal;
  float3 beta;
  float3 refl;
  float pdf;
  VertexType type;

  float3 f(const Vertex &v) const {
    switch (type) {
    case Diffuse:
      return refl / M_PI;
      break;

    case Specular:
      return float3(1.0f, 1.0f, 1.0f);
      break;

    case Glossy:
      return float3(1.0f, 1.0f, 1.0f);

    default:
      return float3(0.0f, 0.0f, 0.0f);
    }
  }
};
} // namespace

inline float sign(float f) {
  return f < 0 ? -1 : 1;
}

inline float3 reflect(float3 I , float3 N) {
  return I - 2*vdot(I, N) * N;
}

inline float3 refract(float3 I, float3 N, float eta) {
  float NdotI = vdot(N, I);
  float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
  if (k < 0.0f)
      return float3(0, 0, 0);
  else
      return eta * I - (eta * NdotI + sqrtf(k)) * N;
}

inline float pow5(float val) {
  return val * val * val * val * val;
}

inline float fresnel_schlick(float3 H, float3 norm, float n1) {
  float r0 = n1 * n1;
  return r0 + (1-r0)*pow5(1 - vdot(H, norm));
}

void progressBar(int tick, int total, int width = 50) {
    float ratio = 100.0f * tick / total;
    float count = width * tick / total;
    std::string bar(width, ' ');
    std::fill(bar.begin(), bar.begin() + count, '+');
    printf("[ %6.2f %% ] [ %s ]%c", ratio, bar.c_str(), tick == total ? '\n' : '\r');
}

void pathtrace(int x, int y, int width, int height, 
               const Mesh &mesh, const std::vector<tinyobj::material_t> &materials, const Accel &accel,
               std::vector<Vertex> *eyeVert) {
    // Clear vertices.
    eyeVert->clear();

    // Simple camera. change eye pos and direction fit to .obj model.         
    float px = x + uniformFloat(-0.5, 0.5);
    float py = y + uniformFloat(-0.5, 0.5);
    float3 rayDir = float3((px / (float)width) - 0.5f, (py / (float)height) - 0.5f, -1.0f);
    rayDir.normalize();

    // Camera position.
    float3 rayOrg = float3(0.0f, 5.0f, 15.0f);

    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    double pdf = 1.0;
    for (int b = 0; b < uMaxBounces; ++b) {
        // Intersection test.
        nanort::Ray ray;
        float kFar = 1.0e+30f;
        ray.min_t = 0.001f;
        ray.max_t = kFar;

        ray.dir[0] = rayDir[0]; ray.dir[1] = rayDir[1]; ray.dir[2] = rayDir[2];
        ray.org[0] = rayOrg[0]; ray.org[1] = rayOrg[1]; ray.org[2] = rayOrg[2];

        nanort::TriangleIntersector<> triangle_intersector(mesh.vertices, mesh.faces);
        nanort::BVHTraceOptions trace_options;
        bool hit = accel.Traverse(ray, trace_options, triangle_intersector);

        if (!hit) {
            break;
        }

        // Russian Roulette
        float rr_fac = 1.0f;
        if(b > 3) {
            float rr_rand = uniformFloat(0, 1);
            float termination_probability = 0.2f;
            if (rr_rand < termination_probability) {
                break;
            }
            rr_fac = 1.0 - termination_probability;
        }
        pdf *= 1.0 / rr_fac;

        // Shading normal
        unsigned int fid = triangle_intersector.intersection.prim_id;
        float3 norm(0,0,0);
        if (mesh.facevarying_normals) {
        float3 normals[3];
        for(int vId = 0; vId < 3; vId++) {
            normals[vId][0] = mesh.facevarying_normals[9*fid + 3*vId + 0];
            normals[vId][1] = mesh.facevarying_normals[9*fid + 3*vId + 1];
            normals[vId][2] = mesh.facevarying_normals[9*fid + 3*vId + 2];
        }
        float u =  triangle_intersector.intersection.u;
        float v =  triangle_intersector.intersection.v;
        norm = (1.0 - u - v) * normals[0] + u  * normals[1] + v * normals[2];
        norm.normalize();
        }

        // Flip normal torwards incoming ray for backface shading
        float3 originalNorm = norm;
        if(vdot(norm, rayDir) > 0) {
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
        float inside = sign(vdot(rayDir, originalNorm)); // 1 for inside, -1 for outside
        // Assume ior of medium outside of objects = 1.0
        float n1 = inside < 0 ? 1.0 / ior : ior;
        float n2 = 1.0 / n1;
          
        float fresnel = fresnel_schlick(-rayDir, norm, (n1-n2)/(n1+n2));

        // Compute probabilities for each surface interaction.
        // Specular is just regular reflectiveness * fresnel.
        float rhoS = vdot(float3(1, 1, 1)/3.0f, specularColor) * fresnel;
        // If we don't have a specular reflection, choose either diffuse or transmissive
        // Mix them based on the dissolve value of the material
        float rhoD = vdot(float3(1, 1, 1)/3.0f, diffuseColor) * (1.0 - fresnel) * (1.0 - mat.dissolve);
        float rhoR = vdot(float3(1, 1, 1)/3.0f, refractionColor) * (1.0 - fresnel) * mat.dissolve;
          
        float rhoE = vdot(float3(1, 1, 1)/3.0f, emissiveColor);


        // Normalize probabilities so they sum to 1.0
        float totalrho = rhoS + rhoD + rhoR + rhoE;
        // No scattering event is likely, just stop here
        if(totalrho < 0.0001) {
        break;
        }

        rhoS /= totalrho;
        rhoD /= totalrho;
        rhoR /= totalrho;
        rhoE /= totalrho;
          
        // Choose an interaction based on the calculated probabilities
        float rand = uniformFloat(0, 1);
        float3 outDir;
        float3 refl;
        VertexType vtype;
        float pdfOmega = 0.0;
        // REFLECT glossy
        if (rand < rhoS)
        {
        outDir = reflect(rayDir, norm);
        refl = diffuseColor;
        pdfOmega = rhoS;
        vtype = Specular;
        }
        // REFLECT diffuse
        else if (rand < rhoS + rhoD)
        {
        // Sample cosine weighted hemisphere
        outDir = directionCosTheta(norm, &pdfOmega);
        // That's why there is no cos factor in here
        refl = diffuseColor;
        pdfOmega *= rhoD;
        vtype = Diffuse;
        }
        // REFRACT
        else if (rand < rhoD + rhoS + rhoR)
        {
        outDir = refract(rayDir, -inside * originalNorm, n1);
        refl = refractionColor;
        pdfOmega = rhoR;
        vtype = Specular;
        } 
        // EMIT
        else 
        {
            // If ray hits a light, stop path tracing.
            break;
        }

        // Convert pdf of solid angle sampling to face location sampling.
        float dist = triangle_intersector.intersection.t;
        float dot = std::abs(vdot(norm, rayDir));
        pdfOmega *= dot / dist;
        pdf *= pdfOmega;

        // Calculate new ray start position and set outgoing direction.
        rayOrg += rayDir * triangle_intersector.intersection.t;
        rayDir = outDir;

        // Add a new vertex for camera subpath.
        Vertex vertex;
        vertex.position = rayOrg;
        vertex.normal = norm;
        vertex.beta = throughput;
        vertex.refl = refl;
        vertex.pdf = pdf;
        vertex.type = vtype;
        eyeVert->push_back(vertex);

        throughput *= refl;
    }
}

void lighttrace(const Mesh &mesh, const std::vector<tinyobj::material_t> &materials, const Accel &accel,
                std::vector<Vertex> *lightVert) {
    // Clear vertices.
    lightVert->clear();

    // Sample light source.
    int num_light = 0;
    float light_area = 0.0;
    std::vector<float> areas;
    std::vector<int> light_ids;
    for (int i = 0; i < mesh.num_faces; i++) {
        int material_id = mesh.material_ids[i];
        const float *Le = materials[material_id].emission;
        if (std::max(Le[0], std::max(Le[1], Le[2])) <= 0.1f) continue;

        float3 v[3];
        for (int k = 0; k < 3; k++) {
            float x = mesh.vertices[mesh.faces[i * 3 + k] * 3 + 0];
            float y = mesh.vertices[mesh.faces[i * 3 + k] * 3 + 1];
            float z = mesh.vertices[mesh.faces[i * 3 + k] * 3 + 2];
            v[k] = float3(x, y, z);
        }

        light_ids.push_back(i);
        areas.push_back(0.5f * vcross(v[2] - v[0], v[1] - v[0]).length());
        light_area += areas[areas.size() - 1];
    }

    // Sample light.
    float r = rand() / (float)RAND_MAX;
    float accum = 0.0f;
    int light_id = -1;
    for (int i = 0; i < light_ids.size(); i++) {
        accum += areas[i] / light_area;
        if (accum >= r) {
            light_id = light_ids[i];
            break;
        }
    }

    // Sample light position.
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    if (u1 + u2) {
        u1 = 1.0f - u1;
        u2 = 1.0f - u2;
    }

    float3 v[3], n[3];
    for (int k = 0; k < 3; k++) {
        float x = mesh.vertices[mesh.faces[light_id * 3 + k] * 3 + 0];
        float y = mesh.vertices[mesh.faces[light_id * 3 + k] * 3 + 1];
        float z = mesh.vertices[mesh.faces[light_id * 3 + k] * 3 + 2];
        v[k] = float3(x, y, z);
        float nx = mesh.facevarying_normals[light_id * 9 + k * 3 + 0];
        float ny = mesh.facevarying_normals[light_id * 9 + k * 3 + 1];
        float nz = mesh.facevarying_normals[light_id * 9 + k * 3 + 2];
        n[k] = float3(nx, ny, nz);
    }
    float3 rayOrg = (1.0f - u1 - u2) * v[0] + u1 * v[1] + u2 * v[2];
    float3 norm = (1.0f - u1 - u2) * n[0] + u1 * n[1] + u2 * n[2];
    float pdfPos = 1.0f / light_area;

    // Sample light direction.
    float pdfDir;
    float3 rayDir = directionCosTheta(norm, &pdfDir);

    // Store vertex on light.
    Vertex vertex;
    vertex.position = rayOrg;
    vertex.normal = normalize(norm);
    vertex.beta = float3(materials[mesh.material_ids[light_id]].emission);
    vertex.pdf = pdfPos * pdfDir;
    vertex.type = Light;
    lightVert->push_back(vertex);

    // Light tracing.
    const float *Le = materials[mesh.material_ids[light_id]].emission;
    float3 throughput(Le[0], Le[1], Le[2]);
    std::vector<Vertex> vertices;
    double pdf = pdfPos * pdfDir;
    for (int b = 0; b < uMaxBounces; b++) {
        // Intersection test.
        nanort::Ray ray;
        float kFar = 1.0e+30f;
        ray.min_t = 0.001f;
        ray.max_t = kFar;

        ray.dir[0] = rayDir[0]; ray.dir[1] = rayDir[1]; ray.dir[2] = rayDir[2];
        ray.org[0] = rayOrg[0]; ray.org[1] = rayOrg[1]; ray.org[2] = rayOrg[2];

        nanort::TriangleIntersector<> triangle_intersector(mesh.vertices, mesh.faces);
        nanort::BVHTraceOptions trace_options;
        bool hit = accel.Traverse(ray, trace_options, triangle_intersector);

        if (!hit) {
            break;
        }

        // Russian Roulette
        float rr_fac = 1.0f;
        if(b > 3) {
            float rr_rand = uniformFloat(0, 1);
            float termination_probability = 0.2f;
            if (rr_rand < termination_probability) {
                break;
            }
            rr_fac = 1.0 - termination_probability;
        }
        pdf *= 1.0 / rr_fac;

        // Shading normal
        unsigned int fid = triangle_intersector.intersection.prim_id;
        float3 norm(0,0,0);
        if (mesh.facevarying_normals) {
        float3 normals[3];
        for(int vId = 0; vId < 3; vId++) {
            normals[vId][0] = mesh.facevarying_normals[9*fid + 3*vId + 0];
            normals[vId][1] = mesh.facevarying_normals[9*fid + 3*vId + 1];
            normals[vId][2] = mesh.facevarying_normals[9*fid + 3*vId + 2];
        }
        float u =  triangle_intersector.intersection.u;
        float v =  triangle_intersector.intersection.v;
        norm = (1.0 - u - v) * normals[0] + u  * normals[1] + v * normals[2];
        norm.normalize();
        }

        // Flip normal torwards incoming ray for backface shading
        float3 originalNorm = norm;
        if(vdot(norm, rayDir) > 0) {
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
        float inside = sign(vdot(rayDir, originalNorm)); // 1 for inside, -1 for outside
        // Assume ior of medium outside of objects = 1.0
        float n1 = inside < 0 ? 1.0 / ior : ior;
        float n2 = 1.0 / n1;
          
        float fresnel = fresnel_schlick(-rayDir, norm, (n1-n2)/(n1+n2));

        // Compute probabilities for each surface interaction.
        // Specular is just regular reflectiveness * fresnel.
        float rhoS = vdot(float3(1, 1, 1)/3.0f, specularColor) * fresnel;
        // If we don't have a specular reflection, choose either diffuse or transmissive
        // Mix them based on the dissolve value of the material
        float rhoD = vdot(float3(1, 1, 1)/3.0f, diffuseColor) * (1.0 - fresnel) * (1.0 - mat.dissolve);
        float rhoR = vdot(float3(1, 1, 1)/3.0f, refractionColor) * (1.0 - fresnel) * mat.dissolve;
          
        float rhoE = vdot(float3(1, 1, 1)/3.0f, emissiveColor);


        // Normalize probabilities so they sum to 1.0
        float totalrho = rhoS + rhoD + rhoR + rhoE;
        // No scattering event is likely, just stop here
        if(totalrho < 0.0001) {
        break;
        }

        rhoS /= totalrho;
        rhoD /= totalrho;
        rhoR /= totalrho;
        rhoE /= totalrho;
          
        // Choose an interaction based on the calculated probabilities
        float rand = uniformFloat(0, 1);
        float3 outDir;
        float3 refl;
        VertexType vtype;
        float pdfOmega = 0.0;
        // REFLECT glossy
        if (rand < rhoS)
        {
        outDir = reflect(rayDir, norm);
        refl = specularColor;
        pdfOmega = rhoS;
        vtype = Specular;
        }
        // REFLECT diffuse
        else if (rand < rhoS + rhoD)
        {
        // Sample cosine weighted hemisphere
        outDir = directionCosTheta(norm, &pdfOmega);
        // That's why there is no cos factor in here
        refl = diffuseColor;
        pdfOmega *= rhoD;
        vtype = Diffuse;
        }
        // REFRACT
        else if (rand < rhoD + rhoS + rhoR)
        {
        outDir = refract(rayDir, -inside * originalNorm, n1);
        refl = refractionColor;
        pdfOmega = rhoR;
        vtype = Specular;
        } 
        // EMIT
        else 
        {
            // If ray hits a light, stop tracing.
            break;
        }

        // Convert pdf of solid angle sampling to face location sampling.
        float dist = triangle_intersector.intersection.t;
        float dot = std::abs(vdot(norm, rayDir));
        pdfOmega *= dot / dist;
        pdf *= pdfOmega;

        // Calculate new ray start position and set outgoing direction.
        rayOrg += rayDir * triangle_intersector.intersection.t;
        rayDir = outDir;        

        // Add a new vertex for camera subpath.
        Vertex vertex;
        vertex.position = rayOrg;
        vertex.normal = norm;
        vertex.beta = throughput;
        vertex.refl = refl;
        vertex.pdf = pdf;
        vertex.type = vtype;
        lightVert->push_back(vertex);

        throughput *= refl;
    }
}

float pdfAt(const std::vector<const Vertex*> &path, int prev, int curr, int next) {
    const int pathLen = path.size();
    if (prev < 0 || prev >= pathLen) {
        return 1.0f;
    }

    if (next < 0 || next >= pathLen) {
        return 1.0f;
    }

    float3 wi = path[prev]->position - path[curr]->position;
    float3 wo = path[next]->position - path[curr]->position;
    float dist = wo.length();

    wi.normalize();
    wo.normalize();

    float pdf = 1.0f;
    switch (path[curr]->type) {
    case Diffuse:
        pdf = vdot(wi, wo);
        pdf = pdf > 0.0f ? pdf : 0.0f;
        break;

    default:
        pdf = 1.0f;
    }

    pdf *= vdot(path[curr]->normal, wo) / dist;

    if (pdf == 0.0f) pdf = 1.0f;
    return pdf;
}

float weightMIS(const std::vector<Vertex> &eyeVert, const std::vector<Vertex> &lightVert,
                int eyeId, int lightId, const Mesh &mesh, const Accel &accel) {
    if (eyeId == 0 && lightId == 0) return 1.0f;

    std::vector<const Vertex*> path;
    for (int i = 0; i <= eyeId; i++) {
        path.push_back(&eyeVert[i]);
    }

    for (int i = lightId; i >= 0; i--) {
        path.push_back(&lightVert[i]);
    }
    const int pathLen = path.size();

    float mis = 0.0;
    float prob = 1.0f;
    for (int i = eyeId; i >= 1; i--) {
        float pdfRev = pdfAt(path, i + 2, i + 1, i);
        float pdfFwd = pdfAt(path, i + 1, i, i - 1);
        prob *= pdfFwd / pdfRev;

        if (path[i]->type == Specular || path[i + 1]->type == Specular) continue;
        mis += prob * prob;
    }

    prob = 1.0f;
    for (int i = eyeId + 1; i < pathLen; i++) {
        float pdfRev = pdfAt(path, i - 2, i - 1, i);
        float pdfFwd = pdfAt(path, i - 1, i, i + 1);
        prob *= pdfFwd / pdfRev;

        if (path[i]->type == Specular || path[i - 1]->type == Specular) continue;
        mis += prob * prob;
    }

    if (mis != 0.0f) {
        mis = 1.0f / (1.0 + mis);
    }
    return mis;    
}

float calcG(const Vertex &v1, const Vertex &v2, const Mesh &mesh, const Accel &accel) {
    float3 to = v2.position - v1.position;
    float dist = to.length();
    to = to / dist;

    nanort::Ray ray;
    ray.org[0] = v1.position[0];
    ray.org[1] = v1.position[1];
    ray.org[2] = v1.position[2];
    ray.dir[0] = to[0];
    ray.dir[1] = to[1];
    ray.dir[2] = to[2];
    ray.min_t = 0.001f;
    ray.max_t = 1.0e30f;

    nanort::TriangleIntersector<> triangle_intersector(mesh.vertices, mesh.faces);
    nanort::BVHTraceOptions trace_options;
    bool hit = accel.Traverse(ray, trace_options, triangle_intersector);
    if (!hit) {
        return 0.0f;
    }

    if (std::abs(dist - triangle_intersector.intersection.t) > 0.001f) {
        return 0.0f;
    }

    float dot1 = vdot(to, v1.normal);
    float dot2 = vdot(-to, v2.normal);
    if (dot1 > 0.0f && dot2 > 0.0f) {
        float ret = dot1 * dot2 / (dist * dist);
        return ret;
    }

    return 0.0f;
}

float3 connectPath(const std::vector<Vertex> &eyeVert, const std::vector<Vertex> &lightVert,
                   const Mesh &mesh, const Accel &accel) {
    float3 color(0.0f, 0.0f, 0.0f);
    for (int e = 0; e < eyeVert.size(); e++) {
        for (int l = 0; l < lightVert.size(); l++) {
            float3 L(0.0f, 0.0f, 0.0f);
            const Vertex &ev = eyeVert[e];
            const Vertex &lv = lightVert[l];
            if (lv.pdf == 0.0f || ev.pdf == 0.0f) continue;

            if (l == 0) {
                L = ev.beta * ev.f(lv) * calcG(ev, lv, mesh, accel) * lv.beta / (lv.pdf * ev.pdf);
            } else {
                L = ev.beta * ev.f(lv) * calcG(ev, lv, mesh, accel) * lv.f(ev) * lv.beta / (lv.pdf * ev.pdf);
            }

            float mis = weightMIS(eyeVert, lightVert, e, l, mesh, accel);
            color += L * mis;
        }
    }
    return color;
}

int main(int argc, char** argv)
{
  int width = 256;
  int height = 256;

  float scale = 1.0f;

  std::string objFilename = "cornellbox_suzanne.obj";

  if (argc > 1) {
    objFilename = std::string(argv[1]);
  }

  if (argc > 2) {
    scale = atof(argv[2]);
  }

  #ifdef _OPENMP
  printf("Using OpenMP: yes!\n");
  #else
  printf("Using OpenMP: no!\n");
  #endif

  bool ret = false;

  // Load scene obj file
  Mesh mesh;
  std::vector<tinyobj::material_t> materials;
  ret = LoadObj(mesh, materials, objFilename.c_str(), scale);
  if (!ret) {
    fprintf(stderr, "Failed to load [ %s ]\n", objFilename.c_str());
    return -1;
  }

  nanort::BVHBuildOptions build_options; // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  timerutil t;
  t.start();

  nanort::TriangleMesh triangle_mesh(mesh.vertices, mesh.faces);
  nanort::TriangleSAHPred triangle_pred(mesh.vertices, mesh.faces);

  printf("num_triangles = %zu\n", mesh.num_faces);
  printf("faces = %p\n", mesh.faces);

  nanort::BVHAccel<nanort::TriangleMesh, nanort::TriangleSAHPred, nanort::TriangleIntersector<> > accel;
  ret = accel.Build(mesh.num_faces, build_options, triangle_mesh, triangle_pred);
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
  #ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 1)
  #endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float3 finalColor = float3(0, 0, 0);
      for(int i = 0; i < SPP; ++i) {
        
        std::vector<Vertex> eyeVert;
        pathtrace(x, y, width, height, mesh, materials, accel, &eyeVert); 

        std::vector<Vertex> lightVert;
        lighttrace(mesh, materials, accel, &lightVert);

        finalColor += connectPath(eyeVert, lightVert, mesh, accel);
      }

      finalColor *= 1.0 / SPP;

      // Gamme Correct
      finalColor[0] = pow(finalColor[0], 1.0/2.2);
      finalColor[1] = pow(finalColor[1], 1.0/2.2);
      finalColor[2] = pow(finalColor[2], 1.0/2.2);

      rgb[3 * ((height - y - 1) * width + x) + 0] = finalColor[0];
      rgb[3 * ((height - y - 1) * width + x) + 1] = finalColor[1];
      rgb[3 * ((height - y - 1) * width + x) + 2] = finalColor[2];

    }
    progressBar(y + 1, height);
  }

  // Save image.
  SaveImage("render.exr", &rgb.at(0), width, height);
  // Save Raw Image that can be opened by tools like GIMP
  SaveImageRaw("render.data", &rgb.at(0), width, height);

  return 0;
}
