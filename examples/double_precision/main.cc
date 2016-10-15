#include "tiny_obj_loader.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "nanort.h"

#include <iostream>

#define USE_MULTIHIT_RAY_TRAVERSAL (0)

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

struct double3 {
  double3() {}
  double3(double xx, double yy, double zz) {
    x = xx;
    y = yy;
    z = zz;
  }
  double3(const double *p) {
    x = p[0];
    y = p[1];
    z = p[2];
  }

  double3 operator*(double f) const { return double3(x * f, y * f, z * f); }
  double3 operator-(const double3 &f2) const {
    return double3(x - f2.x, y - f2.y, z - f2.z);
  }
  double3 operator*(const double3 &f2) const {
    return double3(x * f2.x, y * f2.y, z * f2.z);
  }
  double3 operator+(const double3 &f2) const {
    return double3(x + f2.x, y + f2.y, z + f2.z);
  }
  double3 &operator+=(const double3 &f2) {
    x += f2.x;
    y += f2.y;
    z += f2.z;
    return (*this);
  }
  double3 operator/(const double3 &f2) const {
    return double3(x / f2.x, y / f2.y, z / f2.z);
  }
  double operator[](int i) const { return (&x)[i]; }
  double &operator[](int i) { return (&x)[i]; }

  double3 neg() { return double3(-x, -y, -z); }

  double length() { return sqrtf(x * x + y * y + z * z); }

  void normalize() {
    double len = length();
    if (fabs(len) > 1.0e-6) {
      double inv_len = 1.0 / len;
      x *= inv_len;
      y *= inv_len;
      z *= inv_len;
    }
  }

  double x, y, z;
  // double pad;  // for alignment
};

//inline double3 operator*(double f, const double3 &v) {
//  return double3(v.x * f, v.y * f, v.z * f);
//}

inline double3 vcross(double3 a, double3 b) {
  double3 c;
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
  return c;
}

//inline double vdot(double3 a, double3 b) {
//  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
//}


typedef struct {
  size_t num_vertices;
  size_t num_faces;
  double *vertices;              /// [xyz] * num_vertices
  double *facevarying_normals;   /// [xyz] * 3(triangle) * num_faces
  double *facevarying_tangents;  /// [xyz] * 3(triangle) * num_faces
  double *facevarying_binormals; /// [xyz] * 3(triangle) * num_faces
  double *facevarying_uvs;       /// [xyz] * 3(triangle) * num_faces
  double *facevarying_vertex_colors;   /// [xyz] * 3(triangle) * num_faces
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

void calcNormal(double3& N, double3 v0, double3 v1, double3 v2)
{
  double3 v10 = v1 - v0;
  double3 v20 = v2 - v0;

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

  int ret = SaveEXR(rgb, width, height, /* RGB*/3, filename);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "EXR save error: %d\n", ret);
  } else {
    printf("Saved image to [ %s ]\n", filename);
  }
}

bool LoadObj(Mesh &mesh, const char *filename, float scale, const char* mtl_path) {
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;

  std::string err = tinyobj::LoadObj(shapes, materials, filename, mtl_path);

  if (!err.empty()) {
    std::cerr << err << std::endl;
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
  mesh.vertices = new double[num_vertices * 3];
  mesh.faces = new unsigned int[num_faces * 3];
  mesh.material_ids = new unsigned int[num_faces];
  memset(mesh.material_ids, 0, sizeof(int) * num_faces);
  mesh.facevarying_normals = new double[num_faces * 3 * 3];
  mesh.facevarying_uvs = new double[num_faces * 3 * 2];
  memset(mesh.facevarying_uvs, 0, sizeof(double) * 2 * 3 * num_faces);

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

        double3 n0, n1, n2;

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

        double3 v0, v1, v2;

        v0[0] = shapes[i].mesh.positions[3 * f0 + 0];
        v0[1] = shapes[i].mesh.positions[3 * f0 + 1];
        v0[2] = shapes[i].mesh.positions[3 * f0 + 2];

        v1[0] = shapes[i].mesh.positions[3 * f1 + 0];
        v1[1] = shapes[i].mesh.positions[3 * f1 + 1];
        v1[2] = shapes[i].mesh.positions[3 * f1 + 2];

        v2[0] = shapes[i].mesh.positions[3 * f2 + 0];
        v2[1] = shapes[i].mesh.positions[3 * f2 + 1];
        v2[2] = shapes[i].mesh.positions[3 * f2 + 2];

        double3 N;
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

        double3 n0, n1, n2;

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

#if USE_MULTIHIT_RAY_TRAVERSAL
void IdToCol(float col[3], int mid)
{
    float table[8][3] = {
        { 1.0f, 0.0f, 0.0f },
        { 0.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 0.0f },
        { 1.0f, 0.0f, 1.0f },
        { 0.0f, 1.0f, 1.0f },
        { 1.0f, 1.0f, 0.0f },
        { 1.0f, 1.0f, 1.0f },
        { 0.5f, 0.5f, 0.5f }
    };

    int id = mid % 8; 

    col[0] = table[id][0];
    col[1] = table[id][1];
    col[2] = table[id][2];
}
#endif

} // namespace


int main(int argc, char** argv)
{
  int width = 1024;
  int height = 1024;

  float scale = 1.0f;

  std::string objFilename = "../common/cornellbox_suzanne.obj";
  std::string mtlPath = "../common/";

  if (argc > 1) {
    objFilename = std::string(argv[1]);
  }

  if (argc > 2) {
    scale = atof(argv[2]);
  }

  if (argc > 3) {
    mtlPath = std::string(mtlPath);
  }

  bool ret = false;

  Mesh mesh;
  ret = LoadObj(mesh, objFilename.c_str(), scale, mtlPath.c_str());
  if (!ret) {
    fprintf(stderr, "Failed to load [ %s ]\n", objFilename.c_str());
    return -1;
  }

  nanort::BVHBuildOptions<double> build_options; // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  timerutil t;
  t.start();

  nanort::TriangleMesh<double> triangle_mesh(mesh.vertices, mesh.faces, sizeof(double) * 3);
  nanort::TriangleSAHPred<double> triangle_pred(mesh.vertices, mesh.faces, sizeof(double) * 3);

  printf("num_triangles = %lu\n", mesh.num_faces);
  printf("faces = %p\n", mesh.faces);

  nanort::BVHAccel<double, nanort::TriangleMesh<double>, nanort::TriangleSAHPred<double>, nanort::TriangleIntersector<double> > accel;
  ret = accel.Build(mesh.num_faces, build_options, triangle_mesh, triangle_pred);
  assert(ret);

  t.end();
  printf("  BVH build time: %f secs\n", t.msec() / 1000.0);


  nanort::BVHBuildStatistics stats = accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  double bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);
 
  std::vector<float> rgb(width * height * 3, 0.0f);


  // VR panorama camera rendering.
  // Assume .obj data is modeled in [m] and camera center is placed at (0, 0, 0)
  // FYI, cornellbox_suzanne.obj is roughly within 10^3 [m], and floor is placed at -1.7 [m]

  float ipd = 0.0635; // inter-pupil distance [m]

  // Shoot rays.
  #ifdef _OPENMP
  #pragma omp parallel for
  #endif
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {

      // Upper half: left eye
      // Lower half: right eye
      bool is_left = (y < (height/2));

      float screen_y = 2.0f * (static_cast<float>(y) / static_cast<float>(height)) - 1.0f;

      float theta = 2.0f * M_PI * (static_cast<float>(x) / static_cast<float>(width)); // [0, 2 pi]
      float theta_offset = theta + ( is_left ? 0.0f : M_PI );
      float phi = (fmodf( 2.0f * ( 0.5f * screen_y + 0.5f ) , 1.0f ) - 0.5f ) * M_PI;

      nanort::Ray<double> ray;
      ray.org[0] = 0.5f * ipd * (-cosf(theta_offset));
      ray.org[1] = 0.0f;
      ray.org[2] = 0.5f * ipd * (sinf(theta_offset));

      double3 dir;
      dir[0] = cosf(phi) * -sinf(theta);
      dir[1] = sinf(phi);
      dir[2] = cosf(phi) * -cosf(theta);
      dir.normalize();
      ray.dir[0] = dir[0];
      ray.dir[1] = dir[1];
      ray.dir[2] = dir[2];

      float kFar = 1.0e+30f;
      ray.min_t = 0.0f;
      ray.max_t = kFar;

#if !USE_MULTIHIT_RAY_TRAVERSAL 
      nanort::TriangleIntersector<double, nanort::TriangleIntersection<double> > triangle_intersector(mesh.vertices, mesh.faces, sizeof(double) * 3);
      nanort::BVHTraceOptions trace_options;
      bool hit = accel.Traverse(ray, trace_options, triangle_intersector);
      if (hit) {
        // Write your shader here.
        double3 normal(0.0f, 0.0f, 0.0f);
        unsigned int fid = triangle_intersector.intersection.prim_id;
        if (mesh.facevarying_normals) {
          normal[0] = mesh.facevarying_normals[9*fid+0];
          normal[1] = mesh.facevarying_normals[9*fid+1];
          normal[2] = mesh.facevarying_normals[9*fid+2];
        }
        // Flip Y
        rgb[3 * ((height - y - 1) * width + x) + 0] = std::fabs(normal[0]);
        rgb[3 * ((height - y - 1) * width + x) + 1] = std::fabs(normal[1]);
        rgb[3 * ((height - y - 1) * width + x) + 2] = std::fabs(normal[2]);
      }
#else // multi-hit ray traversal.
      nanort::StackVector<nanort::TriangleIntersector, 128> isects;
      int max_isects = 8;
      nanort::BVHTraceOptions trace_options;
      bool hit = accel.MultiHitTraverse(ray, trace_options, max_isects, &isects);
      if (hit) {
        float col[3];
        IdToCol(col, isects->size()-1);
        //if (isects.size() >= 3) {
        //  for (int i = 0; i < (int)isects.size(); i++) {
        //    printf("t[%d]: %f\n", i, isects[i].t);
        //  }
        //  printf("max: %d\n", (int)isects.size());
        //}

        // Flip Y
        rgb[3 * ((height - y - 1) * width + x) + 0] = col[0];
        rgb[3 * ((height - y - 1) * width + x) + 1] = col[1];
        rgb[3 * ((height - y - 1) * width + x) + 2] = col[2];
      }
#endif

    }
  }

  // Save image.
  SaveImage("render.exr", &rgb.at(0), width, height);
  // Save Raw Image that can be opened by tools like GIMP
  SaveImageRaw("render.data", &rgb.at(0), width, height);

  return 0;
}
