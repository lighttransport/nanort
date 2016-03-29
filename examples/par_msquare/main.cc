#define NANORT_IMPLEMENTATION
#include "nanort.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PAR_MSQUARES_IMPLEMENTATION
#include "par_msquares.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

// PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
// http://www.pcg-random.org/
typedef struct {
  unsigned long long state;
  unsigned long long inc; // not used?
} pcg32_state_t;

#define PCG32_INITIALIZER                                                      \
  { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

float pcg32_random(pcg32_state_t *rng) {
  unsigned long long oldstate = rng->state;
  rng->state = oldstate * 6364136223846793005ULL + rng->inc;
  unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  unsigned int rot = oldstate >> 59u;
  unsigned int ret = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));

  return (float)((double)ret / (double)4294967296.0);
}

void pcg32_srandom(pcg32_state_t *rng, uint64_t initstate, uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1U) | 1U;
  pcg32_random(rng);
  rng->state += initstate;
  pcg32_random(rng);
}

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
  float3 operator/(const float3 &f2) const {
    return float3(x / f2.x, y / f2.y, z / f2.z);
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

//inline float vdot(float3 a, float3 b) {
//  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
//}

void calcNormal(float3 &N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N.normalize();
}

unsigned char fclamp(float x) {
  int i = (int)(powf(x, 1.0 / 2.2) * 256.0f);
  if (i > 255)
    i = 255;
  if (i < 0)
    i = 0;

  return (unsigned char)i;
}

void SaveImagePNG(const char *filename, const float *rgb, int width,
                  int height) {

  std::vector<unsigned char> ldr(width * height * 3);
  for (size_t i = 0; i < (size_t)(width * height * 3); i++) {
    ldr[i] = fclamp(rgb[i]);
  }

  int len = stbi_write_png(filename, width, height, 3, &ldr.at(0), width * 3);
  if (len < 1) {
    printf("Failed to save image\n");
    exit(-1);
  }
}

void BuildCameraFrame(float3 &corner, float3 &du, float3 &dv, const float3 &eye,
                      const float3 &lookat, const float3 &up, int width,
                      int height, float fov) {
  float flen =
      (0.5f * (double)height / tanf(0.5f * (double)(fov * M_PI / 180.0f)));
  float3 look;
  look = lookat - eye;
  du = vcross(look, up);
  du.normalize();

  dv = vcross(look, du);
  dv.normalize();

  look.normalize();
  look = flen * look + eye;

  corner = look - 0.5f * (width * du + height * dv);
}

typedef struct {
  std::vector<float> vertices;
  std::vector<unsigned int> faces;
  std::vector<int> facegroups;
} Mesh;

bool BuildMSQ(Mesh &meshOut, int &imgW, int &imgH, const char *filename) {
  int n;
  int cellsize = 4;
  float threshold = 0.3;

  // Load grayscale image.
  unsigned char *data = stbi_load(filename, &imgW, &imgH, &n, 1);
  if (data == NULL) {
    printf("Failed to load %s\n", filename);
    return false;
  }
  assert(n == 1);

  // Convert to float image
  std::vector<float> graydata(imgW * imgH);
  for (size_t i = 0; i < (size_t)(imgW * imgH); i++) {
    graydata[i] = data[i] / 255.0f;
  }

  free(data);

  printf("w x h = %d x %d\n", imgW, imgH);
  par_msquares_meshlist *mlist = par_msquares_from_grayscale(
      &graydata.at(0), imgW, imgH, cellsize, threshold,
      PAR_MSQUARES_DUAL | PAR_MSQUARES_HEIGHTS);
  int numMeshes = par_msquares_get_count(mlist);
  printf("numMeshes = %d\n", numMeshes);

  assert(numMeshes > 0);
  size_t vtxoffset = 0;
  int ntriangles = 0;
  for (int m = 0; m < numMeshes; m++) {
    par_msquares_mesh const *mesh = par_msquares_get_mesh(mlist, m);
    meshOut.facegroups.push_back(ntriangles);
    ntriangles += mesh->ntriangles;
    printf("numTriangles = %d\n", mesh->ntriangles);
    printf("numVerts = %d\n", mesh->npoints);
    printf("dim = %d\n", mesh->dim);

    float scale = 10.0f;
    for (int i = 0; i < mesh->npoints; i++) {
      if (mesh->dim == 2) {
        meshOut.vertices.push_back(scale * mesh->points[2 * i + 0]);
        meshOut.vertices.push_back(scale * mesh->points[2 * i + 1]);
        meshOut.vertices.push_back(0.0f);
      } else {
        // Zup -> Yup
        meshOut.vertices.push_back(scale * mesh->points[3 * i + 0]);
        meshOut.vertices.push_back(0.125f * scale *
                                   mesh->points[3 * i + 2]); // @fixme
        meshOut.vertices.push_back(scale * mesh->points[3 * i + 1]);
      }
    }

    for (int i = 0; i < mesh->ntriangles; i++) {
      if (mesh->triangles[3 * i + 0] > mesh->npoints) {
        exit(-1);
      }
      meshOut.faces.push_back(vtxoffset + mesh->triangles[3 * i + 0]);
      meshOut.faces.push_back(vtxoffset + mesh->triangles[3 * i + 1]);
      meshOut.faces.push_back(vtxoffset + mesh->triangles[3 * i + 2]);
    }

    vtxoffset += mesh->npoints;
  }

  par_msquares_free(mlist);

  return true;
}

void OrthoBasis(float3 basis[3], const float3 &n) {
  basis[2] = n;
  basis[1].x = 0.0;
  basis[1].y = 0.0;
  basis[1].z = 0.0;

  if ((n.x < 0.6) && (n.x > -0.6)) {
    basis[1].x = 1.0;
  } else if ((n.y < 0.6) && (n.y > -0.6)) {
    basis[1].y = 1.0;
  } else if ((n.z < 0.6) && (n.z > -0.6)) {
    basis[1].z = 1.0;
  } else {
    basis[1].x = 1.0;
  }

  basis[0] = vcross(basis[1], basis[2]);
  basis[0].normalize();

  basis[1] = vcross(basis[2], basis[0]);
  basis[1].normalize();
}

float3 ShadeAO(const float3 &P, const float3 &N, pcg32_state_t *rng,
               nanort::BVHAccel &accel, const float *vertices,
               const unsigned int *faces) {
  const int ntheta = 16;
  const int nphi = 32;

  float3 basis[3];
  OrthoBasis(basis, N);

  float occlusion = 0.0f;

  for (int j = 0; j < ntheta; j++) {
    for (int i = 0; i < nphi; i++) {
      float r0 = pcg32_random(rng);
      float r1 = pcg32_random(rng);
      double theta = sqrt(r0);
      double phi = 2.0f * M_PI * r1;

      double x = cos(phi) * theta;
      double y = sin(phi) * theta;
      double z = sqrt(1.0 - theta * theta);

      // local -> global
      double rx = x * basis[0].x + y * basis[1].x + z * basis[2].x;
      double ry = x * basis[0].y + y * basis[1].y + z * basis[2].y;
      double rz = x * basis[0].z + y * basis[1].z + z * basis[2].z;

      nanort::Ray ray;

      ray.org[0] = P[0] + rx * 0.0001f;
      ray.org[1] = P[1] + ry * 0.0001f;
      ray.org[2] = P[2] + rz * 0.0001f;
      ray.dir[0] = rx;
      ray.dir[1] = ry;
      ray.dir[2] = rz;
      ray.minT = 0.0f;
      ray.maxT = std::numeric_limits<float>::max();

      nanort::Intersection occIsect;
      nanort::BVHTraceOptions traceOptions;
      bool hit = accel.Traverse(&occIsect, vertices, faces, ray, traceOptions);
      if (hit) {
        occlusion += 1.0f;
      }
    }
  }

  occlusion = (ntheta * nphi - occlusion) / (float)(ntheta * nphi);

  float3 col;
  col[0] = occlusion;
  col[1] = occlusion;
  col[2] = occlusion;

  return col;
}

} // namespace

int main(int argc, char **argv) {
  int width = 512;
  int height = 513;

  if (argc < 2) {
    printf("Needs grayscale image\n");
    return 0;
  }

  int imgW = -1;
  int imgH = -1;
  Mesh mesh;
  bool ret = BuildMSQ(mesh, imgW, imgH, argv[1]);
  assert(ret);

  nanort::BVHBuildOptions options; // Use default option
  options.cacheBBox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", options.minLeafPrimitives);
  printf("    SAH binsize         : %d\n", options.binSize);

  timerutil t;
  t.start();

  nanort::BVHAccel accel;
  ret = accel.Build(&mesh.vertices.at(0), &mesh.faces.at(0),
                    mesh.faces.size() / 3, options);
  assert(ret);

  t.end();
  printf("  BVH build time: %f secs\n", t.msec() / 1000.0);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();

  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.numLeafNodes);
  printf("    # of branch nodes: %d\n", stats.numBranchNodes);
  printf("  Max tree depth     : %d\n", stats.maxTreeDepth);
  float bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  std::vector<float> rgb(width * height * 3, 0.0f);

  float3 eye, lookat, up;
  eye[0] = 5.0f;
  eye[1] = 12.5f;
  eye[2] = 20.0f;

  lookat[0] = 5.0f;
  lookat[1] = 0.0f;
  lookat[2] = 0.0f;

  up[0] = 0.0f;
  up[1] = 1.0f;
  up[2] = 0.0f;

  float3 corner, du, dv;
  BuildCameraFrame(corner, du, dv, eye, lookat, up, width, height, 45.0f);
  printf("corner = %f, %f, %f\n", corner[0], corner[1], corner[2]);

  const int numLines = 32;
  int numYBlocks = height / numLines;
  if (numYBlocks < 1)
    numYBlocks = 1;

#ifdef _OPENMP
  // Simple dynamic task processing.
  int counter = 0;
  int nthreads = omp_get_max_threads();

  printf("OMP # of trhreads = %d\n", nthreads);

#pragma omp parallel for
  for (int th = 0; th < nthreads; th++) {

    pcg32_state_t rng;
    pcg32_srandom(&rng, th, 0);

    while (1) {

      int yy = 0;

// @todo { replace with atomic if OMP 3.x is available }
#pragma omp critical
      {
        counter++;
        yy = counter;
      }

      if ((yy * numLines) >= height)
        break;

      int ybegin = yy * numLines;
      int yend = std::min(height, (yy + 1) * numLines);
      for (int y = ybegin; y < yend; y++) {

#else
  {
    {
      pcg32_state_t rng;
      pcg32_srandom(&rng, 0, 0);

      for (int y = 0; y < height; y++) {
#endif

        for (int x = 0; x < width; x++) {

          // Simple camera. change eye pos and direction fit to .obj model.

          nanort::Ray ray;
          ray.org[0] = eye[0];
          ray.org[1] = eye[1];
          ray.org[2] = eye[2];

          float3 dir;
          float pu = x + 0.5f;
          float pv = y + 0.5f;

          dir = corner + pu * du + pv * dv - eye;
          dir.normalize();
          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          nanort::Intersection isect;
          float tFar = 1.0e+30f;
          ray.minT = 0.0f;
          ray.maxT = tFar;
          nanort::BVHTraceOptions traceOptions;
          bool hit = accel.Traverse(&isect, &mesh.vertices.at(0),
                                    &mesh.faces.at(0), ray, traceOptions);
          if (hit) {
            // Write your shader here.
            float3 P;
            float3 N;
            unsigned int fid = isect.faceID;
            unsigned int f0, f1, f2;
            f0 = mesh.faces[3 * fid + 0];
            f1 = mesh.faces[3 * fid + 1];
            f2 = mesh.faces[3 * fid + 2];
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
            calcNormal(N, v0, v1, v2);

            P[0] = ray.org[0] + isect.t * ray.dir[0];
            P[1] = ray.org[1] + isect.t * ray.dir[1];
            P[2] = ray.org[2] + isect.t * ray.dir[2];

            float3 aoCol = ShadeAO(P, N, &rng, accel, &mesh.vertices.at(0),
                                   &mesh.faces.at(0));
            if (fid < (unsigned int)mesh.facegroups[1]) {
                // Ocean
                aoCol = aoCol.x * float3(0,0.25,0.5);
            } else {
                // Land
                aoCol = aoCol.x * float3(0,0.9,0.5);
            }
            rgb[3 * (y * width + x) + 0] = aoCol[0];
            rgb[3 * (y * width + x) + 1] = aoCol[1];
            rgb[3 * (y * width + x) + 2] = aoCol[2];
          }
        }
      }
    }
  }

  SaveImagePNG("render.png", &rgb.at(0), width, height);

  return 0;
}
