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


#ifdef _MSC_VER
#pragma warning(disable : 4018)
#pragma warning(disable : 4244)
#pragma warning(disable : 4189)
#pragma warning(disable : 4996)
#pragma warning(disable : 4267)
#pragma warning(disable : 4477)
#endif

#include "render.h"

#include <chrono>  // C++11
#include <sstream>
#include <thread>  // C++11
#include <vector>

#include <iostream>

#include "../../nanort.h"
#include "matrix.h"

#define ESON_IMPLEMENTATION
#include "eson.h"

#ifdef USE_MULTITHREADING
#define TINYOBJ_LOADER_OPT_IMPLEMENTATION
#include "tinyobj_loader_opt.h"
#else
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#endif

#include "trackball.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef WIN32
#undef min
#undef max
#endif

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
  unsigned int ret = (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));

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
  size_t num_vertices;
  size_t num_faces;
  std::vector<float> vertices;               /// [xyz] * num_vertices
  std::vector<float> facevarying_normals;    /// [xyz] * 3(triangle) * num_faces
  std::vector<float> facevarying_tangents;   /// [xyz] * 3(triangle) * num_faces
  std::vector<float> facevarying_binormals;  /// [xyz] * 3(triangle) * num_faces
  std::vector<float> facevarying_uvs;        /// [xy]  * 3(triangle) * num_faces
  std::vector<float>
      vertex_colors;                         /// [rgb] * num_vertices
  std::vector<unsigned int> faces;         /// triangle x num_faces
  std::vector<unsigned int> material_ids;  /// index x num_faces
} Mesh;

struct Material {
  // float ambient[3];
  float diffuse[3];
  float specular[3];
  // float reflection[3];
  // float refraction[3];
  int id;
  int diffuse_texid;
  int specular_texid;
  // int reflection_texid;
  // int transparency_texid;
  // int bump_texid;
  // int normal_texid;  // normal map
  // int alpha_texid;  // alpha map

  Material() {
    // ambient[0] = 0.0;
    // ambient[1] = 0.0;
    // ambient[2] = 0.0;
    diffuse[0] = 0.5;
    diffuse[1] = 0.5;
    diffuse[2] = 0.5;
    specular[0] = 0.5;
    specular[1] = 0.5;
    specular[2] = 0.5;
    // reflection[0] = 0.0;
    // reflection[1] = 0.0;
    // reflection[2] = 0.0;
    // refraction[0] = 0.0;
    // refraction[1] = 0.0;
    // refraction[2] = 0.0;
    id = -1;
    diffuse_texid = -1;
    specular_texid = -1;
    // reflection_texid = -1;
    // transparency_texid = -1;
    // bump_texid = -1;
    // normal_texid = -1;
    // alpha_texid = -1;
  }
};

struct Texture {
  int width;
  int height;
  int components;
  unsigned char* image;

  Texture() {
    width = -1;
    height = -1;
    components = -1;
    image = NULL;
  }
};

Mesh gMesh;
std::vector<Material> gMaterials;
std::vector<Texture> gTextures;
nanort::BVHAccel<float> gAccel;

typedef nanort::real3<float> float3;

inline float3 Lerp3(float3 v0, float3 v1, float3 v2, float u, float v) {
  return (1.0f - u - v) * v0 + u * v1 + v * v2;
}

inline void CalcNormal(float3& N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}

void BuildCameraFrame(float3* origin, float3* corner, float3* u, float3* v,
                      float quat[4], float eye[3], float lookat[3], float up[3],
                      float fov, int width, int height) {
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

nanort::Ray<float> GenerateRay(const float3& origin, const float3& corner,
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

void FetchTexture(int tex_idx, float u, float v, float* col) {
  assert(tex_idx >= 0);
  Texture& texture = gTextures[tex_idx];
  int tx = u * texture.width;
  int ty = (1.0f - v) * texture.height;
  int idx_offset = (ty * texture.width + tx) * texture.components;
  col[0] = texture.image[idx_offset + 0] / 255.f;
  col[1] = texture.image[idx_offset + 1] / 255.f;
  col[2] = texture.image[idx_offset + 2] / 255.f;
}

static std::string GetBaseDir(const std::string &filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

int LoadTexture(const std::string& filename) {
  if (filename.empty()) return -1;

  printf("  Loading texture : %s\n", filename.c_str());
  Texture texture;

  int w, h, n;
  unsigned char* data = stbi_load(filename.c_str(), &w, &h, &n, 0);
  if (data) {
    texture.width = w;
    texture.height = h;
    texture.components = n;

    size_t n_elem = w * h * n;
    texture.image = new unsigned char[n_elem];
    for (size_t i = 0; i < n_elem; i++) {
      texture.image[i] = data[i];
    }

    gTextures.push_back(texture);
    return gTextures.size() - 1;
  }

  printf("  Failed to load : %s\n", filename.c_str());
  return -1;
}

static const char *mmap_file(size_t *len, const char* filename)
{
  (*len) = 0;
#ifdef _WIN32
  HANDLE file = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
  assert(file != INVALID_HANDLE_VALUE);

  HANDLE fileMapping = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
  assert(fileMapping != INVALID_HANDLE_VALUE);
 
  LPVOID fileMapView = MapViewOfFile(fileMapping, FILE_MAP_READ, 0, 0, 0);
  auto fileMapViewChar = (const char*)fileMapView;
  assert(fileMapView != NULL);

  LARGE_INTEGER fileSize;
  fileSize.QuadPart = 0;
  GetFileSizeEx(file, &fileSize);

  (*len) = static_cast<size_t>(fileSize.QuadPart);
  return fileMapViewChar;

#else

  FILE* f = fopen(filename, "rb" );
  if (!f) {
    fprintf(stderr, "Failed to open file : %s\n", filename);
    return nullptr;
  }
  fseek(f, 0, SEEK_END);
  long fileSize = ftell(f);
  fclose(f);

  if (fileSize < 16) {
    fprintf(stderr, "Empty or invalid .obj : %s\n", filename);
    return nullptr;
  }

  struct stat sb;
  char *p;
  int fd;

  fd = open (filename, O_RDONLY);
  if (fd == -1) {
    perror ("open");
    return nullptr;
  }

  if (fstat (fd, &sb) == -1) {
    perror ("fstat");
    return nullptr;
  }

  if (!S_ISREG (sb.st_mode)) {
    fprintf (stderr, "%s is not a file\n", "lineitem.tbl");
    return nullptr;
  }

  p = (char*)mmap (0, fileSize, PROT_READ, MAP_SHARED, fd, 0);

  if (p == MAP_FAILED) {
    perror ("mmap");
    return nullptr;
  }

  if (close (fd) == -1) {
    perror ("close");
    return nullptr;
  }

  (*len) = fileSize;

  return p;
  
#endif
}

static bool gz_load(std::vector<char>* buf, const char* filename)
{
#ifdef ENABLE_ZLIB
    gzFile file;
    file = gzopen (filename, "r");
    if (! file) {
        fprintf (stderr, "gzopen of '%s' failed: %s.\n", filename,
                 strerror (errno));
        exit (EXIT_FAILURE);
        return false;
    }
    while (1) {
        int err;                    
        int bytes_read;
        unsigned char buffer[1024];
        bytes_read = gzread (file, buffer, 1024);
        buf->insert(buf->end(), buffer, buffer + 1024); 
        //printf ("%s", buffer);
        if (bytes_read < 1024) {
            if (gzeof (file)) {
                break;
            }
            else {
                const char * error_string;
                error_string = gzerror (file, & err);
                if (err) {
                    fprintf (stderr, "Error: %s.\n", error_string);
                    exit (EXIT_FAILURE);
                    return false;
                }
            }
        }
    }
    gzclose (file);
    return true;
#else
  return false;
#endif
}

#ifdef ENABLE_ZSTD
static off_t fsize_X(const char *filename)
{
    struct stat st;
    if (stat(filename, &st) == 0) return st.st_size;
    /* error */
    printf("stat: %s : %s \n", filename, strerror(errno));
    exit(1);
}

static FILE* fopen_X(const char *filename, const char *instruction)
{
    FILE* const inFile = fopen(filename, instruction);
    if (inFile) return inFile;
    /* error */
    printf("fopen: %s : %s \n", filename, strerror(errno));
    exit(2);
}

static void* malloc_X(size_t size)
{
    void* const buff = malloc(size);
    if (buff) return buff;
    /* error */
    printf("malloc: %s \n", strerror(errno));
    exit(3);
}
#endif

static bool zstd_load(std::vector<char>* buf, const char* filename)
{
#ifdef ENABLE_ZSTD
    off_t const buffSize = fsize_X(filename);
    FILE* const inFile = fopen_X(filename, "rb");
    void* const buffer = malloc_X(buffSize);
    size_t const readSize = fread(buffer, 1, buffSize, inFile);
    if (readSize != (size_t)buffSize) {
        printf("fread: %s : %s \n", filename, strerror(errno));
        exit(4);
    }
    fclose(inFile);

    unsigned long long const rSize = ZSTD_getDecompressedSize(buffer, buffSize);
    if (rSize==0) {
        printf("%s : original size unknown \n", filename);
        exit(5);
    }

    buf->resize(rSize);

    size_t const dSize = ZSTD_decompress(buf->data(), rSize, buffer, buffSize);

    if (dSize != rSize) {
        printf("error decoding %s : %s \n", filename, ZSTD_getErrorName(dSize));
        exit(7);
    }

    free(buffer);

    return true;
#else
  return false;
#endif
}

static const char* get_file_data(size_t *len, const char* filename)
{

  const char *ext = strrchr(filename, '.');

  size_t data_len = 0;
  const char* data = nullptr;

  if (strcmp(ext, ".gz") == 0) {
    // gzipped data.

    std::vector<char> buf;
    bool ret = gz_load(&buf, filename);

    if (ret) {
      char *p = static_cast<char*>(malloc(buf.size() + 1));  // @fixme { implement deleter }
      memcpy(p, &buf.at(0), buf.size());
      p[buf.size()] = '\0';
      data = p;
      data_len = buf.size();
    }

  } else if (strcmp(ext, ".zst") == 0) {
    printf("zstd\n");
    // Zstandard data.

    std::vector<char> buf;
    bool ret = zstd_load(&buf, filename);

    if (ret) {
      char *p = static_cast<char*>(malloc(buf.size() + 1));  // @fixme { implement deleter }
      memcpy(p, &buf.at(0), buf.size());
      p[buf.size()] = '\0';
      data = p;
      data_len = buf.size();
    }
  } else {
    
    data = mmap_file(&data_len, filename);

  }

  (*len) = data_len;
  return data;
}

bool LoadObj(Mesh& mesh, const char* filename, float scale) {

  std::string err;

#if defined(USE_MULTITHREADING)
  tinyobj_opt::attrib_t attrib;
  std::vector<tinyobj_opt::shape_t> shapes;
  std::vector<tinyobj_opt::material_t> materials;
#else
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
#endif

  std::string basedir = GetBaseDir(filename) + "/";
  const char* basepath = (basedir.compare("/") == 0) ? NULL : basedir.c_str();

  auto t_start = std::chrono::system_clock::now();

#if defined(USE_MULTITHREADING)

  size_t data_len = 0;
  const char* data = get_file_data(&data_len, filename);
  if (data == nullptr) {
    printf("failed to load file\n");
    exit(-1);
    return false;
  }

  tinyobj_opt::LoadOption option;
  bool ret = parseObj(&attrib, &shapes, &materials, data, data_len, option);
#else
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename,
                              basepath, /* triangulate */ true);
#endif

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
  mesh.num_faces = num_faces;
  mesh.num_vertices = num_vertices;
  mesh.vertices.resize(num_vertices * 3, 0.0f);
  mesh.vertex_colors.resize(num_vertices * 3, 1.0f);
  mesh.faces.resize(num_faces * 3, 0);
  mesh.material_ids.resize(num_faces, 0);
  mesh.facevarying_normals.resize(num_faces * 3 * 3, 0.0f);
  mesh.facevarying_uvs.resize(num_faces * 3 * 2, 0.0f);

  // @todo {}
  // mesh.facevarying_tangents = NULL;
  // mesh.facevarying_binormals = NULL;

  //size_t vertexIdxOffset = 0;
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
          shapes[i].mesh.indices[3 * f + 0].vertex_index;
      mesh.faces[3 * (faceIdxOffset + f) + 1] =
          shapes[i].mesh.indices[3 * f + 1].vertex_index;
      mesh.faces[3 * (faceIdxOffset + f) + 2] =
          shapes[i].mesh.indices[3 * f + 2].vertex_index;

      mesh.material_ids[faceIdxOffset + f] = shapes[i].mesh.material_ids[f];
    }

    if (attrib.normals.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0].normal_index;
        f1 = shapes[i].mesh.indices[3 * f + 1].normal_index;
        f2 = shapes[i].mesh.indices[3 * f + 2].normal_index;

        if (f0 > 0 && f1 > 0 && f2 > 0) {
          float3 n0, n1, n2;

          n0[0] = attrib.normals[3 * f0 + 0];
          n0[1] = attrib.normals[3 * f0 + 1];
          n0[2] = attrib.normals[3 * f0 + 2];

          n1[0] = attrib.normals[3 * f1 + 0];
          n1[1] = attrib.normals[3 * f1 + 1];
          n1[2] = attrib.normals[3 * f1 + 2];

          n2[0] = attrib.normals[3 * f2 + 0];
          n2[1] = attrib.normals[3 * f2 + 1];
          n2[2] = attrib.normals[3 * f2 + 2];

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

          v0[0] = attrib.vertices[3 * f0 + 0];
          v0[1] = attrib.vertices[3 * f0 + 1];
          v0[2] = attrib.vertices[3 * f0 + 2];

          v1[0] = attrib.vertices[3 * f1 + 0];
          v1[1] = attrib.vertices[3 * f1 + 1];
          v1[2] = attrib.vertices[3 * f1 + 2];

          v2[0] = attrib.vertices[3 * f2 + 0];
          v2[1] = attrib.vertices[3 * f2 + 1];
          v2[2] = attrib.vertices[3 * f2 + 2];

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

        v0[0] = attrib.vertices[3 * f0 + 0];
        v0[1] = attrib.vertices[3 * f0 + 1];
        v0[2] = attrib.vertices[3 * f0 + 2];

        v1[0] = attrib.vertices[3 * f1 + 0];
        v1[1] = attrib.vertices[3 * f1 + 1];
        v1[2] = attrib.vertices[3 * f1 + 2];

        v2[0] = attrib.vertices[3 * f2 + 0];
        v2[1] = attrib.vertices[3 * f2 + 1];
        v2[2] = attrib.vertices[3 * f2 + 2];

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

          n0[0] = attrib.texcoords[2 * f0 + 0];
          n0[1] = attrib.texcoords[2 * f0 + 1];

          n1[0] = attrib.texcoords[2 * f1 + 0];
          n1[1] = attrib.texcoords[2 * f1 + 1];

          n2[0] = attrib.texcoords[2 * f2 + 0];
          n2[1] = attrib.texcoords[2 * f2 + 1];

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
  gMaterials.resize(materials.size());
  gTextures.resize(0);
  for (size_t i = 0; i < materials.size(); i++) {
    gMaterials[i].diffuse[0] = materials[i].diffuse[0];
    gMaterials[i].diffuse[1] = materials[i].diffuse[1];
    gMaterials[i].diffuse[2] = materials[i].diffuse[2];
    gMaterials[i].specular[0] = materials[i].specular[0];
    gMaterials[i].specular[1] = materials[i].specular[1];
    gMaterials[i].specular[2] = materials[i].specular[2];

    gMaterials[i].id = i;

    // map_Kd
    gMaterials[i].diffuse_texid = LoadTexture(materials[i].diffuse_texname);
    // map_Ks
    gMaterials[i].specular_texid = LoadTexture(materials[i].specular_texname);
  }

  return true;
}

bool Renderer::LoadObjMesh(const char* obj_filename, float scene_scale) {
  return LoadObj(gMesh, obj_filename, scene_scale);
}

template <typename T>
inline eson::Value createEsonValue(const std::vector<T>& src, size_t n_elem) {
  assert(src.size() == n_elem);
  return eson::Value((uint8_t*)&(src.at(0)), sizeof(T) * n_elem);
}

template <typename T>
inline eson::Value createEsonValue(const T* src, size_t n_elem) {
  return eson::Value((uint8_t*)&(src[0]), sizeof(T) * n_elem);
}

template <typename T>
void recoverEsonValue(eson::Value v, const std::string& name,
                      std::vector<T>& dst, size_t n_elem) {
  eson::Binary binary = v.Get(name).Get<eson::Binary>();
  const T* pointer = reinterpret_cast<T*>(const_cast<uint8_t*>(binary.ptr));
  dst.resize(n_elem);
  for (size_t i = 0; i < n_elem; i++) {
    dst[i] = pointer[i];
  }
}

template <typename T>
void recoverEsonValue(eson::Value v, const std::string& name, T* dst,
                      size_t n_elem) {
  eson::Binary binary = v.Get(name).Get<eson::Binary>();

  const T* pointer = reinterpret_cast<T*>(const_cast<uint8_t*>(binary.ptr));
  for (int i = 0; i < n_elem; i++) {
    dst[i] = pointer[i];
  }
}

bool Renderer::SaveEsonMesh(const char* eson_filename) {
  std::cout << "[SaveESON] " << eson_filename << std::endl;
  eson::Object root;

  size_t num_vertices = gMesh.num_vertices;
  size_t num_faces = gMesh.num_faces;

  // Mesh
  root["num_vertices"] = eson::Value((int64_t)num_vertices);
  root["num_faces"] = eson::Value((int64_t)num_faces);
  root["vertices"] = createEsonValue(gMesh.vertices, num_vertices * 3);
  root["facevarying_normals"] =
      createEsonValue(gMesh.facevarying_normals, num_faces * 3 * 3);
  //   root["facevarying_tangents"] =
  //   createEsonValue(gMesh.facevarying_tangents, num_faces * 3 * 3);
  //   root["facevarying_binormals"] =
  //   createEsonValue(gMesh.facevarying_binormals, num_faces * 3 * 3);
  root["facevarying_uvs"] =
      createEsonValue(gMesh.facevarying_uvs, num_faces * 2 * 3);
  //   root["facevarying_vertex_colors"] =
  //   createEsonValue(gMesh.facevarying_vertex_colors , num_faces * 3 * 3);
  root["faces"] = createEsonValue(gMesh.faces, num_faces * 3);
  root["material_ids"] = createEsonValue(gMesh.material_ids, num_faces);

  // Materials
  root["num_materials"] = eson::Value((int64_t)gMaterials.size());
  for (int i = 0; i < gMaterials.size(); i++) {
    Material& material = gMaterials[i];
    std::stringstream ss;
    ss << "material" << i << "_";
    std::string pf = ss.str();

    root[pf + "diffuse"] = createEsonValue(material.diffuse, 3);
    root[pf + "specular"] = createEsonValue(material.specular, 3);
    root[pf + "id"] = eson::Value((int64_t)material.id);
    root[pf + "diffuse_texid"] = eson::Value((int64_t)material.diffuse_texid);
    root[pf + "specular_texid"] = eson::Value((int64_t)material.specular_texid);
  }

  // Textures
  root["num_textures"] = eson::Value((int64_t)gTextures.size());
  for (int i = 0; i < gTextures.size(); i++) {
    Texture& texture = gTextures[i];
    std::stringstream ss;
    ss << "texture" << i << "_";
    std::string pf = ss.str();

    root[pf + "width"] = eson::Value((int64_t)texture.width);
    root[pf + "height"] = eson::Value((int64_t)texture.height);
    root[pf + "components"] = eson::Value((int64_t)texture.components);
    root[pf + "image"] = createEsonValue(
        texture.image, texture.width * texture.height * texture.components);
  }

  eson::Value v = eson::Value(root);
  int64_t size = v.Size();
  std::vector<uint8_t> buf(size);
  uint8_t* ptr = &buf[0];
  ptr = v.Serialize(ptr);
  assert((ptr - &buf[0]) == size);

  FILE* fp = fopen(eson_filename, "wb");
  if (!fp) {
    return false;
  }
  fwrite(&buf[0], 1, size, fp);
  fclose(fp);

  return true;
}

bool Renderer::LoadEsonMesh(const char* eson_filename) {
  std::vector<uint8_t> buf;

  std::cout << "[LoadESON] " << eson_filename << std::endl;

  FILE* fp = fopen(eson_filename, "rb");
  if (!fp) {
    return false;
  }

  fseek(fp, 0, SEEK_END);
  size_t len = ftell(fp);
  rewind(fp);
  buf.resize(len);
  len = fread(&buf[0], 1, len, fp);
  fclose(fp);

  eson::Value v;

  std::string err = eson::Parse(v, &buf[0]);
  if (!err.empty()) {
    std::cout << "Err: " << err << std::endl;
    exit(1);
  }

  // std::cout << "[LoadESON] # of shapes in .obj : " << shapes.size() <<
  // std::endl;

  int64_t num_vertices = v.Get("num_vertices").Get<int64_t>();
  int64_t num_faces = v.Get("num_faces").Get<int64_t>();
  printf("# of vertices: %d\n", int(num_vertices));

  // Mesh
  gMesh.num_vertices = num_vertices;
  gMesh.num_faces = num_faces;
  recoverEsonValue(v, "vertices", gMesh.vertices, num_vertices * 3);
  recoverEsonValue(v, "facevarying_normals", gMesh.facevarying_normals,
                   num_faces * 3 * 3);
  // recoverEsonValue(v, "facevarying_tangents", gMesh.facevarying_tangents,
  // num_faces * 3 * 3);
  // recoverEsonValue(v, "facevarying_binormals", gMesh.facevarying_binormals,
  // num_faces * 3 * 3);
  recoverEsonValue(v, "facevarying_uvs", gMesh.facevarying_uvs,
                   num_faces * 2 * 3);
  // recoverEsonValue(v, "facevarying_vertex_colors",
  // gMesh.facevarying_vertex_colors, num_faces * 3 * 3);
  recoverEsonValue(v, "faces", gMesh.faces, num_faces * 3);
  recoverEsonValue(v, "material_ids", gMesh.material_ids, num_faces);

  // Materials
  int64_t num_materials = v.Get("num_materials").Get<int64_t>();
  gMaterials.resize(num_materials);
  for (int i = 0; i < gMaterials.size(); i++) {
    Material& material = gMaterials[i];
    std::stringstream ss;
    ss << "material" << i << "_";
    std::string pf = ss.str();

    recoverEsonValue(v, pf + "diffuse", material.diffuse, 3);
    recoverEsonValue(v, pf + "specular", material.specular, 3);
    material.id = v.Get(pf + "id").Get<int64_t>();
    material.diffuse_texid = v.Get(pf + "diffuse_texid").Get<int64_t>();
    material.specular_texid = v.Get(pf + "specular_texid").Get<int64_t>();
  }

  // Textures
  int64_t num_textures = v.Get("num_textures").Get<int64_t>();
  gTextures.resize(num_textures);
  for (int i = 0; i < gTextures.size(); i++) {
    Texture& texture = gTextures[i];
    std::stringstream ss;
    ss << "texture" << i << "_";
    std::string pf = ss.str();

    texture.width = static_cast<int>(v.Get(pf + "width").Get<int64_t>());
    texture.height = static_cast<int>(v.Get(pf + "height").Get<int64_t>());
    texture.components = static_cast<int>(v.Get(pf + "components").Get<int64_t>());

    size_t n_elem = texture.width * texture.height * texture.components;
    texture.image = new unsigned char[n_elem];
    recoverEsonValue(v, pf + "image", texture.image, n_elem);
  }

  return true;
}

bool Renderer::BuildBVH() {
  std::cout << "[Build BVH] " << std::endl;

  nanort::BVHBuildOptions<float> build_options;  // Use default option
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  auto t_start = std::chrono::system_clock::now();

  nanort::TriangleMesh<float> triangle_mesh(gMesh.vertices.data(),
                                            gMesh.faces.data(), sizeof(float) * 3);
  nanort::TriangleSAHPred<float> triangle_pred(gMesh.vertices.data(),
                                               gMesh.faces.data(), sizeof(float) * 3);

  printf("num_triangles = %lu\n", gMesh.num_faces);

  bool ret = gAccel.Build(gMesh.num_faces, triangle_mesh,
                          triangle_pred, build_options);
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

bool Renderer::Render(float* rgba, float* aux_rgba, int* sample_counts,
                      float quat[4], const RenderConfig& config,
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

        // draw dash line to aux buffer for progress.
        // for (int x = 0; x < config.width; x++) {
        //  float c = (x / 8) % 2;
        //  aux_rgba[4*(y*config.width+x)+0] = c;
        //  aux_rgba[4*(y*config.width+x)+1] = c;
        //  aux_rgba[4*(y*config.width+x)+2] = c;
        //  aux_rgba[4*(y*config.width+x)+3] = 0.0f;
        //}

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
              gMesh.vertices.data(), gMesh.faces.data(), sizeof(float) * 3);
          nanort::TriangleIntersection<float> isect;
          bool hit = gAccel.Traverse(ray, triangle_intersector, &isect);
          if (hit) {
            float3 p;
            p[0] =
                ray.org[0] + isect.t * ray.dir[0];
            p[1] =
                ray.org[1] + isect.t * ray.dir[1];
            p[2] =
                ray.org[2] + isect.t * ray.dir[2];

            config.positionImage[4 * (y * config.width + x) + 0] = p.x();
            config.positionImage[4 * (y * config.width + x) + 1] = p.y();
            config.positionImage[4 * (y * config.width + x) + 2] = p.z();
            config.positionImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.varycoordImage[4 * (y * config.width + x) + 0] =
                isect.u;
            config.varycoordImage[4 * (y * config.width + x) + 1] =
                isect.v;
            config.varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
            config.varycoordImage[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = isect.prim_id;

            float3 N;
            if (gMesh.facevarying_normals.size() > 0) {
              float3 n0, n1, n2;
              n0[0] = gMesh.facevarying_normals[9 * prim_id + 0];
              n0[1] = gMesh.facevarying_normals[9 * prim_id + 1];
              n0[2] = gMesh.facevarying_normals[9 * prim_id + 2];
              n1[0] = gMesh.facevarying_normals[9 * prim_id + 3];
              n1[1] = gMesh.facevarying_normals[9 * prim_id + 4];
              n1[2] = gMesh.facevarying_normals[9 * prim_id + 5];
              n2[0] = gMesh.facevarying_normals[9 * prim_id + 6];
              n2[1] = gMesh.facevarying_normals[9 * prim_id + 7];
              n2[2] = gMesh.facevarying_normals[9 * prim_id + 8];
              N = Lerp3(n0, n1, n2, isect.u,
                        isect.v);
            } else {
              unsigned int f0, f1, f2;
              f0 = gMesh.faces[3 * prim_id + 0];
              f1 = gMesh.faces[3 * prim_id + 1];
              f2 = gMesh.faces[3 * prim_id + 2];

              float3 v0, v1, v2;
              v0[0] = gMesh.vertices[3 * f0 + 0];
              v0[1] = gMesh.vertices[3 * f0 + 1];
              v0[2] = gMesh.vertices[3 * f0 + 2];
              v1[0] = gMesh.vertices[3 * f1 + 0];
              v1[1] = gMesh.vertices[3 * f1 + 1];
              v1[2] = gMesh.vertices[3 * f1 + 2];
              v2[0] = gMesh.vertices[3 * f2 + 0];
              v2[1] = gMesh.vertices[3 * f2 + 1];
              v2[2] = gMesh.vertices[3 * f2 + 2];
              CalcNormal(N, v0, v1, v2);
            }

            config.normalImage[4 * (y * config.width + x) + 0] =
                0.5f * N[0] + 0.5f;
            config.normalImage[4 * (y * config.width + x) + 1] =
                0.5f * N[1] + 0.5f;
            config.normalImage[4 * (y * config.width + x) + 2] =
                0.5f * N[2] + 0.5f;
            config.normalImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.depthImage[4 * (y * config.width + x) + 0] =
                isect.t;
            config.depthImage[4 * (y * config.width + x) + 1] =
                isect.t;
            config.depthImage[4 * (y * config.width + x) + 2] =
                isect.t;
            config.depthImage[4 * (y * config.width + x) + 3] = 1.0f;

            float3 vcol(1.0f, 1.0f, 1.0f);
            if (gMesh.vertex_colors.size() > 0) {
              unsigned int f0, f1, f2;
              f0 = gMesh.faces[3 * prim_id + 0];
              f1 = gMesh.faces[3 * prim_id + 1];
              f2 = gMesh.faces[3 * prim_id + 2];

              float3 c0, c1, c2;
              c0[0] = gMesh.vertex_colors[3 * f0 + 0];
              c0[1] = gMesh.vertex_colors[3 * f0 + 1];
              c0[2] = gMesh.vertex_colors[3 * f0 + 2];
              c1[0] = gMesh.vertex_colors[3 * f1 + 0];
              c1[1] = gMesh.vertex_colors[3 * f1 + 1];
              c1[2] = gMesh.vertex_colors[3 * f1 + 2];
              c2[0] = gMesh.vertex_colors[3 * f2 + 0];
              c2[1] = gMesh.vertex_colors[3 * f2 + 1];
              c2[2] = gMesh.vertex_colors[3 * f2 + 2];

              vcol = Lerp3(c0, c1, c2, isect.u,
                         isect.v);

              config.vertexColorImage[4 * (y * config.width + x) + 0] = vcol[0];
              config.vertexColorImage[4 * (y * config.width + x) + 1] = vcol[1];
              config.vertexColorImage[4 * (y * config.width + x) + 2] = vcol[2];
            }

            float3 UV;
            if (gMesh.facevarying_uvs.size() > 0) {
              float3 uv0, uv1, uv2;
              uv0[0] = gMesh.facevarying_uvs[6 * prim_id + 0];
              uv0[1] = gMesh.facevarying_uvs[6 * prim_id + 1];
              uv1[0] = gMesh.facevarying_uvs[6 * prim_id + 2];
              uv1[1] = gMesh.facevarying_uvs[6 * prim_id + 3];
              uv2[0] = gMesh.facevarying_uvs[6 * prim_id + 4];
              uv2[1] = gMesh.facevarying_uvs[6 * prim_id + 5];

              UV = Lerp3(uv0, uv1, uv2, isect.u,
                         isect.v);

              config.texcoordImage[4 * (y * config.width + x) + 0] = UV[0];
              config.texcoordImage[4 * (y * config.width + x) + 1] = UV[1];
            }

            float NdotV = fabsf(vdot(N, dir));

            // Fetch material & texture
            unsigned int material_id =
                gMesh.material_ids[isect.prim_id];

          
            float diffuse_col[3] = {0.5f, 0.5f, 0.5f};
            float specular_col[3] = {0.0f, 0.0f, 0.0f};

            if (material_id < gMaterials.size()) {

              int diffuse_texid = gMaterials[material_id].diffuse_texid;
              if (diffuse_texid >= 0) {
                FetchTexture(diffuse_texid, UV[0], UV[1], diffuse_col);
              } else {
                diffuse_col[0] = gMaterials[material_id].diffuse[0];
                diffuse_col[1] = gMaterials[material_id].diffuse[1];
                diffuse_col[2] = gMaterials[material_id].diffuse[2];
              }

              int specular_texid = gMaterials[material_id].specular_texid;
              if (specular_texid >= 0) {
                FetchTexture(specular_texid, UV[0], UV[1], specular_col);
              } else {
                specular_col[0] = gMaterials[material_id].specular[0];
                specular_col[1] = gMaterials[material_id].specular[1];
                specular_col[2] = gMaterials[material_id].specular[2];
              }
            }

            if (config.pass == 0) {
              rgba[4 * (y * config.width + x) + 0] = NdotV * diffuse_col[0];
              rgba[4 * (y * config.width + x) + 1] = NdotV * diffuse_col[1];
              rgba[4 * (y * config.width + x) + 2] = NdotV * diffuse_col[2];
              rgba[4 * (y * config.width + x) + 3] = 1.0f;
              sample_counts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              rgba[4 * (y * config.width + x) + 0] += NdotV * diffuse_col[0];
              rgba[4 * (y * config.width + x) + 1] += NdotV * diffuse_col[1];
              rgba[4 * (y * config.width + x) + 2] += NdotV * diffuse_col[2];
              rgba[4 * (y * config.width + x) + 3] += 1.0f;
              sample_counts[y * config.width + x]++;
            }

          } else {
            {
              if (config.pass == 0) {
                // clear pixel
                rgba[4 * (y * config.width + x) + 0] = 0.0f;
                rgba[4 * (y * config.width + x) + 1] = 0.0f;
                rgba[4 * (y * config.width + x) + 2] = 0.0f;
                rgba[4 * (y * config.width + x) + 3] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 0] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 1] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 2] = 0.0f;
                aux_rgba[4 * (y * config.width + x) + 3] = 0.0f;
                sample_counts[y * config.width + x] =
                    1;  // Set 1 for the first pass
              } else {
                sample_counts[y * config.width + x]++;
              }

              // No super sampling
              config.normalImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.normalImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.normalImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.normalImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.positionImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.depthImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.texcoordImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.varycoordImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 0] = 1.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 1] = 1.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 2] = 1.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 3] = 1.0f;
            }
          }
        }

        for (int x = 0; x < config.width; x++) {
          aux_rgba[4 * (y * config.width + x) + 0] = 0.0f;
          aux_rgba[4 * (y * config.width + x) + 1] = 0.0f;
          aux_rgba[4 * (y * config.width + x) + 2] = 0.0f;
          aux_rgba[4 * (y * config.width + x) + 3] = 0.0f;
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
