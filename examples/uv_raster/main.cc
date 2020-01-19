/*
The MIT License (MIT)

Copyright (c) 2020 Light Transport Entertainment, Inc.

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
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

// ../common/
#include "picojson.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

// ../common/
#include "tiny_obj_loader.h"

// ../../
#include "nanort.h"

#include "image_saver.hh"

typedef nanort::real3<float> float3;

namespace example {

namespace {

template <typename T>
inline T Lerp(T v0, T v1, T v2, float u, float v) {
  return (1.0f - u - v) * v0 + u * v1 + v * v2;
}

inline void CalcNormal(float3& N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}

}  // namespace

struct Config {
  std::string obj_filename;

  std::string output_basename = "output_";

  // Image resolution.
  int width = 1024;
  int height = 1024;

  // UV region. [left, right, top, bottom]
  float uv_region[4] = {0.0f, 1.0f, 0.0f, 1.0f};

  // When .obj contains multiple shapes, which shape to raster?
  int shape_id = 0;

  bool debug_uvmesh = true;  // Output UV mesh as .obj?

  // default = center of texel(pixel).
  // Offset ray's position using this value.
  float texel_offset[2] = {0.5f, 0.5f};

  bool flip_y = true; // true to compensate wavefront .obj's v texture coordinate.
  bool flip_x = false;

  int num_threads = -1;  // -1 = use # of system threads.
};

typedef struct {
  std::vector<float> vertices;  /// [xyz] * num_vertices
  std::vector<float>
      facevarying_normals;  /// [xyz] * 3(triangle) * num_triangle_faces
  // std::vector<float> facevarying_tangents;   /// [xyz]
  // std::vector<float> facevarying_binormals;  /// [xyz]
  std::vector<float> facevarying_uvs;  /// [xy] x 3 x num_triangle_faces
  std::vector<int> material_ids;       /// index x num_triangle_faces

  // List of triangle vertex indices. For NanoRT BVH
  std::vector<unsigned int>
      triangulated_indices;  /// 3(triangle) x num_triangle_faces

  // List of original vertex indices. For UV interpolation
  std::vector<unsigned int>
      face_indices;  /// length = sum(for each face_num_verts[i])

  // Offset to `face_indices` for a given face_id.
  std::vector<uint64_t> face_index_offsets;  /// length = face_num_verts.size()

  std::vector<unsigned char> face_num_verts;  /// # of vertices per face

  // face ID for each triangle.
  std::vector<uint64_t> face_ids;  /// index x num_triangle_faces

  // Triangule fin ID of a face(e.g. 0 for triangle primitive. 0 or 1 for quad
  // primitive(tessellated into two-triangles)
  std::vector<uint8_t> face_triangle_fin_ids;  /// index x num_triangle_faces

  //
  // For UV raster.
  //

  // UV to position
  std::vector<float> uv_vertices;  /// [xyz] * 3 * num_triangle_faces
  std::vector<unsigned int>
      uv_face_indices;  /// Triangle facevarying indices for `uv_vertices`. This
                        /// is simply uv_face_indices[i] = i. length = 3 *
                        /// num_triangle_faces.

} Mesh;

static bool LoadConfig(Config* config, const char* filename) {
  std::ifstream is(filename);
  if (!is) {
    std::cerr << "Cannot open " << filename << std::endl;
    return false;
  }

  std::istream_iterator<char> input(is);
  std::string err;
  picojson::value v;
  input = picojson::parse(v, input, std::istream_iterator<char>(), &err);
  if (!err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!v.is<picojson::object>()) {
    std::cerr << "Not a JSON object" << std::endl;
    return false;
  }

  picojson::object o = v.get<picojson::object>();

  if (o.find("obj_filename") != o.end()) {
    if (o["obj_filename"].is<std::string>()) {
      config->obj_filename = o["obj_filename"].get<std::string>();
    }
  }

  if (o.find("width") != o.end()) {
    if (o["width"].is<double>()) {
      config->width = int(o["width"].get<double>());
    }
  }

  if (o.find("height") != o.end()) {
    if (o["height"].is<double>()) {
      config->height = int(o["height"].get<double>());
    }
  }

  if (o.find("shape_id") != o.end()) {
    if (o["shape_id"].is<double>()) {
      config->shape_id = int(o["shape_id"].get<double>());
    }
  }

  if (o.find("flip_y") != o.end()) {
    if (o["flip_y"].is<bool>()) {
      config->flip_y = o["flip_y"].get<bool>();
    }
  }

  if (o.find("flip_x") != o.end()) {
    if (o["flip_x"].is<bool>()) {
      config->flip_x = o["flip_x"].get<bool>();
    }
  }

  if (o.find("output_basename") != o.end()) {
    if (o["output_basename"].is<std::string>()) {
      config->output_basename = o["output_basename"].get<std::string>();
    }
  }

  if (o.find("texel_offset") != o.end()) {
    if (o["texel_offset"].is<picojson::array>()) {
      picojson::array arr = o["texel_offset"].get<picojson::array>();
      if (arr.size() == 2) {
        config->texel_offset[0] = static_cast<float>(arr[0].get<double>());
        config->texel_offset[1] = static_cast<float>(arr[1].get<double>());
      }
    }
  }

  if (o.find("uv_region") != o.end()) {
    if (o["uv_region"].is<picojson::array>()) {
      picojson::array arr = o["uv_region"].get<picojson::array>();
      if (arr.size() == 4) {
        config->uv_region[0] = static_cast<float>(arr[0].get<double>());
        config->uv_region[1] = static_cast<float>(arr[1].get<double>());
        config->uv_region[2] = static_cast<float>(arr[2].get<double>());
        config->uv_region[3] = static_cast<float>(arr[3].get<double>());
      }
    }
  }

  return true;
}

static std::string GetBaseDir(const std::string& filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

static void SetupVerticesForUVRaster(Mesh* mesh) {
  mesh->uv_vertices.clear();

  // Set UV coord as vertex position.
  for (size_t i = 0; i < mesh->triangulated_indices.size(); i++) {
    const float vx = mesh->facevarying_uvs[2 * i + 0];
    const float vy = mesh->facevarying_uvs[2 * i + 1];
    const float vz = 0.0f;

    mesh->uv_vertices.push_back(vx);
    mesh->uv_vertices.push_back(vy);
    mesh->uv_vertices.push_back(vz);
  }

  // fill verterx indices.(facevarying indices)
  // uv_face_indices[i] = i;
  mesh->uv_face_indices.resize(mesh->triangulated_indices.size());
  std::iota(mesh->uv_face_indices.begin(), mesh->uv_face_indices.end(), 0);
}

static bool LoadObj(Mesh& mesh, const char* filename, int shape_id) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn;
  std::string err;

  std::string basedir = GetBaseDir(filename) + "/";
  const char* basepath = (basedir.compare("/") == 0) ? NULL : basedir.c_str();

  auto t_start = std::chrono::system_clock::now();

  // We support triangles or quads, so disable triangulation.
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                              filename, basepath, /* triangulate */ false);

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = t_end - t_start;

  if (!warn.empty()) {
    std::cout << warn << std::endl;
  }

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  if (!ret) {
    std::cerr << "Failed to load or parse .obj\n";
    return false;
  }

  if (attrib.texcoords.empty()) {
    std::cerr << ".obj does not contain texture coordinates.\n" << std::endl;
    return false;
  }

  std::cout << "[LoadOBJ] Parse time : " << ms.count() << " [msecs]"
            << std::endl;

  std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
  std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size()
            << std::endl;

  if (shapes.empty()) {
    std::cerr << "Empty shape in .obj\n";
    return false;
  }

  if (shape_id >= int(shapes.size())) {
    std::cerr << "shape_id is out-of-range. Use the first shape\n";
    shape_id = 0;
  }

  size_t num_vertices = 0;
  size_t num_faces = 0;

  num_vertices = attrib.vertices.size() / 3;
  printf("  vertices: %ld\n", attrib.vertices.size() / 3);

  size_t s = size_t(shape_id);

  {
    printf("  shape[%d].name = %s\n", int(s), shapes[s].name.c_str());
    printf("  shape[%d].indices: %ld\n", int(s), shapes[s].mesh.indices.size());
    printf("  shape[%d].material_ids: %ld\n", int(s),
           shapes[s].mesh.material_ids.size());

    num_faces += shapes[s].mesh.num_face_vertices.size();

    // Check if a face is triangle or quad.
    for (size_t k = 0; k < shapes[s].mesh.num_face_vertices.size(); k++) {
      if ((shapes[s].mesh.num_face_vertices[k] == 3) ||
          (shapes[s].mesh.num_face_vertices[k] == 4)) {
        // ok
      } else {
        std::cerr << "face contains invalid polygons."
                  << std::to_string(shapes[s].mesh.num_face_vertices[k])
                  << std::endl;
      }
    }
  }

  std::cout << "[LoadOBJ] # of faces: " << num_faces << std::endl;
  std::cout << "[LoadOBJ] # of vertices: " << num_vertices << std::endl;

  // Shape -> Mesh
  mesh.vertices.resize(num_vertices * 3, 0.0f);

  mesh.vertices.clear();
  for (size_t i = 0; i < attrib.vertices.size(); i++) {
    mesh.vertices.push_back(attrib.vertices[i]);
  }

  mesh.triangulated_indices.clear();
  mesh.face_indices.clear();
  mesh.face_index_offsets.clear();
  mesh.face_num_verts.clear();
  mesh.face_ids.clear();
  mesh.face_triangle_fin_ids.clear();
  mesh.material_ids.clear();
  mesh.facevarying_normals.clear();
  mesh.facevarying_uvs.clear();

  // Flattened indices for easy facevarying normal/uv setup
  std::vector<tinyobj::index_t> triangulated_indices;

  size_t face_id_offset = 0;
  {
    size_t offset = 0;
    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      uint8_t npoly = shapes[s].mesh.num_face_vertices[f];

      mesh.face_num_verts.push_back(npoly);
      mesh.face_index_offsets.push_back(mesh.face_indices.size());

      if (npoly == 4) {
        //
        // triangulate
        // For easier UV coordinate calculation, use (p0, p1, p2), (p2, p3, p0)
        // split
        //
        // p0------p3
        // | \      |
        // |  \     |
        // |   \    |
        // |    \   |
        // |     \  |
        // |      \ |
        // p1 ---- p2
        //
        int32_t f0 = shapes[s].mesh.indices[offset + 0].vertex_index;
        int32_t f1 = shapes[s].mesh.indices[offset + 1].vertex_index;
        int32_t f2 = shapes[s].mesh.indices[offset + 2].vertex_index;
        int32_t f3 = shapes[s].mesh.indices[offset + 3].vertex_index;

        assert(f0 >= 0);
        assert(f1 >= 0);
        assert(f2 >= 0);
        assert(f3 >= 0);

        mesh.triangulated_indices.push_back(uint32_t(f0));
        mesh.triangulated_indices.push_back(uint32_t(f1));
        mesh.triangulated_indices.push_back(uint32_t(f2));

        mesh.triangulated_indices.push_back(uint32_t(f2));
        mesh.triangulated_indices.push_back(uint32_t(f3));
        mesh.triangulated_indices.push_back(uint32_t(f0));

        mesh.face_indices.push_back(uint32_t(f0));
        mesh.face_indices.push_back(uint32_t(f1));
        mesh.face_indices.push_back(uint32_t(f2));
        mesh.face_indices.push_back(uint32_t(f3));

        mesh.face_ids.push_back(face_id_offset + f);
        mesh.face_ids.push_back(face_id_offset + f);

        mesh.face_triangle_fin_ids.push_back(0);
        mesh.face_triangle_fin_ids.push_back(1);

        mesh.material_ids.push_back(shapes[s].mesh.material_ids[f]);
        mesh.material_ids.push_back(shapes[s].mesh.material_ids[f]);

        // for computing normal/uv in the later stage
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 0]);
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 1]);
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 2]);

        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 2]);
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 3]);
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 0]);

      } else {
        int32_t f0 = shapes[s].mesh.indices[offset + 0].vertex_index;
        int32_t f1 = shapes[s].mesh.indices[offset + 1].vertex_index;
        int32_t f2 = shapes[s].mesh.indices[offset + 2].vertex_index;

        assert(f0 >= 0);
        assert(f1 >= 0);
        assert(f2 >= 0);

        mesh.triangulated_indices.push_back(uint32_t(f0));
        mesh.triangulated_indices.push_back(uint32_t(f1));
        mesh.triangulated_indices.push_back(uint32_t(f2));

        mesh.face_indices.push_back(uint32_t(f0));
        mesh.face_indices.push_back(uint32_t(f1));
        mesh.face_indices.push_back(uint32_t(f2));

        mesh.face_ids.push_back(face_id_offset + f);
        mesh.face_triangle_fin_ids.push_back(0);
        mesh.material_ids.push_back(shapes[s].mesh.material_ids[f]);

        // for computing normal/uv in the later stage
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 0]);
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 1]);
        triangulated_indices.push_back(shapes[s].mesh.indices[offset + 2]);
      }

      offset += npoly;
    }

    face_id_offset += shapes[s].mesh.num_face_vertices.size();
  }

  // Setup normal/uv
  if (attrib.normals.size() > 0) {
    for (size_t f = 0; f < triangulated_indices.size() / 3; f++) {
      int f0, f1, f2;

      f0 = triangulated_indices[3 * f + 0].normal_index;
      f1 = triangulated_indices[3 * f + 1].normal_index;
      f2 = triangulated_indices[3 * f + 2].normal_index;

      if ((f0 >= 0) && (f1 >= 0) && (f2 >= 0)) {
        float3 n0, n1, n2;

        n0[0] = attrib.normals[3 * size_t(f0) + 0];
        n0[1] = attrib.normals[3 * size_t(f0) + 1];
        n0[2] = attrib.normals[3 * size_t(f0) + 2];

        n1[0] = attrib.normals[3 * size_t(f1) + 0];
        n1[1] = attrib.normals[3 * size_t(f1) + 1];
        n1[2] = attrib.normals[3 * size_t(f1) + 2];

        n2[0] = attrib.normals[3 * size_t(f2) + 0];
        n2[1] = attrib.normals[3 * size_t(f2) + 1];
        n2[2] = attrib.normals[3 * size_t(f2) + 2];

        mesh.facevarying_normals.push_back(n0[0]);
        mesh.facevarying_normals.push_back(n0[1]);
        mesh.facevarying_normals.push_back(n0[2]);

        mesh.facevarying_normals.push_back(n1[0]);
        mesh.facevarying_normals.push_back(n1[1]);
        mesh.facevarying_normals.push_back(n1[2]);

        mesh.facevarying_normals.push_back(n2[0]);
        mesh.facevarying_normals.push_back(n2[1]);
        mesh.facevarying_normals.push_back(n2[2]);

      } else {  // face contains invalid normal index. calc geometric normal.
        f0 = triangulated_indices[3 * f + 0].vertex_index;
        f1 = triangulated_indices[3 * f + 1].vertex_index;
        f2 = triangulated_indices[3 * f + 2].vertex_index;

        float3 v0, v1, v2;

        v0[0] = attrib.vertices[3 * size_t(f0) + 0];
        v0[1] = attrib.vertices[3 * size_t(f0) + 1];
        v0[2] = attrib.vertices[3 * size_t(f0) + 2];

        v1[0] = attrib.vertices[3 * size_t(f1) + 0];
        v1[1] = attrib.vertices[3 * size_t(f1) + 1];
        v1[2] = attrib.vertices[3 * size_t(f1) + 2];

        v2[0] = attrib.vertices[3 * size_t(f2) + 0];
        v2[1] = attrib.vertices[3 * size_t(f2) + 1];
        v2[2] = attrib.vertices[3 * size_t(f2) + 2];

        float3 N;
        CalcNormal(N, v0, v1, v2);

        mesh.facevarying_normals.push_back(N[0]);
        mesh.facevarying_normals.push_back(N[1]);
        mesh.facevarying_normals.push_back(N[2]);

        mesh.facevarying_normals.push_back(N[0]);
        mesh.facevarying_normals.push_back(N[1]);
        mesh.facevarying_normals.push_back(N[2]);

        mesh.facevarying_normals.push_back(N[0]);
        mesh.facevarying_normals.push_back(N[1]);
        mesh.facevarying_normals.push_back(N[2]);
      }
    }
  } else {
    // calc geometric normal
    for (size_t f = 0; f < triangulated_indices.size() / 3; f++) {
      int f0, f1, f2;

      f0 = triangulated_indices[3 * f + 0].vertex_index;
      f1 = triangulated_indices[3 * f + 1].vertex_index;
      f2 = triangulated_indices[3 * f + 2].vertex_index;

      float3 v0, v1, v2;

      v0[0] = attrib.vertices[3 * size_t(f0) + 0];
      v0[1] = attrib.vertices[3 * size_t(f0) + 1];
      v0[2] = attrib.vertices[3 * size_t(f0) + 2];

      v1[0] = attrib.vertices[3 * size_t(f1) + 0];
      v1[1] = attrib.vertices[3 * size_t(f1) + 1];
      v1[2] = attrib.vertices[3 * size_t(f1) + 2];

      v2[0] = attrib.vertices[3 * size_t(f2) + 0];
      v2[1] = attrib.vertices[3 * size_t(f2) + 1];
      v2[2] = attrib.vertices[3 * size_t(f2) + 2];

      float3 N;
      CalcNormal(N, v0, v1, v2);

      mesh.facevarying_normals.push_back(N[0]);
      mesh.facevarying_normals.push_back(N[1]);
      mesh.facevarying_normals.push_back(N[2]);

      mesh.facevarying_normals.push_back(N[0]);
      mesh.facevarying_normals.push_back(N[1]);
      mesh.facevarying_normals.push_back(N[2]);

      mesh.facevarying_normals.push_back(N[0]);
      mesh.facevarying_normals.push_back(N[1]);
      mesh.facevarying_normals.push_back(N[2]);
    }
  }

  if (attrib.texcoords.size() > 0) {
    for (size_t f = 0; f < triangulated_indices.size() / 3; f++) {
      int f0, f1, f2;

      f0 = triangulated_indices[3 * f + 0].texcoord_index;
      f1 = triangulated_indices[3 * f + 1].texcoord_index;
      f2 = triangulated_indices[3 * f + 2].texcoord_index;

      // check if all index has texcoord.
      if ((f0 >= 0) && (f1 >= 0) && (f2 >= 0)) {
        float3 n0, n1, n2;

        n0[0] = attrib.texcoords[2 * size_t(f0) + 0];
        n0[1] = attrib.texcoords[2 * size_t(f0) + 1];

        n1[0] = attrib.texcoords[2 * size_t(f1) + 0];
        n1[1] = attrib.texcoords[2 * size_t(f1) + 1];

        n2[0] = attrib.texcoords[2 * size_t(f2) + 0];
        n2[1] = attrib.texcoords[2 * size_t(f2) + 1];

        mesh.facevarying_uvs.push_back(n0[0]);
        mesh.facevarying_uvs.push_back(n0[1]);

        mesh.facevarying_uvs.push_back(n1[0]);
        mesh.facevarying_uvs.push_back(n1[1]);

        mesh.facevarying_uvs.push_back(n2[0]);
        mesh.facevarying_uvs.push_back(n2[1]);
      } else {
        std::cerr << "face does not have texcoord index\n";
        return false;
      }
    }
  }

  return true;
}

static bool SaveObj(
    const std::string& filename,
    const std::vector<float>& facevarying_vertices,  // [xyz]
    const std::vector<float>& facevarying_uvcoords,  // [uv]
    const std::vector<uint32_t>& indices)  // assume triangulated indices
{
  assert((facevarying_vertices.size() / 3) == 3 * (indices.size() / 3));
  assert((facevarying_uvcoords.size() / 2) == 3 * (indices.size() / 3));

  std::ofstream ofs(filename);

  if (!ofs) {
    return false;
  }

  for (size_t i = 0; i < facevarying_vertices.size() / 3; i++) {
    ofs << "v " << facevarying_vertices[3 * i + 0] << " "
        << facevarying_vertices[3 * i + 1] << " "
        << facevarying_vertices[3 * i + 2] << "\n";
  }

  for (size_t i = 0; i < facevarying_uvcoords.size() / 2; i++) {
    ofs << "vt " << facevarying_uvcoords[2 * i + 0] << " "
        << facevarying_uvcoords[2 * i + 1] << "\n";
  }

  for (size_t i = 0; i < indices.size() / 3; i++) {
    // .obj index starts with 1.
    uint32_t id0 = 3 * uint32_t(i) + 0 + 1;
    uint32_t id1 = 3 * uint32_t(i) + 1 + 1;
    uint32_t id2 = 3 * uint32_t(i) + 2 + 1;

    ofs << "f " << id0 << "/" << id0 << " " << id1 << "/" << id1 << " " << id2
        << "/" << id2 << "\n";
  }

  return true;
}

}  // namespace example

int main(int argc, char** argv) {
  std::string config_filename = "config.json";
  if (argc > 1) {
    config_filename = argv[1];
  }

  example::Config config;

  if (!example::LoadConfig(&config, config_filename.c_str())) {
    std::cerr << "Failed to load config file: " << config_filename << "\n";
    return EXIT_FAILURE;
  }

  example::Mesh mesh;

  bool ret = LoadObj(mesh, config.obj_filename.c_str(), config.shape_id);

  if (!ret) {
    return EXIT_FAILURE;
  }

  SetupVerticesForUVRaster(&mesh);

  if (config.debug_uvmesh) {
    ret = example::SaveObj("uvmesh.obj", mesh.uv_vertices, mesh.facevarying_uvs,
                           mesh.uv_face_indices);

    if (!ret) {
      return EXIT_FAILURE;
    }

    std::cout << "Wrote uvmesh for debug\n";
  }

  nanort::BVHAccel<float> accel;
  {
    const auto start_t = std::chrono::system_clock::now();
    nanort::BVHBuildOptions<float> build_options;  // Use default option
    build_options.cache_bbox = false;

    nanort::TriangleMesh<float> triangle_mesh(mesh.uv_vertices.data(),
                                              mesh.uv_face_indices.data(),
                                              sizeof(float) * 3);
    nanort::TriangleSAHPred<float> triangle_pred(mesh.uv_vertices.data(),
                                                 mesh.uv_face_indices.data(),
                                                 sizeof(float) * 3);

    size_t num_faces = mesh.uv_face_indices.size() / 3;

    ret = accel.Build(static_cast<unsigned int>(num_faces), triangle_mesh,
                      triangle_pred, build_options);
    assert(ret);

    const auto end_t = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> ms = end_t - start_t;

    std::cout << "BVH build time : " << ms.count() << " [ms]\n";

    // Bounding box = UV range.
    float bmin[3], bmax[3];
    accel.BoundingBox(bmin, bmax);
    std::cout << "  UV x(left, right)    : " << bmin[0] << ", " << bmax[0] << "\n";
    std::cout << "  UV y(top, bottom)    : " << bmin[1] << ", " << bmax[1] << "\n";
  }

  int num_threads = int(std::thread::hardware_concurrency());
  if (config.num_threads > 0) {
    num_threads = config.num_threads;
  }

  num_threads = std::max(1, std::min(1024, num_threads));

  std::vector<std::thread> workers;
  std::atomic<int> counter(0); // for Y

  const auto start_t = std::chrono::system_clock::now();

  // AOVs
  size_t num_pixels = size_t(config.width * config.height);

  std::vector<float> aov_positions(3 * num_pixels,
                                   0.0f);  // in local(object) coord
  std::vector<float> aov_normals(3 * num_pixels,
                                 0.0f);  // shading normal in object coord
  // TODO(LTE): Geometric normal AOV

  std::vector<int> aov_face_ids(num_pixels, -1);

  std::cout << "UV region = " << config.uv_region[0] << ", " << config.uv_region[1] << ", " << config.uv_region[2] << ", " << config.uv_region[3] << "\n";

  for (size_t t = 0; t < size_t(num_threads); t++) {
    workers.push_back(std::thread([&]() {
      int y = 0;
      while ((y = counter++) < config.height) {
        for (int x = 0; x < config.width; x++) {
          // Simple camera. change eye pos and direction fit to .obj model.

          nanort::Ray<float> ray;

          const float usize = (config.uv_region[1] - config.uv_region[0]);
          const float vsize = (config.uv_region[3] - config.uv_region[2]);

          ray.org[0] = config.uv_region[0] + (x * usize + config.texel_offset[0]) / float(config.width);
          ray.org[1] = config.uv_region[2] + (y * vsize + config.texel_offset[1]) / float(config.height);
          ray.org[2] = 1.0f;

          float3 dir;
          dir[0] = 0.0f;
          dir[1] = 0.0f;
          dir[2] = -1.0f;

          ray.dir[0] = dir[0];
          ray.dir[1] = dir[1];
          ray.dir[2] = dir[2];

          float kFar = 1.0e+30f;
          ray.min_t = 0.0f;
          ray.max_t = kFar;

          nanort::TriangleIntersector<> triangle_intersector(
              mesh.uv_vertices.data(), mesh.uv_face_indices.data(),
              /* stride */ sizeof(float) * 3);
          nanort::TriangleIntersection<> isect;
          bool hit = accel.Traverse(ray, triangle_intersector, &isect);
          if (hit) {

            int px = config.flip_x ? (config.width - x - 1) : x;
            int py = config.flip_y ? (config.height - y - 1) : y;

            size_t pixel_idx = size_t(py * config.width + px);

            //
            // Write your shader here.
            //

            float3 normal(0.0f, 0.0f, 0.0f);
            unsigned int fid = isect.prim_id;
            if (mesh.facevarying_normals.size() > 0) {
              float3 n0, n1, n2;
              n0[0] = mesh.facevarying_normals[9 * fid + 0];
              n0[1] = mesh.facevarying_normals[9 * fid + 1];
              n0[2] = mesh.facevarying_normals[9 * fid + 2];
              n1[0] = mesh.facevarying_normals[9 * fid + 3];
              n1[1] = mesh.facevarying_normals[9 * fid + 4];
              n1[2] = mesh.facevarying_normals[9 * fid + 5];
              n2[0] = mesh.facevarying_normals[9 * fid + 6];
              n2[1] = mesh.facevarying_normals[9 * fid + 7];
              n2[2] = mesh.facevarying_normals[9 * fid + 8];

              float3 Ns = example::Lerp(n0, n1, n2, isect.u, isect.v);
              aov_normals[3 * pixel_idx + 0] = Ns[0];
              aov_normals[3 * pixel_idx + 1] = Ns[1];
              aov_normals[3 * pixel_idx + 2] = Ns[2];
            }

            float3 position(0.0f, 0.0f, 0.0f);
            {
              float3 v0, v1, v2;
              size_t f0 = mesh.triangulated_indices[3 * fid + 0];
              size_t f1 = mesh.triangulated_indices[3 * fid + 1];
              size_t f2 = mesh.triangulated_indices[3 * fid + 2];

              v0[0] = mesh.vertices[3 * f0 + 0];
              v0[1] = mesh.vertices[3 * f0 + 1];
              v0[2] = mesh.vertices[3 * f0 + 2];
              v1[0] = mesh.vertices[3 * f1 + 0];
              v1[1] = mesh.vertices[3 * f1 + 1];
              v1[2] = mesh.vertices[3 * f1 + 2];
              v2[0] = mesh.vertices[3 * f2 + 0];
              v2[1] = mesh.vertices[3 * f2 + 1];
              v2[2] = mesh.vertices[3 * f2 + 2];

              float3 P = example::Lerp(v0, v1, v2, isect.u, isect.v);
              aov_positions[3 * pixel_idx + 0] = P[0];
              aov_positions[3 * pixel_idx + 1] = P[1];
              aov_positions[3 * pixel_idx + 2] = P[2];
            }

            aov_face_ids[pixel_idx] = int(fid);
          }
        }
      }
    }));
  }

  for (auto& t : workers) {
    t.join();
  }

  const auto end_t = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = end_t - start_t;

  std::cout << "raster time : " << ms.count() << " [ms]\n";

  if (!example::SaveFloatEXR(aov_positions.data(), uint32_t(config.width),
                             uint32_t(config.height), /* channels */ 3,
                             config.output_basename + "position.exr")) {
    std::cerr << "Failed to save position(P) EXR\n";
    return EXIT_FAILURE;
  }

  if (!example::SaveFloatEXR(aov_normals.data(), uint32_t(config.width),
                             uint32_t(config.height), /* channels */ 3,
                             config.output_basename + "normal.exr")) {
    std::cerr << "Failed to save normal(Ns) EXR\n";
    return EXIT_FAILURE;
  }

  if (!example::SaveIntEXR(aov_face_ids.data(), uint32_t(config.width),
                           uint32_t(config.height), /* channels */ 1,
                           config.output_basename + "faceid.exr")) {
    std::cerr << "Failed to save face id EXR\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
