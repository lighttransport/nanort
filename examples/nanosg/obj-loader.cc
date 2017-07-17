#include "obj-loader.h"
#include "../../nanort.h" // for float3

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

namespace example {

typedef nanort::real3<float> float3;

inline void CalcNormal(float3& N, float3 v0, float3 v1, float3 v2) {
  float3 v10 = v1 - v0;
  float3 v20 = v2 - v0;

  N = vcross(v20, v10);
  N = vnormalize(N);
}

static std::string GetBaseDir(const std::string &filepath) {
  if (filepath.find_last_of("/\\") != std::string::npos)
    return filepath.substr(0, filepath.find_last_of("/\\"));
  return "";
}

static int LoadTexture(const std::string& filename, std::vector<Texture> *textures) {
  if (filename.empty()) return -1;

  std::cout << "  Loading texture : " << filename << std::endl;
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

    free(data);

    textures->push_back(texture);
    return textures->size() - 1;
  }

  std::cout << "  Failed to load : " << filename << std::endl;
  return -1;
}

bool LoadObj(const std::string &filename, float scale, std::vector<Mesh<float> > *meshes, std::vector<Material> *out_materials, std::vector<Texture> *out_textures) {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string err;

  std::string basedir = GetBaseDir(filename) + "/";
  const char* basepath = (basedir.compare("/") == 0) ? NULL : basedir.c_str();

  //auto t_start = std::chrono::system_clock::now();

  bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename.c_str(),
                              basepath, /* triangulate */ true);

  //auto t_end = std::chrono::system_clock::now();
  //std::chrono::duration<double, std::milli> ms = t_end - t_start;

  if (!err.empty()) {
    std::cerr << err << std::endl;
    return false;
  }

  //std::cout << "[LoadOBJ] Parse time : " << ms.count() << " [msecs]"
  //          << std::endl;

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
  //mesh->vertices.resize(num_vertices * 3, 0.0f);
  //mesh->faces.resize(num_faces * 3, 0);
  //mesh->material_ids.resize(num_faces, 0);
  //mesh->facevarying_normals.resize(num_faces * 3 * 3, 0.0f);
  //mesh->facevarying_uvs.resize(num_faces * 3 * 2, 0.0f);

  // TODO(LTE): Implement tangents and binormals

  //size_t faceIdxOffset = 0;

  //for (size_t i = 0; i < attrib.vertices.size(); i++) {
  //  mesh->vertices[i] = scale * attrib.vertices[i];
  //}

  for (size_t i = 0; i < shapes.size(); i++) {
    Mesh<float> mesh;

    const size_t num_faces = shapes[i].mesh.indices.size() / 3;
    mesh.faces.resize(num_faces * 3);
    mesh.material_ids.resize(num_faces);
    mesh.facevarying_normals.resize(num_faces * 3 * 3);
    mesh.facevarying_uvs.resize(num_faces * 3 * 2);
    mesh.vertices.resize(num_faces * 3 * 3);

    for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
    
      // reorder vertices. may create duplicated vertices.
      unsigned int f0 = shapes[i].mesh.indices[3 * f + 0].vertex_index;
      unsigned int f1 = shapes[i].mesh.indices[3 * f + 1].vertex_index;
      unsigned int f2 = shapes[i].mesh.indices[3 * f + 2].vertex_index;

      mesh.vertices[9 * f + 0] = attrib.vertices[3 * f0 + 0];
      mesh.vertices[9 * f + 1] = attrib.vertices[3 * f0 + 1];
      mesh.vertices[9 * f + 2] = attrib.vertices[3 * f0 + 2];

      mesh.vertices[9 * f + 3] = attrib.vertices[3 * f1 + 0];
      mesh.vertices[9 * f + 4] = attrib.vertices[3 * f1 + 1];
      mesh.vertices[9 * f + 5] = attrib.vertices[3 * f1 + 2];

      mesh.vertices[9 * f + 6] = attrib.vertices[3 * f2 + 0];
      mesh.vertices[9 * f + 7] = attrib.vertices[3 * f2 + 1];
      mesh.vertices[9 * f + 8] = attrib.vertices[3 * f2 + 2];

      mesh.faces[3 * f + 0] = 3 * f + 0;
      mesh.faces[3 * f + 1] = 3 * f + 1;
      mesh.faces[3 * f + 2] = 3 * f + 2;

      mesh.material_ids[f] = shapes[i].mesh.material_ids[f];
    }

    if (attrib.normals.size() > 0) {
      for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
        int f0, f1, f2;

        f0 = shapes[i].mesh.indices[3 * f + 0].normal_index;
        f1 = shapes[i].mesh.indices[3 * f + 1].normal_index;
        f2 = shapes[i].mesh.indices[3 * f + 2].normal_index;

        if (f0 > 0 && f1 > 0 && f2 > 0) {
          float n0[3], n1[3], n2[3];

          n0[0] = attrib.normals[3 * f0 + 0];
          n0[1] = attrib.normals[3 * f0 + 1];
          n0[2] = attrib.normals[3 * f0 + 2];

          n1[0] = attrib.normals[3 * f1 + 0];
          n1[1] = attrib.normals[3 * f1 + 1];
          n1[2] = attrib.normals[3 * f1 + 2];

          n2[0] = attrib.normals[3 * f2 + 0];
          n2[1] = attrib.normals[3 * f2 + 1];
          n2[2] = attrib.normals[3 * f2 + 2];

          mesh.facevarying_normals[3 * (3 * f + 0) + 0] =
              n0[0];
          mesh.facevarying_normals[3 * (3 * f + 0) + 1] =
              n0[1];
          mesh.facevarying_normals[3 * (3 * f + 0) + 2] =
              n0[2];

          mesh.facevarying_normals[3 * (3 * f + 1) + 0] =
              n1[0];
          mesh.facevarying_normals[3 * (3 * f + 1) + 1] =
              n1[1];
          mesh.facevarying_normals[3 * (3 * f + 1) + 2] =
              n1[2];

          mesh.facevarying_normals[3 * (3 * f + 2) + 0] =
              n2[0];
          mesh.facevarying_normals[3 * (3 * f + 2) + 1] =
              n2[1];
          mesh.facevarying_normals[3 * (3 * f + 2) + 2] =
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

          mesh.facevarying_normals[3 * (3 * f + 0) + 0] =
              N[0];
          mesh.facevarying_normals[3 * (3 * f + 0) + 1] =
              N[1];
          mesh.facevarying_normals[3 * (3 * f + 0) + 2] =
              N[2];

          mesh.facevarying_normals[3 * (3 * f + 1) + 0] =
              N[0];
          mesh.facevarying_normals[3 * (3 * f + 1) + 1] =
              N[1];
          mesh.facevarying_normals[3 * (3 * f + 1) + 2] =
              N[2];

          mesh.facevarying_normals[3 * (3 * f + 2) + 0] =
              N[0];
          mesh.facevarying_normals[3 * (3 * f + 2) + 1] =
              N[1];
          mesh.facevarying_normals[3 * (3 * f + 2) + 2] =
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

        mesh.facevarying_normals[3 * (3 * f + 0) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * f + 0) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * f + 0) + 2] = N[2];

        mesh.facevarying_normals[3 * (3 * f + 1) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * f + 1) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * f + 1) + 2] = N[2];

        mesh.facevarying_normals[3 * (3 * f + 2) + 0] = N[0];
        mesh.facevarying_normals[3 * (3 * f + 2) + 1] = N[1];
        mesh.facevarying_normals[3 * (3 * f + 2) + 2] = N[2];
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

          mesh.facevarying_uvs[2 * (3 * f + 0) + 0] = n0[0];
          mesh.facevarying_uvs[2 * (3 * f + 0) + 1] = n0[1];

          mesh.facevarying_uvs[2 * (3 * f + 1) + 0] = n1[0];
          mesh.facevarying_uvs[2 * (3 * f + 1) + 1] = n1[1];

          mesh.facevarying_uvs[2 * (3 * f + 2) + 0] = n2[0];
          mesh.facevarying_uvs[2 * (3 * f + 2) + 1] = n2[1];
        }
      }
    }

    meshes->push_back(mesh);
  }

  // material_t -> Material and Texture
  out_materials->resize(materials.size());
  out_textures->resize(0);
  for (size_t i = 0; i < materials.size(); i++) {
    (*out_materials)[i].diffuse[0] = materials[i].diffuse[0];
    (*out_materials)[i].diffuse[1] = materials[i].diffuse[1];
    (*out_materials)[i].diffuse[2] = materials[i].diffuse[2];
    (*out_materials)[i].specular[0] = materials[i].specular[0];
    (*out_materials)[i].specular[1] = materials[i].specular[1];
    (*out_materials)[i].specular[2] = materials[i].specular[2];

    (*out_materials)[i].id = i;

    // map_Kd
    (*out_materials)[i].diffuse_texid = LoadTexture(materials[i].diffuse_texname, out_textures);
    // map_Ks
    (*out_materials)[i].specular_texid = LoadTexture(materials[i].specular_texname, out_textures);
  }

  return true;
}

} // namespace example
