#include "serialize.h"

#define ESON_IMPLEMENTATION
#include "../common/eson.h"

#include <iostream>

namespace example {

template <typename T>
inline eson::Value createEsonValue(const std::vector<T>& src, size_t n_elem) {
  assert(src.size() == n_elem);
  return eson::Value(reinterpret_cast<const uint8_t*>(&(src.at(0))), sizeof(T) * n_elem);
}

template <typename T>
inline eson::Value createEsonValue(const T* src, size_t n_elem) {
  return eson::Value(reinterpret_cast<const uint8_t*>(&(src[0])), sizeof(T) * n_elem);
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
  for (size_t i = 0; i < n_elem; i++) {
    dst[i] = pointer[i];
  }
}

bool SaveSceneToEson(const std::string &eson_filename, const Scene &scene) {
  std::cout << "[SaveESON] " << eson_filename << std::endl;
  eson::Object root;

  size_t num_vertices = scene.mesh.vertices.size() / 3;
  size_t num_faces = scene.mesh.faces.size() / 3;

  // Mesh
  root["num_vertices"] = eson::Value(int64_t(num_vertices));
  root["num_faces"] = eson::Value(int64_t(num_faces));
  root["vertices"] = createEsonValue(scene.mesh.vertices, num_vertices * 3);
  root["facevarying_normals"] =
      createEsonValue(scene.mesh.facevarying_normals, num_faces * 3 * 3);
  //   root["facevarying_tangents"] =
  //   createEsonValue(gMesh.facevarying_tangents, num_faces * 3 * 3);
  //   root["facevarying_binormals"] =
  //   createEsonValue(gMesh.facevarying_binormals, num_faces * 3 * 3);
  root["facevarying_uvs"] =
      createEsonValue(scene.mesh.facevarying_uvs, num_faces * 2 * 3);
  //   root["facevarying_vertex_colors"] =
  //   createEsonValue(gMesh.facevarying_vertex_colors , num_faces * 3 * 3);
  root["faces"] = createEsonValue(scene.mesh.faces, num_faces * 3);
  root["material_ids"] = createEsonValue(scene.mesh.material_ids, num_faces);

  // Materials
  root["num_materials"] = eson::Value(int64_t(scene.materials.size()));
  for (size_t i = 0; i < scene.materials.size(); i++) {
    const Material& material = scene.materials[i];
    std::stringstream ss;
    ss << "material" << i << "_";
    std::string pf = ss.str();

    root[pf + "diffuse"] = createEsonValue(material.diffuse, 3);
    //root[pf + "specular"] = createEsonValue(material.specular, 3);
    root[pf + "id"] = eson::Value(int64_t(material.id));
    root[pf + "diffuse_texid"] = eson::Value(int64_t(material.diffuse_texid));
    root[pf + "vdisp_texid"] = eson::Value(int64_t(material.vdisp_texid));
  }

  // Textures
  root["num_textures"] = eson::Value(int64_t(scene.textures.size()));
  for (size_t i = 0; i < scene.textures.size(); i++) {
    const Texture& texture = scene.textures[i];
    std::stringstream ss;
    ss << "texture" << i << "_";
    std::string pf = ss.str();

    root[pf + "width"] = eson::Value(int64_t(texture.width));
    root[pf + "height"] = eson::Value(int64_t(texture.height));
    root[pf + "components"] = eson::Value(int64_t(texture.components));
    root[pf + "image"] = createEsonValue(
        texture.image.data(), size_t(texture.width * texture.height * texture.components));
  }

  eson::Value v = eson::Value(root);
  int64_t size = int64_t(v.Size());
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  uint8_t* ptr = &buf[0];
  ptr = v.Serialize(ptr);
  assert((ptr - &buf[0]) == size);

  FILE* fp = fopen(eson_filename.c_str(), "wb");
  if (!fp) {
    return false;
  }
  fwrite(&buf[0], 1, size_t(size), fp);
  fclose(fp);

  return true;
}

bool LoadSceneFromEson(const std::string &eson_filename, Scene *scene) {
  std::vector<uint8_t> buf;

  std::cout << "[LoadESON] " << eson_filename << std::endl;

  FILE* fp = fopen(eson_filename.c_str(), "rb");
  if (!fp) {
    return false;
  }

  fseek(fp, 0, SEEK_END);
  size_t len = size_t(ftell(fp));
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
  scene->mesh.vertices.resize(size_t(num_vertices * 3));
  scene->mesh.facevarying_normals.resize(size_t(num_faces * 3 * 3));
  scene->mesh.facevarying_uvs.resize(size_t(num_faces * 2 * 3));
  scene->mesh.faces.resize(size_t(num_faces * 3));
  scene->mesh.material_ids.resize(static_cast<size_t>(num_faces));

  recoverEsonValue(v, "vertices", scene->mesh.vertices.data(), size_t(num_vertices * 3));
  recoverEsonValue(v, "facevarying_normals", scene->mesh.facevarying_normals.data(),
                   size_t(num_faces * 3 * 3));
  // recoverEsonValue(v, "facevarying_tangents", gMesh.facevarying_tangents,
  // num_faces * 3 * 3);
  // recoverEsonValue(v, "facevarying_binormals", gMesh.facevarying_binormals,
  // num_faces * 3 * 3);
  recoverEsonValue(v, "facevarying_uvs", scene->mesh.facevarying_uvs.data(),
                   size_t(num_faces * 2 * 3));
  // recoverEsonValue(v, "facevarying_vertex_colors",
  // gMesh.facevarying_vertex_colors, num_faces * 3 * 3);
  recoverEsonValue(v, "faces", scene->mesh.faces.data(), size_t(num_faces * 3));
  recoverEsonValue(v, "material_ids", scene->mesh.material_ids.data(), size_t(num_faces));

  // Materials
  int64_t num_materials = v.Get("num_materials").Get<int64_t>();
  scene->materials.resize(static_cast<size_t>(num_materials));
  for (size_t i = 0; i < scene->materials.size(); i++) {
    Material& material = scene->materials[i];
    std::stringstream ss;
    ss << "material" << i << "_";
    std::string pf = ss.str();

    recoverEsonValue(v, pf + "diffuse", material.diffuse, 3);
    //recoverEsonValue(v, pf + "specular", material.specular, 3);
    material.id = int(v.Get(pf + "id").Get<int64_t>());
    material.diffuse_texid = int(v.Get(pf + "diffuse_texid").Get<int64_t>());
    material.vdisp_texid = int(v.Get(pf + "vdisp_texid").Get<int64_t>());
  }

  // Textures
  int64_t num_textures = v.Get("num_textures").Get<int64_t>();
  scene->textures.resize(static_cast<size_t>(num_textures));
  for (size_t i = 0; i < scene->textures.size(); i++) {
    Texture& texture = scene->textures[i];
    std::stringstream ss;
    ss << "texture" << i << "_";
    std::string pf = ss.str();

    texture.width = static_cast<int>(v.Get(pf + "width").Get<int64_t>());
    texture.height = static_cast<int>(v.Get(pf + "height").Get<int64_t>());
    texture.components = static_cast<int>(v.Get(pf + "components").Get<int64_t>());

    size_t n_elem = size_t(texture.width * texture.height * texture.components);
    texture.image.resize(n_elem);
    recoverEsonValue(v, pf + "image", texture.image.data(), n_elem);
  }

  return true;
}

} // namespace example


