#include <array>
#include <glm.hpp>
#include <iostream>
#include <vector>

#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>

#include "nanort.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef _MSC_VER
#define STBI_MSC_SECURE_CRT
#endif
#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include "pbr_maths.hh"

#include "gltf-loader.h"
#include "material.h"
#include "mesh.h"
#include "utility.h"

using float_precision = float;

void loadSampler(pbr_maths::sampler2D<float_precision>& sampler,
                 const stbi_uc* data, int w, int h, int c) {
  sampler.width = w;
  sampler.height = h;
  sampler.pixels = new pbr_maths::sampler2D<float_precision>::pixel[w * h];
  sampler.linearFiltering = true;
  for (size_t i = 0; i < w * h; ++i) {
    sampler.pixels[i].r = data[i * c + 0];
    sampler.pixels[i].g = data[i * c + 1];
    sampler.pixels[i].b = data[i * c + 2];
    if (c == 4) sampler.pixels[i].a = data[i * c + 3];
  }
}

// This permit to load metal and roughness maps from different textures (instead
// of the standard GREEN/BLUE combined one)  The shader expect theses two maps
// as a combined one, so we are building it ourselves
void loadCombineMetalRoughSampler(
    pbr_maths::sampler2D<float_precision>& sampler, const stbi_uc* metalData,
    int mw, int mh, int mc, const stbi_uc* roughnessData, int rw, int rh,
    int rc) {
  assert(mw == rw);
  assert(mh == rh);

  sampler.pixels = new pbr_maths::sampler2D<float_precision>::pixel[mw * mh];

  for (size_t i = 0; i < mw * mh; ++i) {
    // We don't really care about these ones
    sampler.pixels[i].r = 0;
    sampler.pixels[i].a = 255;

    sampler.pixels[i].g = roughnessData[i * rc];
    sampler.pixels[i].b = metalData[i * mc];
  }
}

// PLEASE RESPECT ORDER LEFT, RIGHT, UP, DOWN, FRONT, BACK
void loadSamplerCube(pbr_maths::samplerCube<float_precision>& cubemap,
                     const std::array<std::string, 6>& files) {
  for (size_t i{0}; i < 6; ++i) {
    cubemap.faces[i].linearFiltering = true;
    const auto file = files[i];
    int w, h, c;
    auto mapData = stbi_load(file.c_str(), &w, &h, &c, 0);
    if (mapData)
      loadSampler(cubemap.faces[i], mapData, w, h, c);
    else {
      std::cerr << "Cannot load " << file << " as part of the cubemap!";
      exit(-1);
    }
    stbi_image_free(mapData);
  }
}

int main() {
  pbr_maths::sampler2D<float_precision> normalMap, baseColorMap, emissiveMap,
      brdfLUT;
  //  int w, h, c;
  // auto normdata = stbi_load("./MetalPlates02_nrm.jpg", &w, &h, &c, 0);
  // if (normdata) loadSampler(normalMap, normdata, w, h, c);

  // auto colData = stbi_load("./MetalPlates02_col.jpg", &w, &h, &c, 0);
  // if (colData) loadSampler(baseColorMap, colData, w, h, c);

  /* We can compute the BRDF ourselves. Will not load Look up table texture
  If depending on an external texture for ILB BRDF, you can supply a LUT texture
  describing the BRDF outputs from roughness and angle (see Unreal PBR paper
  section Image-Based Lighting/Environment BRDF */

  // auto brdfLUTData = stbi_load("./brdfLUT.png", &w, &h, &c, 0);
  // if (brdfLUTData) loadSampler(brdfLUT, brdfLUTData, w, h, c);

  // brdfLUT.boundsOperation = pbr_maths::sampler2D::outOfBounds::clamp;

  pbr_maths::sampler2D<float_precision> metalRoughMap;
  // int rw, rh, rc, mw, mh, mc;
  // auto roughData = stbi_load("./MetalPlates02_rgh.jpg", &rw, &rh, &rc, 0);
  // auto metalData = stbi_load("./MetalPlates02_met.jpg", &mw, &mh, &mc, 0);
  // if (roughData && metalData)
  //  loadCombineMetalRoughSampler(metalRoughMap, metalData, mw, mh, mc,
  //                               roughData, rw, rh, rc);

  pbr_maths::samplerCube<float_precision> specularEnvMap, diffuseEnvMap;

  loadSamplerCube(diffuseEnvMap, {"diffuse_left_0.jpg", "diffuse_right_0.jpg",
                                  "diffuse_top_0.jpg", "diffuse_bottom_0.jpg",
                                  "diffuse_front_0.jpg", "diffuse_back_0.jpg"});
  loadSamplerCube(specularEnvMap,
                  {"environment_left_0.jpg", "environment_right_0.jpg",
                   "environment_top_0.jpg", "environment_bottom_0.jpg",
                   "environment_front_0.jpg", "environment_back_0.jpg"});

  // Data was copied into the sampler objects, free the images
  // stbi_image_free(normdata);
  // stbi_image_free(colData);
  // stbi_image_free(roughData);
  // stbi_image_free(metalData);

  // No LUT
  // stbi_image_free(brdfLUTData);

  PbrMaterial<float> material;
  material.metalness = 1;
  material.roughness = 1;
  material.albedo.r = 1;
  material.albedo.g = 1;
  material.albedo.b = 1;
  material.emissiveness = 1;
  material.ILBContrib = 0.25f;

  std::vector<PointLight> lights;

  lights.emplace_back(glm::vec3{0.f, 3.f, 0.25f}, Color<float>{2, 2, 2});

  Mesh<float> mesh;
  std::vector<Texture> textures;

  example::LoadGLTF("./DamagedHelmet.glb", 0.25, mesh, textures);
  mesh.num_vertices = mesh.vertices.size();
  mesh.num_faces = (unsigned)mesh.faces.size() / 3;

  auto baseColorIt =
      std::find_if(textures.begin(), textures.end(), [](const Texture& text) {
        return text.use == Texture::usage::baseColor;
      });

  auto metalRoughIt =
      std::find_if(textures.begin(), textures.end(), [](const Texture& text) {
        return text.use == Texture::usage::metal_rough;
      });

  auto normalIt = std::find_if(
      textures.begin(), textures.end(),
      [](const Texture& text) { return text.use == Texture::usage::normal; });

  auto emitIt = std::find_if(
      textures.begin(), textures.end(),
      [](const Texture& text) { return text.use == Texture::usage::emit; });

  if (baseColorIt != textures.end()) {
    loadSampler(baseColorMap, baseColorIt->image, baseColorIt->width,
                baseColorIt->height, baseColorIt->components);
    baseColorMap.boundsOperation =
        pbr_maths::sampler2D<float_precision>::outOfBounds::wrap;
  }

  if (normalIt != textures.end()) {
    loadSampler(normalMap, normalIt->image, normalIt->width, normalIt->height,
                normalIt->components);
    normalMap.boundsOperation =
        pbr_maths::sampler2D<float_precision>::outOfBounds::wrap;
  }

  if (metalRoughIt != textures.end()) {
    loadSampler(metalRoughMap, metalRoughIt->image, metalRoughIt->width,
                metalRoughIt->height, metalRoughIt->components);
    metalRoughMap.boundsOperation =
        pbr_maths::sampler2D<float_precision>::outOfBounds::wrap;
  }

  if (emitIt != textures.end()) {
    loadSampler(emissiveMap, emitIt->image, emitIt->width, emitIt->height,
                emitIt->components);
    emissiveMap.boundsOperation =
        pbr_maths::sampler2D<float_precision>::outOfBounds::wrap;
  }

  for (auto texture : textures) delete[] texture.image;
  textures.clear();

  nanort::BVHBuildOptions<float> build_options;
  build_options.cache_bbox = false;

  printf("  BVH build option:\n");
  printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
  printf("    SAH binsize         : %d\n", build_options.bin_size);

  nanort::TriangleMesh<float> object(mesh.vertices.data(), mesh.faces.data(),
                                     3 * sizeof(float));
  nanort::TriangleSAHPred<float> object_pred(
      mesh.vertices.data(), mesh.faces.data(), 3 * sizeof(float));

  nanort::BVHAccel<float> accel;
  accel.Build(mesh.num_faces, object, object_pred, build_options);

  nanort::BVHBuildStatistics stats = accel.GetStatistics();
  printf("  BVH statistics:\n");
  printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
  printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
  printf("  Max tree depth     : %d\n", stats.max_tree_depth);
  float bmin[3], bmax[3];
  accel.BoundingBox(bmin, bmax);
  printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
  printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);

  const size_t width = 2048;
  const size_t height = 2048;

  std::vector<pixel> img(width * height);
  // memset(img.data(), 255, img.size() * sizeof(pixel));

  glm::mat4 viewRotate(1.0f);

  viewRotate =
      glm::rotate(viewRotate, glm::radians(-90.0f), glm::vec3(1.0f, 0.f, 0.f));
  glm::vec3 org{0, 1, 0};
  // lights[0].position = org;

#ifdef _OPENMP
  printf("This program was buit with OpenMP support\n");
  printf("NanoRT main loop using #pragma omp parallel for\n");
#pragma omp parallel for
#endif
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      // access pixel we are going to calculate
      auto& pixel = img[(y * width) + x];
      pixel.r = pixel.g = pixel.b = 0;
      nanort::BVHTraceOptions trace_options;

      nanort::Ray<float> camRay;
      camRay.org[0] = org.x;
      camRay.org[1] = org.y;
      camRay.org[2] = org.z;

      glm::vec3 dir;
      dir.x = (x / (float)width) - 0.5f;
      dir.y = (y / (float)height) - 0.5f;
      dir.z = -1;

      dir = viewRotate * glm::vec4(dir, 1);

      glm::normalize(dir);

      camRay.dir[0] = dir.x;
      camRay.dir[1] = dir.y;
      camRay.dir[2] = dir.z;

      float kFar = 1.0e+30f;
      camRay.min_t = 0.0f;
      camRay.max_t = kFar;

      nanort::TriangleIntersector<float, nanort::TriangleIntersection<float> >
          triangle_intersector(mesh.vertices.data(), mesh.faces.data(),
                               sizeof(float) * 3);

      nanort::TriangleIntersection<float> isect;

      if (accel.Traverse(camRay, triangle_intersector, &isect)) {
        glm::vec3 hit = org + isect.t * dir;
        glm::vec2 uv{0, 0};

        if (mesh.has_uvs())
          uv = mesh.getTextureCoord(isect.prim_id, isect.u, isect.v);

        // 1) shoot ray to point light
        for (const auto& light : lights) {
          nanort::Ray<float> lightRay;

          glm::vec3 dir = light.position - hit;
          glm::normalize(dir);
          hit += 0.00001f * dir;  // some bias

          memcpy(lightRay.org, toArr<float>(hit).data(), 3 * sizeof(float));
          memcpy(lightRay.dir, toArr<float>(dir).data(), 3 * sizeof(float));

          // 2) if nothing was hit, draw pixel with shader
          if (!accel.Traverse(lightRay, triangle_intersector, &isect)) {
            // This object represent a fragment shader, and is literally a
            // translation in C++ from an official example from khronos
            pbr_maths::PBRShaderCPU<float_precision> shader;

            // Fill in the uniform/varying variables
            shader.u_Camera.x = camRay.org[0];
            shader.u_Camera.y = camRay.org[1];
            shader.u_Camera.z = camRay.org[2];

            shader.u_MetallicRoughnessValues =
                glm::vec2{material.metalness, material.roughness};
            shader.u_BaseColorFactor = glm::vec4{
                material.albedo.r, material.albedo.g, material.albedo.b, 1};
            shader.v_Position = hit;
            shader.u_EmissiveFactor =
                glm::vec3{material.emissiveness, material.emissiveness,
                          material.emissiveness};

            shader.u_LightDirection = dir;
            shader.u_LightColor =
                glm::vec3(light.color.r, light.color.g, light.color.b);

            // Interpolate the normals
            glm::vec3 n0{mesh.facevarying_normals[3 * 3 * isect.prim_id + 0],
                         mesh.facevarying_normals[3 * 3 * isect.prim_id + 1],
                         mesh.facevarying_normals[3 * 3 * isect.prim_id + 2]};
            glm::vec3 n1{mesh.facevarying_normals[3 * 3 * isect.prim_id + 3],
                         mesh.facevarying_normals[3 * 3 * isect.prim_id + 4],
                         mesh.facevarying_normals[3 * 3 * isect.prim_id + 5]};
            glm::vec3 n2{mesh.facevarying_normals[3 * 3 * isect.prim_id + 6],
                         mesh.facevarying_normals[3 * 3 * isect.prim_id + 7],
                         mesh.facevarying_normals[3 * 3 * isect.prim_id + 8]};
            auto computedNormal = glm::normalize(
                (1.0f - isect.u - isect.v) * n0 + isect.u * n1 + isect.v * n2);

            shader.v_Normal = computedNormal;
            shader.v_UV = uv;

            if (normalMap.pixels) {
              shader.useNormalMap = true;
              shader.u_NormalSampler = normalMap;
            } else {
              shader.useNormalMap = false;
            }

            if (baseColorMap.pixels) {
              shader.useBaseColorMap = true;
              shader.u_BaseColorSampler = baseColorMap;
            } else {
              shader.useBaseColorMap = false;
            }

            if (metalRoughMap.pixels) {
              shader.useMetalRoughMap = true;
              shader.u_MetallicRoughnessSampler = metalRoughMap;
            } else {
              shader.useMetalRoughMap = false;
            }

            if (emissiveMap.pixels) {
              shader.useEmissiveMap = true;
              shader.u_EmissiveSampler = emissiveMap;
            } else {
              shader.useEmissiveMap = false;
            }

            if (brdfLUT.pixels) {
              shader.use_ILB_BRDF_LUT = !shader.forceBRDFCompute;
              shader.u_brdfLUT = brdfLUT;
            } else {
              shader.use_ILB_BRDF_LUT = false;
            }
            if (diffuseEnvMap.faces[0].pixels &&
                specularEnvMap.faces[0].pixels) {
              shader.useILB = true;
              shader.u_DiffuseEnvSampler = diffuseEnvMap;
              shader.u_SpecularEnvSampler = specularEnvMap;
              shader.u_ScaleIBLAmbient = {
                  material.ILBContrib, material.ILBContrib, material.ILBContrib,
                  material.ILBContrib};
            } else {
              shader.useILB = false;
            }

            //"Execute shader" on the current pixel, for the current light
            // source
            shader.main();

            // Accumulate the color
            pixel.r =
                std::min<int>(255, pixel.r + int(255 * shader.gl_FragColor.r));
            pixel.g =
                std::min<int>(255, pixel.g + int(255 * shader.gl_FragColor.g));
            pixel.b =
                std::min<int>(255, pixel.b + int(255 * shader.gl_FragColor.b));
          }
        }
      }

      pixel.a = 255;
    }

  stbi_flip_vertically_on_write(true);  // Flip Y
  stbi_write_png("out.png", width, height, 4, (void*)img.data(), 0);

  normalMap.releasePixels();
  baseColorMap.releasePixels();
  metalRoughMap.releasePixels();
  specularEnvMap.releasePixels();
  diffuseEnvMap.releasePixels();
  // brdfLUT.releasePixels();

  return 0;
}
