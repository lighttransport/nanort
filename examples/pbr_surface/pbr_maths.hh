#pragma once

#include <chrono>
#include <glm.hpp>

#define HAS_NORMALS

// This is inspired by the sample implementation from khronos found here :
// https://github.com/KhronosGroup/glTF-WebGL-PBR/blob/master/shaders/pbr-frag.glsl

namespace pbr_maths {

constexpr static float const c_MinRoughness = 0.04;
#ifndef M_PI
constexpr static float const M_PI = 3.141592653589793;
#endif
// GLSL data types
using glm::mat3;
using glm::mat4;
using glm::vec2;
using glm::vec3;
using glm::vec4;

// GLSL funtions
using glm::clamp;
using glm::cross;
using glm::fract;
using glm::length;
using glm::max;
using glm::mix;
using glm::normalize;
using glm::step;

/// GLSL sampler2D object
struct sampler2D {
  /// Represent a single pixel on a texture
  struct pixel {
    /// Each byte is a componant. Value from 0 to 255
    uint8_t r, g, b, a;

    /// Return a number between 0 and 1 that correspound to the byte vale
    /// between 0 to 255
    static float to_float(uint8_t v) { return float(v) / 255.f; }

    /// Convert this object to a glm::vec4 (like a color in GLSL) implicitly
    operator glm::vec4() const {
      return {to_float(r), to_float(g), to_float(b), to_float(a)};
    }
  };

  /// Width of the texture
  size_t width = 0;
  /// Height of the texture
  size_t height = 0;
  /// The actual pixel array
  pixel* pixels = nullptr;

  pixel getPixel(size_t x, size_t y) const { return pixels[y * width + x]; }

  pixel getPixel(std::tuple<size_t, size_t> coord) const {
    return getPixel(std::get<0>(coord), std::get<1>(coord));
  }

  std::tuple<size_t, size_t> getPixelUV(const vec2& uv) const {
    // wrap uvs
    vec2 buv;

    auto in_bound = [](float a) {
      clamp(a, 0.f, 0.99999f);
      return a;
    };

    buv.x = in_bound(uv.x);
    buv.y = in_bound(uv.y);

    // get best matching pixel coordinates
    auto px_x = size_t(buv.x * width);
    auto px_y = size_t(buv.y * height);

    return std::make_tuple(px_x, px_y);
  }

  pixel getPixel(const vec2& uv) const
  { return getPixel(getPixelUV(uv));
  }

  /// Activate texture filtering
  mutable bool linearFiltering = false;

  /// Clear the pixel array
  void releasePixels() {
    delete[] pixels;
    pixels = nullptr;
  }
};

/// Simple linear interpolation on floats
float lerp(float a, float b, float f) { return a + f * (b - a); }



/// Replicate the texture2D function from GLSL
vec4 texture2D(const sampler2D& sampler, const vec2& uv) {
  auto pixelUV = sampler.getPixelUV(uv);
  auto px_x = std::get<0>(pixelUV);
  auto px_y = std::get<1>(pixelUV);

  // TODO linear interpolation on pixel values
  if (sampler.linearFiltering) {
    const auto textureSize = vec2(sampler.width, sampler.height);
    const auto texelSize = vec2(1.0f / textureSize.x, 1.0f / textureSize.y);

    vec4 tl = sampler.getPixel(uv);
    vec4 tr = sampler.getPixel(uv + vec2(texelSize.x, 0));
    vec4 bl = sampler.getPixel(uv + vec2(0, texelSize.y));
    vec4 br = sampler.getPixel(uv + texelSize);

    vec2 f = fract(uv * textureSize);
    vec4 tA = mix(tl, tr, f.x);
    vec4 tB = mix(bl, br, f.x);
    return mix(tA, tB, f.y);
  }

  // return the selected pixel
  return sampler.pixels[px_y * sampler.width + px_x];
}

/// Cubemap sampler object
struct samplerCube {
  /// Indexes for the faces of the cubemap
  enum cubemap_faces : size_t { LEFT, RIGHT, UP, DOWN, FRONT, BACK };
  sampler2D& getFace(cubemap_faces f) { return faces[f]; }
  sampler2D faces[6];

  void releasePixels() {
    for (auto& face : faces) face.releasePixels();
  }
};

void convert_xyz_to_cube_uv(float x, float y, float z, int* index, float* u,
                            float* v) {
  float absX = fabs(x);
  float absY = fabs(y);
  float absZ = fabs(z);

  int isXPositive = x > 0 ? 1 : 0;
  int isYPositive = y > 0 ? 1 : 0;
  int isZPositive = z > 0 ? 1 : 0;

  float maxAxis, uc, vc;

  // POSITIVE X
  if (isXPositive && absX >= absY && absX >= absZ) {
    // u (0 to 1) goes from +z to -z
    // v (0 to 1) goes from -y to +y
    maxAxis = absX;
    uc = -z;
    vc = y;
    *index = 0;
  }
  // NEGATIVE X
  if (!isXPositive && absX >= absY && absX >= absZ) {
    // u (0 to 1) goes from -z to +z
    // v (0 to 1) goes from -y to +y
    maxAxis = absX;
    uc = z;
    vc = y;
    *index = 1;
  }
  // POSITIVE Y
  if (isYPositive && absY >= absX && absY >= absZ) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from +z to -z
    maxAxis = absY;
    uc = x;
    vc = -z;
    *index = 2;
  }
  // NEGATIVE Y
  if (!isYPositive && absY >= absX && absY >= absZ) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -z to +z
    maxAxis = absY;
    uc = x;
    vc = z;
    *index = 3;
  }
  // POSITIVE Z
  if (isZPositive && absZ >= absX && absZ >= absY) {
    // u (0 to 1) goes from -x to +x
    // v (0 to 1) goes from -y to +y
    maxAxis = absZ;
    uc = x;
    vc = y;
    *index = 4;
  }
  // NEGATIVE Z
  if (!isZPositive && absZ >= absX && absZ >= absY) {
    // u (0 to 1) goes from +x to -x
    // v (0 to 1) goes from -y to +y
    maxAxis = absZ;
    uc = -x;
    vc = y;
    *index = 5;
  }

  // Convert range from -1 to 1 to 0 to 1
  *u = 0.5f * (uc / maxAxis + 1.0f);
  *v = 0.5f * (vc / maxAxis + 1.0f);
}

vec4 textureCube(const samplerCube& sampler, vec3 direction) {
  int i;
  vec2 uv;
  normalize(direction);
  convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, &i, &uv.s,
                         &uv.t);
  uv.t = 1 - uv.t;
  return texture2D(sampler.faces[i], uv);
}

// This datastructure is used throughough the code here
struct PBRInfo {
  float NdotL;  // cos angle between normal and light direction
  float NdotV;  // cos angle between normal and view direction
  float NdotH;  // cos angle between normal and half vector
  float LdotH;  // cos angle between light direction and half vector
  float VdotH;  // cos angle between view direction and half vector
  float perceptualRoughness;  // roughness value, as authored by the model
                              // creator (input to shader)
  float metalness;            // metallic value at the surface
  vec3 reflectance0;          // full reflectance color (normal incidence angle)
  vec3 reflectance90;         // reflectance color at grazing angle
  float alphaRoughness;       // roughness mapped to a more linear change in the
                              // roughness (proposed by [2])
  vec3 diffuseColor;          // color contribution from diffuse lighting
  vec3 specularColor;         // color contribution from specular lighting
};

#define MANUAL_SRGB ;

/// Object that represent the fragment shader, and all of it's global state, but
/// in C++. The intended use is that you set all the parameters that *should* be
/// global for the shader and set by OpenGL call when using the program
struct PBRShaderCPU {
  /// Virtual output : color of the "fragment" (aka: the pixel here)
  vec4 gl_FragColor;

  // TODO This is just a pass-through function. This will de pend on the color
  // space used on the fed textures...
  vec4 SRGBtoLINEAR(vec4 srgbIn) {
#ifdef MANUAL_SRGB
#ifdef SRGB_FAST_APPROXIMATION
    vec3 linOut = pow(srgbIn.xyz, vec3(2.2));
#else   // SRGB_FAST_APPROXIMATION
    vec3 bLess = step(vec3(0.04045), vec3(srgbIn));
    vec3 linOut =
        mix(vec3(srgbIn) / vec3(12.92),
            pow((vec3(srgbIn) + vec3(0.055)) / vec3(1.055), vec3(2.4)), bLess);
#endif  // SRGB_FAST_APPROXIMATION
    return vec4(linOut, srgbIn.w);
    ;
#else   // MANUAL_SRGB
    return srgbIn;
#endif  // MANUAL_SRGB
  }
  // Calculation of the lighting contribution from an optional Image Based Light
  // source.
  // Precomputed Environment Maps are required uniform inputs and are computed
  // as outlined in [1]. See our README.md on Environment Maps [3] for
  // additional discussion.
  vec3 getIBLContribution(PBRInfo pbrInputs, vec3 n, vec3 reflection) {
    float mipCount = 9.0;  // resolution of 512x512
    float lod = pbrInputs.perceptualRoughness * mipCount;
    // retrieve a scale and bias to F0. See [1], Figure 3
    vec3 brdf = vec3(SRGBtoLINEAR(
        texture2D(u_brdfLUT,
                  vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness))));

    vec3 diffuseLight = vec3(SRGBtoLINEAR(textureCube(u_DiffuseEnvSampler, n)));

#ifdef USE_TEX_LOD
    vec3 specularLight =
        SRGBtoLINEAR(textureCubeLodEXT(u_SpecularEnvSampler, reflection, lod))
            .rgb;
#else
    vec3 specularLight =
        vec3(SRGBtoLINEAR(textureCube(u_SpecularEnvSampler, reflection)));
#endif

    vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
    vec3 specular = specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);

    // For presentation, this allows us to disable IBL terms
    diffuse *= u_ScaleIBLAmbient.x;
    specular *= u_ScaleIBLAmbient.y;

    return diffuse + specular;
  }

  void main() {
    // Metallic and Roughness material properties are packed together
    // In glTF, these factors can be specified by fixed scalar values
    // or from a metallic-roughness map
    float perceptualRoughness = u_MetallicRoughnessValues.y;
    float metallic = u_MetallicRoughnessValues.x;

    if (useMetalRoughMap) {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b'
      // channel. This layout intentionally reserves the 'r' channel for
      // (optional) occlusion map data
      vec4 mrSample = texture2D(u_MetallicRoughnessSampler, v_UV);

      // NOTE: G channel of the map is used for roughness, B channel is used for
      // metalness
      perceptualRoughness = mrSample.g * perceptualRoughness;
      metallic = mrSample.b * metallic;
    }

    perceptualRoughness = clamp(perceptualRoughness, c_MinRoughness, 1.0f);
    metallic = clamp(metallic, 0.0f, 1.0f);
    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness [2].
    float alphaRoughness = perceptualRoughness * perceptualRoughness;

    // The albedo may be defined from a base texture or a flat color
    vec4 baseColor;
    if (useBaseColorMap) {
      baseColor =
          SRGBtoLINEAR(texture2D(u_BaseColorSampler, v_UV)) * u_BaseColorFactor;
    } else {
      baseColor = u_BaseColorFactor;
    }

    vec3 f0 = vec3(0.04);
    vec3 diffuseColor = vec3(baseColor) * (vec3(1.0) - f0);
    diffuseColor *= 1.0 - metallic;
    vec3 specularColor = mix(f0, vec3(baseColor), metallic);

    // Compute reflectance.
    float reflectance =
        max(max(specularColor.r, specularColor.g), specularColor.b);

    // For typical incident reflectance range (between 4% to 100%) set the
    // grazing reflectance to 100% for typical fresnel effect. For very low
    // reflectance range on highly diffuse objects (below 4%), incrementally
    // reduce grazing reflecance to 0%.
    float reflectance90 = clamp(reflectance * 25.0, 0.0, 1.0);
    vec3 specularEnvironmentR0 = specularColor;
    vec3 specularEnvironmentR90 = vec3(1.0, 1.0, 1.0) * reflectance90;

    vec3 n = getNormal();  // normal at surface point
    vec3 v = normalize(u_Camera -
                       v_Position);  // Vector from surface point to camera
    vec3 l = normalize(u_LightDirection);  // Vector from surface point to light
    vec3 h = normalize(l + v);             // Half vector between both l and v
    vec3 reflection = -normalize(reflect(v, n));

    float NdotL = clamp(dot(n, l), 0.001f, 1.0f);
    float NdotV = clamp(std::abs(dot(n, v)), 0.001f, 1.0f);
    float NdotH = clamp(dot(n, h), 0.0f, 1.0f);
    float LdotH = clamp(dot(l, h), 0.0f, 1.0f);
    float VdotH = clamp(dot(v, h), 0.0f, 1.0f);

    // Hey, modern C++ uniform initialiazation syntax just works!
    PBRInfo pbrInputs = PBRInfo{NdotL,
                                NdotV,
                                NdotH,
                                LdotH,
                                VdotH,
                                perceptualRoughness,
                                metallic,
                                specularEnvironmentR0,
                                specularEnvironmentR90,
                                alphaRoughness,
                                diffuseColor,
                                specularColor};

    // Calculate the shading terms for the microfacet specular shading model
    vec3 F = specularReflection(pbrInputs);
    float G = geometricOcclusion(pbrInputs);
    float D = microfacetDistribution(pbrInputs);

    // Calculation of analytical lighting contribution
    vec3 diffuseContrib = (1.0f - F) * diffuse(pbrInputs);
    vec3 specContrib = F * G * D / (4.0f * NdotL * NdotV);
    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the
    // light (cosine law)
    vec3 color = NdotL * u_LightColor * (diffuseContrib + specContrib);

    // Calculate lighting contribution from image based lighting source (IBL)

    if (useILB) {
      const auto contrib = getIBLContribution(pbrInputs, n, reflection);
      color += contrib;
    }
    // Apply optional PBR terms for additional (optional) shading

    if (useOcclusionMap) {
      float ao = texture2D(u_OcclusionSampler, v_UV).r;
      color = mix(color, color * ao, u_OcclusionStrength);
    }

    if (useEmissiveMap) {
      vec3 emissive = SRGBtoLINEAR(
          vec4(vec3(texture2D(u_EmissiveSampler, v_UV)) * u_EmissiveFactor, 1));
      color += emissive;
    }

    // This section uses mix to override final color for reference app
    // visualization of various parameters in the lighting equation.
    color = mix(color, F, u_ScaleFGDSpec.x);
    color = mix(color, vec3(G), u_ScaleFGDSpec.y);
    color = mix(color, vec3(D), u_ScaleFGDSpec.z);
    color = mix(color, specContrib, u_ScaleFGDSpec.w);

    color = mix(color, diffuseContrib, u_ScaleDiffBaseMR.x);
    color = mix(color, vec3(baseColor), u_ScaleDiffBaseMR.y);
    color = mix(color, vec3(metallic), u_ScaleDiffBaseMR.z);
    color = mix(color, vec3(perceptualRoughness), u_ScaleDiffBaseMR.w);

    gl_FragColor = vec4(pow(color, vec3(1.0 / 2.2)), baseColor.a);
  }

  // Find the normal for this fragment, pulling either from a predefined normal
  // map
  // or from the interpolated mesh normal and tangent attributes.
  vec3 getNormal() {
  // Retrieve the tangent space matrix
#ifndef HAS_TANGENTS
    /*    vec3 pos_dx = dFdx(v_Position);
    vec3 pos_dy = dFdy(v_Position);
    vec3 tex_dx = dFdx(vec3(v_UV, 0.0));
    vec3 tex_dy = dFdy(vec3(v_UV, 0.0));
    vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) /
             (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t)*/
    ;
#ifdef HAS_NORMALS
    vec3 ng = normalize(v_Normal);
#else
    vec3 ng = cross(pos_dx, pos_dy);
#endif

    // This is some random hack to calculate "a" tangent vector
    vec3 t;

    vec3 c1 = cross(ng, vec3(0.0, 0.0, 1.0));
    vec3 c2 = cross(ng, vec3(0.0, 1.0, 0.0));

    if (length(c1) > length(c2)) {
      t = c1;
    } else {
      t = c2;
    }

    t = normalize(t - ng * dot(ng, t));
    vec3 b = normalize(cross(ng, t));
    mat3 tbn = mat3(t, b, ng);
#else  // HAS_TANGENTS
    mat3 tbn = v_TBN;
#endif
    vec3 n;
    if (useNormalMap) {
      n = vec3(texture2D(u_NormalSampler, v_UV));
      n = normalize(
          tbn * ((2.0f * n - 1.0f) * vec3(u_NormalScale, u_NormalScale, 1.0)));
    } else {
      // The tbn matrix is linearly interpolated, so we need to re-normalize
      n = normalize(tbn[2]);
    }
    return n;
  }

  // Basic Lambertian diffuse
  // Implementation from Lambert's Photometria
  // https://archive.org/details/lambertsphotome00lambgoog See also [1],
  // Equation 1
  vec3 diffuse(PBRInfo pbrInputs) {
    return {pbrInputs.diffuseColor / float(M_PI)};
  }

  // The following equation models the Fresnel reflectance term of the spec
  // equation (aka F()) Implementation of fresnel from [4], Equation 15
  vec3 specularReflection(PBRInfo pbrInputs) {
    return pbrInputs.reflectance0 +
           (pbrInputs.reflectance90 - pbrInputs.reflectance0) *
               std::pow(clamp(1.0f - pbrInputs.VdotH, 0.0f, 1.0f), 5.0f);
  }

  // This calculates the specular geometric attenuation (aka G()),
  // where rougher material will reflect less light back to the viewer.
  // This implementation is based on [1] Equation 4, and we adopt their
  // modifications to alphaRoughness as input as originally proposed in [2].
  float geometricOcclusion(PBRInfo pbrInputs) {
    float NdotL = pbrInputs.NdotL;
    float NdotV = pbrInputs.NdotV;
    float r = pbrInputs.alphaRoughness;

    float attenuationL =
        2.0 * NdotL / (NdotL + sqrt(r * r + (1.0 - r * r) * (NdotL * NdotL)));
    float attenuationV =
        2.0 * NdotV / (NdotV + sqrt(r * r + (1.0 - r * r) * (NdotV * NdotV)));
    return attenuationL * attenuationV;
  }

  // The following equation(s) model the distribution of microfacet normals
  // across the area being drawn (aka D()) Implementation from "Average
  // Irregularity Representation of a Roughened Surface for Ray Reflection" by
  // T. S. Trowbridge, and K. P. Reitz Follows the distribution function
  // recommended in the SIGGRAPH 2013 course notes from EPIC Games [1],
  // Equation 3.
  float microfacetDistribution(PBRInfo pbrInputs) {
    float roughnessSq = pbrInputs.alphaRoughness * pbrInputs.alphaRoughness;
    float f =
        (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH +
        1.0;
    return roughnessSq / (M_PI * f * f);
  }

    // Global stuff pasted from glsl file

#define uniform
#define varying

  uniform vec3 u_LightDirection;
  uniform vec3 u_LightColor;

  bool useILB = false;
  uniform samplerCube u_DiffuseEnvSampler;
  uniform samplerCube u_SpecularEnvSampler;
  uniform sampler2D u_brdfLUT;

  bool useBaseColorMap = false;
  uniform sampler2D u_BaseColorSampler;

  bool useNormalMap = false;
  uniform sampler2D u_NormalSampler;
  uniform float u_NormalScale = 1;

  bool useEmissiveMap = false;
  uniform sampler2D u_EmissiveSampler;
  uniform vec3 u_EmissiveFactor;

  bool useMetalRoughMap = false;
  uniform sampler2D u_MetallicRoughnessSampler;

  bool useOcclusionMap = false;
  uniform sampler2D u_OcclusionSampler;
  uniform float u_OcclusionStrength = 1;

  uniform vec2 u_MetallicRoughnessValues = {1, 1};
  uniform vec4 u_BaseColorFactor;

  uniform vec3 u_Camera;

  // debugging flags used for shader output of intermediate PBR variables
  uniform vec4 u_ScaleDiffBaseMR{0};
  uniform vec4 u_ScaleFGDSpec{0};
  uniform vec4 u_ScaleIBLAmbient{0};

  varying vec3 v_Position;

  varying vec2 v_UV;

#ifdef HAS_NORMALS
#ifdef HAS_TANGENTS
  varying mat3 v_TBN;
#else
  varying vec3 v_Normal;
#endif
#endif
#undef uniform
#undef varying
};

}  // namespace pbr_maths
