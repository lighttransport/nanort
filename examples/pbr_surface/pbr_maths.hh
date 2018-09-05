#pragma once
#include <chrono>
#include <glm.hpp>

#define HAS_NORMALS

#define ADD_TEMPLATED_TYPES(fp_type)                          \
  using t_vec2 = glm::vec<2, fp_type, glm::precision::highp>; \
  using t_vec3 = glm::vec<3, fp_type, glm::precision::highp>; \
  using t_vec4 = glm::vec<4, fp_type, glm::precision::highp>;

// This is inspired by the sample implementation from khronos found here :
// https://github.com/KhronosGroup/glTF-WebGL-PBR/blob/master/shaders/pbr-frag.glsl

/// Contains data-structures, free functions and object for PBR shader
/// computation
namespace pbr_maths {

// Compile time constants
constexpr static double const c_MinRoughness = 0.04;
#ifndef M_PI
constexpr static double const M_PI = 3.141592653589793;
#endif

// GLSL data types
using glm::mat3;
using glm::mat4;
// using glm::t_vec2;
// using glm::vec3;
// using glm::t_vec4;

// GLSL functions
using glm::clamp;
using glm::cross;
using glm::fract;
using glm::length;
using glm::max;
using glm::mix;
using glm::normalize;
using glm::step;

/// GLSL sampler2D object
template <typename fp_type>
struct sampler2D {
  ADD_TEMPLATED_TYPES(fp_type);

  enum class outOfBounds { clamp, wrap };
  outOfBounds boundsOperation;

  /// Represent a single pixel on a texture
  struct pixel {
    /// Each byte is a component. Value from 0 to 255
    uint8_t r, g, b, a;

    /// Return a number between 0 and 1 that correspond to the byte vale
    /// between 0 to 255
    static fp_type to_float(uint8_t v) { return fp_type(v) / 255.f; }

    /// Convert this object to a glm::t_vec4 (like a color in GLSL) implicitly
    operator t_vec4() const {
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

  std::tuple<size_t, size_t> getPixelUV(const t_vec2& uv) const {
    // wrap uvs
    t_vec2 buv;

    auto in_bound = [=](float a) {
      switch (boundsOperation) {
        case outOfBounds::clamp:
          clamp(a, 0.f, 0.99999f);
          break;

        case outOfBounds::wrap:
          if (a > 1) a = fmod(a, 1.0f);
          if (a < 0) {
            a = 1 - fmod(abs(a), 1.0f);
          }
          break;
      }
      return a;
    };

    buv.x = in_bound(uv.x);
    buv.y = in_bound(uv.y);

    // get best matching pixel coordinates
    auto px_x = std::min(width - 1, size_t(buv.x * width));
    auto px_y = std::min(height - 1, size_t(buv.y * height));

    return std::make_tuple(px_x, px_y);
  }

  pixel getPixel(const t_vec2& uv) const { return getPixel(getPixelUV(uv)); }

  /// Activate texture filtering
  mutable bool linearFiltering = false;

  /// Clear the pixel array
  void releasePixels() {
    delete[] pixels;
    pixels = nullptr;
  }
};

/// Simple linear interpolation on floats
template <typename fp_type>
fp_type lerp(fp_type a, fp_type b, fp_type f) {
  return a + f * (b - a);
}

/// Replicate the texture2D function from GLSL
template <typename fp_type, typename uv_vec>
glm::vec<4, fp_type, glm::precision::highp> texture2D(
    const sampler2D<fp_type>& sampler, const uv_vec& uv) {
  ADD_TEMPLATED_TYPES(fp_type)
  auto pixelUV = sampler.getPixelUV(uv);
  auto px_x = std::get<0>(pixelUV);
  auto px_y = std::get<1>(pixelUV);

  // TODO linear interpolation on pixel values
  if (sampler.linearFiltering) {
    const auto textureSize = t_vec2(sampler.width, sampler.height);
    const auto texelSize = t_vec2(1.0f / textureSize.x, 1.0f / textureSize.y);

    t_vec4 tl = sampler.getPixel(uv);
    t_vec4 tr = sampler.getPixel(uv + t_vec2(texelSize.x, 0));
    t_vec4 bl = sampler.getPixel(uv + t_vec2(0, texelSize.y));
    t_vec4 br = sampler.getPixel(uv + texelSize);

    t_vec2 f = fract(uv * textureSize);
    t_vec4 tA = mix(tl, tr, f.x);
    t_vec4 tB = mix(bl, br, f.x);
    return t_vec4(mix(tA, tB, f.y));
  }

  // return the selected pixel
  return t_vec4(sampler.pixels[px_y * sampler.width + px_x]);
}

/// Cubemap sampler object
template <typename fp_type>
struct samplerCube {
  /// Indexes for the faces of the cubemap
  enum cubemap_faces : size_t { LEFT, RIGHT, UP, DOWN, FRONT, BACK };
  sampler2D<fp_type>& getFace(cubemap_faces f) { return faces[f]; }
  sampler2D<fp_type> faces[6];

  void releasePixels() {
    for (auto& face : faces) face.releasePixels();
  }
};

/// Get the cooresponding UV and face index of a cubemap from a ray/normal
/// vector
template <typename fp_type>
void convert_xyz_to_cube_uv(fp_type x, fp_type y, fp_type z, int* index,
                            fp_type* u, fp_type* v) {
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

template <typename fp_type>
glm::vec<4, fp_type, glm::precision::highp> textureCube(
    const samplerCube<fp_type>& sampler,
    glm::vec<3, fp_type, glm::precision::highp> direction) {
  ADD_TEMPLATED_TYPES(fp_type);
  int i;
  t_vec2 uv;
  normalize(direction);
  convert_xyz_to_cube_uv(direction.x, direction.y, direction.z, &i, &uv.s,
                         &uv.t);
  uv.t = 1 - uv.t;
  return texture2D(sampler.faces[i], uv);
}

// This data-structure is used throughout the code here
template <typename fp_type>
struct PBRInfo {
  ADD_TEMPLATED_TYPES(fp_type);
  fp_type NdotL;  // cos angle between normal and light direction
  fp_type NdotV;  // cos angle between normal and view direction
  fp_type NdotH;  // cos angle between normal and half vector
  fp_type LdotH;  // cos angle between light direction and half vector
  fp_type VdotH;  // cos angle between view direction and half vector
  fp_type perceptualRoughness;  // roughness value, as authored by the model
                                // creator (input to shader)
  fp_type metalness;            // metallic value at the surface
  t_vec3 reflectance0;     // full reflectance color (normal incidence angle)
  t_vec3 reflectance90;    // reflectance color at grazing angle
  fp_type alphaRoughness;  // roughness mapped to a more linear change in the
                           // roughness (proposed by [2])
  t_vec3 diffuseColor;     // color contribution from diffuse lighting
  t_vec3 specularColor;    // color contribution from specular lighting
};

#define MANUAL_SRGB ;

/// Object that represent the fragment shader, and all of it's global state, but
/// in C++. The intended use is that you set all the parameters that *should* be
/// global for the shader and set by OpenGL call when using the program
template <typename fp_type>
struct PBRShaderCPU {
  ADD_TEMPLATED_TYPES(fp_type);

  /// Virtual output : color of the "fragment" (aka: the pixel here)
  t_vec4 gl_FragColor;

  // TODO This is just a pass-through function. This will depend on the color
  // space used on the fed textures...
  t_vec4 SRGBtoLINEAR(t_vec4 srgbIn) {
#ifdef MANUAL_SRGB
#ifdef SRGB_FAST_APPROXIMATION
    t_vec3 linOut = pow(srgbIn.xyz, t_vec3(2.2));
#else   // SRGB_FAST_APPROXIMATION
    t_vec3 bLess = step(t_vec3(0.04045), t_vec3(srgbIn));
    t_vec3 linOut =
        mix(t_vec3(srgbIn) / t_vec3(12.92),
            pow((t_vec3(srgbIn) + t_vec3(0.055)) / t_vec3(1.055), t_vec3(2.4)),
            bLess);
#endif  // SRGB_FAST_APPROXIMATION
    return t_vec4(linOut, srgbIn.w);
    ;
#else   // MANUAL_SRGB
    return srgbIn;
#endif  // MANUAL_SRGB
  }

  // Code from "Image Based Lighting" section of the Unreal PBR article. Based
  // on the C++ re-implementation of the integration code from
  // https://github.com/HectorMF/BRDFGenerator (MIT)
  fp_type RadicalInverse_VdC(unsigned int bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return fp_type(bits) * 2.3283064365386963e-10;
  }

  t_vec2 Hammersley(unsigned int i, unsigned int N) {
    return t_vec2(fp_type(i) / fp_type(N), RadicalInverse_VdC(i));
  }

  t_vec3 ImportanceSampleGGX(t_vec2 Xi, float roughness, t_vec3 N) {
    fp_type a = roughness * roughness;

    fp_type phi = 2.0 * M_PI * Xi.x;
    fp_type cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    fp_type sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // from spherical coordinates to cartesian coordinates
    t_vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // from tangent-space vector to world-space sample vector
    t_vec3 up =
        abs(N.z) < 0.999 ? t_vec3(0.0, 0.0, 1.0) : t_vec3(1.0, 0.0, 0.0);
    t_vec3 tangent = normalize(cross(up, N));
    t_vec3 bitangent = cross(N, tangent);

    t_vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
  }

  float GeometrySchlickGGX(fp_type NdotV, fp_type roughness) {
    fp_type a = roughness;
    fp_type k = (a * a) / 2.0;

    fp_type nom = NdotV;
    fp_type denom = NdotV * (1.0 - k) + k;

    return nom / denom;
  }

  float GeometrySmith(fp_type roughness, fp_type NoV, fp_type NoL) {
    fp_type ggx2 = GeometrySchlickGGX(NoV, roughness);
    fp_type ggx1 = GeometrySchlickGGX(NoL, roughness);

    return ggx1 * ggx2;
  }

  t_vec2 IntegrateBRDF(fp_type NdotV, fp_type roughness, unsigned int samples) {
    t_vec3 V;
    V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0;
    V.z = NdotV;

    fp_type A = 0.0;
    fp_type B = 0.0;

    t_vec3 N = t_vec3(0.0, 0.0, 1.0);

    for (unsigned int i = 0u; i < samples; ++i) {
      t_vec2 Xi = Hammersley(i, samples);
      t_vec3 H = ImportanceSampleGGX(Xi, roughness, N);
      t_vec3 L = normalize(2.0f * dot(V, H) * H - V);

      fp_type NoL = max(L.z, fp_type(0.0));
      fp_type NoH = max(H.z, fp_type(0.0));
      fp_type VoH = max(dot(V, H), fp_type(0.0));
      fp_type NoV = max(dot(N, V), fp_type(0.0));

      if (NoL > 0.0) {
        fp_type G = GeometrySmith(roughness, NoV, NoL);

        fp_type G_Vis = (G * VoH) / (NoH * NoV);
        fp_type Fc = pow(1.0 - VoH, 5.0);

        A += (1.0 - Fc) * G_Vis;
        B += Fc * G_Vis;
      }
    }

    return t_vec2(A / fp_type(samples), B / fp_type(samples));
  }

  // Calculation of the lighting contribution from an optional Image Based Light
  // source.
  // Precomputed Environment Maps are required uniform inputs and are computed
  // as outlined in [1]. See our README.md on Environment Maps [3] for
  // additional discussion.

  t_vec3 getIBLContribution(PBRInfo<fp_type> pbrInputs, t_vec3 n,
                            t_vec3 reflection) {
    float mipCount = 9.0f;  // resolution of 512x512
    float lod = pbrInputs.perceptualRoughness * mipCount;
    // retrieve a scale and bias to F0. See [1], Figure 3
    t_vec2 brdf;

    if (use_ILB_BRDF_LUT) {
      auto brdf3 = t_vec3(SRGBtoLINEAR(texture2D(
          u_brdfLUT,
          t_vec2(pbrInputs.NdotV, 1.0 - pbrInputs.perceptualRoughness))));

      // Only use RG channel
      brdf.x = brdf3.x;
      brdf.y = brdf3.y;
    } else {
      brdf = t_vec2(IntegrateBRDF(pbrInputs.NdotV,
                                  1.0 - pbrInputs.perceptualRoughness,
                                  brdfResolution));
    }

    t_vec3 diffuseLight =
        t_vec3(SRGBtoLINEAR(textureCube(u_DiffuseEnvSampler, n)));

#ifdef USE_TEX_LOD
    t_vec3 specularLight =
        SRGBtoLINEAR(textureCubeLodEXT(u_SpecularEnvSampler, reflection, lod))
            .rgb;
#else
    t_vec3 specularLight =
        t_vec3(SRGBtoLINEAR(textureCube(u_SpecularEnvSampler, reflection)));
#endif

    t_vec3 diffuse = diffuseLight * pbrInputs.diffuseColor;
    t_vec3 specular =
        specularLight * (pbrInputs.specularColor * brdf.x + brdf.y);

    // For presentation, this allows us to disable IBL terms
    diffuse *= u_ScaleIBLAmbient.x;
    specular *= u_ScaleIBLAmbient.y;

    return diffuse + specular;
  }

  void main() {
    // Metallic and Roughness material properties are packed together
    // In glTF, these factors can be specified by fixed scalar values
    // or from a metallic-roughness map
    fp_type perceptualRoughness = u_MetallicRoughnessValues.y;
    fp_type metallic = u_MetallicRoughnessValues.x;

    if (useMetalRoughMap) {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b'
      // channel. This layout intentionally reserves the 'r' channel for
      // (optional) occlusion map data
      t_vec4 mrSample = texture2D(u_MetallicRoughnessSampler, v_UV);

      // NOTE: G channel of the map is used for roughness, B channel is used for
      // metalness
      perceptualRoughness = mrSample.g * perceptualRoughness;
      metallic = mrSample.b * metallic;
    }

    perceptualRoughness =
        clamp(perceptualRoughness, fp_type(c_MinRoughness), fp_type(1.0));
    metallic = clamp(metallic, fp_type(0.0), fp_type(1.0));
    // Roughness is authored as perceptual roughness; as is convention,
    // convert to material roughness by squaring the perceptual roughness [2].
    fp_type alphaRoughness = perceptualRoughness * perceptualRoughness;

    // The albedo may be defined from a base texture or a flat color
    t_vec4 baseColor;
    if (useBaseColorMap) {
      baseColor =
          SRGBtoLINEAR(texture2D(u_BaseColorSampler, v_UV)) * u_BaseColorFactor;
    } else {
      baseColor = u_BaseColorFactor;
    }

    t_vec3 f0 = t_vec3(0.04f);
    t_vec3 diffuseColor = t_vec3(baseColor) * (t_vec3(1.0f) - f0);
    diffuseColor *= 1.0 - metallic;
    t_vec3 specularColor = mix(f0, t_vec3(baseColor), metallic);

    // Compute reflectance.
    fp_type reflectance =
        max(max(specularColor.r, specularColor.g), specularColor.b);

    // For typical incident reflectance range (between 4% to 100%) set the
    // grazing reflectance to 100% for typical fresnel effect. For very low
    // reflectance range on highly diffuse objects (below 4%), incrementally
    // reduce grazing reflectance to 0%.
    fp_type reflectance90 =
        clamp(reflectance * fp_type(25.0), fp_type(0.0), fp_type(1.0));
    t_vec3 specularEnvironmentR0 = specularColor;
    t_vec3 specularEnvironmentR90 =
        t_vec3(fp_type(1), fp_type(1), fp_type(1)) * reflectance90;

    t_vec3 n = getNormal();  // normal at surface point
    t_vec3 v = normalize(u_Camera -
                         v_Position);  // Vector from surface point to camera
    t_vec3 l =
        normalize(u_LightDirection);  // Vector from surface point to light
    t_vec3 h = normalize(l + v);      // Half vector between both l and v
    t_vec3 reflection = -normalize(reflect(v, n));

    fp_type NdotL = clamp(dot(n, l), 0.001f, 1.0f);
    fp_type NdotV = clamp(std::abs(dot(n, v)), 0.001f, 1.0f);
    fp_type NdotH = clamp(dot(n, h), 0.0f, 1.0f);
    fp_type LdotH = clamp(dot(l, h), 0.0f, 1.0f);
    fp_type VdotH = clamp(dot(v, h), 0.0f, 1.0f);

    // Hey, modern C++ uniform initialization syntax just works!
    PBRInfo<fp_type> pbrInputs = PBRInfo<fp_type>{NdotL,
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
    t_vec3 F = specularReflection(pbrInputs);
    fp_type G = geometricOcclusion(pbrInputs);
    fp_type D = microfacetDistribution(pbrInputs);

    // Calculation of analytical lighting contribution
    t_vec3 diffuseContrib = (fp_type(1.0) - F) * diffuse(pbrInputs);
    t_vec3 specContrib = F * G * D / (fp_type(4.0) * NdotL * NdotV);
    // Obtain final intensity as reflectance (BRDF) scaled by the energy of the
    // light (cosine law)
    t_vec3 color = NdotL * u_LightColor * (diffuseContrib + specContrib);

    // Calculate lighting contribution from image based lighting source (IBL)

    if (useILB) {
      const auto contrib = getIBLContribution(pbrInputs, n, reflection);
      color += contrib;
    }
    // Apply optional PBR terms for additional (optional) shading

    if (useOcclusionMap) {
      fp_type ao = texture2D(u_OcclusionSampler, v_UV).r;
      color = mix(color, color * ao, u_OcclusionStrength);
    }

    if (useEmissiveMap) {
      t_vec3 emissive = SRGBtoLINEAR(t_vec4(
          t_vec3(texture2D(u_EmissiveSampler, v_UV)) * u_EmissiveFactor, 1));
      color += emissive;
    }

    // This section uses mix to override final color for reference app
    // visualization of various parameters in the lighting equation.
    color = mix(color, F, u_ScaleFGDSpec.x);
    color = mix(color, t_vec3(G), u_ScaleFGDSpec.y);
    color = mix(color, t_vec3(D), u_ScaleFGDSpec.z);
    color = mix(color, specContrib, u_ScaleFGDSpec.w);

    color = mix(color, diffuseContrib, u_ScaleDiffBaseMR.x);
    color = mix(color, t_vec3(baseColor), u_ScaleDiffBaseMR.y);
    color = mix(color, t_vec3(metallic), u_ScaleDiffBaseMR.z);
    color = mix(color, t_vec3(perceptualRoughness), u_ScaleDiffBaseMR.w);

    gl_FragColor = t_vec4(pow(color, t_vec3(1.0f / 2.2f)), baseColor.a);
  }

  // Find the normal for this fragment, pulling either from a predefined normal
  // map
  // or from the interpolated mesh normal and tangent attributes.
  t_vec3 getNormal() {
    // Retrieve the tangent space matrix
#ifndef HAS_TANGENTS
    /*    t_vec3 pos_dx = dFdx(v_Position);
    t_vec3 pos_dy = dFdy(v_Position);
    t_vec3 tex_dx = dFdx(t_vec3(v_UV, 0.0));
    t_vec3 tex_dy = dFdy(t_vec3(v_UV, 0.0));
    t_vec3 t = (tex_dy.t * pos_dx - tex_dx.t * pos_dy) /
             (tex_dx.s * tex_dy.t - tex_dy.s * tex_dx.t)*/
    ;
#ifdef HAS_NORMALS
    t_vec3 ng = normalize(v_Normal);
#else
    t_vec3 ng = cross(pos_dx, pos_dy);
#endif

    // This is some random hack to calculate "a" tangent vector
    t_vec3 t;

    t_vec3 c1 = cross(ng, t_vec3(0.0f, 0.0f, 1.0f));
    t_vec3 c2 = cross(ng, t_vec3(0.0f, 1.0f, 0.0f));

    if (length(c1) > length(c2)) {
      t = c1;
    } else {
      t = c2;
    }

    t = normalize(t - ng * dot(ng, t));
    t_vec3 b = normalize(cross(ng, t));
    mat3 tbn = mat3(t, b, ng);
#else  // HAS_TANGENTS
    mat3 tbn = v_TBN;
#endif
    t_vec3 n;
    if (useNormalMap) {
      n = t_vec3(texture2D(u_NormalSampler, v_UV));
      n = normalize(tbn * ((2.0f * n - 1.0f) *
                           t_vec3(u_NormalScale, u_NormalScale, 1.0)));
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
  // t_vec3 diffuse(PBRInfo pbrInputs) {
  //  return {pbrInputs.diffuseColor / float(M_PI)};
  //}

  // use Disney's equation for diffuse
  // https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
  // See section 5.3 (correct for too dark diffuse on the edges in comparison
  // with the above, commented-out function)
  t_vec3 diffuse(PBRInfo<fp_type> pbrInputs) {
    fp_type f90 =
        2.0f * pbrInputs.LdotH * pbrInputs.LdotH * pbrInputs.alphaRoughness -
        0.5f;

    return (pbrInputs.diffuseColor / fp_type(M_PI)) *
           (1.0f + f90 * fp_type(pow((1.0f - pbrInputs.NdotL), 5.0f))) *
           (1.0f + f90 * fp_type(pow((1.0f - pbrInputs.NdotV), 5.0f)));
  }

  // The following equation models the Fresnel reflectance term of the spec
  // equation (aka F()) Implementation of fresnel from [4], Equation 15
  t_vec3 specularReflection(PBRInfo<fp_type> pbrInputs) {
    return pbrInputs.reflectance0 +
           (pbrInputs.reflectance90 - pbrInputs.reflectance0) *
               std::pow(clamp(1.0f - pbrInputs.VdotH, 0.0f, 1.0f), 5.0f);
  }

  // This calculates the specular geometric attenuation (aka G()),
  // where rougher material will reflect less light back to the viewer.
  // This implementation is based on [1] Equation 4, and we adopt their
  // modifications to alphaRoughness as input as originally proposed in [2].
  fp_type geometricOcclusion(PBRInfo<fp_type> pbrInputs) {
    fp_type NdotL = pbrInputs.NdotL;
    fp_type NdotV = pbrInputs.NdotV;
    fp_type r = pbrInputs.alphaRoughness;

    fp_type attenuationL =
        2.0f * NdotL / (NdotL + sqrt(r * r + (1.0f - r * r) * (NdotL * NdotL)));
    fp_type attenuationV =
        2.0f * NdotV / (NdotV + sqrt(r * r + (1.0f - r * r) * (NdotV * NdotV)));
    return attenuationL * attenuationV;
  }

  // The following equation(s) model the distribution of microfacet normals
  // across the area being drawn (aka D()) Implementation from "Average
  // Irregularity Representation of a Roughened Surface for Ray Reflection" by
  // T. S. Trowbridge, and K. P. Reitz Follows the distribution function
  // recommended in the SIGGRAPH 2013 course notes from EPIC Games [1],
  // Equation 3.
  fp_type microfacetDistribution(PBRInfo<fp_type> pbrInputs) {
    fp_type roughnessSq = pbrInputs.alphaRoughness * pbrInputs.alphaRoughness;
    fp_type f =
        (pbrInputs.NdotH * roughnessSq - pbrInputs.NdotH) * pbrInputs.NdotH +
        1.0f;
    return roughnessSq / (M_PI * f * f);
  }

  // Global state of the original shader enclosed into member variables
  t_vec3 u_LightDirection;
  t_vec3 u_LightColor;

  bool useILB = false;
  bool use_ILB_BRDF_LUT = false;
  bool forceBRDFCompute = true;
  int brdfResolution = 256;
  samplerCube<fp_type> u_DiffuseEnvSampler;
  samplerCube<fp_type> u_SpecularEnvSampler;
  sampler2D<fp_type> u_brdfLUT;

  bool useBaseColorMap = false;
  sampler2D<fp_type> u_BaseColorSampler;

  bool useNormalMap = false;
  sampler2D<fp_type> u_NormalSampler;
  float u_NormalScale = 1;

  bool useEmissiveMap = false;
  sampler2D<fp_type> u_EmissiveSampler;
  t_vec3 u_EmissiveFactor;

  bool useMetalRoughMap = false;
  sampler2D<fp_type> u_MetallicRoughnessSampler;

  bool useOcclusionMap = false;
  sampler2D<fp_type> u_OcclusionSampler;
  float u_OcclusionStrength = 1;

  t_vec2 u_MetallicRoughnessValues = {1, 1};
  t_vec4 u_BaseColorFactor;

  t_vec3 u_Camera;

  // debugging flags used for shader output of intermediate PBR variables
  t_vec4 u_ScaleDiffBaseMR{0};
  t_vec4 u_ScaleFGDSpec{0};
  t_vec4 u_ScaleIBLAmbient{0};

  t_vec3 v_Position;

  t_vec2 v_UV;

#ifdef HAS_NORMALS
#ifdef HAS_TANGENTS
  mat3 v_TBN;
#else
  t_vec3 v_Normal;
#endif
#endif
};

}  // namespace pbr_maths

#undef ADD_TEMPLATED_TYPES
