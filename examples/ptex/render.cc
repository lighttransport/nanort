/*
The MIT License (MIT)

Copyright (c) 2015 - 2019 Light Transport Entertainment, Inc.

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

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "trackball.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Ptex
#include "PtexReader.h"

#ifdef WIN32
#undef min
#undef max
#endif

namespace example {

// From ptex source code. -----------------------------------------------

/*
PTEX SOFTWARE
Copyright 2014 Disney Enterprises, Inc.  All rights reserved

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.

  * The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
    Studios" or the names of its contributors may NOT be used to
    endorse or promote products derived from this software without
    specific prior written permission from Walt Disney Pictures.

Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
*/

void DumpFaceInfo(const Ptex::FaceInfo& f) {
  Ptex::Res res = f.res;
  std::cout << "  res: " << int(res.ulog2) << ' ' << int(res.vlog2) << " ("
            << res.u() << " x " << res.v() << ")"
            << "  adjface: " << f.adjfaces[0] << ' ' << f.adjfaces[1] << ' '
            << f.adjfaces[2] << ' ' << f.adjfaces[3]
            << "  adjedge: " << f.adjedge(0) << ' ' << f.adjedge(1) << ' '
            << f.adjedge(2) << ' ' << f.adjedge(3) << "  flags:";
  // output flag names
  if (f.flags == 0)
    std::cout << " (none)";
  else {
    if (f.isSubface()) std::cout << " subface";
    if (f.isConstant()) std::cout << " constant";
    if (f.isNeighborhoodConstant()) std::cout << " nbconstant";
    if (f.hasEdits()) std::cout << " hasedits";
  }
  std::cout << std::endl;
}

void DumpTiling(PtexFaceData* dh) {
  std::cout << "  tiling: ";
  if (dh->isTiled()) {
    Ptex::Res res = dh->tileRes();
    std::cout << "ntiles = " << dh->res().ntiles(res)
              << ", res = " << int(res.ulog2) << ' ' << int(res.vlog2) << " ("
              << res.u() << " x " << res.v() << ")\n";
  } else if (dh->isConstant()) {
    std::cout << "  (constant)" << std::endl;
  } else {
    std::cout << "  (untiled)" << std::endl;
  }
}

void DumpData(PtexTexture* r, int faceid, bool dumpall) {
  int levels = 1;
  if (dumpall) {
    Ptex::PtexReader* R = static_cast<Ptex::PtexReader*>(r);
    if (R) levels = R->header().nlevels;
  }

  const Ptex::FaceInfo& f = r->getFaceInfo(faceid);
  int nchan = r->numChannels();
  float* pixel = (float*)malloc(sizeof(float) * nchan);
  Ptex::Res res = f.res;
  while (levels && res.ulog2 >= 1 && res.vlog2 >= 1) {
    int ures = res.u(), vres = res.v();
    std::cout << "  data (" << ures << " x " << vres << ")";
    if (f.isConstant()) {
      ures = vres = 1;
    }
    bool isconst = (ures == 1 && vres == 1);
    if (isconst)
      std::cout << ", const: ";
    else
      std::cout << ":";
    for (int vi = 0; vi < vres; vi++) {
      for (int ui = 0; ui < ures; ui++) {
        if (!isconst) std::cout << "\n    (" << ui << ", " << vi << "): ";
        r->getPixel(faceid, ui, vi, pixel, 0, nchan, res);
        for (int c = 0; c < nchan; c++) {
          printf(" %.3f", pixel[c]);
        }
      }
    }
    std::cout << std::endl;
    res.ulog2--;
    res.vlog2--;
    levels--;
  }
  free(pixel);
}

template <typename T>
class DumpMetaArrayVal {
 public:
  void operator()(Ptex::PtexMetaData* meta, const char* key) {
    const T* val = 0;
    int count = 0;
    meta->getValue(key, val, count);
    for (int i = 0; i < count; i++) {
      if (i % 10 == 0 && (i || count > 10)) std::cout << "\n  ";
      std::cout << "  " << val[i];
    }
  }
};

void DumpMetaData(Ptex::PtexMetaData* meta) {
  std::cout << "meta:" << std::endl;
  for (int i = 0; i < meta->numKeys(); i++) {
    const char* key;
    Ptex::MetaDataType type;
    meta->getKey(i, key, type);
    std::cout << "  " << key << " type=" << Ptex::MetaDataTypeName(type);
    switch (type) {
      case Ptex::mdt_string: {
        const char* val = 0;
        meta->getValue(key, val);
        std::cout << "  \"" << val << "\"";
      } break;
      case Ptex::mdt_int8:
        DumpMetaArrayVal<int8_t>()(meta, key);
        break;
      case Ptex::mdt_int16:
        DumpMetaArrayVal<int16_t>()(meta, key);
        break;
      case Ptex::mdt_int32:
        DumpMetaArrayVal<int32_t>()(meta, key);
        break;
      case Ptex::mdt_float:
        DumpMetaArrayVal<float>()(meta, key);
        break;
      case Ptex::mdt_double:
        DumpMetaArrayVal<double>()(meta, key);
        break;
    }
    std::cout << std::endl;
  }
}

void DumpInternal(Ptex::PtexTexture* tx) {
  Ptex::PtexReader* r = static_cast<Ptex::PtexReader*>(tx);

  const Ptex::Header& h = r->header();
  const Ptex::ExtHeader& eh = r->extheader();
  std::cout << "Header:\n"
            << "  magic: ";

  if (h.magic == Ptex::Magic)
    std::cout << "'Ptex'" << std::endl;
  else
    std::cout << h.magic << std::endl;

  std::cout << "  version: " << h.version << '.' << h.minorversion << std::endl
            << "  meshtype: " << h.meshtype << std::endl
            << "  datatype: " << h.datatype << std::endl
            << "  alphachan: " << int(h.alphachan) << std::endl
            << "  nchannels: " << h.nchannels << std::endl
            << "  nlevels: " << h.nlevels << std::endl
            << "  nfaces: " << h.nfaces << std::endl
            << "  extheadersize: " << h.extheadersize << std::endl
            << "  faceinfosize: " << h.faceinfosize << std::endl
            << "  constdatasize: " << h.constdatasize << std::endl
            << "  levelinfosize: " << h.levelinfosize << std::endl
            << "  leveldatasize: " << h.leveldatasize << std::endl
            << "  metadatazipsize: " << h.metadatazipsize << std::endl
            << "  metadatamemsize: " << h.metadatamemsize << std::endl
            << "  ubordermode: " << eh.ubordermode << std::endl
            << "  vbordermode: " << eh.vbordermode << std::endl
            << "  lmdheaderzipsize: " << eh.lmdheaderzipsize << std::endl
            << "  lmdheadermemsize: " << eh.lmdheadermemsize << std::endl
            << "  lmddatasize: " << eh.lmddatasize << std::endl
            << "  editdatasize: " << eh.editdatasize << std::endl
            << "  editdatapos: " << eh.editdatapos << std::endl;

  std::cout << "Level info:\n";
  for (int i = 0; i < h.nlevels; i++) {
    const Ptex::LevelInfo& l = r->levelinfo(i);
    std::cout << "  Level " << i << std::endl
              << "    leveldatasize: " << l.leveldatasize << std::endl
              << "    levelheadersize: " << l.levelheadersize << std::endl
              << "    nfaces: " << l.nfaces << std::endl;
  }
}

int CheckAdjacency(PtexTexture* tx) {
  int result = 0;
  bool noinfo = true;

  for (int fid = 0; fid < tx->numFaces(); fid++) {
    const Ptex::FaceInfo& finfo = tx->getFaceInfo(fid);

    for (int e = 0; e < 4; ++e) {
      if (finfo.adjface(e) >= 0) {
        noinfo = false;

        const Ptex::FaceInfo& adjf = tx->getFaceInfo(finfo.adjface(e));

        int oppfid = adjf.adjface(finfo.adjedge(e));

        // trivial match
        if (oppfid == fid) continue;

        // subface case
        if (finfo.isSubface() && !adjf.isSubface()) {
          // neighbor face might be pointing to "other" subface
          if (oppfid == finfo.adjface((e + 1) % 4)) continue;
        }

        std::cerr << "face " << fid << " edge " << e
                  << " has incorrect adjacency\n";
        ++result;
      }
    }
  }

  if (noinfo) {
    std::cerr << "\"" << tx->path()
              << "\" does not appear to have"
                 "any adjacency information.\n";
    ++result;
  }

  if (result == 0) {
    std::cout << "Adjacency information appears consistent.\n";
  }

  return result;
}

// -----------------------------------------------------------------------

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
  unsigned int ret =
      (xorshifted >> rot) | (xorshifted << ((-static_cast<int>(rot)) & 31));

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
  // num_triangle_faces = indices.size() / 3
  std::vector<float> vertices;       /// [xyz] * num_vertices
  std::vector<float> vertex_colors;  /// [rgb] * num_vertices

  std::vector<float>
      facevarying_normals;  /// [xyz] * 3(triangle) * num_triangle_faces
  std::vector<float>
      facevarying_tangents;  /// [xyz] * 3(triangle) * num_triangle_faces
  std::vector<float>
      facevarying_binormals;  /// [xyz] * 3(triangle) * num_triangle_faces
  std::vector<float>
      facevarying_uvs;  /// [xy]  * 3(triangle) * num_triangle_faces

  std::vector<unsigned int> material_ids;  /// index x num_triangle_faces

  // List of triangle vertex indices. For NanoRT BVH
  std::vector<unsigned int>
      triangulated_indices;  /// 3(triangle) x num_triangle_faces

  // List of original vertex indices. For UV interpolation
  std::vector<unsigned int>
      face_indices;  /// length = sum(for each face_num_verts[i])

  // Offset to `face_indices` for a given face_id.
  std::vector<unsigned int>
      face_index_offsets;  /// length = face_num_verts.size()

  std::vector<unsigned char> face_num_verts;  /// # of vertices per face

  // face ID for each triangle. For ptex textureing.
  std::vector<unsigned int> face_ids;  /// index x num_triangle_faces

  // Triangule ID of a face(e.g. 0 for triangle primitive. 0 or 1 for quad
  // primitive(tessellated into two-triangles)
  std::vector<uint8_t> face_triangle_ids;  /// index x num_triangle_faces

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

// Find intersection and compute varycentric coordinates of ray-quad.
//
// Assume plane is the convex planar.
//
// http://graphics.cs.kuleuven.be/publications/LD05ERQIT/LD05ERQIT_paper.pdf
//
// v01-----v11
//  |\      |
//  | \     |
//  |  \    |
//  |   \   |
//  |    \  |
//  |     \ |
// v00 ----v10
//
bool RayQuadIntersection(const float3& rayorg, const float3& raydir,
                         const float3& v00, const float3& v10,
                         const float3& v11, const float3& v01, float tuv[3]) {
  // Reject rays using the barycentric coordinates of the intersection point
  // with respect to `t`.
  const float3 e01 = v10 - v00;
  const float3 e03 = v01 - v00;
  const float3 p = vcross(raydir, e03);
  const float det = vdot(e01, p);

  if (std::fabs(det) < std::numeric_limits<float>::epsilon()) {
    return false;
  }

  const float3 T = rayorg - v00;
  const float a = vdot(T, p) / det;
  if (a < 0.0f) {
    return false;
  }
  // uncomment if you can reorder vertices(see the paper for details.
  // if (a > 1.0f) {
  //  return false;
  //}

  const float3 Q = vcross(T, e01);
  const float b = vdot(raydir, Q) / det;
  if (b < 0.0f) {
    return false;
  }
  // uncomment if you can reorder vertices(see the paper for details.
  // if (b > 1.0f) {
  //  return false;
  //}

  // reject rays using the barycentric coordinates of
  // the intersection point with respect to `tt`
  if ((a + b) > 1.0f) {
    const float3 e23 = v01 - v11;
    const float3 e21 = v10 - v11;
    const float3 pp = vcross(raydir, e21);
    const float dett = vdot(e23, pp);

    if (std::fabs(dett) < std::numeric_limits<float>::epsilon()) {
      return false;
    }

    const float3 TT = rayorg - v11;
    const float aa = vdot(TT, pp) / dett;
    if (aa < 0.0f) {
      std::cerr << "aa = " << std::to_string(aa) << "\n";
      return false;
    }

    const float3 QQ = vcross(TT, e23);
    const float bb = vdot(raydir, QQ) / dett;
    if (bb < 0.0f) {
      std::cerr << "bb = " << std::to_string(bb) << "\n";
      return false;
    }
  }

  // Compute the ray parameter of the intersection point
  const float t = vdot(e03, Q) / det;
  if (t < 0.0f) {
    std::cerr << "t = " << std::to_string(t) << "\n";
    return false;
  }

  // Compute the barycentric coordinate of v11
  const float3 e02 = v11 - v00;
  const float3 n = vcross(e01, e03);

  const float abs_nx = std::fabs(n[0]);
  const float abs_ny = std::fabs(n[1]);
  const float abs_nz = std::fabs(n[2]);

  float a11;
  float b11;

  if ((abs_nx >= abs_ny) && (abs_nx >= abs_nz)) {
    a11 = (e02[1] * e03[2] - e02[2] * e03[1]) / n[0];
    b11 = (e01[1] * e02[2] - e01[2] * e02[1]) / n[0];
  } else if ((abs_ny >= abs_nx) && (abs_ny >= abs_nz)) {
    a11 = (e02[1] * e03[0] - e02[0] * e03[2]) / n[1];
    b11 = (e01[1] * e02[0] - e01[0] * e02[2]) / n[1];
  } else {
    a11 = (e02[0] * e03[1] - e02[1] * e03[0]) / n[2];
    b11 = (e01[0] * e02[1] - e01[1] * e02[0]) / n[2];
  }

  // Compute the barycentric coordinate of the intersection point.
  float u = 0.0f;
  float v = 0.0f;
  if (std::fabs(a11 - 1.0f) < std::numeric_limits<float>::epsilon()) {
    u = a;
    if (std::fabs(b11 - 1.0f) < std::numeric_limits<float>::epsilon()) {
      v = b;
    } else {
      v = b / (u * (b11 - 1.0f) + 1.0f);
    }
  } else if (std::fabs(b11 - 1.0f) < std::numeric_limits<float>::epsilon()) {
    v = b;
    u = a / (v * (a11 - 1.0f) + 1.0f);
  } else {
    const float A = -(b11 - 1.0f);
    const float B = a * (b11 - 1.0f) - b * (a11 - 1.0f) - 1.0f;
    const float C = a;
    const float Delta = B * B - 4.0f * A * C;
    const float Q =
        -0.5f *
        (B + (std::signbit(B) ? -1.0f : 1.0f) *
                 std::sqrt(std::max(std::numeric_limits<float>::min(), Delta)));

    u = Q / A;
    if ((u < 0.0f) || (u > 1.0f)) {
      u = C / Q;
    }

    v = b / (u * (b11 - 1.0f) + 1.0f);
  }

  tuv[0] = t;
  tuv[1] = u;
  tuv[2] = v;
  return true;
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

static std::string GetBaseDir(const std::string& filepath) {
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

bool LoadObj(Mesh& mesh, const char* filename, float scale) {
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

    num_faces += shapes[i].mesh.num_face_vertices.size();

    // Check if a face is triangle or quad.
    for (size_t k = 0; k < shapes[i].mesh.num_face_vertices.size(); k++) {
      if ((shapes[i].mesh.num_face_vertices[k] == 3) ||
          (shapes[i].mesh.num_face_vertices[k] == 4)) {
        // ok
      } else {
        std::cerr << "face contains invalid polygons."
                  << std::to_string(shapes[i].mesh.num_face_vertices[k])
                  << std::endl;
      }
    }
  }

  std::cout << "[LoadOBJ] # of faces: " << num_faces << std::endl;
  std::cout << "[LoadOBJ] # of vertices: " << num_vertices << std::endl;

  // Shape -> Mesh
  mesh.vertices.resize(num_vertices * 3, 0.0f);
  mesh.vertex_colors.resize(num_vertices * 3, 1.0f);

  size_t faceIdxOffset = 0;

  mesh.vertices.clear();
  for (size_t i = 0; i < attrib.vertices.size(); i++) {
    mesh.vertices.push_back(scale * attrib.vertices[i]);
  }

  mesh.vertex_colors.clear();
  for (size_t i = 0; i < attrib.colors.size(); i++) {
    mesh.vertex_colors.push_back(attrib.colors[i]);
  }

  mesh.triangulated_indices.clear();
  mesh.face_indices.clear();
  mesh.face_index_offsets.clear();
  mesh.face_num_verts.clear();
  mesh.face_ids.clear();
  mesh.face_triangle_ids.clear();
  mesh.material_ids.clear();
  mesh.facevarying_normals.clear();
  mesh.facevarying_uvs.clear();

  // Flattened indices for easy facevarying normal/uv setup
  std::vector<tinyobj::index_t> triangulated_indices;

  size_t face_id_offset = 0;
  for (size_t i = 0; i < shapes.size(); i++) {
    size_t offset = 0;
    for (size_t f = 0; f < shapes[i].mesh.num_face_vertices.size(); f++) {
      int npoly = shapes[i].mesh.num_face_vertices[f];

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
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 0].vertex_index);
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 1].vertex_index);
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 2].vertex_index);

        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 2].vertex_index);
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 3].vertex_index);
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 0].vertex_index);

        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 0].vertex_index);
        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 1].vertex_index);
        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 2].vertex_index);
        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 3].vertex_index);

        mesh.face_ids.push_back(face_id_offset + f);
        mesh.face_ids.push_back(face_id_offset + f);

        mesh.face_triangle_ids.push_back(0);
        mesh.face_triangle_ids.push_back(1);

        mesh.material_ids.push_back(shapes[i].mesh.material_ids[offset]);
        mesh.material_ids.push_back(shapes[i].mesh.material_ids[offset]);

        // for computing normal/uv in the later stage
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 0]);
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 1]);
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 2]);

        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 2]);
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 3]);
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 0]);

      } else {
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 0].vertex_index);
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 1].vertex_index);
        mesh.triangulated_indices.push_back(
            shapes[i].mesh.indices[offset + 2].vertex_index);

        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 0].vertex_index);
        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 1].vertex_index);
        mesh.face_indices.push_back(
            shapes[i].mesh.indices[offset + 2].vertex_index);

        mesh.face_ids.push_back(face_id_offset + f);
        mesh.face_triangle_ids.push_back(0);
        mesh.material_ids.push_back(shapes[i].mesh.material_ids[f]);

        // for computing normal/uv in the later stage
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 0]);
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 1]);
        triangulated_indices.push_back(shapes[i].mesh.indices[offset + 2]);
      }

      offset += npoly;
    }

    face_id_offset += shapes[i].mesh.num_face_vertices.size();
  }

  // Setup normal/uv
  if (attrib.normals.size() > 0) {
    for (size_t f = 0; f < triangulated_indices.size() / 3; f++) {
      int f0, f1, f2;

      f0 = triangulated_indices[3 * f + 0].normal_index;
      f1 = triangulated_indices[3 * f + 1].normal_index;
      f2 = triangulated_indices[3 * f + 2].normal_index;

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

      if (f0 > 0 && f1 > 0 && f2 > 0) {
        float3 n0, n1, n2;

        n0[0] = attrib.texcoords[2 * f0 + 0];
        n0[1] = attrib.texcoords[2 * f0 + 1];

        n1[0] = attrib.texcoords[2 * f1 + 0];
        n1[1] = attrib.texcoords[2 * f1 + 1];

        n2[0] = attrib.texcoords[2 * f2 + 0];
        n2[1] = attrib.texcoords[2 * f2 + 1];

        mesh.facevarying_uvs.push_back(n0[0]);
        mesh.facevarying_uvs.push_back(n0[1]);

        mesh.facevarying_uvs.push_back(n1[0]);
        mesh.facevarying_uvs.push_back(n1[1]);

        mesh.facevarying_uvs.push_back(n2[0]);
        mesh.facevarying_uvs.push_back(n2[1]);
      }
    }
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

bool Renderer::LoadPtex(const std::string& ptex_filename, const bool dump) {
  Ptex::String error;
  _ptex.reset(Ptex::PtexTexture::open(ptex_filename.c_str(), error));
  // Ptex::PtexPtr<Ptex::PtexTexture> r(
  //    Ptex::PtexTexture::open(ptex_filename.c_str(), error));

  if (!_ptex) {
    std::cerr << error.c_str() << std::endl;
    return false;
  }

  bool checkadjacency = true;
  if (checkadjacency) {
    int retcode = CheckAdjacency(_ptex);
    if (retcode != 0) {
      return false;
    }
  }

  std::cout << "meshType: " << Ptex::MeshTypeName(_ptex->meshType())
            << std::endl;
  std::cout << "dataType: " << Ptex::DataTypeName(_ptex->dataType())
            << std::endl;
  std::cout << "numChannels: " << _ptex->numChannels() << std::endl;
  std::cout << "alphaChannel: ";
  if (_ptex->alphaChannel() == -1)
    std::cout << "(none)" << std::endl;
  else
    std::cout << _ptex->alphaChannel() << std::endl;
  std::cout << "uBorderMode: " << Ptex::BorderModeName(_ptex->uBorderMode())
            << std::endl;
  std::cout << "vBorderMode: " << Ptex::BorderModeName(_ptex->vBorderMode())
            << std::endl;
  std::cout << "edgeFilterMode: "
            << Ptex::EdgeFilterModeName(_ptex->edgeFilterMode()) << std::endl;
  std::cout << "numFaces: " << _ptex->numFaces() << std::endl;
  std::cout << "hasEdits: " << (_ptex->hasEdits() ? "yes" : "no") << std::endl;
  std::cout << "hasMipMaps: " << (_ptex->hasMipMaps() ? "yes" : "no")
            << std::endl;

  bool dumpfaceinfo = true;
  bool dumpdata = true;
  bool dumptiling = true;
  bool dumpinternal = true;
  bool dumpmeta = true;
  bool dumpalldata = true;

  PtexPtr<PtexMetaData> meta(_ptex->getMetaData());
  if (meta) {
    std::cout << "numMetaKeys: " << meta->numKeys() << std::endl;
    if (dumpmeta && meta->numKeys()) DumpMetaData(meta);
  }

  if (dump) {
    if (dumpfaceinfo || dumpdata || dumptiling) {
      uint64_t texels = 0;
      for (int i = 0; i < _ptex->numFaces(); i++) {
        std::cout << "face " << i << ":";
        const Ptex::FaceInfo& f = _ptex->getFaceInfo(i);
        DumpFaceInfo(f);
        texels += f.res.size();

        if (dumptiling) {
          PtexPtr<PtexFaceData> dh(_ptex->getData(i, f.res));
          DumpTiling(dh);
        }
        if (dumpdata) DumpData(_ptex, i, dumpalldata);
      }
      std::cout << "texels: " << texels << std::endl;
    }

    if (dumpinternal) DumpInternal(_ptex);
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
                                            gMesh.triangulated_indices.data(),
                                            sizeof(float) * 3);
  nanort::TriangleSAHPred<float> triangle_pred(
      gMesh.vertices.data(), gMesh.triangulated_indices.data(),
      sizeof(float) * 3);

  printf("num_triangles = %lu\n", gMesh.triangulated_indices.size() / 3);

  bool ret = gAccel.Build(gMesh.triangulated_indices.size() / 3, triangle_mesh,
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

void Renderer::ShadePtex(int filter, bool lerp, float sharpness,
                         bool noedgeblend, int start_channel, int channels,
                         int face_id, float u, float v, float uw1, float vw1,
                         float uw2, float vw2, float width, float blur,
                         float rgba[4]) {
  if (!_ptex) {
    rgba[0] = 0.5f;
    rgba[1] = 0.0f;
    rgba[2] = 0.0f;
    rgba[3] = 1.0f;

    return;
  }

  if (face_id < 0) {
    rgba[0] = 0.0f;
    rgba[1] = 0.0f;
    rgba[2] = 0.0f;
    rgba[3] = 1.0f;
    return;
  }

  channels = std::max(0, channels);
  start_channel = std::max(0, start_channel);

  int max_channels = _ptex->numChannels();
  if (start_channel >= max_channels) {
    start_channel = max_channels;
  }

  if ((start_channel + channels) >= max_channels) {
    channels = max_channels - start_channel;
  }

  Ptex::PtexFilter::FilterType ftype = Ptex::PtexFilter::f_bicubic;
  if (filter == 0) {
    ftype = Ptex::PtexFilter::f_point;
  } else if (filter == 1) {
    ftype = Ptex::PtexFilter::f_bilinear;
  } else if (filter == 2) {
    ftype = Ptex::PtexFilter::f_box;
  } else if (filter == 3) {
    ftype = Ptex::PtexFilter::f_gaussian;
  } else if (filter == 4) {
    ftype = Ptex::PtexFilter::f_bicubic;
  } else if (filter == 5) {
    ftype = Ptex::PtexFilter::f_bspline;
  } else if (filter == 6) {
    ftype = Ptex::PtexFilter::f_catmullrom;
  } else if (filter == 7) {
    ftype = Ptex::PtexFilter::f_mitchell;
  }

  // TODO(LTE): Do not creat filter every time `ShadePtex` was called.
  Ptex::PtexFilter::Options opts(ftype, lerp, sharpness, noedgeblend);
  Ptex::PtexPtr<PtexFilter> f(Ptex::PtexFilter::getFilter(_ptex, opts));

  f->eval(rgba, /* first channel*/ start_channel, /* nchannels */ channels,
          face_id, u, v, uw1, vw1, uw2, vw2, width, blur);

  if (_ptex->numChannels() == 1) {
    // grayscale to rgb
    rgba[1] = rgba[0];
    rgba[2] = rgba[0];
  }

  rgba[3] = 1.0f;
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
              gMesh.vertices.data(), gMesh.triangulated_indices.data(),
              sizeof(float) * 3);
          nanort::TriangleIntersection<float> isect;
          bool hit = gAccel.Traverse(ray, triangle_intersector, &isect);
          if (hit) {
            float3 p;
            p[0] = ray.org[0] + isect.t * ray.dir[0];
            p[1] = ray.org[1] + isect.t * ray.dir[1];
            p[2] = ray.org[2] + isect.t * ray.dir[2];

            config.positionImage[4 * (y * config.width + x) + 0] = p.x();
            config.positionImage[4 * (y * config.width + x) + 1] = p.y();
            config.positionImage[4 * (y * config.width + x) + 2] = p.z();
            config.positionImage[4 * (y * config.width + x) + 3] = 1.0f;

            unsigned int prim_id = isect.prim_id;

            int face_id = gMesh.face_ids[prim_id];
            config.faceIdImage[(y * config.width + x)] = face_id;

            // compute varicentric UV coord.
            float baryUV[2] = {0.0f, 0.0f};

            uint32_t npolys = gMesh.face_num_verts[face_id];
            if (npolys == 3) {
              std::cout << "triangle\n";
              baryUV[0] = isect.u;
              baryUV[1] = isect.v;
            } else if (npolys == 4) {
              size_t face_offset_idx = gMesh.face_index_offsets[face_id];

              size_t i00 = gMesh.face_indices[face_offset_idx + 0];
              size_t i10 = gMesh.face_indices[face_offset_idx + 1];
              size_t i11 = gMesh.face_indices[face_offset_idx + 2];
              size_t i01 = gMesh.face_indices[face_offset_idx + 3];

              float3 v00, v10, v11, v01;
              v00[0] = gMesh.vertices[3 * i00 + 0];
              v00[1] = gMesh.vertices[3 * i00 + 1];
              v00[2] = gMesh.vertices[3 * i00 + 2];

              v10[0] = gMesh.vertices[3 * i10 + 0];
              v10[1] = gMesh.vertices[3 * i10 + 1];
              v10[2] = gMesh.vertices[3 * i10 + 2];

              v11[0] = gMesh.vertices[3 * i11 + 0];
              v11[1] = gMesh.vertices[3 * i11 + 1];
              v11[2] = gMesh.vertices[3 * i11 + 2];

              v01[0] = gMesh.vertices[3 * i01 + 0];
              v01[1] = gMesh.vertices[3 * i01 + 1];
              v01[2] = gMesh.vertices[3 * i01 + 2];

              float3 rayorg, raydir;
              rayorg[0] = ray.org[0];
              rayorg[1] = ray.org[1];
              rayorg[2] = ray.org[2];

              raydir[0] = ray.dir[0];
              raydir[1] = ray.dir[1];
              raydir[2] = ray.dir[2];

              // raydir must be normalized
              raydir = vnormalize(raydir);

              // Compute ray intersection again is a bit redundant.
              float quad_tuv[3];
              if (!RayQuadIntersection(rayorg, raydir, v00, v10, v11, v01,
                                       quad_tuv)) {
                // Uusually this should not be happen.
                // std::cerr << "Warning. Ray-Quad intersection failed.\n";

                // Still continue to shading as if we hit a triangle.
                baryUV[0] = isect.u;
                baryUV[1] = isect.v;

              } else {
                baryUV[0] = quad_tuv[1];
                baryUV[1] = quad_tuv[2];
              }

#if 0  // to be removed.
              int triangle_id = gMesh.face_triangle_ids[prim_id];
              if (triangle_id == 0) {
                baryUV[0] = isect.u;
                baryUV[1] = isect.v;
              } else {  // assume 1.
                // baryUV[0] = 1.0f-isect.u;
                baryUV[0] = 0.0f;
                baryUV[1] = 1.0f - isect.v;
              }
#endif

            } else {
              // ???
              std::cerr << "???\n";
            }

            config.tri_varycoordImage[4 * (y * config.width + x) + 0] = isect.u;
            config.tri_varycoordImage[4 * (y * config.width + x) + 1] = isect.v;
            config.tri_varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
            config.tri_varycoordImage[4 * (y * config.width + x) + 3] = 1.0f;

            config.varycoordImage[4 * (y * config.width + x) + 0] = baryUV[0];
            config.varycoordImage[4 * (y * config.width + x) + 1] = baryUV[1];
            config.varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
            config.varycoordImage[4 * (y * config.width + x) + 3] = 1.0f;

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
              N = Lerp3(n0, n1, n2, isect.u, isect.v);
            } else {
              unsigned int f0, f1, f2;
              f0 = gMesh.triangulated_indices[3 * prim_id + 0];
              f1 = gMesh.triangulated_indices[3 * prim_id + 1];
              f2 = gMesh.triangulated_indices[3 * prim_id + 2];

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

            config.depthImage[4 * (y * config.width + x) + 0] = isect.t;
            config.depthImage[4 * (y * config.width + x) + 1] = isect.t;
            config.depthImage[4 * (y * config.width + x) + 2] = isect.t;
            config.depthImage[4 * (y * config.width + x) + 3] = 1.0f;

            float3 vcol(1.0f, 1.0f, 1.0f);
            if (gMesh.vertex_colors.size() > 0) {
              unsigned int f0, f1, f2;
              f0 = gMesh.triangulated_indices[3 * prim_id + 0];
              f1 = gMesh.triangulated_indices[3 * prim_id + 1];
              f2 = gMesh.triangulated_indices[3 * prim_id + 2];

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

              vcol = Lerp3(c0, c1, c2, isect.u, isect.v);

              config.vertexColorImage[4 * (y * config.width + x) + 0] = vcol[0];
              config.vertexColorImage[4 * (y * config.width + x) + 1] = vcol[1];
              config.vertexColorImage[4 * (y * config.width + x) + 2] = vcol[2];
            }

            float3 texUV;
            if (gMesh.facevarying_uvs.size() > 0) {
              float3 uv0, uv1, uv2;
              uv0[0] = gMesh.facevarying_uvs[6 * prim_id + 0];
              uv0[1] = gMesh.facevarying_uvs[6 * prim_id + 1];
              uv1[0] = gMesh.facevarying_uvs[6 * prim_id + 2];
              uv1[1] = gMesh.facevarying_uvs[6 * prim_id + 3];
              uv2[0] = gMesh.facevarying_uvs[6 * prim_id + 4];
              uv2[1] = gMesh.facevarying_uvs[6 * prim_id + 5];

              texUV = Lerp3(uv0, uv1, uv2, isect.u, isect.v);

              config.texcoordImage[4 * (y * config.width + x) + 0] = texUV[0];
              config.texcoordImage[4 * (y * config.width + x) + 1] = texUV[1];
            }

            float uu = baryUV[0];
            float vv = baryUV[1];

            float ptexcol[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            // TODO(LTE): Support per-object/per-texture filtering parameters.
            // TODO(LTE): Compute filtering parameters based on pixel
            // footprints(e.g. use ray differentials)
            ShadePtex(config.ptex_filter, config.ptex_lerp,
                      config.ptex_sharpness, config.ptex_noedgeblend,
                      config.ptex_start_channel, config.ptex_channels, face_id,
                      uu, vv, config.ptex_uw1, config.ptex_vw1, config.ptex_uw2,
                      config.ptex_vw2, config.ptex_width, config.ptex_blur,
                      ptexcol);

            if (config.pass == 0) {
              rgba[4 * (y * config.width + x) + 0] = ptexcol[0];
              rgba[4 * (y * config.width + x) + 1] = ptexcol[1];
              rgba[4 * (y * config.width + x) + 2] = ptexcol[2];
              rgba[4 * (y * config.width + x) + 3] = 1.0f;
              sample_counts[y * config.width + x] =
                  1;  // Set 1 for the first pass
            } else {  // additive.
              rgba[4 * (y * config.width + x) + 0] += ptexcol[0];
              rgba[4 * (y * config.width + x) + 1] += ptexcol[1];
              rgba[4 * (y * config.width + x) + 2] += ptexcol[2];
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
              config.tri_varycoordImage[4 * (y * config.width + x) + 0] = 0.0f;
              config.tri_varycoordImage[4 * (y * config.width + x) + 1] = 0.0f;
              config.tri_varycoordImage[4 * (y * config.width + x) + 2] = 0.0f;
              config.tri_varycoordImage[4 * (y * config.width + x) + 3] = 0.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 0] = 1.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 1] = 1.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 2] = 1.0f;
              config.vertexColorImage[4 * (y * config.width + x) + 3] = 1.0f;

              config.faceIdImage[(y * config.width + x)] = -1;
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
