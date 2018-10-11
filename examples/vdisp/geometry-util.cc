#include <cassert>

#include "common-util.h"
#include "geometry-util.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>

namespace example {

inline float lerp(float x, float y, float t) {
  return x + t * (y - x); }

inline float AngleBetween(const float3 &a, const float3 &b)
{
  const float a_len = vlength(a);
  const float b_len = vlength(b);
  if (a_len < 1.0e-6f) {
    return 0.0f;
  }
  if (b_len < 1.0e-6f) {
    return 0.0f;
  }

  const float mag = a_len * b_len;
  if (mag > 1.0e-6f) {
    const float cos_alpha = vdot(a, b) / mag;

    return std::acos(cos_alpha);
  }
  return 0.0f;
}


// Sample texel with bi-linear filtering.
static void FilterTexture(
  const std::vector<float> &image,
  const size_t width,
  const size_t height,
  const float u, const float v,
  float rgb[3])
{
  float uu, vv;

  // clamp
  uu = std::max(u, 0.0f);
  uu = std::min(uu, 1.0f);
  vv = std::max(v, 0.0f);
  vv = std::min(vv, 1.0f);

  float px = width * uu;
  float py = height * vv;

  int x0 = int(px);
  int y0 = int(py);

  if (x0 < 0) x0 = 0;
  if (y0 < 0) y0 = 0;

  if (x0 >= int(width)) x0 = int(width) - 1;
  if (y0 >= int(height)) y0 = int(height) - 1;

  int x1 = ((x0 + 1) >= int(width)) ? (int(width) - 1) : (x0 + 1);
  int y1 = ((y0 + 1) >= int(height)) ? (int(height) - 1) : (y0 + 1);

  float dx = px - float(x0);
  float dy = py - float(y0);

  float w[4];

  w[0] = (1.0f - dx) * (1.0f - dy);
  w[1] = (1.0f - dx) * (dy);
  w[2] = (dx) * (1.0f - dy);
  w[3] = (dx) * (dy);

  int stride = 3;

  int i00 = stride * (y0 * int(width) + x0);
  int i01 = stride * (y0 * int(width) + x1);
  int i10 = stride * (y1 * int(width) + x0);
  int i11 = stride * (y1 * int(width) + x1);

  float texel[4][4];

  for (int i = 0; i < stride; i++) {
    texel[0][i] = image[size_t(i00 + i)];
    texel[1][i] = image[size_t(i10 + i)];
    texel[2][i] = image[size_t(i01 + i)];
    texel[3][i] = image[size_t(i11 + i)];
  }

  for (int i = 0; i < stride; i++) {
    rgb[i] = lerp(lerp(texel[0][i], texel[1][i], dx),
                   lerp(texel[2][i], texel[3][i], dx), dy);
  }

  if (stride == 1) {
    // mono -> RGB
    rgb[1] = rgb[0];
    rgb[2] = rgb[0];
  }
}

#if 0
// Sample texel without filtering.
static void SampleTexture(
  const std::vector<float> &image,
  const size_t width,
  const size_t height,
  const float u, const float v,
  float rgb[3])
{
  float uu, vv;

  // clamp
  uu = std::max(u, 0.0f);
  uu = std::min(uu, 1.0f);
  vv = std::max(v, 0.0f);
  vv = std::min(vv, 1.0f);

  float px = width * uu;
  float py = height * vv;

  int x0 = std::max(0, std::min(int(width - 1), int(px)));
  int y0 = std::max(0, std::min(int(height - 1), int(py)));

  int stride = 3;

  int i00 = stride * (y0 * int(width) + x0);

  for (int i = 0; i < stride; i++) {
    rgb[i] = image[size_t(i00 + i)];
  }

  if (stride == 1) {
    // mono -> RGB
    rgb[1] = rgb[0];
    rgb[2] = rgb[0];
  }

}
#endif

void GeneratePlane(
  size_t u_div,
  size_t v_div,
  const float scale[3],
  std::vector<float> *vertices,
  std::vector<uint32_t> *faces,
  std::vector<float> *facevarying_normals,
  std::vector<float> *facevarying_uvs) {

  assert(u_div > 0);
  assert(v_div > 0);

  float u_step = 1.0f / float(u_div);
  float v_step = 1.0f / float(v_div);

  vertices->clear();
  faces->clear();
  facevarying_normals->clear();
  facevarying_uvs->clear();

  for (size_t z = 0; z <= v_div; z++) {
    float pz = 2.0f * z * v_step - 1.0f;
    for (size_t x = 0; x <= u_div; x++) {

      float px = 2.0f * x * u_step - 1.0f;

      vertices->push_back(px * scale[0]);
      vertices->push_back(0.0f);
      vertices->push_back(pz * scale[2]);
    }
  }

  // generate indices and facevarying attributes.
  //
  // 0       3
  // +-------+---....
  // |      /|
  // |     / |
  // |    /  |
  // |   /   |
  // |  /    |
  // | /     |
  // |/      |
  // +-------+---....
  // 1       2
  // |       |
  // .       .
  // .       .
  //

  size_t stride = u_div + 1;

  for (size_t z = 0; z < v_div; z++) {
    float tv0 = z * v_step;
    float tv1 = (z + 1) * v_step;
    for (size_t x = 0; x < u_div; x++) {
      float tu0 = x * v_step;
      float tu1 = (x + 1) * v_step;

      faces->push_back(uint32_t(z * stride + x));
      faces->push_back(uint32_t((z + 1) * stride+ x));
      faces->push_back(uint32_t(z * stride + (x + 1)));

      faces->push_back(uint32_t((z + 1) * stride + x));
      faces->push_back(uint32_t((z + 1) * stride + (x + 1)));
      faces->push_back(uint32_t(z * stride + (x + 1)));

      for (size_t k = 0; k < 6; k++) {
        facevarying_normals->push_back(0.0f);
        facevarying_normals->push_back(1.0f);
        facevarying_normals->push_back(0.0f);
      }

      // flip V
      facevarying_uvs->push_back(tu0);
      facevarying_uvs->push_back(1.0f - tv0);

      facevarying_uvs->push_back(tu0);
      facevarying_uvs->push_back(1.0f - tv1);

      facevarying_uvs->push_back(tu1);
      facevarying_uvs->push_back(1.0f - tv0);

      facevarying_uvs->push_back(tu0);
      facevarying_uvs->push_back(1.0f - tv1);

      facevarying_uvs->push_back(tu1);
      facevarying_uvs->push_back(1.0f - tv1);

      facevarying_uvs->push_back(tu1);
      facevarying_uvs->push_back(1.0f - tv0);

    }
  }
}

//
// Compute facevarying tangent and facevarying binormal.
//
// Reference:
// http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-13-normal-mapping
void ComputeTangentsAndBinormals(
    const std::vector<float> &vertices, const std::vector<uint32_t> &faces,
    const std::vector<float> &facevarying_texcoords,
    const std::vector<float> &facevarying_normals,
    std::vector<float> *facevarying_tangents,
    std::vector<float> *facevarying_binormals) {
  static_assert(sizeof(float3) == 12, "invalid size for float3");
  assert(facevarying_texcoords.size() > 0);
  assert(facevarying_normals.size() > 0);

  // Temp buffer
  std::vector<float3> tn(vertices.size() / 3);
  memset(&tn.at(0), 0, sizeof(float3) * tn.size());
  std::vector<float3> bn(vertices.size() / 3);
  memset(&bn.at(0), 0, sizeof(float3) * bn.size());

  // Assume all triangle face.
  for (size_t i = 0; i < faces.size() / 3; i++) {

    uint32_t vf0 = faces[3 * i + 0];
    uint32_t vf1 = faces[3 * i + 1];
    uint32_t vf2 = faces[3 * i + 2];

    float v1x = vertices[3 * vf0 + 0];
    float v1y = vertices[3 * vf0 + 1];
    float v1z = vertices[3 * vf0 + 2];

    float v2x = vertices[3 * vf1 + 0];
    float v2y = vertices[3 * vf1 + 1];
    float v2z = vertices[3 * vf1 + 2];

    float v3x = vertices[3 * vf2 + 0];
    float v3y = vertices[3 * vf2 + 1];
    float v3z = vertices[3 * vf2 + 2];

    float w1x = 0.0f;
    float w1y = 0.0f;
    float w2x = 0.0f;
    float w2y = 0.0f;
    float w3x = 0.0f;
    float w3y = 0.0f;

    {
      w1x = facevarying_texcoords[6 * i + 0];
      w1y = facevarying_texcoords[6 * i + 1];
      w2x = facevarying_texcoords[6 * i + 2];
      w2y = facevarying_texcoords[6 * i + 3];
      w3x = facevarying_texcoords[6 * i + 4];
      w3y = facevarying_texcoords[6 * i + 5];
    }

    float x1 = v2x - v1x;
    float x2 = v3x - v1x;
    float y1 = v2y - v1y;
    float y2 = v3y - v1y;
    float z1 = v2z - v1z;
    float z2 = v3z - v1z;

    float s1 = w2x - w1x;
    float s2 = w3x - w1x;
    float t1 = w2y - w1y;
    float t2 = w3y - w1y;

    float r = 1.0;

    if (fabs(double(s1 * t2 - s2 * t1)) > 1.0e-20) {
      r /= (s1 * t2 - s2 * t1);
    }

    float3 tdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r,
                (t2 * z1 - t1 * z2) * r);
    float3 bdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r,
                (s1 * z2 - s2 * z1) * r);

    assert(vf0 < tn.size());
    assert(vf1 < tn.size());
    assert(vf2 < tn.size());

    tn[vf0] += tdir;
    tn[vf1] += tdir;
    tn[vf2] += tdir;

    bn[vf0] += bdir;
    bn[vf1] += bdir;
    bn[vf2] += bdir;
  }

  // normalize * orthogonalize;
  facevarying_tangents->resize(facevarying_normals.size());
  facevarying_binormals->resize(facevarying_normals.size());

  size_t num_faces = faces.size() / 3;

  std::vector<std::thread> workers;

  uint32_t num_threads = std::max(1U, std::thread::hardware_concurrency());

  if (num_faces < num_threads) {
    num_threads = uint32_t(num_faces);
  }

  size_t ndiv = num_faces / num_threads;

  for (size_t t = 0; t < num_threads; t++) {
    workers.emplace_back(std::thread([&, t]() {
      size_t fs = t * ndiv;
      size_t fe = (t == (num_threads - 1)) ? num_faces : std::min((t + 1) * ndiv, num_faces);

      for (size_t i = fs; i < fe; i++) {
        uint32_t vf[3];

        vf[0] = faces[3 * i + 0];
        vf[1] = faces[3 * i + 1];
        vf[2] = faces[3 * i + 2];

        float3 n[3];

        // http://www.terathon.com/code/tangent.html
        assert((9 * i + 8) < facevarying_normals.size());

        {
          n[0][0] = facevarying_normals[9 * i + 0];
          n[0][1] = facevarying_normals[9 * i + 1];
          n[0][2] = facevarying_normals[9 * i + 2];

          n[1][0] = facevarying_normals[9 * i + 3];
          n[1][1] = facevarying_normals[9 * i + 4];
          n[1][2] = facevarying_normals[9 * i + 5];

          n[2][0] = facevarying_normals[9 * i + 6];
          n[2][1] = facevarying_normals[9 * i + 7];
          n[2][2] = facevarying_normals[9 * i + 8];
        }

        for (size_t k = 0; k < 3; k++) {
          float3 Tn = tn[vf[k]];
          float3 Bn = bn[vf[k]];

          if (vlength(Bn) > 0.0f) {
            Bn = vnormalize(Bn);
          }

          // Gram-Schmidt orthogonalize
          Tn = (Tn - n[k] * vdot(n[k], Tn));
          if (vlength(Tn) > 0.0f) {
            Tn = vnormalize(Tn);
          }

          // Calculate handedness
          if (vdot(vcross(n[k], Tn), Bn) < 0.0f) {
            Tn = Tn * -1.0f;
          }

          (*facevarying_tangents)[9 * i + 3 * k + 0] = Tn[0];
          (*facevarying_tangents)[9 * i + 3 * k + 1] = Tn[1];
          (*facevarying_tangents)[9 * i + 3 * k + 2] = Tn[2];

          (*facevarying_binormals)[9 * i + 3 * k + 0] = Bn[0];
          (*facevarying_binormals)[9 * i + 3 * k + 1] = Bn[1];
          (*facevarying_binormals)[9 * i + 3 * k + 2] = Bn[2];
        }
      }
    }));
  }

  for (auto &t : workers) {
    t.join();
  }
}

void ApplyVectorDispacement(const std::vector<float> &vertices,
                      const std::vector<uint32_t> &faces,
                      const std::vector<uint32_t> &material_ids,
                      const std::vector<float> &facevarying_texcoords,
                      const std::vector<float> &facevarying_normals,
                      const std::vector<float> &facevarying_tangents,
                      const std::vector<float> &facevarying_binormals,
                      const uint32_t vdisp_material_id,
                      const std::vector<float> &vdisp_image,  // rgb
                      const size_t vdisp_width, const size_t vdisp_height,
                      const int vdisp_space,
                      const float vdisp_scale,
                      std::vector<float> *displaced_vertices) {

  // not used at the moment.
  (void)facevarying_normals;
  (void)facevarying_tangents;
  (void)facevarying_binormals;

  if (vdisp_space == 0) {
    // world space

    size_t num_verts = vertices.size() / 3;
    size_t num_faces = faces.size() / 3;

    // records the number of references of a shared vertex.
    std::vector<int> counts(num_verts);
    memset(counts.data(), 0, sizeof(int) * num_verts);

    // per-vertex vector displacemenents
    std::vector<float> displacements(num_verts * 3);
    memset(displacements.data(), 0, sizeof(float) * num_verts);

    displaced_vertices->resize(vertices.size());

    // TODO(LTE): parallelize.
    for (size_t f = 0; f < num_faces; f++) {
          
      float uv[3][2];

      uv[0][0] = facevarying_texcoords[6 * f + 0];
      uv[0][1] = facevarying_texcoords[6 * f + 1];

      uv[1][0] = facevarying_texcoords[6 * f + 2];
      uv[1][1] = facevarying_texcoords[6 * f + 3];

      uv[2][0] = facevarying_texcoords[6 * f + 4];
      uv[2][1] = facevarying_texcoords[6 * f + 5];

      float3 dv[3];

      size_t vidx0, vidx1, vidx2;

      vidx0 = faces[3 * f + 0];
      vidx1 = faces[3 * f + 1];
      vidx2 = faces[3 * f + 2];

      float disp0[3] = {0.0f, 0.0f, 0.0f};
      float disp1[3] = {0.0f, 0.0f, 0.0f};
      float disp2[3] = {0.0f, 0.0f, 0.0f};

      const uint32_t material_id = material_ids[f];

      if (material_id == vdisp_material_id) {

        FilterTexture(vdisp_image, vdisp_width, vdisp_height, uv[0][0], uv[0][1], disp0);
        FilterTexture(vdisp_image, vdisp_width, vdisp_height, uv[1][0], uv[1][1], disp1);
        FilterTexture(vdisp_image, vdisp_width, vdisp_height, uv[2][0], uv[2][1], disp2);

        displacements[3 * vidx0 + 0] += vdisp_scale * disp0[0];
        displacements[3 * vidx0 + 1] += vdisp_scale * disp0[1];
        displacements[3 * vidx0 + 2] += vdisp_scale * disp0[2];
        counts[vidx0]++;

        displacements[3 * vidx1 + 0] += vdisp_scale * disp1[0];
        displacements[3 * vidx1 + 1] += vdisp_scale * disp1[1];
        displacements[3 * vidx1 + 2] += vdisp_scale * disp1[2];
        counts[vidx1]++;

        displacements[3 * vidx2 + 0] += vdisp_scale * disp2[0];
        displacements[3 * vidx2 + 1] += vdisp_scale * disp2[1];
        displacements[3 * vidx2 + 2] += vdisp_scale * disp2[2];
        counts[vidx2]++;

      }

    }

    // normalize displacements and add it to vertices
    for (size_t v = 0; v < num_verts; v++) {

      float dv[3] = {0.0f, 0.0f, 0.0f};

      // normalize displacments
      if (counts[v] > 0) {
          float weight = 1.0f / float(counts[v]);

          dv[0] = displacements[3 * v + 0] * weight;
          dv[1] = displacements[3 * v + 1] * weight;
          dv[2] = displacements[3 * v + 2] * weight;
      }


      (*displaced_vertices)[3 * v + 0] = vertices[3 * v + 0] + dv[0]; 
      (*displaced_vertices)[3 * v + 1] = vertices[3 * v + 1] + dv[1]; 
      (*displaced_vertices)[3 * v + 2] = vertices[3 * v + 2] + dv[2]; 

    }

  } else {
    (void)(facevarying_normals);
    (void)(facevarying_tangents);
    (void)(facevarying_binormals);

    assert(0);  // TODO(LTE):
  }
}

// Recompute vertex normal by considering triangle's area and angle.
//
// http://www.bytehazard.com/articles/vertnorm.html
// https://stackoverflow.com/questions/45477806/general-method-for-calculating-smooth-vertex-normals-with-100-smoothness
//
void RecomputeSmoothNormals(
    const std::vector<float> &vertices,
    const std::vector<uint32_t> &faces,
    const bool area_weighting,
    std::vector<float> *facevarying_normals) {

  const size_t num_verts = vertices.size() / 3;
  const size_t num_faces = faces.size() / 3;

  // vertex normals
  std::vector<float3> normals(num_verts);
  memset(normals.data(), 0, sizeof(float3) * num_verts);

  facevarying_normals->resize(num_faces * 3 * 3);
  memset(facevarying_normals->data(), 0, sizeof(float) * num_faces * 3 * 3);

  for (size_t f = 0; f < num_faces; f++) {

    uint32_t f0, f1, f2;
    f0 = faces[3 * f + 0];
    f1 = faces[3 * f + 1];
    f2 = faces[3 * f + 2];

    float3 v0, v1, v2;

    v0[0] = vertices[3 * f0 + 0];
    v0[1] = vertices[3 * f0 + 1];
    v0[2] = vertices[3 * f0 + 2];

    v1[0] = vertices[3 * f1 + 0];
    v1[1] = vertices[3 * f1 + 1];
    v1[2] = vertices[3 * f1 + 2];

    v2[0] = vertices[3 * f2 + 0];
    v2[1] = vertices[3 * f2 + 1];
    v2[2] = vertices[3 * f2 + 2];

    // compute the cross product and add it to each vertex.
    float3 pn;
    pn = vcross(v1 - v0, v2 - v0); // TODO(LTE): Validate handness.

    if (!area_weighting) {
      pn = vnormalize(pn);
    }

    const float a1 = AngleBetween((v1 - v0), (v2 - v0));
    const float a2 = AngleBetween((v2 - v1), (v0 - v1));
    const float a3 = AngleBetween((v0 - v2), (v1 - v2));

    normals[f0] += a1 * pn; 
    normals[f1] += a2 * pn; 
    normals[f2] += a3 * pn; 
  }

  // normalize.
  for (size_t v = 0; v < num_verts; v++) {
    float3 N = vnormalize(normals[v]);

    normals[v] = N;
  }

  for (size_t f = 0; f < num_faces; f++) {

    uint32_t vidx0, vidx1, vidx2;

    vidx0 = faces[3 * f + 0];
    vidx1 = faces[3 * f + 1];
    vidx2 = faces[3 * f + 2];

    float3 n0, n1, n2; 
    n0 = normals[vidx0];
    n1 = normals[vidx1];
    n2 = normals[vidx2];

    (*facevarying_normals)[9 * f + 0] = n0[0];
    (*facevarying_normals)[9 * f + 1] = n0[1];
    (*facevarying_normals)[9 * f + 2] = n0[2];

    (*facevarying_normals)[9 * f + 3] = n1[0];
    (*facevarying_normals)[9 * f + 4] = n1[1];
    (*facevarying_normals)[9 * f + 5] = n1[2];

    (*facevarying_normals)[9 * f + 6] = n2[0];
    (*facevarying_normals)[9 * f + 7] = n2[1];
    (*facevarying_normals)[9 * f + 8] = n2[2];

  }

}

}  // namespace example
