#include <cassert>

#include "common-util.h"
#include "geometry-util.h"

#include <thread>
#include <atomic>
#include <mutex>

namespace example {

inline float lerp(float x, float y, float t) {
  return x + t * (y - x); }


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
  if (vdisp_space == 0) {
    // world space

    size_t num_faces = faces.size() / 3;

    displaced_vertices->resize(num_faces * 3);

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

        for (size_t f = fs; f < fe; f++) {
          
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

          float p[3][3];

          p[0][0] = vertices[3 * vidx0 + 0];
          p[0][1] = vertices[3 * vidx0 + 1];
          p[0][2] = vertices[3 * vidx0 + 2];

          p[1][0] = vertices[3 * vidx1 + 0];
          p[1][1] = vertices[3 * vidx1 + 1];
          p[1][2] = vertices[3 * vidx1 + 2];

          p[2][0] = vertices[3 * vidx2 + 0];
          p[2][1] = vertices[3 * vidx2 + 1];
          p[2][2] = vertices[3 * vidx2 + 2];

          float disp0[3] = {0.0f, 0.0f, 0.0f};
          float disp1[3] = {0.0f, 0.0f, 0.0f};
          float disp2[3] = {0.0f, 0.0f, 0.0f};

          const uint32_t material_id = material_ids[f];

          if (material_id == vdisp_material_id) {
            SampleTexture(vdisp_image, vdisp_width, vdisp_height, uv[0][0], uv[0][1], disp0);
            SampleTexture(vdisp_image, vdisp_width, vdisp_height, uv[1][0], uv[1][1], disp1);
            SampleTexture(vdisp_image, vdisp_width, vdisp_height, uv[2][0], uv[2][1], disp2);
          }

          dv[0][0] = p[0][0] + vdisp_scale * disp0[0];;
          dv[0][1] = p[0][1] + vdisp_scale * disp0[1];;
          dv[0][2] = p[0][2] + vdisp_scale * disp0[2];;

          dv[1][0] = p[1][0] + vdisp_scale * disp1[0];;
          dv[1][1] = p[1][1] + vdisp_scale * disp1[1];;
          dv[1][2] = p[1][2] + vdisp_scale * disp1[2];;

          dv[2][0] = p[2][0] + vdisp_scale * disp2[0];;
          dv[2][1] = p[2][1] + vdisp_scale * disp2[1];;
          dv[2][2] = p[2][2] + vdisp_scale * disp2[2];;

          (*displaced_vertices)[9 * f + 0] = dv[0][0]; 
          (*displaced_vertices)[9 * f + 1] = dv[0][1]; 
          (*displaced_vertices)[9 * f + 2] = dv[0][2]; 

          (*displaced_vertices)[9 * f + 3] = dv[1][0]; 
          (*displaced_vertices)[9 * f + 4] = dv[1][1]; 
          (*displaced_vertices)[9 * f + 5] = dv[1][2]; 

          (*displaced_vertices)[9 * f + 6] = dv[2][0]; 
          (*displaced_vertices)[9 * f + 7] = dv[2][1]; 
          (*displaced_vertices)[9 * f + 8] = dv[2][2]; 

        }
      }));
    }

    for (auto &t : workers) {
      t.join();
    }

  } else {
    (void)(facevarying_normals);
    (void)(facevarying_tangents);
    (void)(facevarying_binormals);

    assert(0);  // TODO(LTE):
  }
}

}  // namespace example
