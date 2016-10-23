#ifndef EXAMPLE_SHADER_H_
#define EXAMPLE_SHADER_H_

namespace example {

class vec3 {
 public:
  vec3() : x(0.0f), y(0.0f), z(0.0f) {}
  vec3(float v) : x(v), y(v), z(v) {}
  vec3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) {}
  //~vec3() {}

  vec3 operator+(const vec3 &f2) const {
    return vec3(x + f2.x, y + f2.y, z + f2.z);
  }
  void operator+=(const vec3 &f2) {
    x += f2.x;
    y += f2.y;
    z += f2.z;
  }
  vec3 operator*(const vec3 &f2) const {
    return vec3(x * f2.x, y * f2.y, z * f2.z);
  }
  vec3 operator/(const vec3 &f2) const {
    return vec3(x / f2.x, y / f2.y, z / f2.z);
  }
  void operator/=(const vec3 &f2) {
    x /= f2.x;
    y /= f2.y;
    z /= f2.z;
  }
  vec3 operator/(const float f) const { return vec3(x / f, y / f, z / f); }

  float x, y, z;
};

inline vec3 operator*(float f, const vec3 &v) {
  return vec3(v.x * f, v.y * f, v.z * f);
}

#define pMax (3)

class HairBSDF {
 public:
  HairBSDF(float h, float eta, const vec3 &sigma_a, float beta_m, float beta_n,
           float alpha);
  ~HairBSDF() {}

  vec3 f(const vec3 &wo, const vec3 &wi) const;
  vec3 Sample_f(const vec3 &wo, vec3 *wi, const float u2[2], float *pdf) const;
  float Pdf(const vec3 &wo, const vec3 &wi) const;

 private:
  // HairBSDF Private Methods
  void ComputeApPdf(float pdfs[pMax + 1], float cosThetaO) const;

  // HairBSDF Private Data
  const float h_, gammaO_, eta_;
  const vec3 sigma_a_;
  const float beta_m_, beta_n_, alpha_;
  float v_[pMax + 1];
  float s_;
  float sin2kAlpha_[3], cos2kAlpha_[3];
};
}

#endif  // EXAMPLE_SHADER_H_
