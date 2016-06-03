#ifndef EXAMPLE_MATRIX_H_
#define EXAMPLE_MATRIX_H_

class Matrix {
public:
  Matrix();
  ~Matrix();

  static void FloatToDoubleMatrix(double out[4][4], float in[4][4]) {
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) {
        out[j][i] = in[j][i];
      }
    }
  }

  static void Print(double m[4][4]);
  static void LookAt(double m[4][4], double eye[3], double lookat[3],
                     double up[3]);
  static void Inverse(double m[4][4]);
  static void Mult(double dst[4][4], double m0[4][4], double m1[4][4]);
  static void MultV(double dst[3], double m[4][4], double v[3]);
};

#endif  // 
