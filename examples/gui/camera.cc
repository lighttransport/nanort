#include "camera.h"

#include <cmath>

#include "matrix.h"
#include "nanort.h"
#include "render-config.h"
#include "trackball.h"

#ifndef M_PI
// Source: http://www.geom.uiuc.edu/~huberty/math5337/groupe/digits.html
#define M_PI \
  3.141592653589793238462643383279502884197169399375105820974944592307816406
#endif

template <typename T>
T deg2rad(const T& ad) {
  return (ad * M_PI) / static_cast<T>(180.0);
}

using float3 = nanort::real3<float>;

void Camera::BaseCamera::setTransformation(
    const example::RenderConfig& config) {
  build_rotmatrix(mat, config.quat);
  float dist = std::abs(config.distance);

  // do for each component
  for (int i = 0; i < 3; i++) {
    // u, v and dir are just the individual columns of the rotation matrix
    float dir_i = mat[i][2];
    float origin_i = config.look_at[i] + dir_i * dist;

    // set origin position in matrix
    mat[i][3] = origin_i;
  }
}

void Camera::setCameraFromStr(example::RenderConfig& config,
                              const std::string& camera_type) {
  namespace cr = Camera::Registry;
  // 1) find out the index
  int idx = -1;
  for (int i = 0; i < cr::numCameraTypes; i++) {
    if (camera_type == cr::cameraTypes[i].typeName) {
      idx = i;
      break;
    }
  }
  if (-1 == idx) {
    printf("(W) could not find '%s' among camera definitions [",
           camera_type.c_str());
    for (int i = 0; i < cr::numCameraTypes; i++) {
      printf("%s", cr::cameraTypes[i].typeName);
      if (i < cr::numCameraTypes - 1) printf(", ");
    }
    idx = 0;
    printf("], using default '%s'\n", cr::cameraTypes[idx].typeName);
  }
  setCameraFromIdx(config, idx);
}

void Camera::setCameraFromIdx(example::RenderConfig& config,
                              int idxNewCamType) {
  namespace cr = Camera::Registry;
  // no change?
  if (!config.camera)
    printf("(I) initializing camera type '%s'\n",
           cr::cameraTypes[idxNewCamType].typeName);
  else if (config.cameraTypeSelection == idxNewCamType)
    return;
  else
    printf("(I) changing from camera type '%s' to '%s'\n",
           cr::cameraTypes[config.cameraTypeSelection].typeName,
           cr::cameraTypes[idxNewCamType].typeName);
  config.cameraTypeSelection = idxNewCamType;

  // get a new instance from the registry
  Camera::BaseCamera* cam = cr::cameraTypes[idxNewCamType].getInstance();
  cam->setTransformation(config);
  // This handling could potentially result in a race condition, since the
  // render thread could still access the camera while being deleted.
  Camera::BaseCamera* camOld = config.camera;
  config.camera = cam;
  delete camOld;
}

void Camera::Pinhole::setTransformation(const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);
  // compute corner
  const float h = static_cast<float>(config.height);
  // the distance to plane where each pixel is shifted by 1 unit
  const float flen = (0.5f * h / tanf(0.5f * deg2rad(config.fov)));
  // do for each component
  for (int i = 0; i < 3; i++) {
    // u, v and dir are just the individual columns of the rotation matrix
    float u_i = mat[i][0];
    float v_i = mat[i][1];
    float dir_i = mat[i][2];

    float look_i = -dir_i * flen;
    // The lower right corner is relative to zero. The half-pixel offsets (-0.5)
    // account for the center of a pixel.
    corner[i] =
        look_i - 0.5f * (static_cast<float>(config.width) * u_i + h * v_i);
  }
}

void Camera::Pinhole::generateRay(nanort::Ray<>& ray, const float xyPic[2]) {
  float3 u;
  float3 v;
  for (int i = 0; i < 3; i++) {
    // u, v and dir are just the individual columns of the rotation matrix
    u[i] = mat[i][0];
    v[i] = mat[i][1];
    ray.org[i] = mat[i][3];
  }

  float3 dir = corner + xyPic[0] * u + xyPic[1] * v;
  nanort::vnormalize(dir);
  for (int i = 0; i < 3; i++) ray.dir[i] = dir[i];
}

void Camera::Orthographic::setTransformation(
    const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);

  const float h = static_cast<float>(config.height);

  // the physical pixel size
  physPixSize = 2.0f * (config.distance * tanf(0.5f * deg2rad(config.fov))) / h;

  // compute corner
  // do for each component
  for (int i = 0; i < 3; i++) {
    // u, v and dir are just the individual columns of the rotation matrix
    float u_i = mat[i][0];
    float v_i = mat[i][1];
    float o_i = mat[i][3];
    // The lower right corner is relative to zero.
    corner[i] = o_i - 0.5f * physPixSize *
                          (static_cast<float>(config.width) * u_i + h * v_i);
  }
}

void Camera::Orthographic::generateRay(nanort::Ray<>& ray,
                                       const float xyPic[2]) {
  for (int i = 0; i < 3; i++) {
    // u, v and dir are just the individual columns of the rotation matrix
    float u_i = mat[i][0];
    float v_i = mat[i][1];
    ray.dir[i] = -mat[i][2];
    ray.org[i] =
        corner[i] + physPixSize * xyPic[0] * u_i + physPixSize * xyPic[1] * v_i;
  }
}

void Camera::SphericalPanorama::setTransformation(
    const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);
  memcpy(mat_T, mat, 16 * sizeof(mat[0][0]));
  Matrix::Inverse(mat_T);

  // compute angle increments per pixel
  const float vfov = deg2rad(config.fov);
  const float h = static_cast<float>(config.height);
  dAngle = vfov / h;

  // the corner angle
  float aspectRatio = static_cast<float>(config.width) / h;
  const float hfov = vfov * aspectRatio;
  // the field-of-view is associated with the vertical dimension
  anglesCorner[1] = -vfov / 2.0f;
  anglesCorner[0] = hfov / 2.0f;
}

void Camera::SphericalPanorama::generateRay(nanort::Ray<float>& ray,
                                            const float xyPic[2]) {
  // the absolute angles
  float angles[2];
  angles[0] = anglesCorner[0] - xyPic[0] * dAngle;
  angles[1] = anglesCorner[1] + xyPic[1] * dAngle;
  // this gives vertical straight lines but bend horizontal lines
  float dir[4] = {-std::cos(angles[1]) * std::sin(angles[0]),
                  std::sin(angles[1]),
                  -std::cos(angles[0]) * std::cos(angles[1])};
  float dir2[4];
  Matrix::MultV(dir2, mat_T, dir);

  for (int i = 0; i < 3; i++) {
    float eye_i = mat[i][3];
    ray.org[i] = eye_i;
    ray.dir[i] = dir2[i];
  }
}

void Camera::Spherical::setTransformation(const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);
  memcpy(mat_T, mat, 16 * sizeof(mat[0][0]));
  Matrix::Inverse(mat_T);

  // compute angle increments per pixel (the +1 comes from the fact that a pixel
  // has an area)
  const float h = static_cast<float>(config.height);
  float aspectRatio = static_cast<float>(config.width) / h;
  // the vertical and horizontal field-of-views
  const float vfov = deg2rad(config.fov);
  const float hfov = vfov * aspectRatio;
  // the angle increment
  dAngle = vfov / h;

  // the corner angle
  // the field-of-view is associated with the vertical dimension
  anglesCorner[1] = -vfov / 2.0;
  anglesCorner[0] = hfov / 2.0f;
}

void Camera::Spherical::generateRay(nanort::Ray<float>& ray,
                                    const float xyPic[2]) {
  // the absolute angles
  float angles[2];
  angles[0] = anglesCorner[0] - xyPic[0] * dAngle;
  angles[1] = anglesCorner[1] + xyPic[1] * dAngle;
  // this gives horizontal straight lines but bend vertical lines (the main
  // difference to SphericalPanorama)
  float dir[4] = {-std::sin(angles[0]),
                  std::cos(angles[0]) * std::sin(angles[1]),
                  -std::cos(angles[0]) * std::cos(angles[1]), 0};
  float dir2[4];
  Matrix::MultV(dir2, mat_T, dir);

  for (int i = 0; i < 3; i++) {
    float eye_i = mat[i][3];
    ray.org[i] = eye_i;
    ray.dir[i] = dir2[i];
  }
}

void Camera::Cylindrical::setTransformation(
    const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);
  memcpy(mat_T, mat, 16 * sizeof(mat[0][0]));
  Matrix::Inverse(mat_T);

  // 1) the "spherical" part (horizontal-direction)
  const float h = static_cast<float>(config.height);
  const float w = static_cast<float>(config.width);
  // compute the angle increments and a corner point
  const float aspectRatio = w / h;
  // the vertical and horizontal field-of-views
  const float vfov = deg2rad(config.fov);
  const float hfov = vfov * aspectRatio;
  dAngle = hfov / w;
  angleCorner = hfov / 2.0;

  // 2) the "pinhole" part (vertical-direction)
  // the distance to plane where each pixel is 1 unit apart
  flen = (0.5f * h / std::tan(0.5f * vfov));
  corner = -h / 2.0f;

  cornerOne = std::tan(vfov / 2.0);
  pxSize = 2 * std::tan(vfov / 2.0) / h;
}

void Camera::Cylindrical::generateRay(nanort::Ray<float>& ray,
                                      const float xyPic[2]) {
  // compute the pixel position on a cylinder
  float angle = angleCorner - xyPic[0] * dAngle;
  float dir1[4] = {-std::sin(angle), pxSize * xyPic[1] - cornerOne,
                   -std::cos(angle), 0};
  float dir11[4];

  // transform the point
  Matrix::MultV(dir11, mat_T, dir1);
  float3 dir2;
  for (int i = 0; i < 3; i++) dir2[i] = dir11[i];
  nanort::vnormalize(dir2);
  for (int i = 0; i < 3; i++) {
    ray.org[i] = mat[i][3];
    ray.dir[i] = dir2[i];
  }
}

void Camera::FishEye::setTransformation(const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);
  memcpy(mat_T, mat, 16 * sizeof(mat[0][0]));
  Matrix::Inverse(mat_T);
  const float h = static_cast<float>(config.height);
  const float w = static_cast<float>(config.width);

  pxCenter[0] = w / 2.0;
  pxCenter[1] = h / 2.0;
  fov = deg2rad(config.fov);
  rFactor = 1.0 / (h < w ? pxCenter[0] : pxCenter[1]);
}

void Camera::FishEye::generateRay(nanort::Ray<float>& ray,
                                  const float xyPic[2]) {
  // pixel position from center
  float pos2center[2] = {pxCenter[0] - xyPic[0], pxCenter[1] - xyPic[1]};
  // distance from center
  float r = std::sqrt(std::pow(pos2center[0], 2) + std::pow(pos2center[1], 2));
  // normalized radius from center
  float rNorm = r * rFactor;

  for (int i = 0; i < 3; i++) ray.org[i] = mat[i][3];

  // normalize
  float p2cNorm[2] = {pos2center[0] / r, pos2center[1] / r};
  // the resulting angle from center to pixel
  float angle = rNorm * fov / 2.0f;
  // limit to +/- 90 deg (a typical effect of fisheye lenses, but the raytracer
  // can do more)
  if (angle > M_PI / 2.0f) {
    for (int i = 0; i < 3; i++) ray.dir[i] = 0;
  } else {
    float dir[4] = {-std::sin(angle) * p2cNorm[0],
                    -std::sin(angle) * p2cNorm[1], -std::cos(angle), 0};
    // transform
    float dir2[4];
    Matrix::MultV(dir2, mat_T, dir);
    for (int i = 0; i < 3; i++) ray.dir[i] = dir2[i];
  }
}
void Camera::FishEyeMKX22::setTransformation(
    const example::RenderConfig& config) {
  // do pre-computations
  BaseCamera::setTransformation(config);
  memcpy(mat_T, mat, 16 * sizeof(mat[0][0]));
  Matrix::Inverse(mat_T);
  const float h = static_cast<float>(config.height);
  const float w = static_cast<float>(config.width);

  pxCenter[0] = w / 2.0f;
  pxCenter[1] = h / 2.0f;
  rFactor = 1.0 / ((h < w ? pxCenter[0] : pxCenter[1]));
}
void Camera::FishEyeMKX22::generateRay(nanort::Ray<float>& ray,
                                       const float* xyPic) {
  // pixel position from center
  float pos2center[2] = {pxCenter[0] - xyPic[0], pxCenter[1] - xyPic[1]};
  // distance from center
  float r = std::sqrt(std::pow(pos2center[0], 2) + std::pow(pos2center[1], 2));
  // normalized radius from center
  float rN1 = r * rFactor;
  float rN2 = rN1 * rN1;
  float rN3 = rN2 * rN1;
  float rN4 = rN2 * rN2;

  for (int i = 0; i < 3; i++) ray.org[i] = mat[i][3];

  // normalize
  float p2cNorm[2] = {pos2center[0] / r, pos2center[1] / r};
  // the resulting angle from center to pixel is a nonlinear function reported
  // in the pdf
  float angle = 1.3202 * rN1 + 1.4539 * rN2 - 2.9949 * rN3 + 2.1007 * rN4;
  // limit to +/- 90 deg (a typical effect of fisheye lenses, but the raytracer
  // can do more)
  if (rN1 > 1 /*angle > deg2rad(220 * 0.5f)*/) {
    for (int i = 0; i < 3; i++) ray.dir[i] = 0;
  } else {
    float dir[4] = {-std::sin(angle) * p2cNorm[0],
                    -std::sin(angle) * p2cNorm[1], -std::cos(angle), 0};
    // transform
    float dir2[4];
    Matrix::MultV(dir2, mat_T, dir);
    for (int i = 0; i < 3; i++) ray.dir[i] = dir2[i];
  }
}
