#ifndef CAMERA_H
#define CAMERA_H

#include <memory>
#include <string>

#include "nanort.h"

namespace example {
struct RenderConfig;
}

//! Implementation of different camera models, cf.
//! [https://docs.chaos.com/display/VMAX/Camera#Camera-cameraTypes].
namespace Camera {

//! A common base class for a camera.
class BaseCamera {
 protected:
  //! The transformation matrix of this camera. This is set by setTransformation
  float mat[4][4];

 public:
  virtual ~BaseCamera() = default;
  //! string identifier
  static const char* getTypeName() { return "basic camera"; }
  //! set the transformation matrix based on the current render config and the
  //! quaternion.
  virtual void setTransformation(const example::RenderConfig& config);
  //! generate a ray based
  virtual void generateRay(nanort::Ray<>& ray, const float xyPic[2]) = 0;
};

template <class T>
static BaseCamera* get() {
  return reinterpret_cast<BaseCamera*>(new T());
};

//! Pinhole camera, the default / standard perspective camera.
class Pinhole : protected BaseCamera {
  using float3 = nanort::real3<float>;
  //! a corner point
  float3 corner = {0, 0, 0};

 public:
  ~Pinhole() override = default;
  //! string identifier
  static const char* getTypeName() { return "perspective"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<>& ray, const float xyPic[2]) override;
};

// using Perspective = Pinhole;

//! An orthographic camera, that is all rays are parallel.
class Orthographic : protected BaseCamera {
  using float3 = nanort::real3<float>;
  //! a corner point
  float3 corner = {0, 0, 0};
  //! physical size of each pixel
  float physPixSize = 0;

 public:
  ~Orthographic() override = default;
  static const char* getTypeName() { return "orthographic"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<>& ray, const float xyPic[2]) override;
};

//! A camera where all pixels are mapped with equal angle increments. Horizontal
//! lines remain straight.
class Spherical : public BaseCamera {
  // the inverse matrix
  float mat_T[4][4];
  // angle size per pixel
  float dAngle = 0;
  // the zero angle
  float anglesCorner[2];

 public:
  ~Spherical() override = default;
  //! string identifier
  static const char* getTypeName() { return "spherical"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<float>& ray, const float xyPic[2]) override;
};

//! A camera where all pixels are mapped with equal angle increments. Vertical
//! lines remain straight. For a field-of-view of 180 this gives a typical
//! panorama view known from photo stitching.
class SphericalPanorama : public BaseCamera {
  // the inverse matrix
  float mat_T[4][4];
  // angle size per pixel
  float dAngle = 0;
  // the zero angle
  float anglesCorner[2];

 public:
  ~SphericalPanorama() override = default;
  static const char* getTypeName() { return "spherical-panorama"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<float>& ray, const float xyPic[2]) override;
};

//! A camera where all horizonal pixels are mapped with equal angle increments
//! and vertical pixels are mapped on a linear grid in space, like a perspective
//! from a pinhole camera. So this camera combines features from both spherical
//! and perspective. Vertical lines remain straight.
class Cylindrical : public BaseCamera {
  //! the inverse matrix
  float mat_T[4][4];
  //! the pixel size at unit distance
  float pxSize = 0;
  //! the vertical startpoint at unit distance
  float cornerOne = 0;
  //! the distance to where a pixel is one unit large
  float flen = 0;
  //! the vertical startpoint
  float corner = 0;
  //! angle size per pixel
  float dAngle = 0;
  //! The zero angle
  float angleCorner = 0;

 public:
  ~Cylindrical() override = default;
  //! string identifier
  static const char* getTypeName() { return "cylindrical"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<float>& ray, const float xyPic[2]) override;
};

//! A linear fish-eye camera model, cf.
//! [http://paulbourke.net/dome/fisheyecorrect/fisheyerectify.pdf]
class FishEye : public BaseCamera {
  //! the inverse matrix
  float mat_T[4][4];
  //! the pixel center
  float pxCenter[2];
  //! the field-of-view
  float fov = 0;
  //! factor to normalize the radius
  float rFactor = 0;

 public:
  ~FishEye() override = default;
  //! string identifier
  static const char* getTypeName() { return "fish-eye"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<float>& ray, const float xyPic[2]) override;
};

//! A nonlinear fish-eye camera model "iZugar MKX22 220 degree", cf.
//! [http://paulbourke.net/dome/fisheyecorrect/fisheyerectify.pdf]
class FishEyeMKX22 : public BaseCamera {
  //! the inverse matrix
  float mat_T[4][4];
  //! the pixel center
  float pxCenter[2];
  //! factor to normalize the radius
  float rFactor = 0;

 public:
  ~FishEyeMKX22() override = default;
  //! string identifier
  static const char* getTypeName() { return "fish-eye MKX22"; }
  void setTransformation(const example::RenderConfig& config) override;
  void generateRay(nanort::Ray<float>& ray, const float xyPic[2]) override;
};

// ========== helper functions ==========

//! Set the camera from given name and modify the config.
void setCameraFromStr(example::RenderConfig& config,
                      const std::string& camera_type = std::string());
//! Set the camera from given index into Registry::cameraTypes and modify the
//! config.
void setCameraFromIdx(example::RenderConfig& config, int idxNewCamType = 0);

// ========== registry of cameras ==========

//! simple register all camera types (makes live easier in other areas, like
//! configuration reading and combo box)
namespace Registry {

//! a camera definition (helps to synchronize different properties)
struct CamDef {
  CamDef(const char* tn, BaseCamera* (*gi)()) : typeName(tn), getInstance(gi) {}
  const char* typeName = nullptr;
  BaseCamera* (*getInstance)() = nullptr;
};

// shortcut macro
#define CAMDEF(CLASS) CamDef(CLASS::getTypeName(), get<CLASS>)

//! Register of all camera types with names and functions to generate a new
//! pointer to an object.
static CamDef cameraTypes[] = {
    CAMDEF(Pinhole),           CAMDEF(Orthographic), CAMDEF(Spherical),
    CAMDEF(SphericalPanorama), CAMDEF(Cylindrical),  CAMDEF(FishEye),
    CAMDEF(FishEyeMKX22),
};
#undef CAMDEF

static int numCameraTypes =
    sizeof(Camera::Registry::cameraTypes) / sizeof(void*);
}  // namespace Registry
}  // namespace Camera

#endif  // CAMERA_H
