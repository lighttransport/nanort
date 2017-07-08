#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

#if __cplusplus > 199711L
#include <chrono>
#endif

#define NOMINMAX
#include "tiny_obj_loader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#include "../../common/tinyexr.h"

#define ENABLE_FP_EXCEPTION (0)
#if ENABLE_FP_EXCEPTION
#ifdef _WIN32
#include <float.h>
void setupFPExceptions() {
	unsigned int cw, newcw;
	_controlfp_s(&cw, 0, 0);
	newcw = ~(_EM_INVALID | _EM_DENORMAL | _EM_ZERODIVIDE | _EM_OVERFLOW |
		_EM_UNDERFLOW);
	_controlfp_s(&cw, newcw, _MCW_EM);
}
#else // Assume x86 SSE + linux or macosx
#include <xmmintrin.h>
void setupFPExceptions() {
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID); // Enale NaN exception
}
#endif
#endif

#include "../../../nanort.h"
#include"Ray.h"
#include"vec3.h"

static const float kEps = 0.001f;
static const float kInf = 1.0e30f;
static const float kPi = 4.0f * std::atan(1.0f);
const float PI = 3.141592;
const float M_PI = 3.141592;

class RandomMT{

public:
	float genrand64_real1(){
		return rand() / (float)(RAND_MAX + 1.0);
	}
	float genrand64_real2(){
		return rand() / (float)(RAND_MAX + 1.0);
	}
	float genrand64_real3(){
		return rand() / (float)(RAND_MAX + 1.0);
	}

	RandomMT(int i){
	}
	~RandomMT(){

	}
};


static const int uMaxBounces = 10;
// static const int SPP = 1000;
static const int SPP = 100;

typedef nanort::BVHAccel<float, nanort::TriangleMesh<float>,
	nanort::TriangleSAHPred<float>,
	nanort::TriangleIntersector<> >
	Accel;

using namespace PTUtility;

namespace {

#if __cplusplus > 199711L
	typedef std::chrono::time_point<std::chrono::system_clock> TimeType;
	inline TimeType tick() { return std::chrono::system_clock::now(); }

	inline double to_duration(TimeType start, TimeType end) {
		return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
			.count() /
			100.0;
	}
#else
#define nullptr NULL

	typedef clock_t TimeType;
	inline TimeType tick() { return clock(); }
	inline double to_duration(TimeType start, TimeType end) {
		return (end - start) / 1000.0;
	}
#endif

	// -----------------------------------------------------------------------------
	// Assertion with message
	// -----------------------------------------------------------------------------

#ifndef __FUNCTION_NAME__
#if defined(_WIN32) || defined(__WIN32__)
#define __FUNCTION_NAME__ __FUNCTION__
#else
#define __FUNCTION_NAME__ __func__
#endif
#endif

#ifndef NDEBUG
#define Assertion(PREDICATE, ...)                                        \
  \
do {                                                                     \
    if (!(PREDICATE)) {                                                  \
      std::cerr << "Asssertion \"" << #PREDICATE << "\" failed in "      \
                << __FILE__ << " line " << __LINE__ << " in function \"" \
                << (__FUNCTION_NAME__) << "\""                           \
                << " : ";                                                \
      fprintf(stderr, __VA_ARGS__);                                      \
      std::cerr << std::endl;                                            \
      std::abort();                                                      \
	    }                                                                    \
  \
}                                                                   \
	  while (false)
#else  // NDEBUG
#define Assertion(PREDICATE, ...) \
  do {                            \
    } while (false)
#endif  // NDEBUG

	// ----------------------------------------------------------------------------
	// Timer class
	// ----------------------------------------------------------------------------

	class Timer {
	public:
		Timer() : start_(), end_() {}

		void start() { start_ = tick(); }

		double stop() {
			end_ = tick();
			return to_duration(start_, end_);
		}

	private:
		TimeType start_, end_;
	};

	// ----------------------------------------------------------------------------
	// 3D vector class
	// ----------------------------------------------------------------------------

	struct float3 {
		float3() : x(0.0f), y(0.0f), z(0.0f) {}
		float3(float xx, float yy, float zz) {
			x = xx;
			y = yy;
			z = zz;
		}
		float3(const float *p) {
			x = p[0];
			y = p[1];
			z = p[2];
		}

		float3 operator*(float f) const { return float3(x * f, y * f, z * f); }
		float3 operator-(const float3 &f2) const {
			return float3(x - f2.x, y - f2.y, z - f2.z);
		}
		float3 operator-() const { return float3(-x, -y, -z); }
		float3 operator*(const float3 &f2) const {
			return float3(x * f2.x, y * f2.y, z * f2.z);
		}
		float3 operator+(const float3 &f2) const {
			return float3(x + f2.x, y + f2.y, z + f2.z);
		}
		float3 &operator+=(const float3 &f2) {
			x += f2.x;
			y += f2.y;
			z += f2.z;
			return (*this);
		}
		float3 &operator*=(const float3 &f2) {
			x *= f2.x;
			y *= f2.y;
			z *= f2.z;
			return (*this);
		}
		float3 &operator*=(const float &f2) {
			x *= f2;
			y *= f2;
			z *= f2;
			return (*this);
		}
		float3 operator/(const float3 &f2) const {
			return float3(x / f2.x, y / f2.y, z / f2.z);
		}
		float3 operator/(const float &f2) const {
			return float3(x / f2, y / f2, z / f2);
		}
		float operator[](int i) const { return (&x)[i]; }
		float &operator[](int i) { return (&x)[i]; }

		float3 neg() { return float3(-x, -y, -z); }

		float length() { return sqrtf(x * x + y * y + z * z); }

		void normalize() {
			float len = length();
			if (fabs(len) > 1.0e-6) {
				float inv_len = 1.0 / len;
				x *= inv_len;
				y *= inv_len;
				z *= inv_len;
			}
		}

		bool isBlack() const { return x == 0.0f && y == 0.0f && z == 0.0f; }

		bool isValid() const {
			if (std::isinf(x) || std::isinf(y) || std::isinf(z)) return false;
			if (std::isnan(x) || std::isnan(y) || std::isnan(z)) return false;
			return true;
		}

		float x, y, z;
	};

	// ----------------------------------------------------------------------------
	// 3D vector utilities
	// ----------------------------------------------------------------------------

	inline float3 normalize(float3 v) {
		v.normalize();
		return v;
	}

	inline float3 operator*(float f, const float3 &v) {
		return float3(v.x * f, v.y * f, v.z * f);
	}

	inline float3 vcross(float3 a, float3 b) {
		float3 c;
		c[0] = a[1] * b[2] - a[2] * b[1];
		c[1] = a[2] * b[0] - a[0] * b[2];
		c[2] = a[0] * b[1] - a[1] * b[0];
		return c;
	}

	inline float vdot(float3 a, float3 b) {
		return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	}

	float3 directionCosTheta(float3 normal, float u1, float u2, float *pdfOmega) {
		float phi = 2.0 * kPi * u2;

		float r = sqrt(u1);

		float x = r * cosf(phi);
		float y = r * sinf(phi);
		float z = sqrtf(1.0 - u1);

		*pdfOmega = z / kPi;

		float3 xDir =
			fabsf(normal.x) < fabsf(normal.y) ? float3(1, 0, 0) : float3(0, 1, 0);
		float3 yDir = normalize(vcross(normal, xDir));
		xDir = vcross(yDir, normal);
		return xDir * x + yDir * y + z * normal;
	}

	// ----------------------------------------------------------------------------
	// Mesh class
	// ----------------------------------------------------------------------------

	typedef struct {
		size_t num_vertices;
		size_t num_faces;
		float *vertices;                   /// [xyz] * num_vertices
		float *facevarying_normals;        /// [xyz] * 3(triangle) * num_faces
		float *facevarying_tangents;       /// [xyz] * 3(triangle) * num_faces
		float *facevarying_binormals;      /// [xyz] * 3(triangle) * num_faces
		float *facevarying_uvs;            /// [xyz] * 3(triangle) * num_faces
		float *facevarying_vertex_colors;  /// [xyz] * 3(triangle) * num_faces
		unsigned int *faces;               /// triangle x num_faces
		unsigned int *material_ids;        /// index x num_faces
	} Mesh;

	void calcNormal(float3 &N, float3 v0, float3 v1, float3 v2) {
		float3 v10 = v1 - v0;
		float3 v20 = v2 - v0;

		N = vcross(v20, v10);
		N.normalize();
	}

	// ----------------------------------------------------------------------------
	// Material class
	// ----------------------------------------------------------------------------

	struct Material {
		float ambient[3];
		float diffuse[3];
		float reflection[3];
		float refraction[3];
		int id;
		int diffuse_texid;
		int reflection_texid;
		int transparency_texid;
		int bump_texid;
		int normal_texid;  // normal map
		int alpha_texid;   // alpha map

		Material() {
			ambient[0] = 0.0;
			ambient[1] = 0.0;
			ambient[2] = 0.0;
			diffuse[0] = 0.5;
			diffuse[1] = 0.5;
			diffuse[2] = 0.5;
			reflection[0] = 0.0;
			reflection[1] = 0.0;
			reflection[2] = 0.0;
			refraction[0] = 0.0;
			refraction[1] = 0.0;
			refraction[2] = 0.0;
			id = -1;
			diffuse_texid = -1;
			reflection_texid = -1;
			transparency_texid = -1;
			bump_texid = -1;
			normal_texid = -1;
			alpha_texid = -1;
		}
	};





	nanort::Ray<float>& As(nanort::Ray<float>& A, const PTUtility::Ray& B){
		for (int i = 0; i < 3; i++){
			A.dir[i] = B.m_Dir[i];
			A.org[i] = B.m_Org[i];
		}
		return A;
	}
	PTUtility::Ray& As(PTUtility::Ray& A, const nanort::Ray<float>& B){
		for (int i = 0; i < 3; i++){
			A.m_Dir[i] = B.dir[i];
			A.m_Org[i] = B.dir[i];
		}
		A.m_Dir.normalize();
		return A;
	}
	float3& As(float3& A, const PTUtility::Vec3& B){
		for (int i = 0; i < 3; i++){
			A[i] = B[i];
		}
		return A;
	}
	PTUtility::Vec3& As(PTUtility::Vec3& A, const float3& B){
		for (int i = 0; i < 3; i++){
			A[i] = B[i];
		}
		return A;
	}




	// ----------------------------------------------------------------------------
	// Image utilities
	// ----------------------------------------------------------------------------

	void SaveImagePNG(const char *filename, const float *rgb, int width,
		int height) {
		unsigned char *bytes = new unsigned char[width * height * 3];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				const int index = y * width + x;
				bytes[index * 3 + 0] = (unsigned char)std::max(
					0.0f, std::min(rgb[index * 3 + 0] * 255.0f, 255.0f));
				bytes[index * 3 + 1] = (unsigned char)std::max(
					0.0f, std::min(rgb[index * 3 + 1] * 255.0f, 255.0f));
				bytes[index * 3 + 2] = (unsigned char)std::max(
					0.0f, std::min(rgb[index * 3 + 2] * 255.0f, 255.0f));
			}
		}
		stbi_write_png(filename, width, height, 3, bytes, width * 3);
		delete[] bytes;
	}

	void SaveImageEXR(const char *filename, const float *rgb, int width,
		int height) {
		int ret = SaveEXR(rgb, width, height, /* RGB */3, filename);
		if (ret != TINYEXR_SUCCESS) {
			fprintf(stderr, "EXR save error: %d\n", ret);
		}
		else {
			printf("Saved image to [ %s ]\n", filename);
		}
	}

	//! OBJ file loader.
	bool LoadObj(Mesh &mesh, std::vector<tinyobj::material_t> &materials,
		const char *filename, float scale, const char* mtl_path) {
		std::vector<tinyobj::shape_t> shapes;
		std::string err;
		bool success = tinyobj::LoadObj(shapes, materials, err, filename, mtl_path);

		if (!err.empty()) {
			std::cerr << err << std::endl;
		}

		if (!success) {
			return false;
		}

		std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
		std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size()
			<< std::endl;

		size_t num_vertices = 0;
		size_t num_faces = 0;
		for (size_t i = 0; i < shapes.size(); i++) {
			printf("  shape[%ld].name = %s\n", i, shapes[i].name.c_str());
			printf("  shape[%ld].indices: %ld\n", i, shapes[i].mesh.indices.size());
			assert((shapes[i].mesh.indices.size() % 3) == 0);
			printf("  shape[%ld].vertices: %ld\n", i, shapes[i].mesh.positions.size());
			assert((shapes[i].mesh.positions.size() % 3) == 0);
			printf("  shape[%ld].normals: %ld\n", i, shapes[i].mesh.normals.size());
			assert((shapes[i].mesh.normals.size() % 3) == 0);

			num_vertices += shapes[i].mesh.positions.size() / 3;
			num_faces += shapes[i].mesh.indices.size() / 3;
		}
		std::cout << "[LoadOBJ] # of faces: " << num_faces << std::endl;
		std::cout << "[LoadOBJ] # of vertices: " << num_vertices << std::endl;

		// @todo { material and texture. }

		// Shape -> Mesh
		mesh.num_faces = num_faces;
		mesh.num_vertices = num_vertices;
		mesh.vertices = new float[num_vertices * 3];
		mesh.faces = new unsigned int[num_faces * 3];
		mesh.material_ids = new unsigned int[num_faces];
		memset(mesh.material_ids, 0, sizeof(int) * num_faces);
		mesh.facevarying_normals = new float[num_faces * 3 * 3];
		mesh.facevarying_uvs = new float[num_faces * 3 * 2];
		memset(mesh.facevarying_uvs, 0, sizeof(float) * 2 * 3 * num_faces);

		// @todo {}
		mesh.facevarying_tangents = NULL;
		mesh.facevarying_binormals = NULL;

		size_t vertexIdxOffset = 0;
		size_t faceIdxOffset = 0;
		for (size_t i = 0; i < shapes.size(); i++) {
			for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
				mesh.faces[3 * (faceIdxOffset + f) + 0] =
					shapes[i].mesh.indices[3 * f + 0];
				mesh.faces[3 * (faceIdxOffset + f) + 1] =
					shapes[i].mesh.indices[3 * f + 1];
				mesh.faces[3 * (faceIdxOffset + f) + 2] =
					shapes[i].mesh.indices[3 * f + 2];

				mesh.faces[3 * (faceIdxOffset + f) + 0] += vertexIdxOffset;
				mesh.faces[3 * (faceIdxOffset + f) + 1] += vertexIdxOffset;
				mesh.faces[3 * (faceIdxOffset + f) + 2] += vertexIdxOffset;

				mesh.material_ids[faceIdxOffset + f] = shapes[i].mesh.material_ids[f];
			}

			for (size_t v = 0; v < shapes[i].mesh.positions.size() / 3; v++) {
				mesh.vertices[3 * (vertexIdxOffset + v) + 0] =
					scale * shapes[i].mesh.positions[3 * v + 0];
				mesh.vertices[3 * (vertexIdxOffset + v) + 1] =
					scale * shapes[i].mesh.positions[3 * v + 1];
				mesh.vertices[3 * (vertexIdxOffset + v) + 2] =
					scale * shapes[i].mesh.positions[3 * v + 2];
			}

			if (shapes[i].mesh.normals.size() > 0) {
				for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
					int f0, f1, f2;

					f0 = shapes[i].mesh.indices[3 * f + 0];
					f1 = shapes[i].mesh.indices[3 * f + 1];
					f2 = shapes[i].mesh.indices[3 * f + 2];

					float3 n0, n1, n2;

					n0[0] = shapes[i].mesh.normals[3 * f0 + 0];
					n0[1] = shapes[i].mesh.normals[3 * f0 + 1];
					n0[2] = shapes[i].mesh.normals[3 * f0 + 2];

					n1[0] = shapes[i].mesh.normals[3 * f1 + 0];
					n1[1] = shapes[i].mesh.normals[3 * f1 + 1];
					n1[2] = shapes[i].mesh.normals[3 * f1 + 2];

					n2[0] = shapes[i].mesh.normals[3 * f2 + 0];
					n2[1] = shapes[i].mesh.normals[3 * f2 + 1];
					n2[2] = shapes[i].mesh.normals[3 * f2 + 2];

					n0.normalize();
					n1.normalize();
					n2.normalize();

					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] = n0[2];

					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] = n1[2];

					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] = n2[2];
				}
			}
			else {
				// calc geometric normal
				for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
					int f0, f1, f2;

					f0 = shapes[i].mesh.indices[3 * f + 0];
					f1 = shapes[i].mesh.indices[3 * f + 1];
					f2 = shapes[i].mesh.indices[3 * f + 2];

					float3 v0, v1, v2;

					v0[0] = shapes[i].mesh.positions[3 * f0 + 0];
					v0[1] = shapes[i].mesh.positions[3 * f0 + 1];
					v0[2] = shapes[i].mesh.positions[3 * f0 + 2];

					v1[0] = shapes[i].mesh.positions[3 * f1 + 0];
					v1[1] = shapes[i].mesh.positions[3 * f1 + 1];
					v1[2] = shapes[i].mesh.positions[3 * f1 + 2];

					v2[0] = shapes[i].mesh.positions[3 * f2 + 0];
					v2[1] = shapes[i].mesh.positions[3 * f2 + 1];
					v2[2] = shapes[i].mesh.positions[3 * f2 + 2];

					float3 N;
					calcNormal(N, v0, v1, v2);

					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 0] = N[0];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 1] = N[1];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 0) + 2] = N[2];

					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 0] = N[0];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 1] = N[1];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 1) + 2] = N[2];

					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 0] = N[0];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 1] = N[1];
					mesh.facevarying_normals[3 * (3 * (faceIdxOffset + f) + 2) + 2] = N[2];
				}
			}

			if (shapes[i].mesh.texcoords.size() > 0) {
				for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
					int f0, f1, f2;

					f0 = shapes[i].mesh.indices[3 * f + 0];
					f1 = shapes[i].mesh.indices[3 * f + 1];
					f2 = shapes[i].mesh.indices[3 * f + 2];

					float3 n0, n1, n2;

					n0[0] = shapes[i].mesh.texcoords[2 * f0 + 0];
					n0[1] = shapes[i].mesh.texcoords[2 * f0 + 1];

					n1[0] = shapes[i].mesh.texcoords[2 * f1 + 0];
					n1[1] = shapes[i].mesh.texcoords[2 * f1 + 1];

					n2[0] = shapes[i].mesh.texcoords[2 * f2 + 0];
					n2[1] = shapes[i].mesh.texcoords[2 * f2 + 1];

					mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 0] = n0[0];
					mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 0) + 1] = n0[1];

					mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 0] = n1[0];
					mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 1) + 1] = n1[1];

					mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 0] = n2[0];
					mesh.facevarying_uvs[2 * (3 * (faceIdxOffset + f) + 2) + 1] = n2[1];
				}
			}

			vertexIdxOffset += shapes[i].mesh.positions.size() / 3;
			faceIdxOffset += shapes[i].mesh.indices.size() / 3;
		}

		return true;
	}

}  // namespace

inline float sign(float f) { return f < 0 ? -1 : 1; }
inline float3 reflect(float3 I, float3 N) { return I - 2 * vdot(I, N) * N; }
inline float3 refract(float3 I, float3 N, float eta) {
	float NdotI = vdot(N, I);
	float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
	if (k < 0.0f)
		return float3(0, 0, 0);
	else
		return eta * I - (eta * NdotI + sqrtf(k)) * N;
}
inline float pow5(float val) { return val * val * val * val * val; }
inline float fresnel_schlick(float3 H, float3 norm, float n1) {
	float r0 = n1 * n1;
	return r0 + (1 - r0) * pow5(1 - vdot(H, norm));
}
bool isLight(const tinyobj::material_t &mat) {
	float3 Le(mat.emission);
	return Le[0] != 0.0f || Le[1] != 0.0f || Le[2] != 0.0f;
}


inline Vec3 Reflect(const Vec3& In, const Vec3& Normal){
	return In - 2.0f * In.dot(Normal) * Normal;
}
inline Vec3 Refract(const Vec3& In, const Vec3& Normal, float eta){
	float NdotI = Normal.dot(In);
	float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
	if (k < 0.0f)
		return Vec3(0, 0, 0);
	else
		return eta * In - (eta * NdotI + sqrtf(k)) * Normal;
}
inline float fresnel_schlick(const Vec3& H, const Vec3& norm, float n1) {
	float r0 = n1 * n1;
	return r0 + (1 - r0) * pow5(1 - H.dot(norm));
}








struct CameraInfo{
	Vec3 _Pos;
	Vec3 _Center;
	float _PlaneWidth;
	float _PlaneHeight;

	Vec3 e1;
	Vec3 e2;

	CameraInfo(const Vec3& Position, const Vec3& PlaneCenter, float PlaneWidth, float PlaneHeight) :
		_Pos(Position), _Center(PlaneCenter), _PlaneWidth(PlaneWidth), _PlaneHeight(PlaneHeight){

		//calculate e1,e2 vector
		Vec3 LK = (_Center - _Pos).normalized();

		if (LK.y() > 0.996){
			e1 = Vec3(1, 0, 0);
			e1 = Vec3(0, 0, 1);
		}
		else{
			e1 = LK.cross(Vec3(1, 0, 0)).normalized();
			e2 = e1.cross(LK).normalized();
		}
	}

	CameraInfo(const Vec3& Position, float PlaneWidth, float PlaneHeight, const Vec3& LookAt, float DistanceToImagePlane) :
		_Pos(Position), _PlaneWidth(PlaneWidth), _PlaneHeight(PlaneHeight){

		_Center = Position + (LookAt - Position).normalized() * DistanceToImagePlane;

		//calculate e1,e2 vector
		Vec3 LK = (_Center - _Pos).normalized();

		if (LK.y() > 0.996){
			e1 = Vec3(1, 0, 0);
			e1 = Vec3(0, 0, 1);
		}
		else{
			e1 = LK.cross(Vec3(0, 1, 0)).normalized();
			e2 = e1.cross(LK).normalized();
		}
	}



	Vec3 GeneratePos(float u, float v)const{
		return _Center + (u - 0.5)*_PlaneWidth*e1 + (v - 0.5)*_PlaneHeight*e2;
	}

};

class Sample{
public:

	static float Sphere(Vec3& result, RandomMT& MT){

		float ph = MT.genrand64_real1() * 2.0f * PI;
		float th = MT.genrand64_real1() * PI;
		th = acos(MT.genrand64_real1() * 2.0f - 1.0f);

		result = Vec3(sinf(th)*cosf(ph), sinf(th)*sinf(ph), cosf(th));
		result.normalize();

		return 0.25f / PI;
	}


	static float HemSphere(Vec3& result, RandomMT& MT){

		float ph = MT.genrand64_real1() * 2.0f * PI;
		float th = MT.genrand64_real1() * PI;
		th = acos(MT.genrand64_real1() * 2.0f - 1.0f);

		result = Vec3(sinf(th)*cosf(ph), sinf(th)*sinf(ph), cosf(th));
		result.normalize();

		return 0.25f / PI;
	}


};







class VolumePathTrace{


public:

	virtual class NextEventEstimation{
		virtual int V_NextEvent(PTUtility::Vec3& CurrentPosition, float& PDF, PTUtility::Vec3& Radiance);

	public:
		int NextEvent(PTUtility::Vec3& CurrentPosition, float& PDF, PTUtility::Vec3& Radiance){
			return V_NextEvent(CurrentPosition, PDF, Radiance);
		}
		NextEventEstimation(){

		}
		virtual ~NextEventEstimation(){

		}

	};
	class NEE_PointLight : public NextEventEstimation{
		virtual int V_NextEvent(PTUtility::Vec3& CurrentPosition, float& PDF, PTUtility::Vec3& Radiance){
			return 1;
		}
		NEE_PointLight(){

		}
		virtual ~NEE_PointLight(){

		}
	};
	class NEE_Shpere : public NextEventEstimation{
		virtual int V_NextEvent(PTUtility::Vec3& CurrentPosition, float& PDF, PTUtility::Vec3& Radiance){
			return 1;
		}
		NEE_Shpere(){

		}
		virtual ~NEE_Shpere(){

		}
	};
	class NEE_Area : public NextEventEstimation{
		virtual int V_NextEvent(PTUtility::Vec3& CurrentPosition, float& PDF, PTUtility::Vec3& Radiance){
			return 1;
		}
		NEE_Area(){

		}
		virtual ~NEE_Area(){

		}
	};


	//Discribe Parameter of Translucent Material
	struct SSSParam{

		static float Default_Phase(const PTUtility::Vec3& in, const PTUtility::Vec3& out){
			return (1.0f) / (4.0f * 3.141592);
		}
		static float HG_Phase(const PTUtility::Vec3& out, const PTUtility::Vec3& in, float g){
			float cos = out.dot(in);
			float G = 0.25f * (1.00001 - g*g) / (3.141592 * pow(1.000001 + g*g - 2.0f * g * cos, 1.5));
			return G;
		}

		Vec3 _Sigma_A;
		Vec3 _Sigma_S;
		float _g;
		float _Index;

		SSSParam(const PTUtility::Vec3& SigmaA, const PTUtility::Vec3& SigmaS, float g, float Index) : _Sigma_A(SigmaA), _Sigma_S(SigmaS), _g(g), _Index(Index){

		}
	};

	//Describe Geometry and DataStructure
	struct Scene{
		const Mesh& _Mesh;
		const nanort::BVHBuildStatistics& _Acc;
		nanort::BVHBuildOptions<float>& _traceOPT;
		nanort::TriangleIntersector<>& TIsc;
	};


private:


	//Rendering Parameter
	struct PTParam{
		Vec3 _Normal;
		Vec3 _Pos;
		Vec3 _Color;
		Ray _NextRay;
		Ray _CRay;
		Vec3 _Weight;
		int _Depth;
		bool _IfLoop;
		bool _NextEventEstimation;

		enum Ray_Stat{
			INSIDE_TRANSLUCENT = 0, 
			OUTSIDE_TRANSLUCENT = 1,
		};
		Ray_Stat _Stat;
		float _Index;

		PTParam() :
			_Weight(1, 1, 1),
			_IfLoop(true), _Depth(0),
			_Normal(0, 0, 1),
			_Pos(0, 0, 0),
			_Color(0, 0, 0),
			_NextRay(Vec3(0, 0, 0), Vec3(0, 0, 1)),
			_CRay(Vec3(0, 0, 0), Vec3(0, 0, 1)), 
			_Stat(OUTSIDE_TRANSLUCENT), 
			_Index(1.0000f), 
			_NextEventEstimation(false){

		}
	};


	int* m_Image;
	int m_ImageX;
	int m_ImageY;





	PTUtility::Vec3 VolumePathTrace::NEESSS(
		const PTUtility::Ray& ray,
		PTParam& pm,
		const SSSParam& mat,
		const Vec3& LightPos){

		const Vec3& St = (mat._Sigma_A + mat._Sigma_S);

		float LightLen = (ray.m_Org - LightPos).norm();
		float phase = SSSParam::HG_Phase(pm._CRay.m_Dir, (ray.m_Org - LightPos).normalized(), mat._g);
		Vec3 RL(exp(-St[0] * LightLen), exp(-St[1] * LightLen), exp(-St[2] * LightLen));
		RL.MultVec(Vec3::DivVec(mat._Sigma_S, mat._Sigma_A + mat._Sigma_S));

		return RL*phase / (LightLen * LightLen);
	}



	Vec3 VolumePathTrace::trace(
		const Mesh& mesh, 
		const nanort::BVHAccel<float, nanort::TriangleMesh<float>, nanort::TriangleSAHPred<float>, nanort::TriangleIntersector<>>& scene, 
		const nanort::TriangleIntersector<>& TriIsec, 
		const nanort::BVHTraceOptions& TraceOPT, 
		PTParam& pm,
		const SSSParam& mat,
		RandomMT& MT
		){

		const int MAX_DEPTH = 100;
		pm._Depth++;
		if (pm._Depth >= MAX_DEPTH){
			pm._IfLoop = false;
			return Vec3::Zero();
		}

		//russian roulette
		float rus_pdf = 1.0f;
		if (pm._Depth > MAX_DEPTH){
			rus_pdf = exp(MAX_DEPTH - pm._Depth);
			if (MT.genrand64_real1() > rus_pdf){
				pm._IfLoop = false;
				return Vec3::Ones();
			}
		}

		Vec3 LightPos(0, 0, -0.5 + 0.2);

		nanort::Ray<float> cray;
		As(cray, pm._CRay);
		cray.min_t = 0.0f;
		cray.max_t = 375000.0f;
		bool Hit = scene.Traverse(cray, TraceOPT, TriIsec);


		int SSStat = -100;
		if (Hit){
			SSStat = 100;
			//if Ray Hits the objects

			//Calculate geometory info at hit position
			Vec3 Color(0, 0, 0);
			float u = TriIsec.intersection.u;
			float v = TriIsec.intersection.v;
			float t = TriIsec.intersection.t;
			unsigned int fid = TriIsec.intersection.prim_id;

			//Caluculate Position
			pm._NextRay.m_Org = pm._CRay.m_Org + (t) * pm._CRay.m_Dir;
			pm._NextRay.m_Dir = pm._CRay.m_Dir.normalized();
			Vec3 pos = pm._NextRay.m_Org;

			//Calculate Normal Vector
			float3 nmtem[3];
			for (int j = 0; j < 3; j++){
				nmtem[j][0] = mesh.facevarying_normals[9 * fid + 3 * j + 0];
				nmtem[j][1] = mesh.facevarying_normals[9 * fid + 3 * j + 1];
				nmtem[j][2] = mesh.facevarying_normals[9 * fid + 3 * j + 2];
			}
			As(pm._Normal, (1.0 - u - v) * nmtem[0] + u * nmtem[1] + v * nmtem[2]);
			pm._Normal.normalize();

			//Normal Vector that    NNormalERay.Dir > 0
			Vec3 NNormal = (pm._Normal.dot(pm._CRay.m_Dir) > 0) ? pm._Normal : -pm._Normal;

			//Scattering coefficient and Absorb coefficient
			Vec3 Ss = mat._Sigma_S;
			Vec3 St = (mat._Sigma_S + mat._Sigma_A);

			//PDF for the distance
			float pdf_dist = 1.0;

			//sample channnel
			int channel = MT.genrand64_real2() * 2.99999999;

			//distance to the next scattering
			float ust = MT.genrand64_real3();
			pdf_dist = -log(ust) / St[channel];
			
			if (pm._Stat == PTParam::Ray_Stat::INSIDE_TRANSLUCENT){
				//If Ray in the object

				if (pdf_dist > t){
					//Ray may exit from the object

					Vec3 Reflec = Reflect(pm._CRay.m_Dir, -NNormal).normalized();
					Vec3 Refrac = Refract(pm._CRay.m_Dir, -NNormal, 1.0f / pm._Index).normalized();
					Vec3 H = (Reflec - Refrac).normalized();
					float ReflectRatio = 1.0 - fresnel_schlick(H, NNormal, 1.0f / pm._Index);
					ReflectRatio = std::max(std::min(1.0f, ReflectRatio), 0.0f);

					if (MT.genrand64_real1() > ReflectRatio){
						//Ray exit from the objects
						pm._Stat = PTParam::Ray_Stat::OUTSIDE_TRANSLUCENT;
						SSStat = 1;
						pm._Index = 1.0f;

						//exit Ray
						pm._NextRay.m_Org += pm._NextRay.m_Dir * 0.001;

						////transmittance
						Vec3 Tr = Vec3(
							exp(-St[0] * t),
							exp(-St[1] * t),
							exp(-St[2] * t));

						float pp = 0.0f;
						for (int i = 0; i < 3; i++){
							pp += Tr[i];
						}
						pp /= 3.0f;

						pm._Weight.MultVec(Tr);
						pm._Weight /= pp;
						pm._Weight /= rus_pdf;

					}
					else{
						//Ray is reflected at the boundery and remains in the objects
						pm._Stat = PTParam::Ray_Stat::INSIDE_TRANSLUCENT;
						SSStat = -1;
						pm._Index = pm._Index;

						pm._NextRay.m_Dir = Reflec;
						pm._NextRay.m_Org = pos + pm._NextRay.m_Dir * 0.001;


					}
				}
				else{
					//Ray is scattered and remains in the objects
					pm._Stat = PTParam::Ray_Stat::INSIDE_TRANSLUCENT;
					SSStat = -2;

					//Ray is scattered
					pm._NextRay.m_Org = pm._CRay.m_Org + pm._CRay.m_Dir * pdf_dist;

					{
						//Next Event Estimation
						Vec3 Nweight = NEESSS(pm._NextRay, pm, mat, LightPos);
						pm._Color += Vec3::MultVec(Nweight, pm._Weight) * 10.0f;
					}


					////transmittance
					Vec3 Tr = Vec3(
						exp(-St[0] * pdf_dist),
						exp(-St[1] * pdf_dist),
						exp(-St[2] * pdf_dist));

					float pp = 0.0f;
					for (int i = 0; i < 3; i++){
						pp += St[i] * Tr[i];
					}
					pp /= 3.0f;
					pm._Weight.MultVec(Tr);
					pm._Weight.MultVec(Ss);
					pm._Weight /= pp;
					pm._Weight /= rus_pdf;

					float pdf = 0.0f;
					////Sample Direction
					pdf = Sample::Sphere(pm._NextRay.m_Dir, MT);
					float phase = mat.HG_Phase(pm._CRay.m_Dir, pm._NextRay.m_Dir, mat._g);
					pm._Weight *= phase;
					pm._Weight /= (pdf);
					

				}
			}
			else if (pm._Stat == PTParam::Ray_Stat::OUTSIDE_TRANSLUCENT){
				//If Ray outside the object


				Vec3 Reflec = Reflect(pm._CRay.m_Dir, -NNormal).normalized();
				Vec3 Refrac = Refract(pm._CRay.m_Dir, -NNormal, pm._Index).normalized();
				Vec3 H = (Reflec - Refrac).normalized();
				float ReflectRatio = 1.0 - fresnel_schlick(H, NNormal, pm._Index);
				ReflectRatio = std::max(std::min(1.0f, ReflectRatio), 0.0f);

				if (ReflectRatio < MT.genrand64_real1()){
					//Ray enter the objects
					pm._Stat = PTParam::Ray_Stat::INSIDE_TRANSLUCENT;
					SSStat = -3;
					pm._Index = mat._Index;

					//Ray Inter the object without interaction
					//Calculate flux
					pm._NextRay = pm._CRay;
					pm._NextRay.m_Org = pos + pm._NextRay.m_Dir * 0.00001;

					{
						//next event estimation
						const Vec3& St = (mat._Sigma_A + mat._Sigma_S);
						float LightLen = (pm._NextRay.m_Org - LightPos).norm();
						Vec3 RL(exp(-St[0] * LightLen), exp(-St[1] * LightLen), exp(-St[2] * LightLen));
						RL.MultVec(pm._Weight);
						pm._Color += RL / (LightLen * LightLen) * 10.0f;
					}

					Sample::Sphere(pm._NextRay.m_Dir, MT);
					if (pm._NextRay.m_Dir.z() < 0.0001){
						pm._IfLoop = false;
						pm._Color = Vec3::Zero();
					}
					pm._Weight *= (4.0f * M_PI);

				}
				else{
					//Ray is reflected at the boundery and remains outside the objects
					pm._Stat = PTParam::Ray_Stat::OUTSIDE_TRANSLUCENT;
					SSStat = 3;
					pm._Index = 1.0f;

					pm._NextRay = pm._CRay;
					pm._NextRay.m_Dir = Reflec;
					pm._NextRay.m_Org = pos + pm._NextRay.m_Dir * 0.00001;


				}
			}
		}
		else{
			//BackGround Color
			if (pm._Depth == 1){
				pm._Color = Vec3(0.28, 0.08, 0.08);
			}
			pm._IfLoop = false;
			SSStat = 0;
		}

		pm._CRay = pm._NextRay;


		return Vec3();
	}

public:

	


	int RenderImage(
		const CameraInfo& camera, 
		const Mesh& mesh, 
		const nanort::BVHAccel<float, nanort::TriangleMesh<float>, nanort::TriangleSAHPred<float>, nanort::TriangleIntersector<>>& scene,
		SSSParam& mat, 
		int NumSample){

		nanort::BVHTraceOptions traceOPT;
		traceOPT.cull_back_face = false;
		nanort::TriangleIntersector<> triIsc(&mesh.vertices[0], &mesh.faces[0], sizeof(float) * 3);


#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
		for (int i = 0; i < m_ImageX*m_ImageY; ++i){

			int IIX = i % m_ImageX;
			int IIY = i / m_ImageY;

			//pixel 
			int imX = i % m_ImageX;
			int imY = i / m_ImageX;

			//world position
			Vec3 Dir = (camera.GeneratePos(IIX / (float)m_ImageX, IIY / (float)m_ImageY) - camera._Pos).normalized();
			Vec3 Org = camera._Pos;
			Vec3 Color(0, 0, 0);

			const int SAMPLE = NumSample;
			for (int s = 0; s < SAMPLE; s++){

				PTParam pm;
				pm._Stat = PTParam::Ray_Stat::OUTSIDE_TRANSLUCENT;
				pm._NextEventEstimation = false;
				pm._CRay = Ray(Org, Dir);
				pm._Color = Vec3(0, 0, 0);
				RandomMT MT(i * 129 + s * 100022 + 1973);

				while (pm._IfLoop){
					trace(mesh, scene, triIsc, traceOPT, pm, mat, MT);
				}
				Color += pm._Color;
			}
			Color /= (float)SAMPLE;

			m_Image[i * 4 + 0] = 255 * Color[0];
			m_Image[i * 4 + 1] = 255 * Color[1];
			m_Image[i * 4 + 2] = 255 * Color[2];
			m_Image[i * 4 + 3] = 255;


		}
		return 1;
	}


	VolumePathTrace(int ImageX, int ImageY, int Nsample) : m_ImageX(ImageX), m_ImageY(ImageY){
		m_Image = new int[ImageX*ImageY * 4];
	}
	~VolumePathTrace(){
		delete[] m_Image;
	}


	const int* const GetImage()const{
		return m_Image;
	}


};





int main(){

	//Create Scene
	nanort::BVHBuildOptions<float> options;
	std::vector<tinyobj::material_t> MatL;
	Mesh Scene;
	LoadObj(Scene, MatL, "OBJ\\sphere.obj", 1.0f, "OBJ\\cube.mtl");

	nanort::TriangleMesh<float> TriMesh(&Scene.vertices[0], &Scene.faces[0], sizeof(float) * 3);
	nanort::TriangleSAHPred<float> TriPred(&Scene.vertices[0], &Scene.faces[0], sizeof(float) * 3);

	nanort::BVHAccel<float, nanort::TriangleMesh<float>, nanort::TriangleSAHPred<float>, nanort::TriangleIntersector<>> acc;
	bool ret = acc.Build(Scene.num_faces, options, TriMesh, TriPred);
	assert(ret);

	nanort::BVHBuildStatistics stat = acc.GetStatistics();

	printf("  BVH statistics:\n");
	printf("    # of leaf   nodes: %d\n", stat.num_leaf_nodes);
	printf("    # of branch nodes: %d\n", stat.num_branch_nodes);
	printf("  Max tree depth   : %d\n", stat.max_tree_depth);


	//Ray Tracing

	CameraInfo camera(Vec3(0, 0, -20), 1.0f, 1.0f, Vec3(0, 0, 0), 18.0f);

	const int WD = 512;
	const int HD = 512;
	char image[WD * HD * 4];

	srand(time(NULL));
	

	VolumePathTrace::SSSParam mat(Vec3(2.00, 2.00, 2.00) * 2.0, Vec3(18.0, 18.0, 18.0) * 2.0, 0.00001, 1.5);
	VolumePathTrace VoLPath(WD,HD, 1);
	VoLPath.RenderImage(camera, Scene, acc, mat, 1);
	for (int i = 0; i < WD * HD * 4; i++){
		int ig = VoLPath.GetImage()[i];
		image[i] = std::max(0, std::min(255, ig));
	}

	/*for (int i = 0; i < WD; i++){
		for (int j = 0; j < HD; j++){

			nanort::Ray<float> ray;
			ray.min_t = 0.0f;
			ray.max_t = 375000.0f;

			Vec3 Dir = (camera.GeneratePos(i / (float)WD, j / (float)HD) - camera._Pos).normalized();
			Vec3 Org = camera._Pos;

			ray.org[0] = Org.x();
			ray.org[1] = Org.y();
			ray.org[2] = Org.z();

			ray.dir[0] = Dir.x();
			ray.dir[1] = Dir.y();
			ray.dir[2] = Dir.z();

			nanort::BVHTraceOptions traceOPT;
			nanort::TriangleIntersector<> triIsc(&Scene.vertices[0], &Scene.faces[0], sizeof(float) * 3);
			bool Hit = acc.Traverse(ray, traceOPT, triIsc);

			if (Hit){
				image[(i* HD + j) * 4 + 0] = 255;
				image[(i* HD + j) * 4 + 1] = 255;
				image[(i* HD + j) * 4 + 2] = 255;
				image[(i* HD + j) * 4 + 3] = 255;
			}
			else{
				image[(i* HD + j) * 4 + 0] = 200;
				image[(i* HD + j) * 4 + 1] = 0;
				image[(i* HD + j) * 4 + 2] = 210;
				image[(i* HD + j) * 4 + 3] = 255;
			}
		}
	}*/

	stbi_write_bmp("result.bmp", WD, HD, 4, image);

	return  0;
}


