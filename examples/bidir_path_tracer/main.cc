#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>

#define NOMINMAX
#include "tiny_obj_loader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include "nanort.h"

static const float kEps = 0.001f;
static const float kInf = 1.0e30f;
static const float kPi  = 4.0f * std::atan(1.0f);

static const int uMaxBounces = 8;
static const int SPP = 100;

typedef nanort::BVHAccel<nanort::TriangleMesh, nanort::TriangleSAHPred, nanort::TriangleIntersector<> > Accel;

namespace {
    
#if __cplusplus > 199711L
#include <chrono>
typedef std::chrono::time_point<std::chrono::system_clock> TimeType;
inline TimeType tick() {
    return std::chrono::sytem_clock::now();
}

inline double to_duration(TypeType start, TimeType end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100.0;
}
#else
typedef clock_t TimeType;
inline TimeType tick() {
    return clock();
}
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
#define Assertion(PREDICATE, ...) \
do { \
    if (!(PREDICATE)) { \
        std::cerr << "Asssertion \"" \
        << #PREDICATE << "\" failed in " << __FILE__ \
        << " line " << __LINE__ \
        << " in function \"" << (__FUNCTION_NAME__) << "\"" \
        << " : "; \
        fprintf(stderr, __VA_ARGS__); \
        std::cerr << std::endl; \
        std::abort(); \
    } \
} while (false)
#else  // NDEBUG
#define Assertion(PREDICATE, ...) do {} while (false)
#endif // NDEBUG


// ----------------------------------------------------------------------------
// Timer class
// ----------------------------------------------------------------------------

class Timer {
public:
    Timer()
        : start_()
        , end_() {
    }

    void start() {
        start_ = tick();
    }

    double stop() {
        end_ = tick();
        return to_duration(start_, end_);
    }

private:
    TimeType start_, end_;
};


// ----------------------------------------------------------------------------
// Random number generator (XOR shift)
// ----------------------------------------------------------------------------

class Random {
public:
    unsigned int nextInt() {
        unsigned int ret = 0u;

        const unsigned int t = seed_[0] ^ (seed_[0] << 11);
        seed_[0] = seed_[1];
        seed_[1] = seed_[2];
        seed_[2] = seed_[3];
        return seed_[3] = (seed_[3] ^ (seed_[3] >> 19)) ^ (t ^ (t >> 8));
    }

    unsigned int nextInt(int n) {
        return nextInt() % n;
    }

    float nextReal() {
        return (float)nextInt() / (float)UINT_MAX;
    }

    Random(unsigned int initial_seed) {
        unsigned int s = initial_seed;
        for (int i = 1; i <= 4; i++){
            seed_[i-1] = s = 1812433253U * (s^(s>>30)) + i;
        }
    }

private:
    unsigned int seed_[4];
};

//static Random rnd((unsigned long)time(0));


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
    float3 operator-() const {
        return float3(-x, -y, -z);
    }
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

    bool isBlack() const {
        return x == 0.0f && y == 0.0f && z == 0.0f;
    }

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
        
    float3 xDir = fabsf(normal.x) < fabsf(normal.y) ? float3(1, 0, 0) : float3(0, 1, 0);
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
    float *vertices;              /// [xyz] * num_vertices
    float *facevarying_normals;   /// [xyz] * 3(triangle) * num_faces
    float *facevarying_tangents;  /// [xyz] * 3(triangle) * num_faces
    float *facevarying_binormals; /// [xyz] * 3(triangle) * num_faces
    float *facevarying_uvs;       /// [xyz] * 3(triangle) * num_faces
    float *facevarying_vertex_colors;   /// [xyz] * 3(triangle) * num_faces
    unsigned int *faces;         /// triangle x num_faces
    unsigned int *material_ids;   /// index x num_faces
} Mesh;

void calcNormal(float3& N, float3 v0, float3 v1, float3 v2)
{
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
    int normal_texid;     // normal map
    int alpha_texid;      // alpha map
        
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
        

// ----------------------------------------------------------------------------
// Image utilities
// ----------------------------------------------------------------------------

void SaveImagePNG(const char *filename, const float *rgb, int width, int height) {
    unsigned char *bytes = new unsigned char[width * height * 3];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int index = y * width + x;
            bytes[index * 3 + 0] = (unsigned char)std::max(0.0f, std::min(rgb[index * 3 + 0] * 255.0f, 255.0f));          
            bytes[index * 3 + 1] = (unsigned char)std::max(0.0f, std::min(rgb[index * 3 + 1] * 255.0f, 255.0f));          
            bytes[index * 3 + 2] = (unsigned char)std::max(0.0f, std::min(rgb[index * 3 + 2] * 255.0f, 255.0f));          
        }
    }
    stbi_write_png(filename, width, height, 3, bytes, width * 3);
    delete[] bytes;
}
    
void SaveImageEXR(const char *filename, const float *rgb, int width, int height) {
    float* image_ptr[3];
    std::vector<float> images[3];
    images[0].resize(width * height);
    images[1].resize(width * height);
    images[2].resize(width * height);
        
    for (int i = 0; i < width * height; i++) {
        images[0][i] = rgb[3*i+0];
        images[1][i] = rgb[3*i+1];
        images[2][i] = rgb[3*i+2];
    }
        
    image_ptr[0] = &(images[2].at(0)); // B
    image_ptr[1] = &(images[1].at(0)); // G
    image_ptr[2] = &(images[0].at(0)); // R
        
    EXRImage image;
    InitEXRImage(&image);
        
    image.num_channels = 3;
    const char* channel_names[] = {"B", "G", "R"}; // must be BGR order.
        
    image.channel_names = channel_names;
    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;
        
    image.pixel_types = (int *)malloc(sizeof(int) * image.num_channels);
    image.requested_pixel_types = (int *)malloc(sizeof(int) * image.num_channels);
    for (int i = 0; i < image.num_channels; i++) {
        image.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
        image.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
    }
        
    const char* err;
    int fail = SaveMultiChannelEXRToFile(&image, filename, &err);
    if (fail) {
        fprintf(stderr, "Error: %s\n", err);
    } else {
        printf("Saved image to [ %s ]\n", filename);
    }
        
    free(image.pixel_types);
    free(image.requested_pixel_types);
}

//! OBJ file loader.    
bool LoadObj(Mesh &mesh, std::vector<tinyobj::material_t>& materials, const char *filename, float scale) {
    std::vector<tinyobj::shape_t> shapes;
    std::string err;
    bool success = tinyobj::LoadObj(shapes, materials, err, filename);
        
    if (!err.empty()) {
        std::cerr << err << std::endl;
    }
        
    if (!success) {
        return false;
    }
        
    std::cout << "[LoadOBJ] # of shapes in .obj : " << shapes.size() << std::endl;
    std::cout << "[LoadOBJ] # of materials in .obj : " << materials.size() << std::endl;
        
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
                    
                f0 = shapes[i].mesh.indices[3*f+0];
                f1 = shapes[i].mesh.indices[3*f+1];
                f2 = shapes[i].mesh.indices[3*f+2];
                    
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
        } else {
            // calc geometric normal
            for (size_t f = 0; f < shapes[i].mesh.indices.size() / 3; f++) {
                int f0, f1, f2;
                    
                f0 = shapes[i].mesh.indices[3*f+0];
                f1 = shapes[i].mesh.indices[3*f+1];
                f2 = shapes[i].mesh.indices[3*f+2];
                    
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
                    
                f0 = shapes[i].mesh.indices[3*f+0];
                f1 = shapes[i].mesh.indices[3*f+1];
                f2 = shapes[i].mesh.indices[3*f+2];
                    
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
    
} // namespace

inline float sign(float f) {
    return f < 0 ? -1 : 1;
}

inline float3 reflect(float3 I , float3 N) {
    return I - 2*vdot(I, N) * N;
}

inline float3 refract(float3 I, float3 N, float eta) {
    float NdotI = vdot(N, I);
    float k = 1.0f - eta * eta * (1.0f - NdotI * NdotI);
    if (k < 0.0f)
        return float3(0, 0, 0);
    else
        return eta * I - (eta * NdotI + sqrtf(k)) * N;
}

inline float pow5(float val) {
    return val * val * val * val * val;
}

inline float fresnel_schlick(float3 H, float3 norm, float n1) {
    float r0 = n1 * n1;
    return r0 + (1-r0)*pow5(1 - vdot(H, norm));
}

bool isLight(const tinyobj::material_t &mat) {
    float3 Le(mat.emission);
    return Le[0] != 0.0f || Le[1] != 0.0f || Le[2] != 0.0f;
}

// ----------------------------------------------------------------------------
// Utilities for BDPT
// ----------------------------------------------------------------------------

enum VertexType {
    Light,
    Lens,
    Surface
};
    
struct Vertex {
    float3 position;
    float3 originalNorm;
    float3 norm;
    float3 beta;
    float3 wo;
    float pdfFwd;
    float pdfRev;
    VertexType type;
    tinyobj::material_t mat;

    bool isDelta() const {
        float3 Ks(mat.specular);
        float3 Kt(mat.transmittance);
        float alpha = mat.shininess;
        //if (mat.shininess != 0.0f) return false;
        if (Ks[0] != 0.0f || Ks[1] != 0.0f || Ks[2] != 0.0f) return true;
        if (Kt[0] != 0.0f || Kt[1] != 0.0f || Kt[2] != 0.0f) return true;
        return false;    
    }
    
    float3 f(const Vertex &v) const {
        float3 wi = v.position - position;
        bool reflect = vdot(wi, norm) * vdot(wo, norm) > 0.0f;

        float3 diffuseColor(mat.diffuse);
        float3 specularColor(mat.specular);
        float3 refractionColor(mat.transmittance);
        float ior = mat.ior;
        
        // Calculate fresnel factor based on ior.
        float inside = sign(vdot(-wo, originalNorm));
        float n1 = inside < 0 ? 1.0 / ior : ior;
        float n2 = 1.0 / n1;
        
        float fresnel = fresnel_schlick(wo, norm, (n1-n2)/(n1+n2));
        
        float rhoS = vdot(float3(1, 1, 1)/3.0f, specularColor) * fresnel;
        float rhoD = vdot(float3(1, 1, 1)/3.0f, diffuseColor) * (1.0 - fresnel) * (1.0 - mat.dissolve);
        float rhoR = vdot(float3(1, 1, 1)/3.0f, refractionColor) * (1.0 - fresnel) * mat.dissolve;

        // Normalize probabilities so they sum to 1.0
        float totalrho = rhoS + rhoD + rhoR;
        // No scattering event is likely, just stop here
        if(totalrho < 0.0001f) {
            return float3(0.0f, 0.0f, 0.0f);
        }
        
        rhoS /= totalrho;
        rhoD /= totalrho;
        rhoR /= totalrho;

        float3 ret(0.0f, 0.0f, 0.0f);
        float weight = 0.0f;
        if (rhoS > 0.0f && reflect) {
            ret += rhoS * float3(0.0f, 0.0f, 0.0f);
            weight += rhoS;
        }

        if (rhoD > 0.0f && reflect) {
            ret += rhoD * diffuseColor / kPi;
            weight += rhoD;
        }
        
        if (rhoR > 0.0f && !reflect) {
            ret += rhoR * float3(0.0f, 0.0f, 0.0f);
            weight += rhoR;
        }

        if (weight != 0.0f) {
            ret = ret / weight;
        }

        return ret;
    }
};
    
class LightSampler {
public:
    LightSampler(const Mesh &mesh, const std::vector<tinyobj::material_t> &materials) {
        // Collect light mesh.
        std::vector<std::pair<float, int> > temp;
        for (int i = 0; i < mesh.num_faces; i++) {
            int material_id = mesh.material_ids[i];
            const float3 Le(materials[material_id].emission);
            if (std::max(Le[0], std::max(Le[1], Le[2])) <= kEps) continue;
                
            float3 v[3];
            for (int k = 0; k < 3; k++) {
                float x = mesh.vertices[mesh.faces[i * 3 + k] * 3 + 0];
                float y = mesh.vertices[mesh.faces[i * 3 + k] * 3 + 1];
                float z = mesh.vertices[mesh.faces[i * 3 + k] * 3 + 2];
                v[k] = float3(x, y, z);
            }

            float area = 0.5f * vcross(v[2] - v[0], v[1] - v[0]).length();
            temp.push_back(std::make_pair(area, i));
            totalArea_ += area;
        }
        std::sort(temp.begin(), temp.end());

        // Compute cumulative density function.
        numLights_ = temp.size();
        cdf_.resize(numLights_);
        ids_.resize(numLights_);
            
        cdf_[0] = temp[0].first / totalArea_;
        ids_[0] = temp[0].second;
        for (int i = 1; i < numLights_; i++) {
            cdf_[i] = cdf_[i - 1] + temp[i].first / totalArea_;
            ids_[i] = temp[i].second;
        }
    }
        
    int sample(const Mesh &mesh, Random &rng, float3 *pos, float3 *norm, float *pdf) const {
        // Sample light with binary search.
        float rand = rng.nextReal();
        int sampleId = std::lower_bound(cdf_.begin(), cdf_.end(), rand) - cdf_.begin();
        sampleId = std::min(sampleId, (int)ids_.size() - 1);
        int lightId = ids_[sampleId];
            
        // Sample light position.
        float u1 = rng.nextReal();
        float u2 = rng.nextReal();
        if (u1 + u2 >= 1.0f) {
            u1 = 1.0f - u1;
            u2 = 1.0f - u2;
        }
            
        float3 v[3], n[3];
        for (int k = 0; k < 3; k++) {
            float x = mesh.vertices[mesh.faces[lightId * 3 + k] * 3 + 0];
            float y = mesh.vertices[mesh.faces[lightId * 3 + k] * 3 + 1];
            float z = mesh.vertices[mesh.faces[lightId * 3 + k] * 3 + 2];
            v[k] = float3(x, y, z);
            float nx = mesh.facevarying_normals[lightId * 9 + k * 3 + 0];
            float ny = mesh.facevarying_normals[lightId * 9 + k * 3 + 1];
            float nz = mesh.facevarying_normals[lightId * 9 + k * 3 + 2];
            n[k] = float3(nx, ny, nz);
        }
            
        *pos  = (1.0f - u1 - u2) * v[0] + u1 * v[1] + u2 * v[2];
        *norm = (1.0f - u1 - u2) * n[0] + u1 * n[1] + u2 * n[2];
        *pdf = 1.0f / totalArea_;
            
        return lightId;
    }

    inline float totalArea() const { return totalArea_; }
        
private:
    std::vector<float> cdf_;
    std::vector<unsigned int> ids_;
    float totalArea_;
    int numLights_;
};

float3 sampleBRDF(const tinyobj::material_t &mat, const float3 &wo,
                  const float3 &origNorm, const float3 &norm, Random &rng, float3 *wi, float *pdf) {
    float3 diffuseColor(mat.diffuse);
    float3 specularColor(mat.specular);
    float3 refractionColor(mat.transmittance);
    float ior = mat.ior;
        
    // Calculate fresnel factor based on ior.
    float inside = sign(vdot(-wo, origNorm)); // 1 for inside, -1 for outside
    // Assume ior of medium outside of objects = 1.0
    float n1 = inside < 0 ? 1.0 / ior : ior;
    float n2 = 1.0 / n1;
        
    float fresnel = fresnel_schlick(wo, norm, (n1-n2)/(n1+n2));
        
    float rhoS = vdot(float3(1, 1, 1)/3.0f, specularColor) * fresnel;
    float rhoD = vdot(float3(1, 1, 1)/3.0f, diffuseColor) * (1.0 - fresnel) * (1.0 - mat.dissolve);
    float rhoR = vdot(float3(1, 1, 1)/3.0f, refractionColor) * (1.0 - fresnel) * mat.dissolve;
                
    // Normalize probabilities so they sum to 1.0
    float totalrho = rhoS + rhoD + rhoR;
    // No scattering event is likely, just stop here
    if(totalrho < 0.0001f) {
        *pdf = 0.0f;
        return float3(0.0f, 0.0f, 0.0f);
    }
        
    rhoS /= totalrho;
    rhoD /= totalrho;
    rhoR /= totalrho;
        
    // Choose an interaction based on the calculated probabilities
    float3 f(1.0f, 1.0f, 1.0f);
    float rand = rng.nextReal();
    *pdf = 1.0f;
    if (rand < rhoS) {
        // Specular reflection
        *wi = reflect(-wo, norm);
        *pdf = rhoS;
        float cosTheta = std::abs(vdot(*wi, norm));
        f = rhoS * specularColor / cosTheta;
    } else if (rand < rhoS + rhoD) {
        // Sample cosine weighted hemisphere
        *wi = directionCosTheta(norm, rng.nextReal(), rng.nextReal(), pdf);
        *pdf *= rhoD;
        f = rhoD * diffuseColor / kPi;
    } else if (rand < rhoD + rhoS + rhoR) {
        *wi = refract(-wo, -inside * origNorm, n1);
        *pdf = rhoR;
        float cosTheta = std::abs(vdot(*wi, norm));
        f *= rhoR * refractionColor / cosTheta;
    }

    return f;
}

float pdfBRDF(const tinyobj::material_t &mat, const float3 &wi,
              const float3 &wo, const float3 &origNorm, const float3 &norm) {
    bool reflect = vdot(wi, norm) * vdot(wo, norm) > 0.0f;

    float3 diffuseColor(mat.diffuse);
    float3 specularColor(mat.specular);
    float3 refractionColor(mat.transmittance);
    float ior = mat.ior;
        
    // Calculate fresnel factor based on ior.
    float inside = sign(vdot(-wo, origNorm)); // 1 for inside, -1 for outside
    // Assume ior of medium outside of objects = 1.0
    float n1 = inside < 0 ? 1.0 / ior : ior;
    float n2 = 1.0 / n1;
        
    float fresnel = fresnel_schlick(wo, norm, (n1-n2)/(n1+n2));
        
    float rhoS = vdot(float3(1, 1, 1)/3.0f, specularColor) * fresnel;
    float rhoD = vdot(float3(1, 1, 1)/3.0f, diffuseColor) * (1.0 - fresnel) * (1.0 - mat.dissolve);
    float rhoR = vdot(float3(1, 1, 1)/3.0f, refractionColor) * (1.0 - fresnel) * mat.dissolve;

    // Normalize probabilities so they sum to 1.0
    float totalrho = rhoS + rhoD + rhoR;
    if(totalrho < 0.0001f) {
        return 0.0f;
    }
        
    rhoS /= totalrho;
    rhoD /= totalrho;
    rhoR /= totalrho;

    float pdf = 0.0f;
    if (rhoS > 0.0f && reflect) {
        pdf += 0.0f;
    }

    if (rhoD > 0.0f && reflect) {
        pdf += rhoD * std::abs(vdot(wi, norm)) / kPi;
    }

    if (rhoR > 0.0f && !reflect) {
        pdf += 0.0f;
    }

    return pdf;
}

void progressBar(int tick, int total, int width = 50) {
    float ratio = 100.0f * tick / total;
    float count = width * tick / total;
    std::string bar(width, ' ');
    std::fill(bar.begin(), bar.begin() + count, '+');
    printf("[ %6.2f %% ] [ %s ]%c", ratio, bar.c_str(), tick == total ? '\n' : '\r');
    std::fflush(stdout);
}

void raytrace(const Mesh &mesh, const Accel &accel, const std::vector<tinyobj::material_t> &materials,
              Random &rng, float3 rayOrg, float3 rayDir, float3 beta, float pdf,
              std::vector<Vertex> *vertices, bool isEyePath) {
    float pdfFwd = pdf, pdfRev = 0.0f;
    for (int b = 0; b < uMaxBounces; ++b) {
        // Intersection test.
        nanort::Ray ray;
        ray.min_t = kEps;
        ray.max_t = kInf;
        
        ray.dir[0] = rayDir[0]; ray.dir[1] = rayDir[1]; ray.dir[2] = rayDir[2];
        ray.org[0] = rayOrg[0]; ray.org[1] = rayOrg[1]; ray.org[2] = rayOrg[2];
        
        nanort::TriangleIntersector<> triangle_intersector(mesh.vertices, mesh.faces);
        nanort::BVHTraceOptions trace_options;
        bool hit = accel.Traverse(ray, trace_options, triangle_intersector);
        
        if (!hit) {
            break;
        }

        // Add a new vertex.
        vertices->push_back(Vertex());
        Vertex& vertex = vertices->at(vertices->size() - 1);
        Vertex& prev   = vertices->at(vertices->size() - 2);
        float3 nextPos = rayOrg + triangle_intersector.intersection.t * rayDir;
                
        // Shading normal
        unsigned int fid = triangle_intersector.intersection.prim_id;
        float3 norm(0,0,0);
        if (mesh.facevarying_normals) {
            float3 normals[3];
            for(int vId = 0; vId < 3; vId++) {
                normals[vId][0] = mesh.facevarying_normals[9*fid + 3*vId + 0];
                normals[vId][1] = mesh.facevarying_normals[9*fid + 3*vId + 1];
                normals[vId][2] = mesh.facevarying_normals[9*fid + 3*vId + 2];
            }
            float u =  triangle_intersector.intersection.u;
            float v =  triangle_intersector.intersection.v;
            norm = (1.0 - u - v) * normals[0] + u  * normals[1] + v * normals[2];
            norm.normalize();
        }
        
        // Flip normal torwards incoming ray for backface shading
        float3 originalNorm = norm;
        if(vdot(norm, rayDir) > 0) {
            norm *= -1;
        }
        
        // Get properties from the material of the hit primitive
        unsigned int matId = mesh.material_ids[fid];
        const tinyobj::material_t &mat = materials[matId];

        if (isLight(mat)) {
            if (!isEyePath) {
                vertices->pop_back();
                break;
            }

            // Store vertex on light.
            float3 Le(mat.emission);
            beta = beta * Le * std::max(0.0f, vdot(originalNorm, -rayDir));

            vertex.position = nextPos;
            vertex.originalNorm = originalNorm;
            vertex.norm = norm;
            vertex.wo = normalize(-rayDir);
            vertex.beta = beta;
            vertex.mat = mat;
            vertex.pdfFwd = pdfFwd;
            vertex.pdfRev = 0.0f;
            vertex.type = Light;
        } else {
            // Store vertex on surface.
            vertex.position = nextPos;
            vertex.originalNorm = originalNorm;
            vertex.norm = norm;
            vertex.wo = normalize(-rayDir);
            vertex.beta = beta;
            vertex.mat = mat;
            vertex.pdfFwd = pdfFwd;
            vertex.pdfRev = 0.0f;
            vertex.type = Surface;
        }

        // Convert measure of pdfFwd.
        float3 to  = vertex.position - prev.position;
        float dist = to.length();
        to = to / dist;
        vertex.pdfFwd *= vdot(to, prev.norm) / (dist * dist);

        if (vertex.type == Light) break;

        float3 outDir;
        float3 f = sampleBRDF(mat, -rayDir, originalNorm, norm, rng, &outDir, &pdfFwd);
        if (pdfFwd == 0.0f) break;

        beta = f * beta * std::abs(vdot(norm, outDir)) / pdfFwd;
        if (beta.isBlack()) break;

        pdfRev = pdfBRDF(mat, outDir, -rayDir, originalNorm, norm);
        
        // Calculate new ray start position and set outgoing direction.
        rayOrg = nextPos;
        rayDir = outDir;

        prev.pdfRev = pdfRev * std::abs(vdot(-to, vertex.norm)) / (dist * dist);
    }
}

void eyeSubpath(int x, int y, int width, int height,
               const Mesh &mesh, const std::vector<tinyobj::material_t> &materials,
               const Accel &accel, Random &rng, std::vector<Vertex> *eyeVert) {
    // Clear vertices.
    eyeVert->clear();
    
    // Simple camera. change eye pos and direction fit to .obj model.
    float px = x + (rng.nextReal() - 0.5f);
    float py = y + (rng.nextReal() - 0.5f);
    float3 rayDir = float3((px / (float)width) - 0.5f, (py / (float)height) - 0.5f, -1.0f);
    rayDir.normalize();
    float3 rayOrg = float3(0.0f, 5.0f, 20.0f);
    
    // Store first vertex.
    Vertex vertex;
    vertex.position = rayOrg;
    vertex.norm = rayDir;
    vertex.beta = float3(1.0f, 1.0f, 1.0f);
    vertex.pdfFwd = 1.0f;
    vertex.type = Lens;
    eyeVert->push_back(vertex);
    
    float3 beta = float3(1.0f, 1.0f, 1.0f);
    float pdf = 1.0f;
    
    raytrace(mesh, accel, materials, rng, rayOrg, rayDir, beta, pdf, eyeVert, true);
}

void lightSubpath(const Mesh &mesh, const std::vector<tinyobj::material_t> &materials, const Accel &accel,
                  Random &rng, const LightSampler &lights, std::vector<Vertex> *lightVert) {
    // Clear vertices.
    lightVert->clear();
    
    // Sample light source.
    float3 rayOrg, norm;
    float pdfPos;
    int lightId = lights.sample(mesh, rng, &rayOrg, &norm, &pdfPos);
    float3 Le(materials[mesh.material_ids[lightId]].emission);
    
    // Sample light direction.
    float pdfDir;
    float3 rayDir = directionCosTheta(norm, rng.nextReal(), rng.nextReal(), &pdfDir);

    // Store vertex on light.
    Vertex vertex;
    vertex.position = rayOrg;
    vertex.norm = normalize(norm);
    vertex.beta = Le / pdfPos;
    vertex.pdfFwd = pdfPos;
    vertex.type = Light;
    lightVert->push_back(vertex);
    
    // Light tracing.
    float3 beta = Le / pdfPos;
    std::vector<Vertex> vertices;
    double pdf = pdfDir;
    
    raytrace(mesh, accel, materials, rng, rayOrg, rayDir, beta, pdf, lightVert, false);
}

float weightMIS(const LightSampler &light, const std::vector<Vertex> &eyeVert, const std::vector<Vertex> &lightVert,
                int numEyeVert, int numLightVert, const Mesh &mesh, const Accel &accel) {
    if (numEyeVert <= 2 && numLightVert == 0) return 1.0f;

    std::vector<Vertex> path;
    for (int i = 0; i < numEyeVert; i++) {
        path.push_back(eyeVert[i]);
    }
    
    for (int i = numLightVert - 1; i >= 0; i--) {
        path.push_back(lightVert[i]);
    }
    const int pathLen = path.size();

    Vertex *ve = numEyeVert - 1 >= 0 ? &path[numEyeVert - 1] : nullptr;
    Vertex *vl = numEyeVert < pathLen ? &path[numEyeVert] : nullptr;
    Vertex *veMinus = numEyeVert - 2 >= 0 ? &path[numEyeVert - 2] : nullptr;
    Vertex *vlMinus = numEyeVert + 1 < pathLen ? &path[numEyeVert + 1] : nullptr;

    if (ve) {
        if (numLightVert == 0) {
            ve->pdfRev = 1.0f / light.totalArea();
        } else if (numLightVert == 1) {
            float3 to = ve->position - vl->position;
            float dist = to.length();
            to = to / dist;
            float pdfDir = std::max(0.0f, vdot(vl->norm, to));

            float dot = vdot(vl->norm, to);
            ve->pdfRev = pdfDir * dot / (dist * dist);
        } else {
            float3 wi = vlMinus->position - vl->position;
            float3 wo = ve->position - vl->position;
            float dist = wo.length();

            wi.normalize();
            wo.normalize();
            float pdfOmega = pdfBRDF(vl->mat, wi, wo, vl->originalNorm, vl->norm);
            ve->pdfRev = pdfOmega * std::abs(vdot(vl->norm, wo)) / (dist * dist);
        }
    }

    if (vl) {
        if (numEyeVert <= 1) {
            printf("Error!!\n");
            assert(numEyeVert >= 2);
        } else {
            float3 wi = veMinus->position - ve->position;
            float3 wo = vl->position - ve->position;
            float dist = wo.length();

            wi.normalize();
            wo.normalize();
            float pdfOmega = pdfBRDF(ve->mat, wi, wo, ve->originalNorm, ve->norm);
            vl->pdfRev = pdfOmega * std::abs(vdot(ve->norm, wo)) / (dist * dist);
        }
    }

    if (veMinus) {
        if (numLightVert == 0) {
            float3 to = veMinus->position - ve->position;
            float dist = to.length();
            to = to / dist;
            float pdfDir = std::max(0.0f, vdot(ve->norm, to));

            float dot = vdot(ve->norm, to);
            veMinus->pdfRev = pdfDir * dot / (dist * dist);
        } else {
            float3 wi = vl->position - ve->position;
            float3 wo = veMinus->position - ve->position;
            float dist = wo.length();

            wi.normalize();
            wo.normalize();
            float pdfOmega = pdfBRDF(ve->mat, wi, wo, ve->originalNorm, ve->norm);
            veMinus->pdfRev = pdfOmega * std::abs(vdot(ve->norm, wo)) / (dist * dist);
        }
    }

    if (vlMinus) {
        if (numEyeVert <= 1) {
            printf("Error!!\n");
            assert(numEyeVert >= 2);
        } else {
            float3 wi = ve->position - vl->position;
            float3 wo = vlMinus->position - vl->position;
            float dist = wo.length();

            wi.normalize();
            wo.normalize();
            float pdfOmega = pdfBRDF(vl->mat, wi, wo, vl->originalNorm, vl->norm);
            vlMinus->pdfRev = pdfOmega * std::abs(vdot(vl->norm, wo)) / (dist * dist);
        }
    }
    
    float mis = 0.0;
    float prob = 1.0f;
    for (int i = numEyeVert - 1; i >= 2; i--) {
        float pdfFwd = path[i].pdfFwd == 0.0f ? 1.0f : path[i].pdfFwd;
        float pdfRev = path[i].pdfRev == 0.0f ? 1.0f : path[i].pdfRev;
        prob *= pdfRev / pdfFwd;
        
        if (path[i].isDelta() || path[i - 1].isDelta()) continue;
        mis += prob * prob;
    }
    
    prob = 1.0f;
    for (int i = numEyeVert; i < pathLen; i++) {
        float pdfFwd = path[i].pdfFwd == 0.0f ? 1.0f : path[i].pdfFwd;
        float pdfRev = path[i].pdfRev == 0.0f ? 1.0f : path[i].pdfRev;
        prob *= pdfRev / pdfFwd;
        
        if (path[i].isDelta() || (i + 1 < pathLen && path[i + 1].isDelta())) continue;
        mis += prob * prob;
    }

    mis = 1.0f / (1.0f + mis);
    return mis;    
}

float calcG(const Vertex &v1, const Vertex &v2, const Mesh &mesh, const Accel &accel) {
    float3 to = v2.position - v1.position;
    float dist = to.length();
    to = to / dist;
    
    nanort::Ray ray;
    ray.org[0] = v1.position[0];
    ray.org[1] = v1.position[1];
    ray.org[2] = v1.position[2];
    ray.dir[0] = to[0];
    ray.dir[1] = to[1];
    ray.dir[2] = to[2];
    ray.min_t = kEps;
    ray.max_t = kInf;
    
    nanort::TriangleIntersector<> triangle_intersector(mesh.vertices, mesh.faces);
    nanort::BVHTraceOptions trace_options;
    bool hit = accel.Traverse(ray, trace_options, triangle_intersector);
    if (!hit) {
        return 0.0f;
    }
    
    if (std::abs(dist - triangle_intersector.intersection.t) > kEps) {
        return 0.0f;
    }
    
    float dot1 = std::max(0.0f, vdot(to, v1.norm));
    float dot2 = std::max(0.0f, vdot(-to, v2.norm));
    return dot1 * dot2 / (dist * dist);
}

float3 connectPath(const std::vector<Vertex> &eyeVert, const std::vector<Vertex> &lightVert,
                   const Mesh &mesh, const Accel &accel, const LightSampler &light) {
    float3 color(0.0f, 0.0f, 0.0f);
    if (eyeVert[eyeVert.size() - 1].type == Light) {
        float mis = weightMIS(light, eyeVert, lightVert, eyeVert.size(), 0, mesh, accel);
        const Vertex &ev = eyeVert[eyeVert.size() - 1];
        color += mis * ev.beta;
    }

    for (int e = 2; e <= eyeVert.size(); e++) {
        const Vertex &ev = eyeVert[e - 1];
        if (ev.isDelta() || ev.type == Light) {
            continue;
        }
        
        for (int l = 1; l <= lightVert.size(); l++) {
            if (e + l - 2 > uMaxBounces) continue;

            const Vertex &lv = lightVert[l - 1];
            if (l != 1 && lv.isDelta()) continue;
            
            float3 L(0.0f, 0.0f, 0.0f);
            if (l == 1) {
                float3 to = lv.position - ev.position;
                float dist = to.length();
                to = to / dist;
                L = ev.beta * ev.f(lv) * lv.beta * std::abs(vdot(lv.norm, -to));
            } else {
                L = ev.beta * ev.f(lv) * lv.f(ev) * lv.beta;
            }

            if (L[0] == 0.0f && L[1] == 0.0f && L[2] == 0.0f) continue;
            L *= calcG(ev, lv, mesh, accel);

            float mis = weightMIS(light, eyeVert, lightVert, e, l, mesh, accel);
            color += mis * L;
            Assertion(color.isValid(), "Either of color elements takes nan or inf!!");
        }
    }

    return color;
}

int main(int argc, char** argv)
{
    int width = 512;
    int height = 512;
    
    float scale = 1.0f;
    
    std::string objFilename = "cornellbox_suzanne.obj";
    
    if (argc > 1) {
        objFilename = std::string(argv[1]);
    }
    
    if (argc > 2) {
        scale = atof(argv[2]);
    }
    
#ifdef _OPENMP
    printf("Using OpenMP: yes!\n");
#else
    printf("Using OpenMP: no!\n");
#endif
    
    bool ret = false;
    
    // Load scene obj file
    Mesh mesh;
    std::vector<tinyobj::material_t> materials;
    ret = LoadObj(mesh, materials, objFilename.c_str(), scale);
    if (!ret) {
        fprintf(stderr, "Failed to load [ %s ]\n", objFilename.c_str());
        return -1;
    }
    
    nanort::BVHBuildOptions build_options; // Use default option
    build_options.cache_bbox = false;
    
    printf("  BVH build option:\n");
    printf("    # of leaf primitives: %d\n", build_options.min_leaf_primitives);
    printf("    SAH binsize         : %d\n", build_options.bin_size);

    Timer t;
    t.start();
    
    nanort::TriangleMesh triangle_mesh(mesh.vertices, mesh.faces);
    nanort::TriangleSAHPred triangle_pred(mesh.vertices, mesh.faces);
    
    printf("num_triangles = %zu\n", mesh.num_faces);
    printf("faces = %p\n", mesh.faces);
    
    nanort::BVHAccel<nanort::TriangleMesh, nanort::TriangleSAHPred, nanort::TriangleIntersector<> > accel;
    ret = accel.Build(mesh.num_faces, build_options, triangle_mesh, triangle_pred);
    assert(ret);
    
    printf("  BVH build time: %f secs\n", t.stop() / 1000.0);
    
    nanort::BVHBuildStatistics stats = accel.GetStatistics();
    
    printf("  BVH statistics:\n");
    printf("    # of leaf   nodes: %d\n", stats.num_leaf_nodes);
    printf("    # of branch nodes: %d\n", stats.num_branch_nodes);
    printf("  Max tree depth     : %d\n", stats.max_tree_depth);
    float bmin[3], bmax[3];
    accel.BoundingBox(bmin, bmax);
    printf("  Bmin               : %f, %f, %f\n", bmin[0], bmin[1], bmin[2]);
    printf("  Bmax               : %f, %f, %f\n", bmax[0], bmax[1], bmax[2]);
    
    std::vector<float> rgb(width * height * 3, 0.0f);
    
    // Construct light sampler.
    LightSampler lights(mesh, materials);
        
    // Shoot rays.
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float3 finalColor = float3(0, 0, 0);
            for(int i = 0; i < SPP; ++i) {
                unsigned long seed = (y * width + x) * SPP + i;
                Random rng(seed);

                std::vector<Vertex> eyeVert;
                eyeSubpath(x, y, width, height, mesh, materials, accel, rng, &eyeVert);
                if (eyeVert.size() <= 1) continue;
                
                std::vector<Vertex> lightVert;
                lightSubpath(mesh, materials, accel, rng, lights, &lightVert);
                
                finalColor += connectPath(eyeVert, lightVert, mesh, accel, lights);
            }
            finalColor *= 1.0 / SPP;
            
            // Gamme Correct
            finalColor[0] = pow(finalColor[0], 1.0/2.2);
            finalColor[1] = pow(finalColor[1], 1.0/2.2);
            finalColor[2] = pow(finalColor[2], 1.0/2.2);
            
            rgb[3 * ((height - y - 1) * width + x) + 0] = finalColor[0];
            rgb[3 * ((height - y - 1) * width + x) + 1] = finalColor[1];
            rgb[3 * ((height - y - 1) * width + x) + 2] = finalColor[2];            
        }
        progressBar(y + 1, height);
    }
    
    // Save image as EXR.
    SaveImageEXR("render.exr", &rgb.at(0), width, height);

    // Save image as PNG.
    SaveImagePNG("render.png", &rgb.at(0), width, height);
    
    return 0;
}
