#ifndef EXAMPLE_GEOMETRY_UTIL_H
#define EXAMPLE_GEOMETRY_UTIL_H

#include <cstdint>
#include <vector>

namespace example {

///
/// Compute tangent and binormal for a given mesh.
/// Assume all faces are triangles.
///
///
void ComputeTangentsAndBinormals(
    const std::vector<float> &vertices, const std::vector<uint32_t> &faces,
    const std::vector<float> &facevarying_texcoords,
    const std::vector<float> &facevarying_normals,
    std::vector<float> *facevarying_tangents,
    std::vector<float> *facevarying_binormals);

///
/// Apply vector displacement to mesh vertex position.
/// Faces whose material id is `vdisp_material_id` will be displaced.
///
/// Assume all faces are triangles.
/// Displacement is applied to shared vertices.
/// displacement with different UV assinged may result in unexpected displacements.
///
/// @param[in] vdisp_space What coordinate is used for vector displacement
/// value(0 = world, 1 = tangent)
/// @param[out] displaced_vertices Displaced vertex position.
///
void ApplyVectorDispacement(
    const std::vector<float> &vertices, const std::vector<uint32_t> &faces,
    const std::vector<uint32_t> &material_ids,
    const std::vector<float> &facevarying_texcoords,
    const std::vector<float> &facevarying_normals,
    const std::vector<float> &facevarying_tangents,
    const std::vector<float> &facevarying_binormals,
    const uint32_t vdisp_material_id,
    const std::vector<float> &vdisp_image,  // rgb
    const size_t vdisp_width, const size_t vdisp_height,
    const int vdisp_channels,
    const float vdisp_scale,
    std::vector<float> *displaced_vertices);  // this is facevarying.

///
/// Recompute smooth vertex normal.
///
/// @param[in] area_weighting Weighting normal by face's area.
///
void RecomputeSmoothNormals(
    const std::vector<float> &vertices,
    const std::vector<uint32_t> &faces,
    const bool area_weighting,
    std::vector<float> *facevarying_normals);

}  // namespace example

#endif  // EXAMPLE_GEOMETRY_UTIL_H
