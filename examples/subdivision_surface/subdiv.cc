#include <fstream>
#include <chrono>
#include <iostream>

#include "subdiv.hh"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif

#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/topologyDescriptor.h>

#ifdef __clang__
#pragma clang diagnostic pop
#endif

using namespace OpenSubdiv;

namespace example {

void subdivide(int subd_level, const ControlQuadMesh &in_mesh, Mesh *out_mesh, bool dump) {

  if (subd_level < 0) {
    subd_level = 0;
  }

  std::cout << "SubD: level = " << subd_level << "\n";

  const auto start_t = std::chrono::system_clock::now();

  int maxlevel = subd_level;

  if (maxlevel > 8) {
    maxlevel = 8;
    std::cout << "SubD: limit subd level to " << maxlevel << "\n";
  }

  typedef Far::TopologyDescriptor Descriptor;

  Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;

  Sdc::Options options;
  options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);
  options.SetFVarLinearInterpolation(Sdc::Options::FVAR_LINEAR_NONE);

  // Populate a topology descriptor with our raw data
  Descriptor desc;
  desc.numVertices = in_mesh.vertices.size() / 3;
  desc.numFaces = in_mesh.verts_per_faces.size();
  desc.numVertsPerFace = in_mesh.verts_per_faces.data();
  desc.vertIndicesPerFace = in_mesh.indices.data();

#if 0  // TODO
    int channelUV = 0;
    int channelColor = 1;

    // Create a face-varying channel descriptor
    Descriptor::FVarChannel channels[2];
    channels[channelUV].numValues = g_nuvs;
    channels[channelUV].valueIndices = g_uvIndices;
    channels[channelColor].numValues = g_ncolors;
    channels[channelColor].valueIndices = g_colorIndices;

    // Add the channel topology to the main descriptor
    desc.numFVarChannels = 2;
    desc.fvarChannels = channels;
#else
  desc.numFVarChannels = 0;
#endif

  // Instantiate a Far::TopologyRefiner from the descriptor
  Far::TopologyRefiner *refiner =
      Far::TopologyRefinerFactory<Descriptor>::Create(
          desc,
          Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

  // Uniformly refine the topology up to 'maxlevel'
  // note: fullTopologyInLastLevel must be true to work with face-varying data
  {
    Far::TopologyRefiner::UniformOptions refineOptions(maxlevel);
    refineOptions.fullTopologyInLastLevel = true;
    refiner->RefineUniform(refineOptions);
  }

  // Allocate and initialize the 'vertex' primvar data (see tutorial 2 for
  // more details).
  std::vector<Vertex> vbuffer(refiner->GetNumVerticesTotal());
  Vertex *verts = &vbuffer[0];

  for (int i = 0; i < in_mesh.vertices.size() / 3; ++i) {
    verts[i].SetPosition(in_mesh.vertices[3 * i + 0],
                         in_mesh.vertices[3 * i + 1],
                         in_mesh.vertices[3 * i + 2]);
  }

#if 0
    // Allocate and initialize the first channel of 'face-varying' primvar data (UVs)
    std::vector<FVarVertexUV> fvBufferUV(refiner->GetNumFVarValuesTotal(channelUV));
    FVarVertexUV * fvVertsUV = &fvBufferUV[0];
    for (int i=0; i<g_nuvs; ++i) {
        fvVertsUV[i].u = in_mesh.facevarying_uvs[2 * i + 0];
        fvVertsUV[i].v = in_mesh.facevarying_uvs[2 * i + 1];
    }

    // Allocate & interpolate the 'face-varying' primvar data (colors)
    std::vector<FVarVertexColor> fvBufferColor(refiner->GetNumFVarValuesTotal(channelColor));
    FVarVertexColor * fvVertsColor = &fvBufferColor[0];
    for (int i=0; i<g_ncolors; ++i) {
        fvVertsColor[i].r = g_colors[i][0];
        fvVertsColor[i].g = g_colors[i][1];
        fvVertsColor[i].b = g_colors[i][2];
        fvVertsColor[i].a = g_colors[i][3];
    }
#endif

  // Interpolate both vertex and face-varying primvar data
  Far::PrimvarRefiner primvarRefiner(*refiner);

  Vertex *srcVert = verts;
  // FVarVertexUV * srcFVarUV = fvVertsUV;
  // FVarVertexColor * srcFVarColor = fvVertsColor;

  for (int level = 1; level <= maxlevel; ++level) {
    Vertex *dstVert = srcVert + refiner->GetLevel(level - 1).GetNumVertices();
    // FVarVertexUV * dstFVarUV = srcFVarUV +
    // refiner->GetLevel(level-1).GetNumFVarValues(channelUV); FVarVertexColor *
    // dstFVarColor = srcFVarColor +
    // refiner->GetLevel(level-1).GetNumFVarValues(channelColor);

    primvarRefiner.Interpolate(level, srcVert, dstVert);
    // primvarRefiner.InterpolateFaceVarying(level, srcFVarUV, dstFVarUV,
    // channelUV); primvarRefiner.InterpolateFaceVarying(level, srcFVarColor,
    // dstFVarColor, channelColor);

    srcVert = dstVert;
    // srcFVarUV = dstFVarUV;
    // srcFVarColor = dstFVarColor;
  }

  {  // Output

    std::ofstream ofs;
    if (dump) {
      ofs.open("subd.obj");
    }

    Far::TopologyLevel const &refLastLevel = refiner->GetLevel(maxlevel);

    int nverts = refLastLevel.GetNumVertices();
    // int nuvs   = refLastLevel.GetNumFVarValues(channelUV);
    // int ncolors= refLastLevel.GetNumFVarValues(channelColor);
    int nfaces = refLastLevel.GetNumFaces();

    std::cout << "nverts = " << nverts << ", nfaces = " << nfaces << "\n";

    // Print vertex positions
    int firstOfLastVerts = refiner->GetNumVerticesTotal() - nverts;

    out_mesh->vertices.resize(nverts * 3);

    for (int vert = 0; vert < nverts; ++vert) {
      float const *pos = verts[firstOfLastVerts + vert].GetPosition();
      ofs << "v " << pos[0] << " " << pos[1] << " " << pos[2] << "\n";
      out_mesh->vertices[3 * vert + 0] = pos[0];
      out_mesh->vertices[3 * vert + 1] = pos[1];
      out_mesh->vertices[3 * vert + 2] = pos[2];
    }

#if 0
        // Print uvs
        int firstOfLastUvs = refiner->GetNumFVarValuesTotal(channelUV) - nuvs;

        for (int fvvert = 0; fvvert < nuvs; ++fvvert) {
            FVarVertexUV const & uv = fvVertsUV[firstOfLastUvs + fvvert];
            printf("vt %f %f\n", uv.u, uv.v);
        }

        // Print colors
        int firstOfLastColors = refiner->GetNumFVarValuesTotal(channelColor) - ncolors;

        for (int fvvert = 0; fvvert < nuvs; ++fvvert) {
            FVarVertexColor const & c = fvVertsColor[firstOfLastColors + fvvert];
            printf("c %f %f %f %f\n", c.r, c.g, c.b, c.a);
        }
#endif

    out_mesh->triangulated_indices.clear();
    out_mesh->face_num_verts.clear();
    out_mesh->face_index_offsets.clear();
    out_mesh->face_ids.clear();
    out_mesh->face_triangle_ids.clear();
    out_mesh->material_ids.clear();

    for (int face = 0; face < nfaces; ++face) {
      Far::ConstIndexArray fverts = refLastLevel.GetFaceVertices(face);
      // Far::ConstIndexArray fuvs   = refLastLevel.GetFaceFVarValues(face,
      // channelUV);

      // all refined Catmark faces should be quads
      // assert(fverts.size()==4 && fuvs.size()==4);
      assert(fverts.size() == 4);

      out_mesh->face_index_offsets.push_back(out_mesh->face_num_verts.size());

      out_mesh->face_num_verts.push_back(fverts.size());

      if (dump) {
        ofs << "f";
      }
      for (int vert = 0; vert < fverts.size(); ++vert) {
        out_mesh->face_indices.push_back(fverts[vert]);

        if (dump) {
          // OBJ uses 1-based arrays...
          ofs << " " << fverts[vert]+1;
        }
      }

      if (dump) {
        ofs << "\n";
      }


      // triangulated face
      out_mesh->triangulated_indices.push_back(fverts[0]);
      out_mesh->triangulated_indices.push_back(fverts[1]);
      out_mesh->triangulated_indices.push_back(fverts[2]);

      out_mesh->triangulated_indices.push_back(fverts[2]);
      out_mesh->triangulated_indices.push_back(fverts[3]);
      out_mesh->triangulated_indices.push_back(fverts[0]);

      // some face attribs.
      out_mesh->face_ids.push_back(face);
      out_mesh->face_ids.push_back(face);

      out_mesh->face_triangle_ids.push_back(0);
      out_mesh->face_triangle_ids.push_back(1);

      // -1 = no material
      out_mesh->material_ids.push_back(-1);
      out_mesh->material_ids.push_back(-1);
    }

  }

  const auto end_t = std::chrono::system_clock::now();
  const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_t - start_t).count();

  std::cout << "SubD time : " << elapsed << " [ms]\n";

  if (dump) {
    std::cout << "dumped subdivided mesh as `subd.obj`\n";
  }
}



}  // namespace example
