#ifndef EXAMPLE_RENDER_LAYER_H_
#define EXAMPLE_RENDER_LAYER_H_

#include <vector>

namespace example {

class RenderLayer {

 public:
  void Clear() {
    std::fill_n(rgba.begin(), rgba.size(), 0.0f);
    std::fill_n(shading_normal.begin(), shading_normal.size(), 0.0f);
    std::fill_n(geometric_normal.begin(), geometric_normal.size(), 0.0f);
    std::fill_n(tangent.begin(), tangent.size(), 0.0f);
    std::fill_n(binormal.begin(), binormal.size(), 0.0f);
    std::fill_n(position.begin(), position.size(), 0.0f);
    std::fill_n(depth.begin(), depth.size(), 0.0f);
    std::fill_n(texcoord.begin(), texcoord.size(), 0.0f);
    std::fill_n(varycoord.begin(), varycoord.size(), 0.0f);
    std::fill_n(vertexColor.begin(), vertexColor.size(), 0.0f);
    std::fill_n(diffuse.begin(), diffuse.size(), 0.0f);
    std::fill_n(count.begin(), count.size(), 0);

  }

  void Resize(size_t w, size_t h) {

    width = w;
    height = h;

    rgba.resize(width * height * 4);
    count.resize(width * height); // scalar
    shading_normal.resize(width * height * 4);
    geometric_normal.resize(width * height * 4);
    tangent.resize(width * height * 4);
    binormal.resize(width * height * 4);
    position.resize(width * height * 4);
    depth.resize(width * height); // scalar
    texcoord.resize(width * height * 4);
    varycoord.resize(width * height * 4);
    vertexColor.resize(width * height * 4);
    diffuse.resize(width * height * 4);

  }

  RenderLayer() {
  }
  ~RenderLayer() {}

  std::vector<float> rgba;
  std::vector<int> count;
  std::vector<float> shading_normal;
  std::vector<float> geometric_normal;
  std::vector<float> tangent;
  std::vector<float> binormal;
  std::vector<float> position;
  std::vector<float> depth;
  std::vector<float> texcoord;
  std::vector<float> varycoord;
  std::vector<float> vertexColor;
  std::vector<float> diffuse;

  size_t width;
  size_t height;

};




};

#endif // EXAMPLE_RENDER_LAYER_H_
