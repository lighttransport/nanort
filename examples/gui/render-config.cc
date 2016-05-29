#include "render-config.h"

#include "picojson.h"

#include <fstream>
#include <istream>

namespace example {

bool LoadRenderConfig(example::RenderConfig *config, const char* filename)
{
  
  std::ifstream is(filename);
  if (is.fail()) {
    std::cerr << "Cannot open " << filename << std::endl;
    return false;
  }

  std::istream_iterator<char> input(is);
  std::string err;
  picojson::value v;
  input = picojson::parse(v, input, std::istream_iterator<char>(), &err);
  if (! err.empty()) {
    std::cerr << err << std::endl;
  }

  if (!v.is<picojson::object>()) {
    std::cerr << "Not a JSON object" << std::endl;
    return false;
  }

  picojson::object o = v.get<picojson::object>();

  if (o.find("obj_filename") != o.end()) {
    if (o["obj_filename"].is<std::string>()) {
      config->obj_filename = o["obj_filename"].get<std::string>();
    }
  }

  if (o.find("scene_scale") != o.end()) {
    if (o["scene_scale"].is<double>()) {
      config->scene_scale = static_cast<float>(o["scene_scale"].get<double>());
    }
  }

  return true;
}

}
