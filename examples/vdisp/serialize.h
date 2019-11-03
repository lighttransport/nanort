#ifndef EXAMPLE_SERIALIZE_H_
#define EXAMPLE_SERIALIZE_H_

#include "scene.h"

#include <string>

namespace example {

// TODO(LTE): serialize envmap.

bool SaveSceneToEson(const std::string &eson_filename, const Scene &scene);
bool LoadSceneFromEson(const std::string &eson_filename, Scene *scene);

} // namespace example

#endif // EXAMPLE_SERIALIZE_H_
