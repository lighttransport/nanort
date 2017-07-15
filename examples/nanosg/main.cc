#include "nanosg.h"
#include "render.h"

#include <cstdio>
#include <cstdlib>

int
main(int argc, char **argv)
{
  nanosg::Scene<float, example::Mesh<float>> scene;

  scene.Commit();

	return EXIT_SUCCESS;
}
