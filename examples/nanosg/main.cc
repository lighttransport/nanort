#include "nanosg.h"

#include <cstdio>
#include <cstdlib>

int
main(int argc, char **argv)
{
  nanosg::Scene<float> scene;

  scene.Commit();

	return EXIT_SUCCESS;
}
