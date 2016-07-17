#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int save_png(const char* filename, int width, int height,
             const unsigned char* rgb) {
  int ret =
      stbi_write_png(filename, width, height, 3, (const void*)rgb, width * 3);

  return ret;
}
