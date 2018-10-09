#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "tinyexr.h"

namespace save_img {

static bool SaveEXR(const float* rgba,const int width,const int height,const char* outfilename) {

  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 3;

  std::vector<float> images[3];
  images[0].resize(size_t(width * height));
  images[1].resize(size_t(width * height));
  images[2].resize(size_t(width * height));

  // Split RGBRGBRGB... into R, G and B layer
  /*for (int i = 0; i < width * height; i++) {
    images[0][width * height - i - 1] = rgba[4*i+0];
    images[1][width * height - i - 1] = rgba[4*i+1];
    images[2][width * height - i - 1] = rgba[4*i+2];
  }*/
  for (int i = 0;i < height;i++) {
    for (int j = 0;j < width;j++) {
      for (int k = 0;k < 3;k++) {
        images[k][size_t((height - i - 1) * width + j)] = rgba[size_t(4 * (i * width + j) + k)];
      }
    }
  }

  float* image_ptr[3];
  image_ptr[0] = &(images[2].at(0)); // B
  image_ptr[1] = &(images[1].at(0)); // G
  image_ptr[2] = &(images[0].at(0)); // R

  image.images = reinterpret_cast<unsigned char**>(image_ptr);
  image.width = width;
  image.height = height;

  header.num_channels = 3;
  header.channels = static_cast<EXRChannelInfo *>(malloc(sizeof(EXRChannelInfo) * size_t(header.num_channels)));
  // Must be (A)BGR order, since most of EXR viewers expect this channel order.
  strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
  strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
  strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';

  header.pixel_types = static_cast<int *>(malloc(sizeof(int) * size_t(header.num_channels)));
  header.requested_pixel_types = static_cast<int *>(malloc(sizeof(int) * size_t(header.num_channels)));
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
  }

  const char* err = nullptr; // or nullptr in C++11 or later.
  int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err); // free's buffer for an error message
    return ret;
  }
  printf("Saved exr file. [ %s ] \n", outfilename);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  return TINYEXR_SUCCESS;

} 

}
