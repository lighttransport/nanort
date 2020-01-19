/*
The MIT License (MIT)

Copyright (c) 2020 Light Transport Entertainment, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <iostream>

#define TINYEXR_IMPLEMENTATION
#include "../common/tinyexr.h"

#include "image_saver.hh"

namespace example {

bool SaveFloatEXR(const float* pixels, uint32_t width, uint32_t height,
                       uint32_t channels, const std::string &outfilename) {
  // Use simple API
  const char *err = nullptr;
  int ret = SaveEXR(pixels, int(width), int(height), int(channels), /* fp16 */0, outfilename.c_str(), &err);

  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save EXR err: %s\n", err);
    FreeEXRErrorMessage(err);  // free's buffer for an error message
    return false;
  }

  printf("Saved exr file. [ %s ] \n", outfilename.c_str());
 
  return true;
}

bool SaveIntEXR(const int* pixels, uint32_t width, uint32_t height,
                       uint32_t channels, const std::string &outfilename) {
  if ((channels < 1) || (channels > 4)) {
    std::cerr << "Channels must be 1, 2, 3 or 4.\n";
    return false;
  }

  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = int(channels);

  std::vector<std::vector<int> > images(channels);

  for (size_t i = 0; i < size_t(channels); i++) {
    images[i].resize(width * height);
  }

  // Split RGBRGBRGB... into R, G and B layer
  for (size_t i = 0; i < size_t(width * height); i++) {
    for (size_t c = 0; c < channels; c++) {
      images[c][i] = pixels[channels * i + c];
    }
  }

  std::vector<int*> image_ptr(channels);
  for (size_t i = 0; i < size_t(channels); i++) {
    // reverse order
    image_ptr[i] = &(images[channels-i].at(0));
  }

  image.images = reinterpret_cast<unsigned char**>(image_ptr.data());
  image.width = int(width);
  image.height = int(height);

  // No premultily alpha.
  std::string channel_names[4] = {"R", "G", "B", "A"};

  header.num_channels = int(channels);
  header.channels =
      static_cast<EXRChannelInfo*>(malloc(sizeof(EXRChannelInfo) * size_t(header.num_channels)));
  // Must be (A)BGR order, since most of EXR viewers expect this channel order.
  for (size_t i = 0; i < size_t(channels); i++) {
    strncpy(header.channels[i].name, channel_names[channels - i].c_str(), 255);
    header.channels[i].name[channel_names[channels - i].size()] = '\0';
  }

  header.pixel_types = static_cast<int*>(malloc(sizeof(int) * size_t(header.num_channels)));
  header.requested_pixel_types =
      static_cast<int*>(malloc(sizeof(int) * size_t(header.num_channels)));
  for (int i = 0; i < header.num_channels; i++) {
    header.pixel_types[i] = TINYEXR_PIXELTYPE_UINT;  // pixel type of input image
    header.requested_pixel_types[i] =
        TINYEXR_PIXELTYPE_UINT;  // pixel type of output image to be stored in
                                // .EXR
  }

  const char* err = nullptr;
  int ret = SaveEXRImageToFile(&image, &header, outfilename.c_str(), &err);

  FreeEXRHeader(&header);

  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "Save Int EXR err: %s\n", err);
    FreeEXRErrorMessage(err);  // free's buffer for an error message
    return false;
  }
  printf("Saved exr file. [ %s ] \n", outfilename.c_str());

  return true;
}

} // namespace example
