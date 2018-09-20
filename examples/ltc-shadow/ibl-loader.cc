#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "ibl-loader.h"
#include "stb_image.h"

namespace example {

static bool FileExists(const std::string &filepath) {
  std::ifstream ifs(filepath);

  if (!ifs) {
    return false;
  }

  return true;
}

static std::string JoinPath(const std::string &dir,
                            const std::string &filename) {
  if (dir.empty()) {
    return filename;
  } else {
    // check '/'
    char lastChar = *dir.rbegin();
    if (lastChar != '/') {
      return dir + std::string("/") + filename;
    } else {
      return dir + filename;
    }
  }
}

// https://stackoverflow.com/questions/29310166/check-if-a-fstream-is-either-a-file-or-directory
static bool IsDirectory(const std::string &filepath) {
  std::ifstream my_file;
  try {
    // Ensure that my_file throws an exception if the fail or bad bit is set.
    my_file.exceptions(std::ios::failbit | std::ios::badbit);
    // std::cout << "Read file '" << argv[i] << "'" << std::endl;
    my_file.open(filepath);
    my_file.seekg(0, std::ios::end);
  } catch (std::ios_base::failure &err) {
    (void)err;
    // std::cerr << "  Exception during open(): " << argv[i] << ": " <<
    // strerror(errno) << std::endl;
    return true;
  }

  try {
    errno = 0;  // reset errno before I/O operation.
    std::string line;
    int n = 0;
    while (std::getline(my_file, line)) {
      // std::cout << "  read line" << std::endl;
      // ...
      n++;
      if (n > 1024) {
        break;
      }
    }
  } catch (std::ios_base::failure &err) {
    (void)err;
    if (errno == 0) {
      // std::cerr << "  Exception during read(), but errono is 0. No real
      // error." << std::endl;
      // exception is likely raised due to EOF, no real error.
    } else {
      return true;
    }
  }

  return false;
}

static std::string GetFileExtension(const std::string &filename) {
  if (filename.find_last_of(".") != std::string::npos)
    return filename.substr(filename.find_last_of(".") + 1);
  return "";
}

bool LoadHDRImage(const std::string &filename, std::vector<float> *out_image,
                  int *out_width, int *out_height, int *out_channels) {
  auto start_t = std::chrono::system_clock::now();

  size_t width = 0, height = 0;
  size_t num_channels = 0;

  // TODO(LTE): Support EXR image
  // TODO(LTE): Support 16bit or 32bit TIFF.

  std::string ext = GetFileExtension(filename);
  if ((ext.compare("rgbm") == 0) || (ext.compare("RGBM") == 0)) {
    // RGBM encoded HDR image with LDR format(usually JPG or PNG)
    // Use STB to load LDR image.

    int image_width, image_height, n;

    unsigned char *data = stbi_load(filename.c_str(), &image_width,
                                    &image_height, &n, STBI_default);

    if (!data) {
      std::cerr << "File not found " << filename << std::endl;
      return false;
    }

    if ((image_width == -1) || (image_height == -1)) {
      stbi_image_free(data);
      return false;
    }

    if (n != 4) {
      std::cerr << "# of channels must be 4(RGBA) : " << filename << std::endl;
      stbi_image_free(data);
      return false;
    }

    // Reconstruct HDR.
    const size_t num_pixels = static_cast<size_t>(image_width * image_height);

    out_image->resize(num_pixels * 3);

    // See Filament's RGMtoLinear().
    for (size_t i = 0; i < num_pixels; i++) {
      float r = data[4 * i + 0] / 255.0f;
      float g = data[4 * i + 1] / 255.0f;
      float b = data[4 * i + 2] / 255.0f;
      float a = data[4 * i + 3] / 255.0f;

      r = r * a * 16.0f;
      g = g * a * 16.0f;
      b = b * a * 16.0f;

      // Gamma to linear space
      (*out_image)[3 * i + 0] = r * r;
      (*out_image)[3 * i + 1] = g * g;
      (*out_image)[3 * i + 2] = b * b;
    }

    stbi_image_free(data);

    width = size_t(image_width);
    height = size_t(image_height);
    num_channels = 3;

  } else {
    std::cerr << "Unsupported file format: " << ext << std::endl;
    return false;
  }

  auto end_t = std::chrono::system_clock::now();
  std::chrono::duration<double, std::milli> ms = end_t - start_t;
  std::cout << "Image loading time : " << ms.count() << " [msecs]" << std::endl;
  ;

  (*out_width) = int(width);
  (*out_height) = int(height);
  (*out_channels) = int(num_channels);

  return true;
}

inline float sRGBToLinear(const float &sRGB) {
  const float a = 0.055f;
  const float a1 = 1.055f;
  const float p = 2.4f;
  float linear;
  if (sRGB <= 0.04045f) {
    linear = sRGB * (1.0f / 12.92f);
  } else {
    linear = std::pow((sRGB + a) / a1, p);
  }
  return linear;
}


bool LoadLDRImage(const std::string &filename, Image *dst) {

  int image_width, image_height, n;

  unsigned char *data = stbi_load(filename.c_str(), &image_width,
                                  &image_height, &n, STBI_default);

  if (!data) {
    std::cerr << "File not found " << filename << std::endl;
    return false;
  }

  if ((image_width == -1) || (image_height == -1)) {
    stbi_image_free(data);
    return false;
  }

  // Reconstruct HDR.
  const size_t num_pixels = static_cast<size_t>(image_width * image_height);

  dst->pixels.resize(num_pixels * n);

  // color
  int col_channels = n;
  if (n == 4) {
    col_channels = 3;
  }
  
  for (size_t i = 0; i < num_pixels; i++) {
    for (size_t c = 0; c < col_channels; c++) {
      float v = data[n * i + c] / 255.0f;

      // sRGB -> linear
      dst->pixels[n * i + c] = sRGBToLinear(v);
    }
  }

  // alpha
  if (n == 4) {
    for (size_t i = 0; i < num_pixels; i++) {
      float alpha = data[n * i + 3] / 255.0f;

      dst->pixels[n * i + 3] = alpha;

    }
  }

  stbi_image_free(data);

  dst->width = image_width;
  dst->height = image_height;
  dst->channels = n;

  return true;
}

// Swap pixels in both x and y 
static void FlipImage(Image &image)
{
  Image tmp = image;

  for (size_t y = 0; y < image.height; y++) {
    size_t dy = (image.height - 1) - y;
    for (size_t x = 0; x < image.height; x++) {
      size_t dx = (image.width - 1) - x;

      for (size_t c = 0; c < image.channels; c++) {
        tmp.pixels[image.channels * (y * image.width + x) + c] = image.pixels[image.channels * (dy * image.width + dx) + c];
      }
    }
  }

  image = tmp;

}

int LoadCubemaps(std::string &dirpath,
                 std::vector<Cubemap> *out_cubemaps) {
  std::cout << "Load ibl from : " << dirpath << std::endl;

  std::array<std::string, 6> cubemap_face_names = {
      {"nx", "px", "ny", "py", "nz", "pz"}};

  int num_levels = 0;

  std::array<Image, 6> cubemap_faces;

  // up to 8 levels.
  for (size_t m = 0; m < 8; m++) {
    std::string prefix = "m" + std::to_string(m) + "_";

    for (size_t f = 0; f < 6; f++) {
      std::string filename = JoinPath(dirpath, prefix + cubemap_face_names[f]);

      filename += ".rgbm";

      std::cerr << "Trying to load file : " << filename << " for face " << std::to_string(f) << std::endl;

      if (!FileExists(filename)) {
        if (f == 0) {
          // No files for this level.
          std::cerr << "It looks file does not exit : " << filename
                    << std::endl;
          return num_levels;
        } else {
          std::cerr << "Image for cubemap face[" << f << "] at level " << m
                    << " not found : " << f << std::endl;
          return num_levels;
        }
      }

      Image image;
      if (!LoadHDRImage(filename, &image.pixels, &(image.width),
                        &(image.height), &(image.channels))) {
        return num_levels;
      }

      if ((image.channels != 3) && (image.channels != 4)) {
        std::cerr << "Image must be RGB or RGBA : " << filename
                  << ", but got channels " << image.channels << std::endl;
        return num_levels;
      }

      // // Work around.
      // // PY, NY image should be flipped in my configuration.
      // if ((f == 2) || (f == 3)) {
      //   FlipImage(image);
      // }

      cubemap_faces[f] = std::move(image);
    }

    Cubemap cubemap(cubemap_faces);

    out_cubemaps->emplace_back(std::move(cubemap));

    num_levels++;
  }

  return num_levels;
}

}  // namespace example
