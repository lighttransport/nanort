#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

#ifdef __cplusplus
extern "C" {
#endif

void *EMSCRIPTEN_KEEPALIVE allocate(size_t n)
{
  void *ptr = malloc(n);
  return ptr;
}

float EMSCRIPTEN_KEEPALIVE func(float *a, size_t n, float *out) { //, float *b) {
  std::vector<float> buf;
  buf.resize(1024*1024*1);

  memcpy(buf.data(), buf.data(), 1024*1024*sizeof(float));

  float sum = 0.0f;
  for (size_t i = 0; i < n; i++) {
    sum += a[i];
  }

  //printf("muda\n");
  //std::cout << "bora\n";

  out[0] = sum;
  //a[0] = sum;

  return sum;
}

#ifdef __cplusplus
}
#endif

#ifndef __EMSCRIPTEN__
int main(int argc, char **argv)
{
  std::vector<float> buf;
  buf.resize(argc);

  printf("argc %d\n", argc);
  for (size_t i = 0; i < argc; i++) {
    printf("argv[%d] = %s\n", i, argv[i]);
  }
  //fopen(argv[0]);
  // dummy
  return 3;
}
#endif

//std::string EMSCRIPTEN_KEEPALIVE func(std::string &img) {
//  return img;
//}
