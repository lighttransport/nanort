#pragma once

struct Texture
{
  enum class usage : unsigned char
  {
    baseColor, normal, emit, metal_rough
  };

  usage use;
  int width, height, components;
  unsigned char* image;
};