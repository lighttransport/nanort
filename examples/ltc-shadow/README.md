# Denoised shadow experiment.

NanoRT implementation of Combining Analytic Direct Illumination and Stochastic Shadows https://eheitzresearch.wordpress.com/705-2/

## Build

### Linux or macOS

```bash
premake5 gmake
make
```

### Windows

```bash
premake5 vs2017
```

## How to generate IBL cubemaps

Use Filament's cmgen to generate prefiltered IBL envmaps(in cubemap, RGBM format)

```
$ cmgen -x ibls input.exr
```

Set path to `ibls/input` to `ibl_dirname` in `config.json` to load IBL cubemaps.

## Third party libraries and its icenses

* picojson : BSD license.
* bt3gui : zlib license.
* glew : BSD/MIT license.
* tinyobjloader : MIT license.
* glm : The Happy Bunny License (Modified MIT License). Copyright (c) 2005 - 2017 G-Truc Creation
* ImGui : The MIT License (MIT). Copyright (c) 2014-2015 Omar Cornut and ImGui contributors
* ImGuizmo : The MIT License (MIT). Copyright (c) 2016 Cedric Guillemet
