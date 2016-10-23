# Raytracing hair primitive example

## Features

* Render hair as cubic Bezier curve.
* Example CyHair loader is provided.  
  * Input points are treated as CatmullRom spline.
* Hair shader based on PBRT-v3
  * http://pbrt.org/hair.pdf

## Run

Download cyhair data from ...
Edit `config.json`, then,

    $ ./bin/native/Release/hair

## Parameters

* `cyhair_filename`(string) : Filename of CyHair format hair file.
* `max_strands`(int) : Maximum number of strands to use(default is -1, which means use all strands)
* `scene_scale`(float3)  : Scale factor for the scene 
* `scene_tranlate`(float3) : Translation for the scene(after scaling by `scene_scale`)

## UI

* left mouse drag = rotate view
* Tab + left mouse drag = zoom view
* Shift + left mouse drag = translate view

## TODO

* [ ] Implement recursive ray-Bezier intersection
* [ ] Implement hair BSDF.
