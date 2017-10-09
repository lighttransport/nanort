# NanoRT + Embree2 compatile API

Drop-in replacement of Embree API For cross-platform raytracing(e.g. raytracing on ARM, PowerPC)

## Version

Based on Embree2 2.17.0 header.

## Status

Minimum and experimental.

Triangle + single ray intersection only.

## Coordinates

* Right-handed
* Geometric normal defined as CCW

## How to use

Simply copy embree2 header files(`include/embree2/`), `nanort.h`, `nanosg.h` and `nanort-embree.cc` to your project.

## TODO

* [ ] Curve/hair
* [ ] Subdivision surface
* [ ] Motion blur
* [ ] Instanciation
* [ ] Stream intersection API
* [ ] Ray stream API
