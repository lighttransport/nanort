# NanoRT + Embree2 compatile API

Drop-in replacement of raytracing engine with Embree API for cross-platform support(e.g. raytracing on ARM, PowerPC)

## Version

Based on Embree2 2.17.0 header.

## Status

Minimum and experimental.

Triangle + single ray intersection only.

## Coordinates

* Right-handed
* Geometric normal defined as CCW

## How to use

Simply copy embree2 header files(`include/embree2/`), `nanort.h`(`../../`), `nanosg.h`(`../nanosg/`) and `nanort-embree.cc` to your project.

## Notes

Current implementation does not consider calling Embree API from a multi-threaded environment.
Application must care of calling Embree API with proper thread locking(except for `rtcIntersect`)

## TODO

* [ ] Curve/hair
* [ ] Subdivision surface
* [ ] Motion blur
* [ ] Instanciation
* [ ] Stream intersection API
* [ ] Ray stream API
* [ ] Reentrant API support(thread-safe API).
* [ ] DLL build of nanort-embree module.

## Licenses

* embree2 header files : Copyright 2009-2017 Intel Corporation. Apache 2.0 license.
