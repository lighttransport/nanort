# TODO(LTE): OpenMP dependency support. https://github.com/mesonbuild/meson/pull/1852
project('path_tracer', 'cpp')
inc = include_directories('../common/', '../../')
sources = ['main.cc'
  , 'tiny_obj_loader.cc'
  ]
executable('path_tracer',
  sources,
  include_directories : inc)
