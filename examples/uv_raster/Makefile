CXX := clang++
CXXFLAGS := -std=c++11 -O2 -g

INC_FLAGS=-I../common/ -I../../
EXTRA_CXXFLAGS=-fsanitize=address -Werror -Weverything -Wno-c++98-compat
EXTRA_LINKFLAGS=-fsanitize=address

.PHONY: all clean

all: uv_raster

OBJS=main.o image_saver.o tiny_obj_loader.o

uv_raster: $(OBJS)
	$(CXX) $(LDFLAGS) $(EXTRA_LINKFLAGS) $(OBJS) -o $@

tiny_obj_loader.o: ../common/tiny_obj_loader.cc
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(INC_FLAGS) -c $< -o $@ 

main.o: main.cc
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(INC_FLAGS) -c $< -o $@ 

image_saver.o: image_saver.cc
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(INC_FLAGS) -c $< -o $@ 

