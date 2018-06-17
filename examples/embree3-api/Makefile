
# Variables
#
# - CXX
# - CXX_FLAGS
# - EMBREE_PATH

# Examples:
#
# - make
# - make CXX=g++
# - make CXX=clang++
# - make CXX=/usr/local/opt/llvm/bin/clang++ EMBREE_PATH=/opt/embree/embree-3.2.0.x86_64.macosx

SHELL := /bin/bash

CXX=g++
CXX_FLAGS=-Wall -pedantic -std=c++11 -O3
EMBREE_PATH=.

all: nanort

main.o: main.cc
	$(CXX) $(CXX_FLAGS) -c $^ -I ${EMBREE_PATH}/include -I ../.. -I ../nanosg -I ../common -o $@

main_openmp.o: main.cc
	$(CXX) $(CXX_FLAGS) -fopenmp -c $^ -I ${EMBREE_PATH}/include -I ../.. -I ../nanosg -I ../common -o $@

obj_loader.o: ../nanosg/obj-loader.cc
	$(CXX) $(CXX_FLAGS) -c $^ -I ../.. -I ../nanosg -I ../common -o $@

obj_loader_openmp.o: ../nanosg/obj-loader.cc
	$(CXX) $(CXX_FLAGS) -fopenmp -c $^ -I ../.. -I ../nanosg -I ../common -o $@

nanort-embree3.o: nanort-embree3.cc
	$(CXX) $(CXX_FLAGS) -c $^ -I ${EMBREE_PATH}/include -I ../.. -I ../nanosg -I ../common -o $@

nanort: main.o obj_loader.o
	$(CXX) $^ -L ${EMBREE_PATH}/lib -lembree3 -o $@

nanort_openmp: main_openmp.o obj_loader_openmp.o
	$(CXX) $^ -L ${EMBREE_PATH}/lib -L ${shell dirname `which $(CXX)`}/../lib -lembree3 -fopenmp -lomp -o $@

run_nanort: nanort
	source ${EMBREE_PATH}/embree-vars.sh
	./nanort

run_nanort_openmp: nanort_openmp
	source ${EMBREE_PATH}/embree-vars.sh
	./nanort_openmp

clean:
	rm -f a.out main nanort nanort_openmp *.png *.o

.PHONY: clean run_nanort run_nanort_openmp
