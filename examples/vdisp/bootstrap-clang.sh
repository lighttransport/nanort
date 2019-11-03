#!/bin/bash

rm -rf build

CXX=clang++ CC=clang cmake -Bbuild -H. -DSANITIZE_ADDRESS=On
