#!/bin/bash
# Run this from within a bash shell
# Use default gcc
#cmake -G "Unix Makefiles" ../../source && ccmake ../../source
# Use g++44 (g++ 4.4) on the GEM cluster
cmake -G "Unix Makefiles" -DCMAKE_CXX_COMPILER="g++44" -DCMAKE_C_COMPILER="gcc44" ../../source && ccmake ../../source
# Use address sanitizer on a machine with g++ 4.8 installed
#cmake -G "Unix Makefiles" -DCMAKE_CXX_COMPILER="g++-4.8" -DCMAKE_C_COMPILER="gcc-4.8" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer" -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address" -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address" -DCMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address"  ../../source && ccmake ../../source
