cmake_minimum_required(VERSION 3.2)
project(OpenCL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework OpenCL")

set(SOURCE_FILES Conductor.cpp main.cpp)

add_executable(VectorDemo ${SOURCE_FILES})