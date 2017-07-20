cmake_minimum_required(VERSION 2.8.12)
project(MultiVarGauss)

find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(${PROJECT_NAME}_VERSION_MAJOR_0)
set(${PROJECT_NAME}_VERSION_MINOR_1)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(
  include
  /usr/include/eigen3)

link_directories(
  )

file(GLOB_RECURSE LIB_SOURCE "src/mvg/*.c*")
add_library(${PROJECT_NAME} SHARED
  ${LIB_SOURCE} src/mixedgaussians.cpp src/multivargauss.cpp src/org_knowrob_gaussian_MixedGaussianInterface.cpp)
target_link_libraries(${PROJECT_NAME}
  json-c)

