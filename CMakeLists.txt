cmake_minimum_required(VERSION 2.8.12)
project(MultiVarGauss)

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

add_executable(multivargauss
  src/multivargauss.cpp)

add_executable(mixedgaussians
  src/mixedgaussians.cpp)

file(GLOB_RECURSE LIB_SOURCE "src/mvg/*.cpp")
add_library(${PROJECT_NAME}
 ${LIB_SOURCE})

target_link_libraries(multivargauss
  ${PROJECT_NAME}
  json-c)

target_link_libraries(mixedgaussians
  ${PROJECT_NAME}
  json-c)
