#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>

#include <Eigen/Dense>

#include <mvg/MixedGaussians.hpp>
#include <mvg/JSON.h>


int main(int argc, char** argv) {
  int nReturnvalue = EXIT_FAILURE;
  
  // TODO: Needs implementation. Some functionality from
  // multivargauss.cpp must go into the MultiVarGauss class first,
  // though; for example the code for handling nominal values and the
  // parts that load data from files.
  
  mvg::MixedGaussians<float> mgGaussians;
  
  // ...
  
  return nReturnvalue;
}
