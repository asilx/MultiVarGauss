#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>

#include <Eigen/Dense>

#include <mvg/MixedGaussians.hpp>
#include <mvg/JSON.h>


int main(int argc, char** argv) {
  int nReturnvalue = EXIT_SUCCESS;
  
  // TODO: Needs implementation. Some functionality from
  // multivargauss.cpp must go into the MultiVarGauss class first,
  // though; for example the code for handling nominal values and the
  // parts that load data from files.
  
  srand(time(NULL));
  
  mvg::MixedGaussians<float> mgGaussians;
  
  unsigned int unSampleDimensions = 2;
  std::vector<std::vector<float>> vecMeans = {{1.0, -0.5}, {1.0, 1.0}};
  std::vector<std::vector<float>> vecSpreads = {{1.0, 1.5}, {0.5, 2.0}};
  std::vector<int> vecSampleCounts = {1500, 1500};
  
  for(unsigned int unI = 0; unI < vecMeans.size(); ++unI) {
    mvg::MultiVarGauss<float>::Ptr mvgGaussian = mvg::MultiVarGauss<float>::create();
    mvg::Dataset::Ptr dsDataset = mvg::Dataset::create();
    mvgGaussian->setDataset(dsDataset);
    mgGaussians.addGaussian(mvgGaussian, 1.0);
    
    for(unsigned int unSample = 0; unSample < vecSampleCounts[unI]; ++unSample) {
      std::vector<float> vecSample;
      
      for(unsigned int unDimension = 0; unDimension < unSampleDimensions; ++unDimension) {
	vecSample.push_back(vecMeans[unI][unDimension] + ((((double)rand() / (double)RAND_MAX) * vecSpreads[unI][unDimension]) - vecSpreads[unI][unDimension] / 2.0));
      }
      
      mvg::MultiVarGauss<float>::addToDataset(dsDataset, vecSample);
    }
  }
  
  mvg::MultiVarGauss<float>::Rect rctBoundingBox = mgGaussians.boundingBox();
  mvg::MultiVarGauss<float>::DensityFunction fncDensity = mgGaussians.densityFunction();
  
  // Two dimensional case
  float fStepSizeX = 0.01;
  float fStepSizeY = 0.01;
  
  for(float fX = rctBoundingBox.vecMin[0]; fX < rctBoundingBox.vecMax[0]; fX += fStepSizeX) {
    for(float fY = rctBoundingBox.vecMin[1]; fY < rctBoundingBox.vecMax[1]; fY += fStepSizeY) {
      float fValue = fncDensity({fX, fY});
      
      std::cout << fX << ", " << fY << ", " << fValue << std::endl;
    }
  }
  
  /*std::cout << "Min :";
  for(float fValue : rctBoundingBox.vecMin) {
    std::cout << "  " << fValue;
  }
  std::cout << std::endl;
  
  std::cout << "Max :";
  for(float fValue : rctBoundingBox.vecMax) {
    std::cout << "  " << fValue;
  }
  std::cout << std::endl;*/
  
  return nReturnvalue;
}
