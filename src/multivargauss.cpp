#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>

#include <Eigen/Dense>

#include <mvg/MultiVarGauss.hpp>
#include <mvg/JSON.h>


unsigned int mvg::MultiVarGaussDriver::nominalValue(unsigned int unRow, std::string strValue) {
  std::map<std::string, unsigned int>& mapRow = mapNominalValues[unRow];
  
  if(mapRow.find(strValue) != mapRow.end()) {
    return mapRow[strValue];
  } else {
    unsigned int unNextHigher = 0;
    
    for(std::map<std::string, unsigned int>::iterator itPair = mapRow.begin();
	itPair != mapRow.end(); ++itPair) {
      if(itPair->second >= unNextHigher) {
	unNextHigher = itPair->second + 1;
      }
    }
    
    mapRow[strValue] = unNextHigher;
    
    return unNextHigher;
  }
}

mvg::MultiVarGauss<float> mvg::MultiVarGaussDriver::createMultiVarGauss(char* inputName)
{
  mvg::MultiVarGauss<float> mvgMain;
  
  srand(time(NULL));
  
  std::string strFile = inputName;
  std::ifstream ifFile(strFile.c_str());
  
  if(ifFile.good()) {
    mvg::JSON jsnJSON;
    std::string strLine;
      
    mvg::Dataset::Ptr dsData = mvg::Dataset::create();
      
    while(std::getline(ifFile, strLine)) {
      jsnJSON.parse(strLine);
      mvg::Property* prRoot = jsnJSON.rootProperty();
	
      if(prRoot->type() == mvg::Property::Array) {
	std::vector<mvg::Property*> vecRows = prRoot->subProperties();
        std::vector<float> vecValues;
	  
	for(unsigned int unRow = 0; unRow < vecRows.size() - 1 /* Ignore Z */; ++unRow) {
	  mvg::Property* prSub = vecRows[unRow];
	    
	  float fValue = 0.0;
	    
          switch(prSub->type()) {
	    case mvg::Property::Integer: fValue = (float)prSub->getInteger(); break;
	    case mvg::Property::Double: fValue = (float)prSub->getDouble(); break;
	    case mvg::Property::String: fValue = (float)nominalValue(unRow, prSub->getString()); break;
	    default: break;
	  }
	    
	  vecValues.push_back(fValue);
	}
	  
	mvg::MultiVarGauss<float>::addToDataset(dsData, vecValues);
      }
    }
    mvgMain.setDataset(dsData);
  }
  else {
    std::cerr << "Couldn't open file '" << strFile << "'" << std::endl;
  }

  return mvgMain;
}


int mvg::MultiVarGaussDriver::runMainMethod(char* inputName)
{
  mvg::MultiVarGauss<float> mvgMain = createMultiVarGauss(inputName);
      
  mvg::MultiVarGauss<float>::DensityFunction fncDensity = mvgMain.densityFunction();
      
  int nBoundary = 2.0;
      
  for(float fX = 0.1; fX <= 1.2; fX += 0.01) {
    for(float fY = -0.5; fY <= 1.0; fY += 0.01) {
      float fValue = fncDensity({0, fX, fY});
	  
      std::cout << fX << ", " << fY << ", " << fValue << std::endl;
    }
  }
      
  return EXIT_SUCCESS;

}

int mvg::MultiVarGaussDriver::runJNIMethod(char* inputName, char* outputName)
{
  std::ofstream outputFile;
  outputFile.open(outputName);


  mvg::MultiVarGauss<float> mvgMain = createMultiVarGauss(inputName);
      
  mvg::MultiVarGauss<float>::DensityFunction fncDensity = mvgMain.densityFunction();
      
  int nBoundary = 2.0;
      
  for(float fX = 0.1; fX <= 1.2; fX += 0.01) {
    for(float fY = -0.5; fY <= 1.0; fY += 0.01) {
      float fValue = fncDensity({0, fX, fY});
	  
      outputFile << fX << ", " << fY << ", " << fValue << std::endl;
    }
  }

  outputFile.close();      

  return EXIT_SUCCESS;

}


/*int main(int argc, char** argv) {
  int nReturnvalue = EXIT_FAILURE;
  
  if(argc > 1) {
    mvg::MultiVarGaussDriver mvgd;
    mvgd.runMainMethod(argv[0]);
    
  } else {
    std::cerr << "Missing argument: filename" << std::endl;
  }
  
  return nReturnvalue;
}*/
