#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <fstream>

#include <mvg/KMeans.h>


bool fileExists(std::string strFilepath) {
  std::ifstream ifFile(strFilepath, std::ios::in);
  
  return ifFile.good();
}


mvg::Dataset::Ptr loadCSV(std::string strFilepath, std::vector<unsigned int> vecUsedIndices = {}) {
  mvg::Dataset::Ptr dsData = nullptr;
  
  std::ifstream ifFile(strFilepath, std::ios::in);
  
  if(ifFile.good()) {
    dsData = mvg::Dataset::create();
    
    std::string strLine;
    std::getline(ifFile, strLine); // Header
    while(std::getline(ifFile, strLine)) {
      std::vector<std::string> vecTokens;
      
      char* cToken = std::strtok((char*)strLine.c_str(), ",");
      while(cToken != nullptr) {
	vecTokens.push_back(cToken);
	cToken = std::strtok(NULL, ",");
      }
      
      std::vector<double> vecData;
      for(unsigned int unI = 0; unI < vecTokens.size(); ++unI) {
	if(vecUsedIndices.size() == 0 || std::find(vecUsedIndices.begin(), vecUsedIndices.end(), unI) != vecUsedIndices.end()) {
	  std::string strToken = vecTokens[unI];
	  
	  try {
	    vecData.push_back(std::stod(strToken));
	  } catch(std::exception& seException) {
	    vecData.push_back(0.0);
	  }
	}
      }
      
      if(vecData.size() > 0) {
	Eigen::VectorXf vxdData(vecData.size());
	for(unsigned int unI = 0; unI < vecData.size(); ++unI) {
	  vxdData[unI] = vecData[unI];
	}
	
	dsData->add(vxdData);
      }
    }
  }
  
  return dsData;
}


int main(int argc, char** argv) {
  int nReturnvalue = EXIT_FAILURE;
  
  if(argc > 1) {
    std::string strFile = argv[1];
    
    if(fileExists(strFile)) {
      std::cout << "Cluster Analysis: '" << strFile << "'" << std::endl;
      
      mvg::KMeans kmMeans;
      mvg::Dataset::Ptr dsData = loadCSV(strFile, {0, 1, 2, 3});
      
      if(dsData) {
	kmMeans.setSource(dsData);
	
	if(kmMeans.calculate(1, 4)) {
	  std::vector<mvg::Dataset::Ptr> vecClusters = kmMeans.clusters();
	  std::vector<std::vector<double>> vecSilhouettes = kmMeans.silhouettes();
	  
	  std::cout << "Optimal cluster count: " << vecSilhouettes.size() << std::endl;
	  
	  for(unsigned int unCluster = 0; unCluster < vecClusters.size(); ++unCluster) {
	    std::cout << std::endl << "Cluster #" << unCluster << ":" << std::flush << std::endl;
	    
	    for(unsigned int unI = 0; unI < vecClusters[unCluster]->count(); ++unI) {
	      for(unsigned int unD = 0; unD < (*vecClusters[unCluster])[unI].size(); ++unD) {
		std::cout << "\t" << std::setprecision(3) << (*vecClusters[unCluster])[unI][unD];
	      }
	      
	      std::cout << std::endl;
	    }
	    
	    // std::cout << "Silhouette #" << unCluster << ":" << std::endl;
	    // for(double dSilhouetteValue : vecSilhouettes[unCluster]) {
	    //   std::cout << "\t" << dSilhouetteValue << std::endl;
	    // }
	  }
	  
	  nReturnvalue = EXIT_SUCCESS;
	}
      }
    } else {
      std::cerr << "Error: File not found ('" << strFile << "')" << std::endl;
    }
  } else {
    std::cerr << "Usage: " << argv[0] << " <data.csv>" << std::endl;
  }
  
  return nReturnvalue;
}
