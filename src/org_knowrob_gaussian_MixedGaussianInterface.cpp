#include <org_knowrob_gaussian_MixedGaussianInterface.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <map>

#include <Eigen/Dense>
#include <mvg/JSON.h>
#include <mvg/MixedGaussians.hpp>
#include <mvg/MultiVarGauss.hpp>

#include <iomanip>
#include <string>

#include <mvg/KMeans.h>
#include <mvg/MixedGaussians.hpp>


bool fileExists(std::string strFilepath) {
  std::ifstream ifFile(strFilepath, std::ios::in);
  
  return ifFile.good();
}


std::string makeOutputFilename(std::string strFileIn) {
  size_t szSlash = strFileIn.find_last_of("/");
  if(szSlash == std::string::npos) {
    szSlash = -1;
  }
  
  size_t szDot = strFileIn.find_first_of(".", szSlash);
  if(szDot == std::string::npos) {
    szDot = strFileIn.length();
  }
  
  szSlash++;
  
  return strFileIn.substr(szSlash, szDot - szSlash) + ".out" + strFileIn.substr(szDot);
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


JNIEXPORT void JNICALL Java_org_knowrob_gaussian_MixedGaussianInterface_createMultiVarGaussians(JNIEnv* env, jobject obj, jstring inputJava, jstring outputJava)
{
   const char *inputString = env->GetStringUTFChars(inputJava, 0);
   const char *outputString = env->GetStringUTFChars(outputJava, 0);

   mvg::MultiVarGaussDriver mvgd;
   mvgd.runJNIMethod(const_cast<char*>(inputString), const_cast<char*>(outputString));

   env->ReleaseStringUTFChars(inputJava, inputString);
   env->ReleaseStringUTFChars(outputJava, outputString);
}

JNIEXPORT void JNICALL Java_org_knowrob_gaussian_MixedGaussianInterface_createMixedGaussians(JNIEnv* env, jobject obj, jstring outputJava)
{
   const char *nativeString = env->GetStringUTFChars(outputJava, JNI_FALSE);

   mvg::MixedGaussiansDriver::runJNIMethod(const_cast<char*>(nativeString));

   env->ReleaseStringUTFChars(outputJava, nativeString);
   
}

JNIEXPORT void JNICALL Java_org_knowrob_gaussian_MixedGaussianInterface_analyzeCluster(JNIEnv* env, jobject obj, jstring inputJava, jstring outputJava)
{
    const char *inputString = env->GetStringUTFChars(inputJava, 0);
    const char *outputString = env->GetStringUTFChars(outputJava, 0);

    std::string strFileIn = inputString;
    std::string strFileOut = outputString;
    
    if(fileExists(strFileIn)) {
      std::cout << "Cluster Analysis: '" << strFileIn << "' --> '" << strFileOut << "'" << std::endl;
      
      mvg::KMeans kmMeans;
      mvg::Dataset::Ptr dsData = loadCSV(strFileIn, {0, 1});
      std::cout << "Dataset: " << dsData->count() << " samples with " << dsData->dimension() << " dimension" << (dsData->dimension() == 1 ? "" : "s") << std::endl;
      
      if(dsData) {
	kmMeans.setSource(dsData);
	std::cout << "Calculating KMeans clusters .. " << std::flush;
	
	if(kmMeans.calculate(1, 5)) {
	  std::cout << "done" << std::endl;
	  
	  std::vector<mvg::Dataset::Ptr> vecClusters = kmMeans.clusters();
	  std::cout << "Optimal cluster count: " << vecClusters.size() << std::endl;
	  
	  unsigned int unSumSamplesUsed = 0;
	  for(unsigned int unI = 0; unI < vecClusters.size(); ++unI) {
	    std::cout << " * Cluster #" << unI << ": " << vecClusters[unI]->count() << " sample" << (vecClusters[unI]->count() == 1 ? "" : "s") << std::endl;
	    unSumSamplesUsed += vecClusters[unI]->count();
	  }
	  
	  unsigned int unRemovedOutliers = dsData->count() - unSumSamplesUsed;
	  if(unRemovedOutliers > 0) {
	    std::cout << "Removed " << unRemovedOutliers << " outlier" << (unRemovedOutliers == 1 ? "" : "s") << std::endl;
	  }
	  
	  mvg::MixedGaussians<double> mgGaussians;
	  
	  for(mvg::Dataset::Ptr dsCluster : vecClusters) {
	    mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
	    mvgGaussian->setDataset(dsCluster);
	    //mvgGaussian->setDataset(dsData);
	    mgGaussians.addGaussian(mvgGaussian, 1.0);
	    //break;
	  }
	  
	  mvg::MultiVarGauss<double>::Rect rctBB = mgGaussians.boundingBox();
          mvg::MultiVarGauss<double>::DensityFunction fncDensity = mgGaussians.densityFunction();
	  
	  std::cout << "Clusters bounding box: [" << rctBB.vecMin[0] << ", " << rctBB.vecMin[1] << "] --> [" << rctBB.vecMax[0] << ", " << rctBB.vecMax[1] << "]" << std::endl;
	  
	  // Two dimensional case
	  float fStepSizeX = 0.01;
	  float fStepSizeY = 0.01;
	  
	  std::cout << "Writing CSV file (step size = [" << fStepSizeX << ", " << fStepSizeY << "]) .. " << std::endl;
	  
	  std::ofstream ofFile(strFileOut, std::ios::out);
	  
	  for(float fX = rctBB.vecMin[0]; fX < rctBB.vecMax[0]; fX += fStepSizeX) {
	    for(float fY = rctBB.vecMin[1]; fY < rctBB.vecMax[1]; fY += fStepSizeY) {
              float fValue = fncDensity({fX, fY});
	      ofFile << fX << ", " << fY << ", " << fValue << std::endl;
	    }
	  }
	  
	  ofFile.close();
	  
	  std::cout << "done" << std::endl;
	  
	} else {
	  std::cout << "failed" << std::endl;
	}
      }
    } else {
      std::cerr << "Error: File not found ('" << strFileIn << "')" << std::endl;
    }
    env->ReleaseStringUTFChars(inputJava, inputString);
    env->ReleaseStringUTFChars(outputJava, outputString);
}

JNIEXPORT jdoubleArray JNICALL Java_org_knowrob_gaussian_MixedGaussianInterface_analyzeTrials(JNIEnv* env, jobject obj, jstring inputPosJava, jstring inputNegJava, jstring outputJava, jint positiveClusters, jint negativeClusters)
{
    const char *inputPosString = env->GetStringUTFChars(inputPosJava, 0);
    const char *inputNegString = env->GetStringUTFChars(inputNegJava, 0);
    const char *outputString = env->GetStringUTFChars(outputJava, 0);

    std::string strPosFile = inputPosString;
    std::string strNegFile = inputNegString;
    std::string strFileOut = outputString;

    int positiveClusterNumber = (int) positiveClusters;
    int negativeClusterNumber = (int) negativeClusters;
    
    if(fileExists(strPosFile) & fileExists(strNegFile)) {
      std::cout << "Trial Analysis: '" << strPosFile << "' & '" << strNegFile << "'" << std::endl;
      
      mvg::Dataset::Ptr dsDataPos = loadCSV(strPosFile, {0, 1});
      mvg::KMeans kmMeansPos;
      std::cout << "Positive Dataset: " << dsDataPos->count() << " samples with " << dsDataPos->dimension() << " dimension" << (dsDataPos->dimension() == 1 ? "" : "s") << std::endl;
      mvg::Dataset::Ptr dsDataNeg = loadCSV(strNegFile, {0, 1});
      mvg::KMeans kmMeansNeg;
      std::cout << "Negative Dataset: " << dsDataNeg->count() << " samples with " << dsDataNeg->dimension() << " dimension" << (dsDataNeg->dimension() == 1 ? "" : "s") << std::endl;
      
      if(dsDataPos && dsDataNeg) {

        std::vector<mvg::Dataset::Ptr> vecClustersPos;
        std::vector<mvg::Dataset::Ptr> vecClustersNeg;
	if (positiveClusterNumber > 1)
        {
          kmMeansPos.setSource(dsDataPos);
	  std::cout << "Calculating positive kMeans clusters .. " << std::flush;
          kmMeansPos.calculate(1, positiveClusterNumber);

          std::cout << "done" << std::endl;
	  
	  vecClustersPos = kmMeansPos.clusters();
	  std::cout << "Positive optimal cluster count: " << vecClustersPos.size() << std::endl;
	  
	  unsigned int unSumSamplesUsed = 0;
	  for(unsigned int unI = 0; unI < vecClustersPos.size(); ++unI) {
	    std::cout << " * Cluster #" << unI << ": " << vecClustersPos[unI]->count() << " sample" << (vecClustersPos[unI]->count() == 1 ? "" : "s") << std::endl;
	    unSumSamplesUsed += vecClustersPos[unI]->count();
	  }
	  
	  unsigned int unRemovedOutliers = dsDataPos->count() - unSumSamplesUsed;
	  if(unRemovedOutliers > 0) {
	    std::cout << "Removed " << unRemovedOutliers << " outlier" << (unRemovedOutliers == 1 ? "" : "s") << " from positive dataset" << std::endl;
	  }
	}

        if (negativeClusterNumber > 1)
        {
          kmMeansNeg.setSource(dsDataNeg);
	  std::cout << "Calculating negative kMeans clusters .. " << std::flush;
          kmMeansNeg.calculate(1, negativeClusterNumber);

          std::cout << "done" << std::endl;
	  
	  vecClustersNeg = kmMeansNeg.clusters();
	  std::cout << "Negative optimal cluster count: " << vecClustersNeg.size() << std::endl;
	  
	  unsigned int unSumSamplesUsed = 0;
	  for(unsigned int unI = 0; unI < vecClustersPos.size(); ++unI) {
	    std::cout << " * Cluster #" << unI << ": " << vecClustersNeg[unI]->count() << " sample" << (vecClustersNeg[unI]->count() == 1 ? "" : "s") << std::endl;
	    unSumSamplesUsed += vecClustersNeg[unI]->count();
	  }
	  
	  unsigned int unRemovedOutliers = dsDataNeg->count() - unSumSamplesUsed;
	  if(unRemovedOutliers > 0) {
	    std::cout << "Removed " << unRemovedOutliers << " outlier" << (unRemovedOutliers == 1 ? "" : "s") << " from negative dataset" << std::endl;
	  }
	}
	
        mvg::MixedGaussians<double> mgGaussiansPos;
        mvg::MixedGaussians<double> mgGaussiansNeg;

	if (positiveClusterNumber > 1)
        {
          for(mvg::Dataset::Ptr dsCluster : vecClustersPos) {
            mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
            mvgGaussian->setDataset(dsCluster);
	    mgGaussiansPos.addGaussian(mvgGaussian, 1.0);
	  }
        }
        else
        {
          mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
          mvgGaussian->setDataset(dsDataPos);
	  mgGaussiansPos.addGaussian(mvgGaussian, 1.0);
        }

        if (negativeClusterNumber > 1)
        {
          for(mvg::Dataset::Ptr dsCluster : vecClustersNeg) {
            mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
            mvgGaussian->setDataset(dsCluster);
	    mgGaussiansNeg.addGaussian(mvgGaussian, 1.0);
	  }
        }
        else
        {
          mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
          mvgGaussian->setDataset(dsDataNeg);
	  mgGaussiansNeg.addGaussian(mvgGaussian, 1.0);
        }
	  
	mvg::MultiVarGauss<double>::Rect rctBBPos = mgGaussiansPos.boundingBox();
        mvg::MultiVarGauss<double>::DensityFunction fncDensityPos = mgGaussiansPos.densityFunction();

        mvg::MultiVarGauss<double>::Rect rctBBNeg = mgGaussiansNeg.boundingBox();
        mvg::MultiVarGauss<double>::DensityFunction fncDensityNeg = mgGaussiansNeg.densityFunction();

        double min_x, min_y, max_x, max_y;

        if(rctBBPos.vecMin[0] > rctBBNeg.vecMin[0]) min_x = rctBBNeg.vecMin[0]; else min_x = rctBBPos.vecMin[0];
        if(rctBBPos.vecMin[1] > rctBBNeg.vecMin[1]) min_y = rctBBNeg.vecMin[1]; else min_y = rctBBPos.vecMin[1];
        if(rctBBPos.vecMax[0] > rctBBNeg.vecMax[0]) max_x = rctBBPos.vecMax[0]; else max_x = rctBBNeg.vecMax[0];
        if(rctBBPos.vecMax[1] > rctBBNeg.vecMax[1]) max_y = rctBBPos.vecMax[1]; else max_y = rctBBNeg.vecMax[1]; 
	  
	std::cout << "Clusters bounding box: [" << min_x << ", " << min_y << "] --> [" << max_x << ", " << max_y << "]" << std::endl;
	  
	// Two dimensional case
	float fStepSizeX = 0.01;
	float fStepSizeY = 0.01;
	  
	std::cout << "Writing CSV file (step size = [" << fStepSizeX << ", " << fStepSizeY << "]) .. " << std::endl;
	  
	std::ofstream ofFile(strFileOut, std::ios::out);
	 
	jdoubleArray maximized_expectation = env->NewDoubleArray(2);
	float maxValue = -1;
	float maxValueIndX = -1;
	float maxValueIndY = -1;
	   
	for(float fX = min_x; fX < max_x; fX += fStepSizeX) {
	  for(float fY = min_y; fY < max_y; fY += fStepSizeY) {
            float fValuePos = fncDensityPos({fX, fY});
            float fValueNeg = fncDensityNeg({fX, fY});
            float fValue = fValuePos - fValueNeg;
            if (fValue < 0) fValue = 0;
            if (fValue > 1) fValue = 1;
            if (fValue != fValue) fValue = 0;
            ofFile << fX << ", " << fY << ", " << fValue << std::endl;
            if(maxValue < fValue)
	    {
               maxValue = fValue;
	       maxValueIndX = fX;
               maxValueIndY = fY;
	    }
	  }
	}
	jdouble *pMax = env->GetDoubleArrayElements(maximized_expectation, NULL);
	pMax[0] = maxValueIndX;
        pMax[1] = maxValueIndY;
        ofFile.close();
	  
	std::cout << "done" << std::endl;
	env->ReleaseStringUTFChars(inputPosJava, inputPosString);
        env->ReleaseStringUTFChars(inputNegJava, inputNegString);
        env->ReleaseStringUTFChars(outputJava, outputString);
	      
	return maximized_expectation;
	
      }
    } else {
      std::cerr << "Error: Input files not found " << std::endl;
    }
    env->ReleaseStringUTFChars(inputPosJava, inputPosString);
    env->ReleaseStringUTFChars(inputNegJava, inputNegString);
    env->ReleaseStringUTFChars(outputJava, outputString);
}
