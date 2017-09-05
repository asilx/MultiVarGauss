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

void clusterizeDataset (mvg::KMeans& kmean, unsigned int& maxCluster, mvg::Dataset::Ptr dsData, std::vector<mvg::Dataset::Ptr> vecCluster)
{
   kmean.setSource(dsData);
   std::cout << "Calculating kMeans clusters .. " << std::flush;
   kmean.calculate(1, maxCluster);

   std::cout << "done" << std::endl;
	  
   vecCluster = kmean.clusters();
   std::cout << "Optimal cluster count: " << vecCluster.size() << std::endl;
	  
   unsigned int unSumSamplesUsed = 0;
   for(unsigned int unI = 0; unI < vecCluster.size(); ++unI) {
      std::cout << " * Cluster #" << unI << ": " << vecCluster[unI]->count() << " sample" << (vecCluster[unI]->count() == 1 ? "" : "s") << std::endl;
      unSumSamplesUsed += vecCluster[unI]->count();
   }
	  
   unsigned int unRemovedOutliers = dsData->count() - unSumSamplesUsed;
   if(unRemovedOutliers > 0) {
      std::cout << "Removed " << unRemovedOutliers << " outlier" << (unRemovedOutliers == 1 ? "" : "s") << " from positive dataset" << std::endl;
   }
}

void addDataSetsToMixedGaussian (unsigned int& maxCluster, mvg::Dataset::Ptr dsData, std::vector<mvg::Dataset::Ptr> vecCluster, mvg::MixedGaussians<double>& mgGaussians)
{
   if (maxCluster > 1)
   {
      for(mvg::Dataset::Ptr dsCluster : vecCluster) {
         mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
         mvgGaussian->setDataset(dsCluster);
	 mgGaussians.addGaussian(mvgGaussian, 1.0);
      }
   }
   else
   {
      mvg::MultiVarGauss<double>::Ptr mvgGaussian = mvg::MultiVarGauss<double>::create();
      mvgGaussian->setDataset(dsData);
      mgGaussians.addGaussian(mvgGaussian, 1.0);
   }
}

void calculateClustersandBoundingBoxes(unsigned int& positiveClusterNumber, unsigned int& negativeClusterNumber, mvg::Dataset::Ptr dsDataPos, mvg::Dataset::Ptr dsDataNeg, 
                                       std::vector<mvg::Dataset::Ptr> vecClustersPos, std::vector<mvg::Dataset::Ptr> vecClustersNeg, mvg::MixedGaussians<double>& mgGaussiansPos, 
                                       mvg::MixedGaussians<double>& mgGaussiansNeg, double& min_x, double& min_y, double& max_x, double& max_y)
{
   mvg::KMeans kmMeansPos;
   mvg::KMeans kmMeansNeg;
   if (positiveClusterNumber > 1)
   {
      clusterizeDataset (kmMeansPos, positiveClusterNumber, dsDataPos, vecClustersPos);
   }

   if (negativeClusterNumber > 1)
   {
      clusterizeDataset (kmMeansPos, positiveClusterNumber, dsDataPos, vecClustersPos);
   }

   addDataSetsToMixedGaussian(positiveClusterNumber, dsDataPos, vecClustersPos, mgGaussiansPos);
   addDataSetsToMixedGaussian(negativeClusterNumber, dsDataNeg, vecClustersPos, mgGaussiansNeg);

   mvg::MultiVarGauss<double>::Rect rctBBPos = mgGaussiansPos.boundingBox();
   mvg::MultiVarGauss<double>::Rect rctBBNeg = mgGaussiansNeg.boundingBox();
   /*if(rctBBPos.vecMin[0] > rctBBNeg.vecMin[0]) min_x = rctBBNeg.vecMin[0]; else min_x = rctBBPos.vecMin[0];
    if(rctBBPos.vecMin[1] > rctBBNeg.vecMin[1]) min_y = rctBBNeg.vecMin[1]; else min_y = rctBBPos.vecMin[1];
    if(rctBBPos.vecMax[0] > rctBBNeg.vecMax[0]) max_x = rctBBPos.vecMax[0]; else max_x = rctBBNeg.vecMax[0];
    if(rctBBPos.vecMax[1] > rctBBNeg.vecMax[1]) max_y = rctBBPos.vecMax[1]; else max_y = rctBBNeg.vecMax[1];*/ 

   min_x = rctBBPos.vecMin[0];//positive bb is enough for visualization. It does matter most.
   max_x = rctBBPos.vecMax[0];
   min_y = rctBBPos.vecMin[1];
   max_y = rctBBPos.vecMax[1];
 
   std::cout << "Clusters bounding box: [" << min_x << ", " << min_y << "] --> [" << max_x << ", " << max_y << "]" << std::endl;
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

    unsigned int positiveClusterNumber = (unsigned int) positiveClusters;
    unsigned int negativeClusterNumber = (unsigned int) negativeClusters;
    
    if(fileExists(strPosFile) & fileExists(strNegFile)) {
      std::cout << "Trial Analysis: '" << strPosFile << "' & '" << strNegFile << "'" << std::endl;
      
      mvg::Dataset::Ptr dsDataPos = loadCSV(strPosFile, {0, 1});
      std::cout << "Positive Dataset: " << dsDataPos->count() << " samples with " << dsDataPos->dimension() << " dimension" << (dsDataPos->dimension() == 1 ? "" : "s") << std::endl;
      mvg::Dataset::Ptr dsDataNeg = loadCSV(strNegFile, {0, 1});
      std::cout << "Negative Dataset: " << dsDataNeg->count() << " samples with " << dsDataNeg->dimension() << " dimension" << (dsDataNeg->dimension() == 1 ? "" : "s") << std::endl;
      
      if(dsDataPos && dsDataNeg) {

        std::vector<mvg::Dataset::Ptr> vecClustersPos;
        std::vector<mvg::Dataset::Ptr> vecClustersNeg;
	
        mvg::MixedGaussians<double> mgGaussiansPos;
        mvg::MixedGaussians<double> mgGaussiansNeg;

        double min_x, min_y, max_x, max_y;
        calculateClustersandBoundingBoxes(positiveClusterNumber, negativeClusterNumber,dsDataPos, dsDataNeg, vecClustersPos, vecClustersNeg, mgGaussiansPos, mgGaussiansNeg, min_x, min_y, max_x, max_y);

        // Two dimensional case
	float fStepSizeX = 0.01;
	float fStepSizeY = 0.01;
	  
	std::cout << "Writing CSV file (step size = [" << fStepSizeX << ", " << fStepSizeY << "]) .. " << std::endl;
	  
	std::ofstream ofFile(strFileOut, std::ios::out);
	 
	jdoubleArray maximized_expectation = env->NewDoubleArray(2);
	float maxValue = -1;
	float maxValueIndX = -1;
	float maxValueIndY = -1;
	   
        mvg::MultiVarGauss<double>::DensityFunction fncDensityPos = mgGaussiansPos.densityFunction();
        mvg::MultiVarGauss<double>::DensityFunction fncDensityNeg = mgGaussiansNeg.densityFunction();
	for(float fX = min_x; fX < max_x; fX += fStepSizeX) {
	  for(float fY = min_y; fY < max_y; fY += fStepSizeY) {
            float fValuePos = fncDensityPos({fX, fY});
            float fValueNeg = fncDensityNeg({fX, fY});
            float fValue = (fValuePos + (1 - fValueNeg))/2;
           

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
        pMax[0] = (double) maxValueIndX;
        pMax[1] = (double) maxValueIndY;
        ofFile.close();
	  
        std::cout << maxValueIndX << "-" << maxValueIndY << std::endl;
	std::cout << "done" << std::endl;
	env->ReleaseStringUTFChars(inputPosJava, inputPosString);
        env->ReleaseStringUTFChars(inputNegJava, inputNegString);
        env->ReleaseStringUTFChars(outputJava, outputString);
	env->ReleaseDoubleArrayElements(maximized_expectation, pMax, NULL);      
	return maximized_expectation;
	
      }
    } else {
      std::cerr << "Error: Input files not found " << std::endl;
    }
    env->ReleaseStringUTFChars(inputPosJava, inputPosString);
    env->ReleaseStringUTFChars(inputNegJava, inputNegString);
    env->ReleaseStringUTFChars(outputJava, outputString);
}

//first two elements are mean. Last four are covariance
JNIEXPORT jdoubleArray JNICALL Java_org_knowrob_gaussian_MixedGaussianInterface_likelyLocationClosest(JNIEnv* env, jobject obj, jstring inputPosJava, jstring inputNegJava,  jint positiveClusters, jint negativeClusters)
{
    const char *inputPosString = env->GetStringUTFChars(inputPosJava, 0);
    const char *inputNegString = env->GetStringUTFChars(inputNegJava, 0);

    std::string strPosFile = inputPosString;
    std::string strNegFile = inputNegString;

    unsigned int positiveClusterNumber = (unsigned int) positiveClusters;
    unsigned int negativeClusterNumber = (unsigned int) negativeClusters;
    
    if(fileExists(strPosFile) & fileExists(strNegFile)) {
      std::cout << "Trial Analysis: '" << strPosFile << "' & '" << strNegFile << "'" << std::endl;
      
      mvg::Dataset::Ptr dsDataPos = loadCSV(strPosFile, {0, 1, 3});
      std::cout << "Positive Dataset: " << dsDataPos->count() << " samples with " << dsDataPos->dimension() << " dimension" << (dsDataPos->dimension() == 1 ? "" : "s") << std::endl;
      mvg::Dataset::Ptr dsDataNeg = loadCSV(strNegFile, {0, 1, 3});
      std::cout << "Negative Dataset: " << dsDataNeg->count() << " samples with " << dsDataNeg->dimension() << " dimension" << (dsDataNeg->dimension() == 1 ? "" : "s") << std::endl;
      
      if(dsDataPos && dsDataNeg) {

        std::vector<mvg::Dataset::Ptr> vecClustersPos;
        std::vector<mvg::Dataset::Ptr> vecClustersNeg;
	
        mvg::MixedGaussians<double> mgGaussiansPos;
        mvg::MixedGaussians<double> mgGaussiansNeg;

        double min_x, min_y, max_x, max_y;
        calculateClustersandBoundingBoxes(positiveClusterNumber, negativeClusterNumber,dsDataPos, dsDataNeg, vecClustersPos, vecClustersNeg, mgGaussiansPos, mgGaussiansNeg, min_x, min_y, max_x, max_y);

        // Two dimensional case
	float fStepSizeX = 0.01;
	float fStepSizeY = 0.01;
	  
	 
	jdoubleArray expectation_gauss = env->NewDoubleArray(6);
	float maxValue = -1;
	float maxValueIndX = -1;
	float maxValueIndY = -1;
       
        mvg::MultiVarGauss<double>::DensityFunction fncDensityPos = mgGaussiansPos.densityFunction();
        mvg::MultiVarGauss<double>::DensityFunction fncDensityNeg = mgGaussiansNeg.densityFunction();
	for(float fX = min_x; fX < max_x; fX += fStepSizeX) {
	  for(float fY = min_y; fY < max_y; fY += fStepSizeY) {
            float fValuePos = fncDensityPos({fX, fY});
            float fValueNeg = fncDensityNeg({fX, fY});
            float fValue = 0;
            
            fValue = (fValuePos + (1- fValueNeg)) / 2;

            if (fValue < 0) fValue = 0;
            if (fValue > 1) fValue = 1;
            if (fValue != fValue) fValue = 0;
            if(maxValue < fValue)
	       maxValue = fValue;
	  }
	}
        //get a gaussian for maximized locations
        mvg::Dataset::Ptr dsDataMax = mvg::Dataset::create();
        for(float fX = min_x; fX < max_x; fX += fStepSizeX) {
	  for(float fY = min_y; fY < max_y; fY += fStepSizeY) {
            float fValuePos = fncDensityPos({fX, fY});
            float fValueNeg = fncDensityNeg({fX, fY});
            float fValue = 0;
            
            fValue = (fValuePos + (1- fValueNeg)) / 2;

            if (fValue < 0) fValue = 0;
            if (fValue > 1) fValue = 1;
            if (fValue != fValue) fValue = 0;

            if(maxValue == fValue)
	    {
               Eigen::VectorXf vxdData(2);
               vxdData[0] = fX;
               vxdData[1] = fY;
               dsDataMax->add(vxdData);
	    }
          }
        }
        mvg::MultiVarGauss<double>::Ptr mvgGaussianMax = mvg::MultiVarGauss<double>::create();
        mvgGaussianMax->setDataset(dsDataMax);
        Eigen::VectorXf meanV = mvgGaussianMax->dataMean();
        Eigen::MatrixXf cV = mvgGaussianMax->covariance();

	jdouble *pMax = env->GetDoubleArrayElements(expectation_gauss, NULL);
        
        pMax[0] = (double) meanV[0];
        pMax[1] = (double) meanV[1];
        pMax[2] = (double) cV(0,0);
        pMax[3] = (double) cV(0,1);
        pMax[4] = (double) cV(1,0);
        pMax[5] = (double) cV(1,1);
	  
        std::cout << maxValueIndX << "-" << maxValueIndY << std::endl;
	std::cout << "done" << std::endl;
	env->ReleaseStringUTFChars(inputPosJava, inputPosString);
        env->ReleaseStringUTFChars(inputNegJava, inputNegString);
	env->ReleaseDoubleArrayElements(expectation_gauss, pMax, NULL);      
	return expectation_gauss;
	
      }
    } else {
      std::cerr << "Error: Input files not found " << std::endl;
    }
    env->ReleaseStringUTFChars(inputPosJava, inputPosString);
    env->ReleaseStringUTFChars(inputNegJava, inputNegString);
}
