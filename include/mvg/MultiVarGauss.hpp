#ifndef __MULTIVARGAUSS_HPP__
#define __MULTIVARGAUSS_HPP__


#include <memory>
#include <iostream>
#include <functional>

#include <Eigen/LU>
#include <Eigen/Dense>
#include <unsupported/Eigen/src/MatrixFunctions/MatrixExponential.h>

#include <mvg/Dataset.hpp>


namespace mvg {
  template<typename T>
  class MultiVarGauss {
  public:
    typedef std::shared_ptr<MultiVarGauss> Ptr;
    typedef std::function<float(std::vector<T>)> DensityFunction;
    
    typedef struct {
      std::vector<T> vecMin;
      std::vector<T> vecMax;
    } Rect;
    
  private:
    typename Dataset::Ptr m_dsData;
    
  protected:
  public:
    MultiVarGauss() : m_dsData(nullptr) {
    }
    
    ~MultiVarGauss() {
    }
    
    unsigned int dataDimension() {
      if(m_dsData) {
	return m_dsData->dimension();
      }
      
      return 0;
    }
    
    Eigen::VectorXf dataMean() {
      unsigned int unSize = this->dataDimension();
      Eigen::VectorXf vxMean = Eigen::VectorXf::Zero(unSize);
      
      for(unsigned int unI = 0; unI < m_dsData->count(); ++unI) {
	vxMean += (*m_dsData)[unI];
      }
      
      vxMean /= m_dsData->count();
      return vxMean;
    }
    
    void setDataset(typename Dataset::Ptr dsData) {
      m_dsData = dsData;
    }
    
    static void addToDataset(mvg::Dataset::Ptr dsDataset, std::vector<float> vecData) {
      Eigen::VectorXf vxdData(vecData.size());
      
      for(unsigned int unI = 0; unI < vecData.size(); ++unI) {
	vxdData[unI] = vecData[unI];
      }
      
      dsDataset->add(vxdData);
    }
    
    void setDataset(std::vector<std::vector<T>> vecData) {
      Dataset::Ptr dsSet = Dataset::create();
      
      for(std::vector<T> vecRow : vecData) {
	this->addToDataset(dsSet, vecRow);
      }
      
      this->setDataset(dsSet);
    }
    
    Eigen::MatrixXf covariance() {
      unsigned int unSize = this->dataDimension();
      
      Eigen::MatrixXf mxCov = Eigen::MatrixXf::Zero(unSize, unSize);
      Eigen::VectorXf vxMean = this->dataMean();
      
      for(unsigned int unI = 0; unI < m_dsData->count(); ++unI) {
	Eigen::VectorXf vxDiff = ((*m_dsData)[unI] - vxMean).conjugate();
	mxCov += vxDiff * vxDiff.adjoint();
      }
      
      mxCov /= m_dsData->count();
      return mxCov;
    }
    
    DensityFunction densityFunction() {
      Eigen::MatrixXf mxCov = this->covariance();
      Eigen::MatrixXf mxCovInverse = mxCov.inverse();
      
      float fDet = mxCov.determinant();
      unsigned int unSize = this->dataDimension();
      
      float fTwoPiToTheN = pow(2 * M_PI, unSize);
      float fTwoPiToTheNTimesDet = fTwoPiToTheN * fDet;
      float fCoefficient = 1.0 / sqrt(fTwoPiToTheNTimesDet);
      
      Eigen::VectorXf vxMean = this->dataMean();
      
      return [mxCovInverse, fCoefficient, vxMean](std::vector<T> vecPoint) -> float {
	Eigen::VectorXf vxX(vecPoint.size());
	
	for(unsigned int unI = 0; unI < vecPoint.size(); ++unI) {
	  vxX[unI] = vecPoint[unI];
	}
	
	Eigen::VectorXf vxDiff = vxX - vxMean;
	float fExp = exp(vxDiff.transpose() * (mxCovInverse / -2) * vxDiff);
	
	return fCoefficient * fExp;
      };
    }
    
    Rect boundingBox() {
      Rect rctBB;
      
      if(m_dsData && m_dsData->count() > 0) {
	unsigned int unSize = this->dataDimension();
	
	rctBB.vecMin.resize(unSize);
	rctBB.vecMax.resize(unSize);
	
	for(unsigned int unI = 0; unI < unSize; ++unI) {
	  rctBB.vecMin[unI] = (*m_dsData)[0][unI];
	  rctBB.vecMax[unI] = (*m_dsData)[0][unI];
	}
	
	for(unsigned int unD = 1; unD < m_dsData->count(); ++unD) {
	  for(unsigned int unI = 0; unI < unSize; ++unI) {
	    if((*m_dsData)[unD][unI] < rctBB.vecMin[unI]) {
	      rctBB.vecMin[unI] = (*m_dsData)[unD][unI];
	    } else if((*m_dsData)[unD][unI] > rctBB.vecMax[unI]) {
	      rctBB.vecMax[unI] = (*m_dsData)[unD][unI];
	    }
	  }
	}
      }
      
      return rctBB;
    }
    
    template<class ... Args>
      static MultiVarGauss::Ptr create(Args ... args) {
      return std::make_shared<MultiVarGauss>(std::forward<Args>(args)...);
    }
  };

  class MultiVarGaussDriver{
  public:    
    int runMainMethod(char* inputName);
    int runJNIMethod(char* inputName, char* fileName);
  private:
    mvg::MultiVarGauss<float> createMultiVarGauss(char* inputName);
    std::map<unsigned int, std::map<std::string, unsigned int>> mapNominalValues;
    unsigned int nominalValue(unsigned int unRow, std::string strValue); 
  };  
}


#endif /* __MULTIVARGAUSS_HPP__ */
