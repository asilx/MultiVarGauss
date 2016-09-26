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
	if(m_dsData->count() > 0) {
	  return (*m_dsData)[0].size();
	}
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
    
    template<class ... Args>
      static MultiVarGauss::Ptr create(Args ... args) {
      return std::make_shared<MultiVarGauss>(std::forward<Args>(args)...);
    }
  };
}


#endif /* __MULTIVARGAUSS_HPP__ */
