#ifndef __MIXEDGAUSSIANS_H__
#define __MIXEDGAUSSIANS_H__


#include <memory>
#include <iostream>

#include <mvg/MultiVarGauss.hpp>


namespace mvg {
  template<typename T>
  class MixedGaussians {
  public:
    typedef std::shared_ptr<MixedGaussians> Ptr;
    
    typedef struct {
      typename MultiVarGauss<T>::Ptr mvgGaussian;
      double dWeight;
      typename MultiVarGauss<T>::DensityFunction fncDensity;
    } Gaussian;
    
  private:
    std::vector<Gaussian> m_vecGaussians;
    
  protected:
  public:
    MixedGaussians() {}
    ~MixedGaussians() {}
    
    void addGaussian(typename MultiVarGauss<T>::Ptr mvgGaussian, double dWeight) {
      m_vecGaussians.push_back({mvgGaussian, dWeight, mvgGaussian->densityFunction()});
    }
    
    T sample(std::vector<T> vecValues) {
      double dWeightSum = 0.0;
      for(Gaussian& gsGaussian : m_vecGaussians) {
	dWeightSum += gsGaussian.dWeight;
      }
      
      // Generate a uniformly distributed random variable to choose
      // the distribution to sample from, based on the distributions'
      // weights.
      //double dRandom = ((double)rand() / (double)RAND_MAX) * dWeightSum;
      
      T tSample = T();
      for(Gaussian& gsGaussian : m_vecGaussians) {
	tSample += gsGaussian.dWeight * gsGaussian.fncDensity(vecValues);
	// if(dRandom > gsGaussian.dWeight) {
	//   dRandom -= gsGaussian.dWeight;
	// } else {
	//   return gsGaussian.fncDensity(vecValues);
	// }
      }
      
      return tSample;
    }
    
    void recalculateDensityFunctions() {
      // The `MultiVarGauss::densityFunction()` function does some
      // computation based on the data inside the MultiVarGauss
      // instance. That's why we're getting the function object
      // instead of calling `densityFunction()` each time we want to
      // sample from the distribution.
      for(Gaussian& gsGaussian : m_vecGaussians) {
	gsGaussian.fncDensity = gsGaussian.mvgGaussian->densityFunction();
      }
    }
    
    typename MultiVarGauss<T>::Rect boundingBox() {
      typename MultiVarGauss<T>::Rect rctBB;
      
      if(m_vecGaussians.size() > 0) {
	rctBB = m_vecGaussians[0].mvgGaussian->boundingBox();
	
	for(Gaussian& gsGaussian : m_vecGaussians) {
	  typename MultiVarGauss<T>::Rect rctCurrent = gsGaussian.mvgGaussian->boundingBox();
	  
	  for(unsigned int unI = 0; unI < rctCurrent.vecMin.size(); ++unI) {
	    if(rctCurrent.vecMin[unI] < rctBB.vecMin[unI]) {
	      rctBB.vecMin[unI] = rctCurrent.vecMin[unI];
	    }
	    
	    if(rctCurrent.vecMax[unI] > rctBB.vecMax[unI]) {
	      rctBB.vecMax[unI] = rctCurrent.vecMax[unI];
	    }
	  }
	}
      }
      
      return rctBB;
    }
    
    typename MultiVarGauss<T>::DensityFunction densityFunction() {
      this->recalculateDensityFunctions();
      
      return [this](std::vector<T> vecValues) -> T {
	return this->sample(vecValues);
      };
    }
    
    template<class ... Args>
    static MixedGaussians<T>::Ptr create(Args ... args) {
      return std::make_shared<MixedGaussians<T>>(std::forward<Args>(args)...);
    }
  };
}


#endif /* __MIXEDGAUSSIANS_H__ */
