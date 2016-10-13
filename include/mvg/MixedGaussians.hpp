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
    
    float sample(std::vector<T> vecValues) {
      double dWeightSum = 0.0;
      for(Gaussian& gsGaussian : m_vecGaussians) {
	dWeightSum += gsGaussian.dWeight;
      }
      
      // Generate a uniformly distributed random variable to choose
      // the distribution to sample from, based on the distributions'
      // weights.
      double dRandom = ((double)rand() / (double)RAND_MAX) * dWeightSum;
      
      for(Gaussian& gsGaussian : m_vecGaussians) {
	if(dRandom > gsGaussian.dWeight) {
	  dRandom -= gsGaussian.dWeight;
	} else {
	  return gsGaussian.fncDensity(vecValues);
	}
      }
      
      return 0.0;
    }
    
    void recalculateDensityFunctions() {
      // The `MultiVarGauss::densityFunction()` function does some
      // computation based on the data inside the MultiVarGauss
      // instance. That's why we're getting the function object
      // instead of calling `densityFunction()` each time we want to
      // sample from the distribution.
      for(Gaussian& gsGaussian : m_vecGaussians) {
	gsGaussian.fncDensity = gsGaussian.mvgGaussian.densityFunction();
      }
    }
    
    typename MultiVarGauss<T>::DensityFunction densityFunction() {
      return [this](std::vector<T>& vecValues) -> float {
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
