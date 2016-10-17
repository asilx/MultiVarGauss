#ifndef __DATASET_HPP__
#define __DATASET_HPP__


#include <memory>
#include <iostream>
#include <vector>

#include <Eigen/Dense>


namespace mvg {
  class Dataset {
  public:
    typedef std::shared_ptr<Dataset> Ptr;
    
  private:
    std::vector<Eigen::VectorXf> m_vecData;
    
  protected:
  public:
    Dataset() {
    }
    
    ~Dataset() {
    }
    
    unsigned int dimension() {
      if(m_vecData.size() > 0) {
	return m_vecData[0].size();
      }
      
      return 0;
    }
    
    void add(Eigen::VectorXf vxData) {
      m_vecData.push_back(vxData);
    }
    
    unsigned int count() {
      return m_vecData.size();
    }
    
    Eigen::VectorXf& operator[](unsigned int unIndex) {
      return m_vecData[unIndex];
    }
    
    template<class ... Args>
      static Dataset::Ptr create(Args ... args) {
      return std::make_shared<Dataset>(std::forward<Args>(args)...);
    }
  };
}


#endif /* __DATASET_HPP__ */
