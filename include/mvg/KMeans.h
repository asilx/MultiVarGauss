#ifndef __KMEANS_H__
#define __KMEANS_H__


#include <memory>
#include <iostream>
#include <string>
#include <fstream>

#include <mpi_kmeans.h>

#include <mvg/Dataset.hpp>


namespace mvg {
  class KMeans {
  public:
    typedef std::shared_ptr<KMeans> Ptr;
    
  private:
    Dataset::Ptr m_dsSource;
    std::vector<Dataset::Ptr> m_vecClusters;
    
  protected:
  public:
    KMeans();
    ~KMeans();
    
    void setSource(Dataset::Ptr dsSource);
    bool calculate(unsigned int unMinClusters, unsigned int unMaxClusters);
    bool calculate(unsigned int unClusters);
    std::vector<Dataset::Ptr> clusters();
    
    double dissimilarity(Eigen::VectorXf evcPoint, unsigned int unCluster);
    std::vector<std::vector<double>> silhouettes();
    double silhouetteAverage(unsigned int unClusters);
    
    template<class ... Args>
      static KMeans::Ptr create(Args ... args) {
      return std::make_shared<KMeans>(std::forward<Args>(args)...);
    }
  };
}


#endif /* __KMEANS_H__ */
