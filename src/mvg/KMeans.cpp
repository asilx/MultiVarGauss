#include <mvg/KMeans.h>


namespace mvg {
  KMeans::KMeans() : m_dsSource(nullptr) {
  }
  
  KMeans::~KMeans() {
  }
  
  void KMeans::setSource(Dataset::Ptr dsSource) {
    m_dsSource = dsSource;
  }
  
  bool KMeans::calculate(unsigned int unMinClusters, unsigned int unMaxClusters) {
    unsigned int unDimension = m_dsSource->dimension();
    
    unsigned int unBestClusterCount = 0;
    double dLowestAverageSilhouetteValue = -1;
    
    for(unsigned int unClusters = unMinClusters; unClusters <= unMaxClusters; ++unClusters) {
      if(unClusters < unDimension - 1 || unClusters > unDimension + 1) {
	double dAverageSilhouetteValue = this->silhouetteAverage(unClusters);
	
	if(dLowestAverageSilhouetteValue == -1 || dAverageSilhouetteValue > dLowestAverageSilhouetteValue) {
	  unBestClusterCount = unClusters;
	  dLowestAverageSilhouetteValue = dAverageSilhouetteValue;
	}
      }
    }
    
    this->calculate(unBestClusterCount);
    
    return (dLowestAverageSilhouetteValue > -1);
  }
  
  bool KMeans::calculate(unsigned int unClusters) {
    if(m_dsSource) {
      unsigned int unDimensions = m_dsSource->dimension();
      
      if(unDimensions > 0) {
	unsigned int unSamples = m_dsSource->count();
	double arrdElements[unDimensions * unSamples];
	
	for(unsigned int unSample = 0; unSample < unSamples; ++unSample) {
	  for(unsigned int unDimension = 0; unDimension < unDimensions; ++unDimension) {
	    arrdElements[unSample * unDimensions + unDimension] = (*m_dsSource)[unSample][unDimension];
	  }
	}
	
	double arrdClusters[unClusters * unDimensions];
	memset(arrdClusters, sizeof(double) * unDimensions * unSamples, 0);
	
	unsigned int arrunAssignment[unSamples];
	
	unsigned int unMaxIter = 100;
	unsigned int unMaxRestart = 10;
	
	double dSumSquaredError = kmeans(arrdClusters, arrdElements, arrunAssignment, unDimensions, unSamples, unClusters, unMaxIter, unMaxRestart);
	
	m_vecClusters.clear();
	for(unsigned int unI = 0; unI < unClusters; ++unI) {
	  m_vecClusters.push_back(Dataset::create());
	}
	
	for(unsigned int unI = 0; unI < unSamples; ++unI) {
	  m_vecClusters[arrunAssignment[unI]]->add((*m_dsSource)[unI]);
	}
	
	return true;
      }
    }
    
    return false;
  }
  
  std::vector<Dataset::Ptr> KMeans::clusters() {
    return m_vecClusters;
  }
  
  double KMeans::dissimilarity(Eigen::VectorXf evcPoint, unsigned int unCluster) {
    unsigned int unCount = 0;
    double dDistance = 0.0;
    
    Dataset::Ptr dsCluster = m_vecClusters[unCluster];
    
    for(unsigned int unI = 0; unI < dsCluster->count(); ++unI) {
      double dSumIntermediate = 0.0;
      
      for(unsigned int unD = 0; unD < evcPoint.size(); ++unD) {
	dSumIntermediate += ((*dsCluster)[unI][unD] - evcPoint[unD]) * ((*dsCluster)[unI][unD] - evcPoint[unD]);
      }
      
      dDistance += sqrt(dSumIntermediate);
    }
    
    return dDistance / (double)dsCluster->count();
  }
  
  std::vector<std::vector<double>> KMeans::silhouettes() {
    std::vector<std::vector<double>> vecSilhouettes;
    
    for(unsigned int unI = 0; unI < m_vecClusters.size(); ++unI) {
      vecSilhouettes.push_back(std::vector<double>());
      
      for(unsigned int unP = 0; unP < m_vecClusters[unI]->count(); ++unP) {
	double dOwnDissimilarity = this->dissimilarity((*m_vecClusters[unI])[unP], unI);
	
	double dLowestOtherDissimilarity = -1;
	for(unsigned int unOC = 0; unOC < m_vecClusters.size(); ++unOC) {
	  if(unOC != unI) {
	    double dOtherDissimilarity = this->dissimilarity((*m_vecClusters[unI])[unP], unOC);
	    
	    if(dLowestOtherDissimilarity < 0 || dOtherDissimilarity < dLowestOtherDissimilarity) {
	      dLowestOtherDissimilarity = dOtherDissimilarity;
	    }
	  }
	}
	
	double dSilhouetteValue = 0;
	if(dOwnDissimilarity < dLowestOtherDissimilarity) {
	  dSilhouetteValue = 1 - dOwnDissimilarity / dLowestOtherDissimilarity;
	} else if(dOwnDissimilarity > dLowestOtherDissimilarity) {
	  dSilhouetteValue = dLowestOtherDissimilarity / dOwnDissimilarity - 1;
	}
	
	vecSilhouettes[unI].push_back(dSilhouetteValue);
      }
    }
    
    return vecSilhouettes;
  }
  
  double KMeans::silhouetteAverage(unsigned int unClusters) {
    this->calculate(unClusters);
    std::vector<std::vector<double>> vecSilhouettes = this->silhouettes();
    
    unsigned int unCount = 0;
    double dSum = 0.0;
    
    for(std::vector<double> vecClusterSilhouette : vecSilhouettes) {
      for(double dSilhouetteValue : vecClusterSilhouette) {
	dSum += dSilhouetteValue;
	unCount++;
      }
    }
    
    return dSum / (double)unCount;
  }
}
