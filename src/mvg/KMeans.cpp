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
      double dAverageSilhouetteValue = this->silhouetteAverage(unClusters);
      
      if(dLowestAverageSilhouetteValue == -1 || dAverageSilhouetteValue > dLowestAverageSilhouetteValue) {
	unBestClusterCount = unClusters;
	dLowestAverageSilhouetteValue = dAverageSilhouetteValue;
      }
    }
    
    this->calculate(unBestClusterCount);
    
    // Throw out any outliers
    unsigned int unMinSamples = m_dsSource->count() / (2.5 * unBestClusterCount);
    std::vector<Dataset::Ptr> vecFilteredClusters;
    
    for(Dataset::Ptr dsCluster : m_vecClusters) {
      if(dsCluster->count() >= unMinSamples) {
	vecFilteredClusters.push_back(dsCluster);
      }
    }
    
    m_vecClusters = vecFilteredClusters;
    
    return m_vecClusters.size() > 0 && (dLowestAverageSilhouetteValue > -1);
  }
  
  bool KMeans::calculate(unsigned int unClusters) {
    if(m_dsSource) {
      unsigned int unDimensions = m_dsSource->dimension();
      
      if(unDimensions > 0) {
	unsigned int unSamples = m_dsSource->count();
	
	if(unSamples > 0) {
	  if(unSamples < unClusters) {
	    // More clusters than samples doesn't make sense
	    unClusters = unSamples;
	  }
	  
	  srand(time(NULL));
	  
	  std::vector<Eigen::VectorXf> vecCentroids;
	  
	  // Initialize random centroids
	  std::vector<unsigned int> vecSampleIndices;
	  while(vecSampleIndices.size() < unClusters) {
	    unsigned int unSampleIndex = (rand() % unSamples);
	    
	    if(std::find(vecSampleIndices.begin(), vecSampleIndices.end(), unSampleIndex) == vecSampleIndices.end()) {
	      vecSampleIndices.push_back(unSampleIndex);
	    }
	  }
	  
	  for(unsigned int unSampleIndex : vecSampleIndices) {
	    vecCentroids.push_back((*m_dsSource)[unSampleIndex]);
	  }
	  
	  unsigned int unIterations = 0;
	  unsigned int unMaxIterations = 10000;
	  std::vector<Eigen::VectorXf> vecOldCentroids;
	  
	  std::map<unsigned int, unsigned int> mapAssignments;
	  
	  bool bFirstRun = true;
	  bool bGoon = true;
	  while(bGoon) {
	    if(!bFirstRun) {
	      if(unIterations <= unMaxIterations) {
		bool bAllEqual = true;
		
		for(unsigned int unCentroid = 0; unCentroid < vecCentroids.size(); ++unCentroid) {
		  double dTolerance = 1e-4;
		  
		  if((vecCentroids[unCentroid] - vecOldCentroids[unCentroid]).norm() > dTolerance) {
		    bAllEqual = false;
		    break;
		  }
		}
		
		bGoon = !bAllEqual;
	      } else {
		bGoon = false;
	      }
	    } else {
	      bFirstRun = false;
	    }
	    
	    if(bGoon) {
	      vecOldCentroids = vecCentroids;
	      unIterations++;
	      
	      // "Assign labels"
	      for(unsigned int unSample = 0; unSample < unSamples; ++unSample) {
		unsigned int unClosestCentroid = 0;
		double dSmallestDistance = -1;
		
		for(unsigned int unCentroid = 0; unCentroid < vecCentroids.size(); ++unCentroid) {
		  double dDistance = ((*m_dsSource)[unSample] - vecCentroids[unCentroid]).norm();
		  
		  if(dSmallestDistance == -1 || dDistance < dSmallestDistance) {
		    unClosestCentroid = unCentroid;
		    dSmallestDistance = dDistance;
		  }
		}
		
		mapAssignments[unSample] = unClosestCentroid;
	      }
	      
	      // Check if all clusters have samples
	      bool bClustersGood = true;
	      for(unsigned int unCentroid = 0; unCentroid < vecCentroids.size(); ++unCentroid) {
		bool bFound = false;
		
		for(std::pair<unsigned int, unsigned int> prAssignment : mapAssignments) {
		  if(prAssignment.second == unCentroid) {
		    bFound = true;
		    break;
		  }
		}
		
		if(!bFound) {
		  bClustersGood = false;
		  break;
		}
	      }
	      
	      if(!bClustersGood) {
		// Randomly re-initialize
		vecCentroids.clear();
		
		std::vector<unsigned int> vecSampleIndices;
		while(vecSampleIndices.size() < unClusters) {
		  unsigned int unSampleIndex = (rand() % unSamples);
		  
		  if(std::find(vecSampleIndices.begin(), vecSampleIndices.end(), unSampleIndex) == vecSampleIndices.end()) {
		    vecSampleIndices.push_back(unSampleIndex);
		  }
		}
		
		for(unsigned int unSampleIndex : vecSampleIndices) {
		  vecCentroids.push_back((*m_dsSource)[unSampleIndex]);
		}
	      } else {
		// Move means
		for(unsigned int unCentroid = 0; unCentroid < vecCentroids.size(); ++unCentroid) {
		  Eigen::VectorXf evcMean = Eigen::VectorXf::Zero(unDimensions);
		  unsigned int unCount = 0;
		  
		  for(std::pair<unsigned int, unsigned int> prAssignment : mapAssignments) {
		    if(prAssignment.second == unCentroid) {
		      evcMean += (*m_dsSource)[prAssignment.first];
		      unCount++;
		    }
		  }
		  
		  vecCentroids[unCentroid] = evcMean / unCount;
		}
	      }
	    }
	  }
	  
	  m_vecClusters.clear();
	  for(unsigned int unI = 0; unI < unClusters; unI++) {
	    m_vecClusters.push_back(Dataset::create());
	  }
	  
	  for(std::pair<unsigned int, unsigned int> prAssignment : mapAssignments) {
	    m_vecClusters[prAssignment.second]->add((*m_dsSource)[prAssignment.first]);
	  }
	  
	  return true;
	}
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
