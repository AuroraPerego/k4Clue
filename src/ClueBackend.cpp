/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 * Licensed under the Apache License, Version 2.0.
 */
#include "ClueBackend.h"

#include "CLUECalorimeterHit.h"
#include "CLUEstering/CLUEstering.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

constexpr float C_MM_NS_SQUARED = 299.792458f * 299.792458f;

template <uint8_t nDim>
struct ClueBackend {
  std::optional<clue::Clusterer<nDim>> clueAlgo;
  std::optional<clue::Queue> queue;
};

template <uint8_t nDim>
ClueBackend<nDim>* createBackend() {
  return new ClueBackend<nDim>();
}

template <uint8_t nDim>
void destroyBackend(ClueBackend<nDim>* backend) {
  delete backend;
}

template <uint8_t nDim>
bool setupBackend(ClueBackend<nDim>* backend, float dc, float rhoc, float dm, float seedDc, int pointsPerBin,
                  ClueCoordinate coordinate) {
  const std::vector<ALPAKA_BACKEND::Device> devices = alpaka::getDevs(clue::Platform{});

  for (const auto& device : devices) {
    std::cout << " - " << alpaka::getName(device) << "\n";
  }

  if (devices.empty()) {
    return false;
  }

  backend->queue = clue::get_queue(devices.front());
  std::cout << "CLUEAlgo will run on device " << alpaka::getName(alpaka::getDev(*backend->queue)) << "\n";

  const float seedingDistance = (seedDc < 0.f) ? dc : seedDc;
  const float outlierDistance = (dm < 0.f) ? dc : dm;

  backend->clueAlgo.emplace(*backend->queue, dc, rhoc, outlierDistance, seedingDistance, pointsPerBin);

  if (coordinate == ClueCoordinate::Polar) {
    std::vector<int> wrappedCoordinates(nDim, 0);
    wrappedCoordinates[1] = 1;
    backend->clueAlgo->setWrappedCoordinates(wrappedCoordinates);
  }

  return true;
}

template <uint8_t nDim>
void fillCluePoints(const std::vector<clue::CLUECalorimeterHit>& clue_hits, ClueCoordinate coordinate,
                    std::vector<float>& floatBuffer) {
  size_t nPoints = clue_hits.size();

  if (coordinate == ClueCoordinate::Cartesian) {
    for (size_t i = 0; i < nPoints; ++i) {
      const auto& position = clue_hits[i].getPosition();
      floatBuffer[i] = position.x;           // Fill x coordinates
      floatBuffer[nPoints + i] = position.y; // Fill y coordinates
      if constexpr (nDim >= 3)
        floatBuffer[nPoints * 2 + i] = position.z; // Fill z coordinates
      if constexpr (nDim >= 4)
        floatBuffer[nPoints * 3 + i] = clue_hits[i].getTime();    // Fill time coordinates
      floatBuffer[nPoints * nDim + i] = clue_hits[i].getEnergy(); // Fill weights
    }
  } else if (coordinate == ClueCoordinate::Polar) {
    for (size_t i = 0; i < nPoints; ++i) {
      float phi = clue_hits[i].getPhi();
      // Normalize phi to [0, 2*pi] for periodic distance calculation
      // (clue::PeriodicEuclideanMetric only works with positive periods)
      if (phi < 0) {
        phi += 2.0f * M_PI;
      }

      floatBuffer[i] = clue_hits[i].getTheta(); // Fill theta coordinates
      floatBuffer[nPoints + i] = phi;           // Fill phi coordinates
      if constexpr (nDim >= 3)
        floatBuffer[nPoints * 2 + i] = clue_hits[i].getPosition().z; // Fill z coordinates
      floatBuffer[nPoints * nDim + i] = clue_hits[i].getEnergy();    // Fill weights
    }
  } // if Cartesian or Polar (else should not happen due to checks in initialize())
}

template <uint8_t nDim>
ResultMap makeAssociationMap(const clue::PointsHost<nDim>& cluePoints) {
  ResultMap associationMap;

  for (uint32_t i = 0; i < static_cast<uint32_t>(cluePoints.size()); ++i) {
    const int32_t clusterIdx = cluePoints.clusterIndexes()[i];
    if (clusterIdx < 0)
      continue;

    if (associationMap.size() <= static_cast<uint32_t>(clusterIdx)) {
      associationMap.resize(clusterIdx + 1);
    }
    associationMap[static_cast<uint32_t>(clusterIdx)].push_back(i);
  }

  return associationMap;
}

template <uint8_t nDim>
ResultMap launchClustering(ClueBackend<nDim>* backend, const std::vector<clue::CLUECalorimeterHit>& hits,
                           ClueCoordinate coordinate) {
  std::vector<float> floatBuffer;
  std::vector<int> intBuffer;
  size_t nPoints = hits.size();
  floatBuffer.resize(nPoints * (nDim + 1));
  intBuffer.resize(nPoints * 2);
  fillCluePoints<nDim>(hits, coordinate, floatBuffer);
  // Construct the PointsSoA object
  clue::PointsHost<nDim> cluePoints(*backend->queue, nPoints, floatBuffer.data(), intBuffer.data());

  if (coordinate == ClueCoordinate::Cartesian) {
    if constexpr (nDim == 4) {
      auto metric = clue::metrics::WeightedEuclidean<nDim>(1.f, 1.f, 1.f, C_MM_NS_SQUARED);
      backend->clueAlgo->make_clusters(*backend->queue, cluePoints, metric);
    } else {
      backend->clueAlgo->make_clusters(*backend->queue, cluePoints);
    }
  } else if (coordinate == ClueCoordinate::Polar) {
    std::array<float, nDim> periods{};
    periods[1] = 2.0f * M_PI;
    clue::PeriodicEuclideanMetric<nDim> metric(periods);
    backend->clueAlgo->make_clusters(*backend->queue, cluePoints, metric);
  }

  return makeAssociationMap<nDim>(cluePoints);
}

ResultMap launchVertexing(ClueBackend<1>* backend, const std::vector<float>& zip, const std::vector<float>& pt) {
  const int n = static_cast<int>(zip.size());
  std::vector<float> floatBuffer(n * 2);
  std::vector<int> intBuffer(n * 2);
  std::copy(zip.begin(), zip.end(), floatBuffer.begin());
  std::copy(pt.begin(), pt.end(), floatBuffer.begin() + n);
  clue::PointsHost<1> cluePoints(*backend->queue, n, floatBuffer.data(), intBuffer.data());

  backend->clueAlgo->make_clusters(*backend->queue, cluePoints);

  return makeAssociationMap<1>(cluePoints);
}

// Explicit instantiations: keep them because definitions are in this .cpp.
template struct ClueBackend<1>;
template struct ClueBackend<2>;
template struct ClueBackend<3>;
template struct ClueBackend<4>;

template ClueBackend<1>* createBackend<1>();
template ClueBackend<2>* createBackend<2>();
template ClueBackend<3>* createBackend<3>();
template ClueBackend<4>* createBackend<4>();

template void destroyBackend<1>(ClueBackend<1>*);
template void destroyBackend<2>(ClueBackend<2>*);
template void destroyBackend<3>(ClueBackend<3>*);
template void destroyBackend<4>(ClueBackend<4>*);

template bool setupBackend<1>(ClueBackend<1>*, float, float, float, float, int, ClueCoordinate);
template bool setupBackend<2>(ClueBackend<2>*, float, float, float, float, int, ClueCoordinate);
template bool setupBackend<3>(ClueBackend<3>*, float, float, float, float, int, ClueCoordinate);
template bool setupBackend<4>(ClueBackend<4>*, float, float, float, float, int, ClueCoordinate);

template ResultMap launchClustering<2>(ClueBackend<2>*, const std::vector<clue::CLUECalorimeterHit>&, ClueCoordinate);
template ResultMap launchClustering<3>(ClueBackend<3>*, const std::vector<clue::CLUECalorimeterHit>&, ClueCoordinate);
template ResultMap launchClustering<4>(ClueBackend<4>*, const std::vector<clue::CLUECalorimeterHit>&, ClueCoordinate);

/*



template <uint8_t nDim>
clue::AssociationMapHost launchClustering(ClueBackend<nDim>* backend,
                                          clue::PointsHost<nDim>& cluePoints,
                                          ClueCoordinate coordinate) {
  if (coordinate == ClueCoordinate::Cartesian) {
    if constexpr (nDim == 4) {
      auto metric = clue::metrics::WeightedEuclidean<nDim>(1.f, 1.f, 1.f, C_MM_NS_SQUARED);
      backend->clueAlgo->make_clusters(*backend->queue, cluePoints, metric);
    } else {
      backend->clueAlgo->make_clusters(*backend->queue, cluePoints);
    }
  } else if (coordinate == ClueCoordinate::Polar) {
    std::array<float, nDim> periods{};
    periods[1] = 2.0f * M_PI;
    clue::PeriodicEuclideanMetric<nDim> metric(periods);
    backend->clueAlgo->make_clusters(*backend->queue, cluePoints, metric);
  }

  return backend->clueAlgo->getClusters(cluePoints);
}

clue::AssociationMapHost launchVertexing(ClueBackend<1>* backend,
                                          clue::PointsHost<1>& cluePoints) {
  backend->clueAlgo->make_clusters(*backend->queue, cluePoints);
  return backend->clueAlgo->getClusters(cluePoints);
}

// explicit instantiations
template struct ClueBackend<1>;
template struct ClueBackend<2>;
template struct ClueBackend<3>;
template struct ClueBackend<4>;

template ClueBackend<1>* createBackend<1>();
template ClueBackend<2>* createBackend<2>();
template ClueBackend<3>* createBackend<3>();
template ClueBackend<4>* createBackend<4>();

template void destroyBackend<1>(ClueBackend<1>*);
template void destroyBackend<2>(ClueBackend<2>*);
template void destroyBackend<3>(ClueBackend<3>*);
template void destroyBackend<4>(ClueBackend<4>*);

template bool setupBackend<1>(ClueBackend<1>*, float, float, float, float, int, ClueCoordinate);
template bool setupBackend<2>(ClueBackend<2>*, float, float, float, float, int, ClueCoordinate);
template bool setupBackend<3>(ClueBackend<3>*, float, float, float, float, int, ClueCoordinate);
template bool setupBackend<4>(ClueBackend<4>*, float, float, float, float, int, ClueCoordinate);

template clue::Queue& backendQueue<1>(ClueBackend<1>*);
template clue::Queue& backendQueue<2>(ClueBackend<2>*);
template clue::Queue& backendQueue<3>(ClueBackend<3>*);
template clue::Queue& backendQueue<4>(ClueBackend<4>*);

template const clue::Queue& backendQueue<1>(const ClueBackend<1>*);
template const clue::Queue& backendQueue<2>(const ClueBackend<2>*);
template const clue::Queue& backendQueue<3>(const ClueBackend<3>*);
template const clue::Queue& backendQueue<4>(const ClueBackend<4>*);

template clue::AssociationMapHost launchClustering<2>(ClueBackend<2>*, clue::PointsHost<2>&, ClueCoordinate);
template clue::AssociationMapHost launchClustering<3>(ClueBackend<3>*, clue::PointsHost<3>&, ClueCoordinate);
template clue::AssociationMapHost launchClustering<4>(ClueBackend<4>*, clue::PointsHost<4>&, ClueCoordinate);*/
