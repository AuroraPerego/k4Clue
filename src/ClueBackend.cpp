/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 * Licensed under the Apache License, Version 2.0.
 */

#include "ClueBackend.h"

#include <array>
#include <cmath>
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
bool setupBackend(ClueBackend<nDim>* backend,
                  float dc,
                  float rhoc,
                  float dm,
                  float seed_dc,
                  int pointsPerBin,
                  ClueCoordinate coordinate) {

  const std::vector<ALPAKA_BACKEND::Device> devices = alpaka::getDevs(clue::Platform{});
  if (devices.empty()) {
    return false;
  }

  backend->queue = clue::get_queue(devices[0]);

  const auto seeding_distance = (seed_dc < 0.f) ? dc : seed_dc;
  const auto outlier_distance = (dm < 0.f) ? dc : dm;

  backend->clueAlgo = std::make_optional<clue::Clusterer<nDim>>(
      *backend->queue, dc, rhoc, outlier_distance, seeding_distance, pointsPerBin);

  if (coordinate == ClueCoordinate::Polar) {
    std::vector<int> coord(nDim, 0);
    coord[1] = 1;
    backend->clueAlgo->setWrappedCoordinates(coord);
  }
  return true;
}

template <uint8_t nDim>
clue::Queue& backendQueue(ClueBackend<nDim>* backend) {
  return *backend->queue;
}

template <uint8_t nDim>
const clue::Queue& backendQueue(const ClueBackend<nDim>* backend) {
  return *backend->queue;
}

template <uint8_t nDim>
clue::AssociationMapHost launchClustering(ClueBackend<nDim>* backend,
                                          clue::PointsHost<nDim>& cluePoints,
                                          ClueCoordinate coordinate) {
  if (coordinate == ClueCoordinate::Cartesian) {
    std::array<float, nDim> periods{};
    periods[1] = 2.0f * M_PI;
    clue::PeriodicEuclideanMetric<nDim> metric(periods);
    backend->clueAlgo->make_clusters(*backend->queue, cluePoints, metric);
  } else if (coordinate == ClueCoordinate::Polar) {
    if constexpr (nDim == 4) {
      auto metric = clue::metrics::WeightedEuclidean<nDim>(1.f, 1.f, 1.f, C_MM_NS_SQUARED);
      backend->clueAlgo->make_clusters(*backend->queue, cluePoints, metric);
    } else {
      backend->clueAlgo->make_clusters(*backend->queue, cluePoints);
    }
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
template clue::AssociationMapHost launchClustering<4>(ClueBackend<4>*, clue::PointsHost<4>&, ClueCoordinate);
