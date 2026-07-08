/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef CLUE_BACKEND_H
#define CLUE_BACKEND_H

#include "CLUEstering/CLUEstering.hpp"

#include <cstdint>

template <uint8_t nDim>
struct ClueBackend;

enum class ClueCoordinate { Cartesian, Polar };

// lifecycle
template <uint8_t nDim>
ClueBackend<nDim>* createBackend();

template <uint8_t nDim>
void destroyBackend(ClueBackend<nDim>* backend);

// setup
template <uint8_t nDim>
bool setupBackend(ClueBackend<nDim>* backend,
                  float dc,
                  float rhoc,
                  float dm,
                  float seed_dc,
                  int pointsPerBin,
                  ClueCoordinate coordinate);

// helpers
template <uint8_t nDim>
clue::Queue& backendQueue(ClueBackend<nDim>* backend);

template <uint8_t nDim>
const clue::Queue& backendQueue(const ClueBackend<nDim>* backend);

// run clustering
template <uint8_t nDim>
clue::AssociationMapHost launchClustering(ClueBackend<nDim>* backend,
                                          clue::PointsHost<nDim>& cluePoints,
                                          ClueCoordinate coordinate);

#endif
