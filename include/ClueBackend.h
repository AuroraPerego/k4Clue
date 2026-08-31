/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef CLUE_BACKEND_H
#define CLUE_BACKEND_H

#include <cstdint>
#include <vector>

template <uint8_t nDim>
struct ClueBackend;

enum class ClueCoordinate { Cartesian, Polar };

using ResultMap = std::vector<std::vector<uint32_t>>;

// lifecycle
template <uint8_t nDim>
ClueBackend<nDim>* createBackend();

template <uint8_t nDim>
void destroyBackend(ClueBackend<nDim>* backend);

// setup
template <uint8_t nDim>
bool setupBackend(ClueBackend<nDim>* backend, float dc, float rhoc, float dm, float seed_dc, int pointsPerBin,
                  ClueCoordinate coordinate = ClueCoordinate::Cartesian);

namespace clue {
class CLUECalorimeterHit;
}

template <uint8_t nDim>
ResultMap launchClustering(ClueBackend<nDim>* backend, const std::vector<clue::CLUECalorimeterHit>& hits,
                           ClueCoordinate coordinate);

ResultMap launchVertexing(ClueBackend<1>* backend, const std::vector<float>& zip, const std::vector<float>& pt);

#endif
