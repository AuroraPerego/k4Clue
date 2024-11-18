/*
 * Copyright (c) 2020-2024 Key4hep-Project.
 *
 * This file is part of Key4hep.
 * See https://key4hep.github.io/key4hep-doc/ for further info.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef points_h
#define points_h

#include <vector>
#include "alpaka/VecArray.h"

using alpakatools::VecArray;

template<uint8_t nDim>
struct Points {

  Points() = default;
  Points(const std::vector<VecArray<float, nDim>>& in_coords,
         const std::vector<float>& in_weight)
      : coords{in_coords}, weight{in_weight}, n{weight.size()} {
    rho.resize(n);
    delta.resize(n);
    nearestHigher.resize(n);
    clusterIndex.resize(n);
    isSeed.resize(n);
  }
  Points(const std::vector<std::vector<float>>& in_coords, const std::vector<float>& in_weight)
      : weight{in_weight}, n{weight.size()} {
    for (const auto& x : in_coords) {
      VecArray<float, nDim> temp_vecarray;
      for (auto value : x) {
        temp_vecarray.push_back_unsafe(value);
      }
      coords.push_back(temp_vecarray);
    }

    rho.resize(n);
    delta.resize(n);
    nearestHigher.resize(n);
    clusterIndex.resize(n);
    isSeed.resize(n);
  }

  void clear() {
    coords.clear();
    layer.clear();
    weight.clear();

    rho.clear();
    delta.clear();
    nearestHigher.clear();
    clusterIndex.clear();
    followers.clear();
    isSeed.clear();

    n = 0;
  }

  std::vector<VecArray<float, nDim>> coords;
  std::vector<float> addCoord;
  std::vector<float> weight;
  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<int> clusterIndex;
  std::vector<int> isSeed;
  // why use int instead of bool?
  // https://en.cppreference.com/w/cpp/container/vector_bool
  // std::vector<bool> behaves similarly to std::vector, but in order to be space efficient, it:
  // Does not necessarily store its elements as a contiguous array (so &v[0] + n != &v[n])

  size_t n;

  // missing in cluestering
  std::vector<int> layer;
  std::vector<std::vector<int>> followers;
};

#endif
