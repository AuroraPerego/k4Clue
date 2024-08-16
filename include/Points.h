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

#include "alpaka/VecArray.h"
#include <vector>

using alpakatools::VecArray;

template <uint8_t Ndim>
struct Points {
  Points() = default;
  Points(const std::vector<VecArray<float, Ndim>>& coords,
         const std::vector<float>& weight)
      : m_coords{coords}, m_weight{weight}, n{weight.size()} {
    m_rho.resize(n);
    m_delta.resize(n);
    m_nearestHigher.resize(n);
    m_clusterIndex.resize(n);
    m_isSeed.resize(n);
  }
  Points(const std::vector<std::vector<float>>& coords, const std::vector<float>& weight)
      : m_weight{weight}, n{weight.size()} {
    for (const auto& x : coords) {
      VecArray<float, Ndim> temp_vecarray;
      for (auto value : x) {
        temp_vecarray.push_back_unsafe(value);
      }
      m_coords.push_back(temp_vecarray);
    }

    m_rho.resize(n);
    m_delta.resize(n);
    m_nearestHigher.resize(n);
    m_clusterIndex.resize(n);
    m_isSeed.resize(n);
  }

  void clear() {
    m_coords.clear();
    m_layer.clear();
    m_weight.clear();

    m_rho.clear();
    m_delta.clear();
    m_nearestHigher.clear();
    m_clusterIndex.clear();
    m_followers.clear();
    m_isSeed.clear();

    n = 0;
  }

  std::vector<VecArray<float, Ndim>> m_coords;
  std::vector<float> m_weight;
  std::vector<float> m_rho;
  std::vector<float> m_delta;
  std::vector<int> m_nearestHigher;
  std::vector<int> m_clusterIndex;
  std::vector<int> m_isSeed;
  // why use int instead of bool?
  // https://en.cppreference.com/w/cpp/container/vector_bool
  // std::vector<bool> behaves similarly to std::vector, but in order to be space efficient, it:
  // Does not necessarily store its elements as a contiguous array (so &v[0] + n != &v[n])

  size_t n;

  // missing in cluestering
  std::vector<int> m_layer;
  std::vector<std::vector<int>> m_followers;
};

#endif
