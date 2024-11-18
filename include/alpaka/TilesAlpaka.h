#ifndef Tiles_Alpaka_h
#define Tiles_Alpaka_h

#include <stdint.h>

#include <algorithm>
#include <alpaka/alpaka.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <type_traits>
#include <vector>

#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "alpaka/TilesConstants.h"
#include "alpaka/VecArray.h"

using alpakatools::VecArray;

template <typename TAcc, typename T>
class TilesAlpaka_T {
 public:
  TilesAlpaka_T(const TAcc& acc) { acc_ = acc; tiles_.resize(T::nTiles); };

  int nTilesPerDim(int dim) const { return tiles::nTilesPerDim<T>(dim); }

  ALPAKA_FN_HOST_ACC inline int getBin(float coord, int dim) const {
    int coord_Bin;
    if (T::wrapped[dim]) { // FIXME with dim it cannot be constexpr
      coord_Bin =
          static_cast<int>(reco::normalizedPhi(coord) / T::tileSize[dim]);
    } else {
      coord_Bin =
          static_cast<int>((coord - T::MinMax[dim][0]) / T::tileSize[dim]);

      // Address the cases of underflow and overflow and underflow
      coord_Bin = alpaka::math::min(acc_, coord_Bin, nTilesPerDim(dim) - 1);
      coord_Bin = alpaka::math::max(acc_, coord_Bin, 0);
    }
    return coord_Bin;
  }

  ALPAKA_FN_HOST_ACC inline int getGlobalBin(
      const VecArray<float, T::nDim>& coords) const {
    int globalBin = getBin(coords[0], 0);
    for (int i = 1; i != T::nDim; ++i) {
      if (T::wrapped[i]) { // FIXME with dim it cannot be constexpr
        globalBin += nTilesPerDim(i) * getBin(coords[i], i);
      } else {
        globalBin += nTilesPerDim(i) * getBin(coords[i], i);
      }
    }
    return globalBin;
  }

  ALPAKA_FN_HOST_ACC inline int getGlobalBinByBin(
      const VecArray<uint32_t, T::nDim>& Bins) const {
    uint32_t globalBin{Bins[0]};
    for (int i = 1; i != T::nDim; ++i) {
      globalBin += nTilesPerDim(i) * Bins[i];
    }
    return globalBin;
  }

  ALPAKA_FN_ACC inline constexpr void fill(
      const VecArray<float, T::nDim>& coords, int i) {
    tiles_[getGlobalBin(coords)].push_back(acc_, i);
  }

  ALPAKA_FN_ACC inline void searchBox(
      const VecArray<VecArray<float, 2>, T::nDim>& sb_extremes,
      VecArray<VecArray<uint32_t, 2>, T::nDim>* search_box) {
    for (int dim = 0; dim != T::nDim; ++dim) {
      VecArray<uint32_t, 2> dim_sb;
      dim_sb.push_back_unsafe(getBin(sb_extremes[dim][0], dim));
      dim_sb.push_back_unsafe(getBin(sb_extremes[dim][1], dim));

      search_box->push_back_unsafe(dim_sb);
    }
  }

  ALPAKA_FN_HOST_ACC inline constexpr auto size() { return T::nTiles; }

  ALPAKA_FN_HOST_ACC inline constexpr void clear() {
    for (int i = 0; i < T::nTiles; ++i) {
      tiles_[i].reset();
    }
  }

  ALPAKA_FN_HOST_ACC inline constexpr VecArray<uint32_t, T::maxTileDepth>&
  operator[](int globalBinId) {
    return tiles_[globalBinId];
  }

 private:
  VecArray<VecArray<uint32_t, T::maxTileDepth>, T::maxNTiles> tiles_;
  const TAcc& acc_;
};

#endif
