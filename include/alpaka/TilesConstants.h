#ifndef TilesConstants_h
#define TilesConstants_h

#include "constexpr_cmath.h"

namespace tiles {
  template <typename TTile>
  constexpr uint32_t nTilesPerDim(const int N) {
      return static_cast<uint32_t>(reco::ceil(TTile::MinMax[N][1] - TTile::MinMax[N][0]) / TTile::tileSize[N]);
  }

  template <typename TTile>
  constexpr uint32_t nTiles() {
      uint32_t nTiles = 0;
      for (uint8_t n = 0; n < TTile::nDim; ++n)
        nTiles += nTilesPerDim<TTile>(n);
      return nTiles;
  }
}

struct CLDBarrelLayerTilesConstants2D {
  static constexpr uint8_t nDim = 2;
  // static constexpr float MinMax[nDim][2] = {{-7400.f,7400.f},{-2210.f, 2210.f}};
  static constexpr float MinMax[nDim][2] = {{-2210.f, 2210.f}, {-M_PI, M_PI}};
  static constexpr float tileSize[nDim] = {15.f, 0.01f};
  static constexpr uint32_t nTiles = tiles::nTiles<CLDBarrelLayerTilesConstants2D>(); // [nDim] = {tileSize[0]/(MinMax[0][1]-MinMax[0][0]),tileSize[1]/(MinMax[1][1]-MinMax[1][0]};
  static constexpr bool wrapped[nDim] = {false, true};
  static constexpr uint32_t maxTileDepth = 40;
  static constexpr uint32_t maxNTiles = 1 << 10;
  static constexpr int nLayers = 40;
};

struct CLDEndcapLayerTilesConstants2D {
  static constexpr uint8_t nDim = 2;
  static constexpr float MinMax[nDim][2] = {{-2455.f, 2455.f},{-2455.f, 2455.f}};
  static constexpr float tileSize[nDim] = {15.f, 15.f};
  //static constexpr uint32_t nTiles[nDim] = {tileSize[0]/(MinMax[0][1]-MinMax[0][0]),tileSize[1]/(MinMax[1][1]-MinMax[1][0]};
  static constexpr uint32_t nTiles = tiles::nTiles<CLDEndcapLayerTilesConstants2D>();
  static constexpr bool wrapped[nDim] = {false, false};
  static constexpr uint32_t maxTileDepth = 40;
  static constexpr uint32_t maxNTiles = 1 << 10;
  static constexpr int nLayers = 80;
};

struct CLICdetBarrelLayerTilesConstants2D {
  static constexpr uint8_t nDim = 2;
  static constexpr float MinMax[nDim][2] = {{-2210.f, 2210.f}, {-M_PI, M_PI}};
  static constexpr float tileSize[nDim] = {35.f, 0.15f};
  //static constexpr uint32_t nTiles[nDim] = {tileSize[0]/(MinMax[0][1]-MinMax[0][0]),tileSize[1]/(MinMax[1][1]-MinMax[1][0]};
  static constexpr uint32_t nTiles = tiles::nTiles<CLICdetBarrelLayerTilesConstants2D>();
  static constexpr bool wrapped[nDim] = {false, true};
  static constexpr uint32_t maxTileDepth = 40;
  static constexpr uint32_t maxNTiles = 1 << 10;
  static constexpr int nLayers = 40;
};

struct CLICdetEndcapLayerTilesConstants2D {
  static constexpr uint8_t nDim = 2;
  static constexpr float MinMax[nDim][2] = {{-1701.f, 1701.f},{-1701.f, 1701.f}};
  static constexpr float tileSize[nDim] = {27.f, 27.f};
  //static constexpr uint32_t nTiles[nDim] = {tileSize[0]/(MinMax[0][1]-MinMax[0][0]),tileSize[1]/(MinMax[1][1]-MinMax[1][0]};
  static constexpr uint32_t nTiles = tiles::nTiles<CLICdetEndcapLayerTilesConstants2D>();
  static constexpr bool wrapped[nDim] = {false, false};
  static constexpr uint32_t maxTileDepth = 40;
  static constexpr uint32_t maxNTiles = 1 << 10;
  static constexpr int nLayers = 80;
};

struct LArBarrelLayerTilesConstants2D {
  static constexpr uint8_t nDim = 2;
  static constexpr float MinMax[nDim][2] = {{-3110.f, 3110.f}, {-M_PI, M_PI}};
  static constexpr float tileSize[nDim] = {50.f, 0.15f};
  //static constexpr uint32_t nTiles[nDim] = {tileSize[0]/(MinMax[0][1]-MinMax[0][0]),tileSize[1]/(MinMax[1][1]-MinMax[1][0]};
  static constexpr uint32_t nTiles = tiles::nTiles<LArBarrelLayerTilesConstants2D>();
  static constexpr bool wrapped[nDim] = {false, true};
  static constexpr uint32_t maxTileDepth = 40;
  static constexpr uint32_t maxNTiles = 1 << 10;
  static constexpr int nLayers = 12;
};

#endif  // TilesConstants_h
