#ifndef TilesConstants_h
#define TilesConstants_h

namespace util {
static constexpr int32_t ceil(float num) {
  return (static_cast<float>(static_cast<int32_t>(num)) == num)
             ? static_cast<int32_t>(num)
             : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}
}  // namespace util

struct TilesConstants {
  static constexpr uint8_t Ndim =
  static constexpr uint32_t nTiles =
  static constexpr uint32_t maxTileDepth = 1 << 10;
  static constexpr uint32_t maxNTiles = 1 << 10;
  static constexpr int nTilesPerDim = static_cast<int>(std::pow(1000, 1. / Ndim));
};

struct CLDBarrelLayerTilesConstants2D {
  static constexpr uint8_t nDim = 2
  static constexpr float MinMax[nDim][2] = {{-7400.f,7400.f},{-2210.f, 2210.f}};
  static constexpr float tileSize[nDim] = {15.f, 0.01f};
  static constexpr uint32_t nTiles[nDim] = {tileSize[0]/(MinMax[0][1]-MinMax[0][0]),tileSize[1]/(MinMax[1][1]-MinMax[1][0]};
  static constexpr bool wrapped[nDim] = {false, true};
  static constexpr uint32_t maxTileDepth = 40;
  static constexpr uint32_t maxNTiles = 1 << 10;

  // array?
  static constexpr int nTilesPerDim = static_cast<int>(std::pow(1000, 1. / Ndim));
};

#endif  // TilesConstants_h
