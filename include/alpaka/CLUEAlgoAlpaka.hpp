#ifndef CLUE_Algo_Alpaka_h
#define CLUE_Algo_Alpaka_h

#include <stdint.h>

#include <algorithm>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/vec/Vec.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Points.h"
#include "alpaka/PointsAlpaka.h"
#include "alpaka/TilesAlpaka.h"
#include "alpaka/CLUEAlpakaKernels.h"
//#include "ConvolutionalKernel.h"

#include "AlpakaCore/config.h"

using alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

template <typename TAcc, typename TTile>
class CLUEAlgoAlpaka_T {
 public:
  CLUEAlgoAlpaka_T() = delete;
  explicit CLUEAlgoAlpaka_T(float dc, float rhoc, float outlierDeltaFactor,
                            Queue queue_, bool verbose)
      : dc_{dc},
        rhoc_{rhoc},
        outlierDeltaFactor_{outlierDeltaFactor},
        verbose_{verbose} {
    init_device(queue_);
    if (verbose_) {
      std::cout << "ClueGaudiAlgorithmWrapper: nTilesPerDim\n";
      for (uint8_t n = 0; n < TTile::nDim; ++n) {
        std::cout << " - dim " << (uint16_t)(n + 1) << " : "
                  << tiles::nTilesPerDim<TTile, n>() << " tiles in the range ("
                  << TTile::MinMax[n][0] << ", " << TTile::MinMax[n][1]
                  << ")\n";
      }
    }
  }

  TilesAlpaka_T<TAcc, TTile>* m_tiles;
  VecArray<int32_t, max_seeds>* m_seeds;
  VecArray<int32_t, max_followers>* m_followers;

  template <typename KernelType>
  std::vector<std::vector<int>> make_clusters(
      Points<TTile::nDim>& h_points, PointsAlpaka<TTile::nDim>& d_points,
      const KernelType& kernel, Queue queue_, std::size_t block_size);

 private:
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;
  bool verbose_;

  // Buffers
  std::optional<alpakatools::device_buffer<Device, TilesAlpaka_T<TAcc, TTile>>>
      d_tiles;
  std::optional<alpakatools::device_buffer<
      Device, alpakatools::VecArray<int32_t, max_seeds>>>
      d_seeds;
  std::optional<alpakatools::device_buffer<
      Device, alpakatools::VecArray<int32_t, max_followers>[]>>
      d_followers;

  // Private methods
  void init_device(Queue queue_);
  void setup(const Points<TTile::nDim>& h_points,
             PointsAlpaka<TTile::nDim>& d_points, Queue queue_,
             std::size_t block_size);
};

template <typename TAcc, typename TTile>
void CLUEAlgoAlpaka_T<TAcc, TTile>::init_device(Queue queue_) {
  d_tiles = alpakatools::make_device_buffer<TilesAlpaka_T<TAcc, TTile>>(queue_);
  d_seeds = alpakatools::make_device_buffer<
      alpakatools::VecArray<int32_t, max_seeds>>(queue_);
  d_followers = alpakatools::make_device_buffer<
      alpakatools::VecArray<int32_t, max_followers>[]>(queue_, reserve);

  // Copy to the public pointers
  m_tiles = (*d_tiles).data();
  m_seeds = (*d_seeds).data();
  m_followers = (*d_followers).data();
}

template <typename TAcc, typename TTile>
void CLUEAlgoAlpaka_T<TAcc, TTile>::setup(const Points<TTile::nDim>& h_points,
                                        PointsAlpaka<TTile::nDim>& d_points,
                                        Queue queue_, std::size_t block_size) {
  const Idx tiles_grid_size =
      alpakatools::divide_up_by(tiles::nTiles<TTile>(), block_size);
  const auto tiles_work_div =
      alpakatools::make_workdiv<Acc1D>(tiles_grid_size, block_size);
  alpaka::exec<Acc1D>(queue_, tiles_work_div, KernelResetTiles{}, m_tiles,
                      tiles::nTiles<TTile>());

  alpaka::memcpy(
      queue_, d_points.coords,
      alpakatools::make_host_view(h_points.m_coords.data(), h_points.n));
  alpaka::memcpy(
      queue_, d_points.weight,
      alpakatools::make_host_view(h_points.m_weight.data(), h_points.n));
  alpaka::memset(queue_, (*d_seeds), 0x00);

  // Define the working division
  const Idx grid_size = alpakatools::divide_up_by(h_points.n, block_size);
  const auto working_div =
      alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
  alpaka::exec<Acc1D>(queue_, working_div, KernelResetFollowers{}, m_followers,
                      h_points.n);
}

// Public methods
template <typename TAcc, typename TTile>
template <typename KernelType>
std::vector<std::vector<int>> CLUEAlgoAlpaka_T<TAcc, TTile>::make_clusters(
    Points<TTile::nDim>& h_points, PointsAlpaka<TTile::nDim>& d_points,
    const KernelType& kernel, Queue queue_, std::size_t block_size) {
#ifdef GPU_DEBUG
  auto start = std::chrono::high_resolution_clock::now();
#endif
  setup(h_points, d_points, queue_, block_size);
#ifdef GPU_DEBUG
  alpaka::wait(queue_);
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "ClueGaudiAlgorithmWrapper: setup:     " << elapsed.count() *1000 << " ms" << std::endl;
#endif

  const Idx grid_size = alpakatools::divide_up_by(h_points.n, block_size);
  auto working_div = alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
#ifdef GPU_DEBUG
  start = std::chrono::high_resolution_clock::now();
#endif
  alpaka::exec<Acc1D>(queue_, working_div, KernelFillTiles{}, d_points.view(),
                      m_tiles, h_points.n);
#ifdef GPU_DEBUG
  alpaka::wait(queue_);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "ClueGaudiAlgorithmWrapper: KernelFillTiles:     " << elapsed.count() *1000 << " ms" << std::endl;
#endif

#ifdef GPU_DEBUG
  start = std::chrono::high_resolution_clock::now();
#endif
  alpaka::exec<Acc1D>(queue_, working_div, KernelCalculateLocalDensity{},
                      m_tiles, d_points.view(), kernel,
                      /* m_domains.data(), */
                      dc_, h_points.n);
#ifdef GPU_DEBUG
  alpaka::wait(queue_);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "ClueGaudiAlgorithmWrapper: KernelCalculateLocalDensity:     " << elapsed.count() *1000 << " ms" << std::endl;
#endif

#ifdef GPU_DEBUG
  start = std::chrono::high_resolution_clock::now();
#endif
  alpaka::exec<Acc1D>(queue_, working_div, KernelCalculateNearestHigher{},
                      m_tiles, d_points.view(),
                      /* m_domains.data(), */
                      outlierDeltaFactor_, dc_, h_points.n);
#ifdef GPU_DEBUG
  alpaka::wait(queue_);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "ClueGaudiAlgorithmWrapper: KernelCalculateNearestHigher:     " << elapsed.count() *1000 << " ms" << std::endl;
#endif

#ifdef GPU_DEBUG
  start = std::chrono::high_resolution_clock::now();
#endif
  alpaka::exec<Acc1D>(queue_, working_div, KernelFindClusters<TTile::nDim>{},
                      m_seeds, m_followers, d_points.view(),
                      outlierDeltaFactor_, dc_, rhoc_, h_points.n);
#ifdef GPU_DEBUG
  alpaka::wait(queue_);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "ClueGaudiAlgorithmWrapper: KernelFindClusters<" << (uint16_t)TTile::nDim <<  ">:     " << elapsed.count() *1000 << " ms" << std::endl;
#endif

  // We change the working division when assigning the clusters
  const Idx grid_size_seeds = alpakatools::divide_up_by(max_seeds, block_size);
  auto working_div_seeds =
      alpakatools::make_workdiv<Acc1D>(grid_size_seeds, block_size);
#ifdef GPU_DEBUG
  start = std::chrono::high_resolution_clock::now();
#endif
  alpaka::exec<Acc1D>(queue_, working_div, KernelAssignClusters<TTile::nDim>{},
                      m_seeds, m_followers, d_points.view());
#ifdef GPU_DEBUG
  alpaka::wait(queue_);
  finish = std::chrono::high_resolution_clock::now();
  elapsed = finish - start;
  std::cout << "ClueGaudiAlgorithmWrapper: KernelAssignClusters<" << (uint16_t)TTile::nDim <<  ">:     " << elapsed.count() *1000 << " ms" << std::endl;
#endif

  alpaka::memcpy(queue_,
                 alpakatools::make_host_view(h_points.m_rho.data(), h_points.n),
                 d_points.rho, static_cast<uint32_t>(h_points.n));
  alpaka::memcpy(
      queue_, alpakatools::make_host_view(h_points.m_delta.data(), h_points.n),
      d_points.delta, static_cast<uint32_t>(h_points.n));
  alpaka::memcpy(
      queue_,
      alpakatools::make_host_view(h_points.m_nearestHigher.data(), h_points.n),
      d_points.nearest_higher, static_cast<uint32_t>(h_points.n));
  alpaka::memcpy(
      queue_,
      alpakatools::make_host_view(h_points.m_clusterIndex.data(), h_points.n),
      d_points.cluster_index, static_cast<uint32_t>(h_points.n));
  alpaka::memcpy(
      queue_, alpakatools::make_host_view(h_points.m_isSeed.data(), h_points.n),
      d_points.is_seed, static_cast<uint32_t>(h_points.n));

  // Wait for all the operations in the queue to finish
  alpaka::wait(queue_);

  return {h_points.m_clusterIndex, h_points.m_isSeed};
}

using CLICdetEndcapCLUEAlgo = CLUEAlgoAlpaka_T<Acc1D, CLICdetEndcapLayerTilesConstants2D>;
using CLICdetBarrelCLUEAlgo = CLUEAlgoAlpaka_T<Acc1D, CLICdetBarrelLayerTilesConstants2D>;
using CLDEndcapCLUEAlgo = CLUEAlgoAlpaka_T<Acc1D, CLDEndcapLayerTilesConstants2D>;
using CLDBarrelCLUEAlgo = CLUEAlgoAlpaka_T<Acc1D, CLDBarrelLayerTilesConstants2D>;
using LArBarrelCLUEAlgo = CLUEAlgoAlpaka_T<Acc1D, LArBarrelLayerTilesConstants2D>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
