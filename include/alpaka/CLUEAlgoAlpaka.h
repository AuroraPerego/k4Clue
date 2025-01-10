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

#include "AlpakaCore/config.h"
#include "Points.h"
#include "alpaka/CLUEAlpakaKernels.h"
#include "alpaka/ConvolutionalKernel.h"
#include "alpaka/PointsAlpaka.h"
#include "alpaka/TilesAlpaka.h"

using alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  template <typename TAcc, typename TTile, typename TConvKernel = GaussianKernel>
  class CLUEAlgoAlpaka_T {
   public:
    CLUEAlgoAlpaka_T()
        : dc_(0.f), rhoc_(0.f), outlierDeltaFactor_(0.f), verbose_(false) {}

    CLUEAlgoAlpaka_T(float dc, float rhoc, float outlierDeltaFactor, Queue queue,
                     bool verbose)
        : dc_{dc},
          rhoc_{rhoc},
          outlierDeltaFactor_{outlierDeltaFactor},
          verbose_{verbose} {
      d_tiles = alpaka::allocAsyncBufIfSupported<TilesAlpaka_T<TAcc, TTile>, Idx>(
          queue, Vec1D{1});
      d_seeds = alpaka::allocAsyncBufIfSupported<
          alpakatools::VecArray<int32_t, max_seeds>, Idx>(queue, Vec1D{1});
      d_followers = alpaka::allocAsyncBufIfSupported<
          alpakatools::VecArray<int32_t, max_followers>, Idx>(queue,
                                                              Vec1D{MAX_POINTS});
      d_points = PointsAlpaka<2>{queue};
      if (verbose_) {
        std::cout << "ClueGaudiAlgorithmWrapper: nTilesPerDim\n";
        for (uint8_t n = 0; n < TTile::nDim; ++n) {
          std::cout << " - dim " << (uint16_t)(n + 1) << " : "
                    << tiles::nTilesPerDim<TTile>(n) << " tiles in the range ("
                    << TTile::MinMax[n][0] << ", " << TTile::MinMax[n][1]
                    << ")\n";
        }
      }
    }

    std::optional<alpaka::Buf<Device, TilesAlpaka_T<TAcc, TTile>, Dim1D, Idx>>
        d_tiles;
    std::optional<alpaka::Buf<Device, VecArray<int32_t, max_seeds>, Dim1D, Idx>>
        d_seeds;
    std::optional<
        alpaka::Buf<Device, VecArray<int32_t, max_followers>, Dim1D, Idx>>
        d_followers;
    std::optional<PointsAlpaka<2>> d_points;

    std::vector<std::vector<int>> makeClusters(Points<2> h_points,
                                               Queue& queue,
                                               std::size_t block_size);
    bool clearAndSetPoints(const Points<2>& h_points, Queue& queue,
                           std::size_t block_size);

    std::map<int, std::vector<int> > getClusters(Points<2> h_points);
    void clearLayerTiles(Queue& queue, std::size_t block_size);

   private:
    float dc_;
    float rhoc_;
    float outlierDeltaFactor_;
    bool verbose_;

  };

  template <typename TAcc, typename TTile, typename TConv>
  bool CLUEAlgoAlpaka_T<TAcc, TTile, TConv>::clearAndSetPoints(
      const Points<2>& h_points, Queue& queue, std::size_t block_size) {
    if (not h_points.size()) return false;

    d_points->nPoints = h_points.size();

    const Idx tiles_grid_size =
        alpakatools::divide_up_by(tiles::nTiles<TTile>(), block_size);
    const auto tiles_work_div =
        alpakatools::make_workdiv<Acc1D>(tiles_grid_size, block_size);
    alpaka::exec<Acc1D>(queue, tiles_work_div, KernelResetTiles{},
                        d_tiles->data()); //, tiles::nTiles<TTile>());

    const Idx grid_size = alpakatools::divide_up_by(h_points.size(), block_size);
    const auto work_div = alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
    alpaka::exec<Acc1D>(queue, work_div, KernelResetFollowers{},
                        d_followers->data(), h_points.size());

    alpaka::memcpy(
        queue, d_points->coords,
        alpakatools::make_host_view(h_points.coords.data(), h_points.size()));
    alpaka::memcpy(
        queue,
        d_points->addCoord,
        alpakatools::make_host_view(h_points.addCoord.data(), h_points.size()));
    alpaka::memcpy(
        queue, d_points->weight,
        alpakatools::make_host_view(h_points.weight.data(), h_points.size()));
    alpaka::memset(queue, *d_seeds, 0x00);
    alpaka::wait(queue);

    return true;
  }

  // Public methods
  template <typename TAcc, typename TTile, typename TConvKernel>
  std::vector<std::vector<int>>
  CLUEAlgoAlpaka_T<TAcc, TTile, TConvKernel>::makeClusters(Points<2> h_points, Queue& queue,
                                                     std::size_t block_size) {
    // #ifdef GPU_DEBUG
    //   auto start = std::chrono::high_resolution_clock::now();
    // #endif
    //   setup(h_points, d_points, queue_, block_size); -> e' diventato il
    //   clearAndSetPoints, da decidere se lo vogliamo qui o nel wrapper
    // #ifdef GPU_DEBUG
    //   alpaka::wait(queue);
    //   auto finish = std::chrono::high_resolution_clock::now();
    //   std::chrono::duration<double> elapsed = finish - start;
    //   std::cout << "ClueGaudiAlgorithmWrapper: setup:     " << elapsed.count()
    //   *1000 << " ms" << std::endl;
    // #endif

    const auto nPoints = d_points->nPoints;

    const Idx grid_size = alpakatools::divide_up_by(nPoints, block_size);
    auto work_div = alpakatools::make_workdiv<Acc1D>(grid_size, block_size);
  #ifdef GPU_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
  #endif
    alpaka::exec<Acc1D>(queue, work_div, KernelFillTiles{},
                       (d_points->coords).data(), d_tiles->data(), nPoints);
  #ifdef GPU_DEBUG
    alpaka::wait(queue);
    auto finish = std::chrono::high_resolution_clock::now();
    auto elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
    std::cout << "ClueGaudiAlgorithmWrapper: KernelFillTiles:     "
              << elapsed << " ms" << std::endl;
  #endif

    #ifdef GPU_DEBUG
      start = std::chrono::high_resolution_clock::now();
    #endif
      TConvKernel kernel(1.f, 0.f, 1.f);
      alpaka::exec<Acc1D>(queue, work_div, KernelCalculateLocalDensity{},
                          d_tiles->data(), d_points->view(), kernel,
                          /* m_domains.data(), */
                          dc_, nPoints);
    #ifdef GPU_DEBUG
      alpaka::wait(queue);
      finish = std::chrono::high_resolution_clock::now();
      elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
      std::cout << "ClueGaudiAlgorithmWrapper: KernelCalculateLocalDensity: "
      << elapsed << " ms" << std::endl;
    #endif

    #ifdef GPU_DEBUG
      start = std::chrono::high_resolution_clock::now();
    #endif
      alpaka::exec<Acc1D>(queue, work_div, KernelCalculateNearestHigher{},
                          d_tiles->data(), d_points->view(),
                          /* m_domains.data(), */
                          outlierDeltaFactor_, dc_, nPoints);
    #ifdef GPU_DEBUG
      alpaka::wait(queue);
      finish = std::chrono::high_resolution_clock::now();
      elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
      std::cout << "ClueGaudiAlgorithmWrapper: KernelCalculateNearestHigher: "
      << elapsed << " ms" << std::endl;
    #endif

    #ifdef GPU_DEBUG
      start = std::chrono::high_resolution_clock::now();
    #endif
      alpaka::exec<Acc1D>(queue, work_div,
      KernelFindClusters{},
                          d_seeds->data(), d_followers->data(), d_points->view(),
                          outlierDeltaFactor_, dc_, rhoc_, nPoints);
    #ifdef GPU_DEBUG
      alpaka::wait(queue);
      finish = std::chrono::high_resolution_clock::now();
      elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
      std::cout << "ClueGaudiAlgorithmWrapper: KernelFindClusters: "
      << elapsed << " ms" << std::endl;
    #endif

      // We change the work division when assigning the clusters
      const Idx grid_size_seeds = alpakatools::divide_up_by(max_seeds,
      block_size); auto work_div_seeds =
          alpakatools::make_workdiv<Acc1D>(grid_size_seeds, block_size);
    #ifdef GPU_DEBUG
      start = std::chrono::high_resolution_clock::now();
    #endif
      alpaka::exec<Acc1D>(queue, work_div_seeds,
      KernelAssignClusters{},
                          d_seeds->data(), d_followers->data(), d_points->view());
    #ifdef GPU_DEBUG
      alpaka::wait(queue);
      finish = std::chrono::high_resolution_clock::now();
      elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count());
      std::cout << "ClueGaudiAlgorithmWrapper: KernelAssignClusters: "
      << elapsed << " ms" << std::endl;
    #endif

    alpaka::memcpy(queue,
                   alpakatools::make_host_view(h_points.rho.data(), nPoints),
                   d_points->rho,
                   nPoints);
    alpaka::memcpy(queue,
                   alpakatools::make_host_view(h_points.delta.data(), nPoints),
                   d_points->delta,
                   nPoints);
    alpaka::memcpy(queue,
                   alpakatools::make_host_view(h_points.nearestHigher.data(), nPoints),
                   d_points->nearest_higher,
                   nPoints);
    alpaka::memcpy(queue,
                   alpakatools::make_host_view(h_points.clusterIndex.data(), nPoints),
                   d_points->cluster_index,
                   nPoints);
    alpaka::memcpy(queue,
                   alpakatools::make_host_view(h_points.isSeed.data(), nPoints),
                   d_points->is_seed, nPoints);

    // Wait for all the operations in the queue to finish
    alpaka::wait(queue);

    return {h_points.clusterIndex, h_points.isSeed};
  }

  template <typename TAcc, typename TTile, typename TConv>
  std::map<int, std::vector<int> > CLUEAlgoAlpaka_T<TAcc, TTile, TConv>::getClusters(Points<2> h_points){
    // cluster all points with same clusterId
    std::map<int, std::vector<int> > clusters;
    for(size_t i = 0; i < h_points.size(); i++) {
      clusters[h_points.clusterIndex[i]].push_back(i);
    }
    return clusters;
  }

  template <typename TAcc, typename TTile, typename TConv>
  void CLUEAlgoAlpaka_T<TAcc, TTile, TConv>::clearLayerTiles(Queue& queue, std::size_t block_size){
    const Idx tiles_grid_size =
        alpakatools::divide_up_by(tiles::nTiles<TTile>(), block_size);
    const auto tiles_work_div =
        alpakatools::make_workdiv<Acc1D>(tiles_grid_size, block_size);
    alpaka::exec<Acc1D>(queue, tiles_work_div, KernelResetTiles{},
                        d_tiles->data());//, tiles::nTiles<TTile>());
  }

  using CLICdetEndcapCLUEAlgo =
      CLUEAlgoAlpaka_T<Acc1D, CLICdetEndcapLayerTilesConstants2D>;
  using CLICdetBarrelCLUEAlgo =
      CLUEAlgoAlpaka_T<Acc1D, CLICdetBarrelLayerTilesConstants2D>;
  using CLDEndcapCLUEAlgo =
      CLUEAlgoAlpaka_T<Acc1D, CLDEndcapLayerTilesConstants2D>;
  using CLDBarrelCLUEAlgo =
      CLUEAlgoAlpaka_T<Acc1D, CLDBarrelLayerTilesConstants2D>;
  using LArBarrelCLUEAlgo =
      CLUEAlgoAlpaka_T<Acc1D, LArBarrelLayerTilesConstants2D>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif
