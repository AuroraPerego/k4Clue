#ifndef CLUE_Alpaka_Kernels_h
#define CLUE_Alpaka_Kernels_h

#include <alpaka/core/Common.hpp>
#include <chrono>
#include <cstdint>

#include "AlpakaCore/workdivision.h"
#include "alpaka/PointsAlpaka.h"
#include "alpaka/TilesAlpaka.h"
#include "alpaka/VecArray.h"
// #include "ConvolutionalKernel.h"

using alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct KernelResetTiles {
    template <typename TAcc, typename TConstants>
    ALPAKA_FN_ACC void operator()(TAcc const&,
                                  TilesAlpaka_T<TAcc, TConstants>* tiles) const {
                                  //uint32_t nTiles) const {
  //    for (auto index : alpaka::uniformElements(acc, nTiles)) {
        tiles->clear();
  //    };
    }
  };

  struct KernelResetFollowers {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int, max_followers>* d_followers,
                                  uint32_t nPoints) const {
      for (auto index : alpaka::uniformElements(acc, nPoints)) {
        d_followers[index].reset(); };
    }
  };

  struct KernelFillTiles {
    template <typename TAcc, typename TConstants, int nDim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, VecArray<float, nDim>* points,
                                  TilesAlpaka_T<TAcc, TConstants>* tiles,
                                  uint32_t nPoints) const {
      for (auto index : alpaka::uniformElements(acc, nPoints)) {
        tiles->fill(points[index], index);
      };
    }
  };

  template <typename TAcc, uint8_t nDim, uint8_t N_,
            typename TConstants, typename KernelType>
  ALPAKA_FN_HOST_ACC void for_recursion(
      const TAcc& acc, VecArray<uint32_t, nDim>& base_vec,
      const VecArray<VecArray<uint32_t, 2>, nDim>& search_box,
      TilesAlpaka_T<TAcc, TConstants>* tiles, PointsAlpakaView<nDim>* dev_points,
      const KernelType& kernel,
      /* const VecArray<VecArray<float, 2>, nDim>& domains, */
      const VecArray<float, nDim>& coords_i, float* rho_i, float dc,
      uint32_t point_id) {
    if constexpr (N_ == 0) {
      int binId{tiles->getGlobalBinByBin(base_vec)};
      // get the size of this bin
      int binSize{static_cast<int>((*tiles)[binId].size())};

      // iterate inside this bin
      for (int binIter = 0; binIter < binSize; ++binIter) {
        uint32_t j = (*tiles)[binId][binIter];
        // query N_{dc_}(i)

        VecArray<float, nDim> coords_j{dev_points->coords[j]};

        float dist_ij_sq = 0.f;
        for (int dim = 0; dim != nDim; ++dim) {
          dist_ij_sq +=
              (coords_j[dim] - coords_i[dim]) * (coords_j[dim] - coords_i[dim]);
        }

        if (dist_ij_sq <= dc * dc) {
          *rho_i +=
              kernel(acc, alpaka::math::sqrt(acc, dist_ij_sq), point_id, j) *
              dev_points->weight[j];
        }

      }  // end of interate inside this bin

      return;
    } else {
      for (unsigned int i = search_box[search_box.capacity() - N_][0];
           i <= search_box[search_box.capacity() - N_][1]; ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion<TAcc, nDim, N_ - 1>(acc, base_vec, search_box, tiles,
                                          dev_points, kernel, coords_i, rho_i, dc,
                                          point_id);
      }
    }
  }

  struct KernelCalculateLocalDensity {
    template <typename TAcc, typename TConstants, uint8_t nDim,
              typename KernelType>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, TilesAlpaka_T<TAcc, TConstants>* dev_tiles,
        PointsAlpakaView<nDim>* dev_points, const KernelType& kernel,
        /* const VecArray<VecArray<float, 2>, nDim>& domains, */
        float dc, uint32_t nPoints) const {
      for (auto i : alpaka::uniformElements(acc, nPoints)) {
        float rho_i = 0.f;
        VecArray<float, nDim> coords_i{dev_points->coords[i]};

        // Get the extremes of the search box
        VecArray<VecArray<float, 2>, nDim> searchbox_extremes;
        for (int dim = 0; dim != nDim; ++dim) {
          VecArray<float, 2> dim_extremes;
          dim_extremes.push_back_unsafe(coords_i[dim] - dc);
          dim_extremes.push_back_unsafe(coords_i[dim] + dc);

          searchbox_extremes.push_back_unsafe(dim_extremes);
        }

        // Calculate the search box
        VecArray<VecArray<uint32_t, 2>, nDim> search_box;
        dev_tiles->searchBox(searchbox_extremes, &search_box);

        VecArray<uint32_t, nDim> base_vec;
        for_recursion<TAcc, nDim, nDim>(acc, base_vec, search_box, dev_tiles,
                                        dev_points, kernel, coords_i, &rho_i, dc,
                                        i);

        dev_points->rho[i] = rho_i;
      };
    }
  };

  template <typename TAcc, uint8_t nDim, uint8_t N_, typename TConstants>
  ALPAKA_FN_HOST_ACC void for_recursion_nearest_higher(
      const TAcc& acc, VecArray<uint32_t, nDim>& base_vec,
      const VecArray<VecArray<uint32_t, 2>, nDim>& s_box,
      TilesAlpaka_T<TAcc, TConstants>* tiles, PointsAlpakaView<nDim>* dev_points,
      /* const VecArray<VecArray<float, 2>, nDim>& domains, */
      const VecArray<float, nDim>& coords_i, float rho_i, float* delta_i,
      int* nh_i, float dm_sq, uint32_t point_id) {
    if constexpr (N_ == 0) {
      int binId{tiles->getGlobalBinByBin(base_vec)};
      // get the size of this bin
      int binSize{(*tiles)[binId].size()};

      // iterate inside this bin
      for (int binIter{}; binIter < binSize; ++binIter) {
        unsigned int j{(*tiles)[binId][binIter]};
        // query N'_{dm}(i)
        float rho_j{dev_points->rho[j]};
        bool found_higher{(rho_j > rho_i)};
        // in the rare case where rho is the same, use detid
        found_higher =
            found_higher || ((rho_j == rho_i) && (rho_j > 0.f) && (j > point_id));

        // Calculate the distance between the two points
        VecArray<float, nDim> coords_j{dev_points->coords[j]};
        float dist_ij_sq{0.f};
        for (int dim{}; dim != nDim; ++dim) {
          dist_ij_sq +=
              (coords_j[dim] - coords_i[dim]) * (coords_j[dim] - coords_i[dim]);
        }

        if (found_higher && dist_ij_sq <= dm_sq) {
          // find the nearest point within N'_{dm}(i)
          if (dist_ij_sq < *delta_i) {
            // update delta_i and nearestHigher_i
            *delta_i = dist_ij_sq;
            *nh_i = j;
          }
        }
      }  // end of interate inside this bin

      return;
    } else {
      for (unsigned int i{s_box[s_box.capacity() - N_][0]};
           i <= s_box[s_box.capacity() - N_][1]; ++i) {
        base_vec[base_vec.capacity() - N_] = i;
        for_recursion_nearest_higher<TAcc, nDim, N_ - 1>(
            acc, base_vec, s_box, tiles, dev_points, coords_i, rho_i, delta_i,
            nh_i, dm_sq, point_id);
      }
    }
  }

  struct KernelCalculateNearestHigher {
    template <typename TAcc, typename TConstants, uint8_t nDim>
    ALPAKA_FN_ACC void operator()(
        const TAcc& acc, TilesAlpaka_T<TAcc, TConstants>* dev_tiles,
        PointsAlpakaView<nDim>* dev_points,
        /* const VecArray<VecArray<float, 2>, nDim>& domains, */
        float outlier_delta_factor, float dc, uint32_t n_points) const {
      float dm{outlier_delta_factor * dc};
      float dm_squared{dm * dm};
      alpakatools::for_each_element_in_grid(acc, n_points, [&](uint32_t i) {
        float delta_i{std::numeric_limits<float>::max()};
        int nh_i{-1};
        VecArray<float, nDim> coords_i{dev_points->coords[i]};
        float rho_i{dev_points->rho[i]};

        // Get the extremes of the search box
        VecArray<VecArray<float, 2>, nDim> searchbox_extremes;
        for (int dim{}; dim != nDim; ++dim) {
          VecArray<float, 2> dim_extremes;
          dim_extremes.push_back_unsafe(coords_i[dim] - dm);
          dim_extremes.push_back_unsafe(coords_i[dim] + dm);

          searchbox_extremes.push_back_unsafe(dim_extremes);
        }

        // Calculate the search box
        VecArray<VecArray<uint32_t, 2>, nDim> search_box;
        dev_tiles->searchBox(searchbox_extremes, &search_box);

        VecArray<uint32_t, nDim> base_vec{};
        for_recursion_nearest_higher<TAcc, nDim, nDim>(
            acc, base_vec, search_box, dev_tiles, dev_points, coords_i, rho_i,
            &delta_i, &nh_i, dm_squared, i);

        dev_points->delta[i] = alpaka::math::sqrt(acc, delta_i);
        dev_points->nearest_higher[i] = nh_i;
      });
    }
  };

  struct KernelFindClusters {
    template <typename TAcc, uint8_t nDim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int32_t, max_seeds>* seeds,
                                  VecArray<int32_t, max_followers>* followers,
                                  PointsAlpakaView<nDim>* dev_points,
                                  float outlier_delta_factor, float d_c,
                                  float rho_c, uint32_t n_points) const {
      alpakatools::for_each_element_in_grid(acc, n_points, [&](uint32_t i) {
        // initialize cluster_index
        dev_points->cluster_index[i] = -1;

        float delta_i{dev_points->delta[i]};
        float rho_i{dev_points->rho[i]};

        // Determine whether the point is a seed or an outlier
        bool is_seed{(delta_i > d_c) && (rho_i >= rho_c)};
        bool is_outlier{(delta_i > outlier_delta_factor * d_c) &&
                        (rho_i < rho_c)};

        if (is_seed) {
          dev_points->is_seed[i] = 1;
          seeds[0].push_back(acc, i);
        } else {
          if (!is_outlier) {
            followers[dev_points->nearest_higher[i]].push_back(acc, i);
          }
          dev_points->is_seed[i] = 0;
        }
      });
    }
  };

  struct KernelAssignClusters {
    template <typename TAcc, uint8_t nDim>
    ALPAKA_FN_ACC void operator()(const TAcc& acc,
                                  VecArray<int, max_seeds>* seeds,
                                  VecArray<int, max_followers>* followers,
                                  PointsAlpakaView<nDim>* dev_points) const {
      const auto n_seeds = seeds->size();
      for (auto idx_cls : alpaka::uniformElements(acc, n_seeds)) {
        int local_stack[256] = {-1};
        int local_stack_size = 0;

        int idx_this_seed = (*seeds)[idx_cls];
        dev_points->cluster_index[idx_this_seed] = idx_cls;
        // push_back idThisSeed to localStack
        local_stack[local_stack_size] = idx_this_seed;
        ++local_stack_size;
        // process all elements in localStack
        while (local_stack_size > 0) {
          // get last element of localStack
          int idx_end_of_local_stack{local_stack[local_stack_size - 1]};
          int temp_cluster_index{
              dev_points->cluster_index[idx_end_of_local_stack]};
          // pop_back last element of localStack
          local_stack[local_stack_size - 1] = -1;
          --local_stack_size;
          const auto& followers_ies{followers[idx_end_of_local_stack]};
          const auto followers_size{followers[idx_end_of_local_stack].size()};
          // loop over followers of last element of localStack
          for (int j{}; j != followers_size; ++j) {
            // pass id to follower
            int follower{followers_ies[j]};
            dev_points->cluster_index[follower] = temp_cluster_index;
            // push_back follower to localStack
            local_stack[local_stack_size] = follower;
            ++local_stack_size;
          }
        }
      };
    }
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
