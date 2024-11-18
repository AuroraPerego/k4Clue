#ifndef Points_Alpaka_h
#define Points_Alpaka_h

#include <cstdint>
#include <memory>

#include "AlpakaCore/config.h"
#include "AlpakaCore/memory.h"
#include "alpaka/VecArray.h"

using alpakatools::VecArray;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr uint32_t max_followers{100};
  constexpr uint32_t max_seeds{100};
  constexpr uint32_t MAX_POINTS{1000000};

  template <uint8_t nDim>
  class PointsAlpakaView {
  public:
    VecArray<float, nDim> *coords;
    float *addCoord;
    float *weight;
    float *rho;
    float *delta;
    int *nearest_higher;
    int *cluster_index;
    int *is_seed;
  };

  template <uint8_t nDim>
  class PointsAlpaka {
    public:
      PointsAlpaka() = delete;
      explicit PointsAlpaka(Queue queue)
          : coords{alpakatools::make_device_buffer<VecArray<float, nDim>[]>(
                queue, MAX_POINTS)},
            addCoord{alpakatools::make_device_buffer<float[]>(queue, MAX_POINTS)},
            weight{alpakatools::make_device_buffer<float[]>(queue, MAX_POINTS)},
            rho{alpakatools::make_device_buffer<float[]>(queue, MAX_POINTS)},
            delta{alpakatools::make_device_buffer<float[]>(queue, MAX_POINTS)},
            nearest_higher{
                alpakatools::make_device_buffer<int[]>(queue, MAX_POINTS)},
            cluster_index{
                alpakatools::make_device_buffer<int[]>(queue, MAX_POINTS)},
            is_seed{alpakatools::make_device_buffer<int[]>(queue, MAX_POINTS)},
            view_dev{alpakatools::make_device_buffer<PointsAlpakaView<nDim>>(queue)} {
        view_dev.data()->coords = coords.data();
        view_dev.data()->addCoord = addCoord.data();
        view_dev.data()->weight = weight.data();
        view_dev.data()->rho = rho.data();
        view_dev.data()->delta = delta.data();
        view_dev.data()->nearest_higher = nearest_higher.data();
        view_dev.data()->cluster_index = cluster_index.data();
        view_dev.data()->is_seed = is_seed.data();
      }

      // Copy constructor/assignment operator
      PointsAlpaka(const PointsAlpaka &) = delete;
      PointsAlpaka &operator=(const PointsAlpaka &) = delete;
      // Move constructor/assignment operator
      PointsAlpaka(PointsAlpaka &&) = default;
      PointsAlpaka &operator=(PointsAlpaka &&) = default;
      // Destructor
      ~PointsAlpaka() = default;

      VecArray<float, nDim> getPoint(uint32_t i) { return coords[i]; };

      alpaka::Buf<Device, VecArray<float, nDim>, Dim1D, Idx> coords;
      // possible additional coordinate
      alpaka::Buf<Device, float, Dim1D, Idx> addCoord;
      alpaka::Buf<Device, float, Dim1D, Idx> weight;
      alpaka::Buf<Device, float, Dim1D, Idx> rho;
      alpaka::Buf<Device, float, Dim1D, Idx> delta;
      alpaka::Buf<Device, int, Dim1D, Idx> nearest_higher;
      alpaka::Buf<Device, int, Dim1D, Idx> cluster_index;
      alpaka::Buf<Device, int, Dim1D, Idx> is_seed;
      uint32_t nPoints;

      PointsAlpakaView<nDim> *view() { return view_dev.data(); };

    private:
      alpakatools::device_buffer<Device, PointsAlpakaView<nDim>> view_dev;
  };
} // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
