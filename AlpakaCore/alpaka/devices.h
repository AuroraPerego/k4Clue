#ifndef AlpakaCore_alpaka_devices_h
#define AlpakaCore_alpaka_devices_h

#include <vector>

#include <alpaka/alpaka.hpp>

#include "AlpakaCore/config.h"

namespace alpakatools {

  // alpaka accelerator platform and devices
  // Note: even when TPlatform is the same as PlatformHost these functions return different objects
  // than host_platform() and host()

  // return the alpaka accelerator platform for the given platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  TPlatform const& platform();

  // return the alpaka accelerator devices for the given platform
  template <typename TPlatform, typename = std::enable_if_t<alpaka::isPlatform<TPlatform>>>
  std::vector<alpaka::Dev<TPlatform>> const& devices();

}  // namespace alpakatools

#endif  // AlpakaCore_alpaka_devices_h
