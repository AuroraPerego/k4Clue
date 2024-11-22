#include <alpaka/alpaka.hpp>

#include "AlpakaCore/host.h"

namespace alpakatools {

  // alpaka host platform and device

  // return the alpaka host platform
  alpaka::PlatformCpu const& host_platform() {
    static const auto platform = alpaka::PlatformCpu{};
    return platform;
  }

  // return the alpaka host device
  alpaka::DevCpu const& host() {
    static const auto host = alpaka::getDevByIdx(host_platform(), 0u);
    // assert on the host index ?
    return host;
  }

}  // namespace alpakatools
