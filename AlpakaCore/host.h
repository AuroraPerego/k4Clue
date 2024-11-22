#ifndef AlpakaCore_host_h
#define AlpakaCore_host_h

#include <alpaka/alpaka.hpp>

namespace alpakatools {

  // returns the alpaka host platform
  alpaka::PlatformCpu const& host_platform();

  // returns the alpaka host device
  alpaka::DevCpu const& host();

}  // namespace cms::alpakatools

#endif  // AlpakaCore_host_h
