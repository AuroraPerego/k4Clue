#ifndef AlpakaCore_getHostCachingAllocator_h
#define AlpakaCore_getHostCachingAllocator_h

#include "AlpakaCore/AllocatorConfig.h"
#include "AlpakaCore/CachingAllocator.h"
#include "AlpakaCore/alpaka/devices.h"
#include "AlpakaCore/config.h"
#include "AlpakaCore/host.h"

namespace alpakatools {

  template <typename TQueue>
  inline CachingAllocator<alpaka_common::Host, TQueue>& getHostCachingAllocator() {
    // thread safe initialisation of the host allocator
    static CachingAllocator<alpaka_common::Host, TQueue> allocator(host(),
                                                                      config::binGrowth,
                                                                      config::minBin,
                                                                      config::maxBin,
                                                                      config::maxCachedBytes,
                                                                      config::maxCachedFraction,
                                                                      false,   // reuseSameQueueAllocations
                                                                      false);  // debug

    // the public interface is thread safe
    return allocator;
  }

}  // namespace alpakatools

#endif  // AlpakaCore_getHostCachingAllocator_h
