// g++ -std=c++20 -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -I/eos/user/a/aperego/fcc/alpaka/include -isystem /cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-477823d53efabc5118f199265eb7ab49/include -Wall src/alpaka/CLUEAlgoAlpaka.cc -c -o CLUEAlgoAlpaka.cc.o -I/eos/user/a/aperego/fcc/k4Clue/include -I/eos/user/a/aperego/fcc/k4Clue -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT  -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#include "AlpakaCore/config.h"
#include "include/alpaka/CLUEAlgoAlpaka.h"

template<typename TAcc>
using CLICdetEndcapLayerTiles = TilesAlpaka_T<TAcc, CLICdetEndcapLayerTilesConstants2D>;
template<typename TAcc>
using CLICdetBarrelLayerTiles = TilesAlpaka_T<TAcc, CLICdetBarrelLayerTilesConstants2D>;
template<typename TAcc>
using CLDEndcapLayerTiles = TilesAlpaka_T<TAcc, CLDEndcapLayerTilesConstants2D>;
template<typename TAcc>
using CLDBarrelLayerTiles = TilesAlpaka_T<TAcc, CLDBarrelLayerTilesConstants2D>;
template<typename TAcc>
using LArBarrelLayerTiles = TilesAlpaka_T<TAcc, LArBarrelLayerTilesConstants2D>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template class CLUEAlgoAlpaka_T<Acc1D, CLICdetEndcapLayerTilesConstants2D>;
  template class CLUEAlgoAlpaka_T<Acc1D, CLICdetBarrelLayerTilesConstants2D>;
  template class CLUEAlgoAlpaka_T<Acc1D, CLDEndcapLayerTilesConstants2D>;
  template class CLUEAlgoAlpaka_T<Acc1D, CLDBarrelLayerTilesConstants2D>;
  template class CLUEAlgoAlpaka_T<Acc1D, LArBarrelLayerTilesConstants2D>;
}
