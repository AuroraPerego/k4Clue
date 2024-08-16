// g++ -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -I/afs/cern.ch/user/a/aperego/public/pixeltrack-standalone/external/alpaka/include -isystem /afs/cern.ch/user/a/aperego/public/pixeltrack-standalone/external/boost/include -Wall CLUEAlgoAlpaka.cc -c -o CLUEAlgoAlpaka.cc.o -I/eos/user/a/aperego/fcc/k4Clue/include -I/eos/user/a/aperego/fcc/k4Clue -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_PRESENT  -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND

#include "include/alpaka/CLUEAlgoAlpaka.hpp"

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

