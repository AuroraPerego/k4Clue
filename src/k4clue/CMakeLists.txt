#[[
Copyright (c) 2020-2024 Key4hep-Project.

This file is part of Key4hep.
See https://key4hep.github.io/key4hep-doc/ for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

# CLUE as Gaudi algorithm

include(ExternalData)
set(ExternalData_URL_TEMPLATES
  "https://key4hep.web.cern.ch:443/testFiles/k4clue/inputData/clic/%(hash)"
)

gaudi_add_module(ClueGaudiAlgorithmWrapper
  SOURCES
    ${PROJECT_SOURCE_DIR}/src/ClueGaudiAlgorithmWrapper.cpp
    ${PROJECT_SOURCE_DIR}/src/CLUECalorimeterHit.cpp
    ${PROJECT_SOURCE_DIR}/src/CLUENtuplizer.cpp
  LINK
    Gaudi::GaudiKernel
    k4FWCore::k4FWCore
    DD4hep::DDCore
    EDM4HEP::edm4hep
    CLUEAlgo_lib
    AlpakaCore_lib
)

target_include_directories(ClueGaudiAlgorithmWrapper PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${alpaka_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${Boost_SOURCE_DIR}/include>
)

# alpaka build flags
target_compile_options(
  ClueGaudiAlgorithmWrapper
  PRIVATE -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)

ExternalData_Add_Test(k4clue_tests NAME gaudiWrapper
         WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
         COMMAND bash -c "k4run ${PROJECT_SOURCE_DIR}/gaudi_opts/clue_gaudi_wrapper.py --EventDataSvc.input DATA{${PROJECT_SOURCE_DIR}/test/input_files/20240905_gammaFromVertex_10GeV_uniform_10events_reco_edm4hep.root} --ClueGaudiAlgorithmWrapperName.OutputLevel 2 --CLUEAnalysis.OutputLevel 2")

set_tests_properties(gaudiWrapper PROPERTIES
                     FAIL_REGULAR_EXPRESSION "ERROR;Failed")

set_property(TEST gaudiWrapper APPEND PROPERTY ENVIRONMENT "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}:${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}:${PROJECT_BINARY_DIR}/src/k4clue:$<$<TARGET_EXISTS:ROOT::Core>:$<TARGET_FILE_DIR:ROOT::Core>>:$<$<TARGET_EXISTS:EDM4HEP::edm4hep>:$<TARGET_FILE_DIR:EDM4HEP::edm4hep>>:$<$<TARGET_EXISTS:podio::podio>:$<TARGET_FILE_DIR:podio::podio>>:$ENV{LD_LIBRARY_PATH}")
set_property(TEST gaudiWrapper APPEND PROPERTY ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/${CMAKE_PROJECT_NAME}/genConfDir:${PROJECT_BINARY_DIR}/src/k4clue/genConfDir:$ENV{PYTHONPATH}")

ExternalData_Add_Target(k4clue_tests)

#g++ \
#  -std=c++20 \
#  -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED \
#  -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND  \
#  -DALPAKA_DISABLE_VENDOR_RNG \
#  -I/eos/user/a/aperego/fcc/k4Clue/include \
#  -I/eos/user/a/aperego/fcc/k4Clue \
#  -Wall ClueGaudiAlgorithmWrapper.cpp \
#  -c \
#  -o CLUEWrapper.cc.o \
#  -I/eos/user/a/aperego/fcc/alpaka/include \
#  -isystem /cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-477823d53efabc5118f199265eb7ab49/include \
#  -isystem /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/gaudi/38.0-u3otlq/include \
#  -isystem /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/k4fwcore/1.0pre19-zlns7f/include \
#  -isystem /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/root/6.28.10-j6yheu/include/root \
#  -isystem /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/podio/0.99-uovuyy/include \
#  -isystem /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/edm4hep/0.10.5-27swdr/include \
#  -isystem /cvmfs/sw.hsf.org/key4hep/releases/2024-03-10/x86_64-almalinux9-gcc11.3.1-opt/dd4hep/1.28-q6ea5f/include
#
