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

# Optionally glob, but only for CMake 3.12 or later: file(GLOB HEADER_LIST
# CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/modern/*.hpp")
set(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/AlpakaCore/*.h" "${PROJECT_SOURCE_DIR}/AlpakaCore/alpaka/*.h")

set(SOURCES
    ${PROJECT_SOURCE_DIR}/AlpakaCore/alpaka/devices.cc
    ${PROJECT_SOURCE_DIR}/AlpakaCore/alpaka/initialise.cc
    ${PROJECT_SOURCE_DIR}/AlpakaCore/host.cc
)

## Make an automatic library - will be static or dynamic based on user setting
add_library(AlpakaCore_lib ${SOURCES} ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(AlpakaCore_lib PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${alpaka_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${Boost_INCLUDE_DIR}>)

# alpaka build flags
target_compile_options(
  AlpakaCore_lib
  PRIVATE -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
          -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_SYNC_BACKEND)

install(TARGETS AlpakaCore_lib ${INSTALL_LIBRARIES}
  EXPORT AlpakaCoreTarget
  DESTINATION "${CMAKE_INSTALL_LIBDIR}")
