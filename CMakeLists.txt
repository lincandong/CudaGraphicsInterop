cmake_minimum_required(VERSION 3.10)
project(TowardsRealtimeRayTracing)
set(CMAKE_CXX_STANDARD 17)

include(Macros.cmake)

set(ENABLE_D3D12 OFF)
set(ENABLE_VULKAN ON)
# set(ENABLE_D3D12 ON)
# set(ENABLE_VULKAN OFF)

if (WIN32 AND ENABLE_D3D12)
    add_definitions(-DENABLE_D3D12)
endif()
if (ENABLE_VULKAN)
    add_definitions(-DENABLE_VULKAN)
endif()

# add external libraries
include_directories(${PROJECT_SOURCE_DIR}/external)

add_subdirectory(common)
add_subdirectory(import_texture_to_cuda)
