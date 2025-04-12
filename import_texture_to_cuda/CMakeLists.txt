set(PROJECT_NAME import_texture_to_cuda)
set(SRCROOT ${PROJECT_SOURCE_DIR}/${PROJECT_NAME})
enable_language(CUDA)

set(SRC
    ${SRCROOT}/main.cpp
    # add more here
)

add_project(${PROJECT_NAME}
            SOURCES ${SRC}
)

if (WIN32 AND ENABLE_D3D12)
target_link_libraries(${PROJECT_NAME} PRIVATE
    d3d12
    dxgi
    dxguid
    d3dcompiler
)
endif()

if (ENABLE_VULKAN)
find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC
    ${Vulkan_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PUBLIC
    ${Vulkan_LIBRARIES}
)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "80")
