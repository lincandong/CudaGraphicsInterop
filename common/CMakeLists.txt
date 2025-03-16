# clean cache at every build
set(COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/common/TextureExtractor.h
    ${PROJECT_SOURCE_DIR}/common/TextureExtractor.cu
    ${PROJECT_SOURCE_DIR}/common/Utility.hpp
    ${PROJECT_SOURCE_DIR}/common/Utility.cpp
    ${PROJECT_SOURCE_DIR}/common/CudaPortable.hpp
    CACHE INTERNAL ""
)

if (WIN32 AND ENABLE_D3D12)
    set(COMMON_SOURCES
        ${COMMON_SOURCES}
        ${PROJECT_SOURCE_DIR}/common/d3d12/TextureExtractorD3D12.h
        ${PROJECT_SOURCE_DIR}/common/d3d12/TextureExtractorD3D12.cu
        ${PROJECT_SOURCE_DIR}/common/d3d12/d3d12_helper.h
        ${PROJECT_SOURCE_DIR}/common/d3d12/d3dx12.h
        CACHE INTERNAL ""
    )
endif()