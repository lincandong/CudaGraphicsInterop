#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <exception>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": \n" \
                  << "\tErrorName: " << cudaGetErrorName(err) << " (" << err << ")\n" \
                  << "\tErrorString: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        throw std::exception("error"); \
    } \
} while(0)

enum class ExtractorAPI {
    D3D12,
    VULKAN
};

class TextureExtractor {
public:
    virtual ExtractorAPI GetAPIType() = 0;
    TextureExtractor() = default;
    virtual ~TextureExtractor() { cleanup(); };
    
    // Deleted copy/move operations
    TextureExtractor(const TextureExtractor&) = delete;
    TextureExtractor& operator=(const TextureExtractor&) = delete;
    TextureExtractor(TextureExtractor&&) = delete;
    TextureExtractor& operator=(TextureExtractor&&) = delete;
    
    // Extract RGB texture data from D3D12 shared handle
    std::vector<glm::vec3> extract(std::string& resourceName, int width, int height);

protected:
    // CUDA objects
    void* devicePtr = nullptr;
    cudaMipmappedArray_t mipArray = nullptr;
    cudaExternalMemory_t extMemory = nullptr;
    cudaArray_t cudaResource = nullptr;
    cudaDeviceProp cudaDevProp;
    cudaStream_t m_streamToRun;
    
    // Texture dimensions
    int width = 0;
    int height = 0;
    int rowPitch = 0;
    int totalBytes = 0;
    
    // in order execution
    virtual bool initialize(std::string& resourceName, int textureWidth, int textureHeight) = 0;
    virtual bool importTextureToCuda(std::string& resourceName) = 0;
    virtual std::vector<glm::vec3> extractTextureData() = 0;
    virtual void cleanup(){};
};

// decode kernels
__global__ void unpackRGB10A2Kernel(cudaTextureObject_t texObj, glm::vec3* output, int width, int height);
__global__ void unpackRGB10A2RawKernel(uint32_t* data, glm::vec3* output, int width, int height);
