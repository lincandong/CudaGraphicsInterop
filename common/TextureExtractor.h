#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <exception>
#include <string>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": \n" \
                  << "\tErrorName: " << cudaGetErrorName(err) << " (" << err << ")\n" \
                  << "\tErrorString: " << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        throw std::runtime_error("error"); \
    } \
} while(0)

enum class TextureFormat
{
    RGB10A2,
    RGBA8,
    RGBA32
};

enum class ExtractorAPI {
    D3D12,
    VULKAN
};

class TextureExtractor {
public:
    virtual ExtractorAPI GetAPIType() = 0;
    TextureExtractor(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes);
    virtual ~TextureExtractor(){};
    
    // Deleted copy/move operations
    TextureExtractor(const TextureExtractor&) = delete;
    TextureExtractor& operator=(const TextureExtractor&) = delete;
    TextureExtractor(TextureExtractor&&) = delete;
    TextureExtractor& operator=(TextureExtractor&&) = delete;

    // initialized before importing texture to cuda
    virtual bool initialize() = 0;
    // cleanup in destructor
    virtual void cleanup() = 0;

    // if initialized, call this to import external texture and store it as devicePtr
    virtual bool importTextureToCuda() = 0;

    void* getDevicePtr() const { return devicePtr; }

    // a helper function to extract texture data from devicePtr
    std::vector<glm::vec3> extractTextureData();

protected:
    bool initialized = false;

    // CUDA objects
    void* devicePtr = nullptr;
    cudaMipmappedArray_t mipArray = nullptr;
    cudaExternalMemory_t extMemory = nullptr;
    cudaArray_t cudaResource = nullptr;
    cudaDeviceProp cudaDevProp;
    
    // Texture description, should be determined at initliazation
    std::string resourceName;
    TextureFormat format;
    int width = 0;
    int height = 0;
    int totalBytes = 0;
    bool packed = false; // if the buffer is packed as uint or simply float
};

// decode kernels
__global__ void unpackRGB32Kernel(cudaTextureObject_t texObj, glm::vec3* output, int width, int height);
__global__ void unpackRGB10A2PackedKernel(uint32_t* data, glm::vec3* output, int width, int height);
