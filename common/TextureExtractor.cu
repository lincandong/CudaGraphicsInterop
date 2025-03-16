#include "TextureExtractor.h"

__global__ void unpackRGB10A2Kernel(cudaTextureObject_t texObj, glm::vec3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float4 pixel = tex2D<float4>(texObj, x, y);
        output[idx] = glm::vec3(pixel.x, pixel.y, pixel.z);
    }
}

__global__ void unpackRGB10A2RawKernel(uint32_t* data, glm::vec3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int inIndex = y * width + x;
        int outIndex = y * width + x;
        uint32_t pixel = data[inIndex];
        output[outIndex] = glm::vec3(
            (pixel & 0x3FF) / 1023.0f,
            ((pixel >> 10) & 0x3FF) / 1023.0f,
            ((pixel >> 20) & 0x3FF) / 1023.0f
        );
    }
}

// Extract RGB texture data from D3D12 shared handle
std::vector<glm::vec3> TextureExtractor::extract(std::string& resourceName, int width, int height) {
    if (!initialize(resourceName, width, height)) {
        std::cerr << "Failed to initialize texture extraction" << std::endl;
        return {};
    }
    
    std::vector<glm::vec3> result = extractTextureData();
    cleanup();
    return result;
}