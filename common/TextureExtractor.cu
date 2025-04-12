#include "TextureExtractor.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>

cudaChannelFormatDesc GetChannelDesc(TextureFormat format) {
    switch (format)
    {
        case TextureFormat::RGBA8:
            return cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        case TextureFormat::RGB10A2:
            return cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102);
        case TextureFormat::RGBA32:
            // return cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102);
            return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
        default:
            throw std::runtime_error("Invalid channel format");
    }
}

size_t GetChannelSize(TextureFormat format) {
    switch (format)
    {
        case TextureFormat::RGBA8:
            return 4;
        case TextureFormat::RGB10A2:
            return 4;
        case TextureFormat::RGBA32:
            return 4 * 4;
        default:
            throw std::runtime_error("Invalid channel format");
    }
}

TextureExtractor::TextureExtractor(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes)
{
    resourceName = resName;
    width = textureWidth;
    height = textureHeight;
    format = textureFormat;
    totalBytes = textureBytes;
}

std::vector<glm::vec3> TextureExtractor::extractTextureData() {
    if (!initialized) {
        return {};
    }

    std::vector<glm::vec3> result(width * height);

    // Allocate device memory
    glm::vec3* d_rgb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rgb, width * height * sizeof(glm::vec3)));

    cudaArray_t cuArray;
    cudaChannelFormatDesc channelDesc = GetChannelDesc(format);
    size_t channelSize = GetChannelSize(format);
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, devicePtr, width * height * channelSize, cudaMemcpyDeviceToDevice));

    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // cudaArray_t baseArray;
    // CUDA_CHECK(cudaGetMipmappedArrayLevel(&baseArray, mipArray, 0));
    // resDesc.resType = cudaResourceTypeArray;
    // resDesc.res.array.array = baseArray;

    // resDesc.resType = cudaResourceTypeMipmappedArray;
    // resDesc.res.mipmap.mipmap = mipArray;

    // resDesc.resType = cudaResourceTypePitch2D;
    // resDesc.res.pitch2D.desc = cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102);
    // resDesc.res.pitch2D.devPtr = devicePtr;
    // resDesc.res.pitch2D.width = width;
    // resDesc.res.pitch2D.height = height;
    // // resDesc.res.pitch2D.pitchInBytes = rowPitch;
    // // resDesc.res.pitch2D.pitchInBytes = (width+2) * sizeof(uint32_t);
    // resDesc.res.pitch2D.pitchInBytes = (width) * sizeof(uint32_t);
    // // resDesc.res.pitch2D.pitchInBytes = (width+2);

    // resDesc.resType = cudaResourceTypeLinear;
    // resDesc.res.linear.desc = cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102);
    // resDesc.res.linear.devPtr = devicePtr;
    // // resDesc.res.linear.sizeInBytes = totalBytes;
    // resDesc.res.linear.sizeInBytes = width * height * sizeof(uint32_t);

    cudaTextureDesc texDesc = {};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;     // No filtering
    texDesc.readMode = cudaReadModeElementType;   // Read raw values
    texDesc.normalizedCoords = 0;                 // Use integer coordinates for fetch

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // Launch conversion kernel
    dim3 threads(16, 16);
    dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);
    unpackRGB32Kernel<<<blocks, threads>>>(texObj, d_rgb, width, height);

    // if (packed)
    // {
    //     unpackRGB10A2PackedKernel<<<blocks, threads>>>((uint32_t*)devicePtr, d_rgb, width, height);
    // }
    // else
    // {
    //     unpackRGB32Kernel<<<blocks, threads>>>(texObj, d_rgb, width, height);
    // }
    
    // Copy result back to host
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(result.data(), d_rgb, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    
    // Clean up device memory
    cudaFree(d_rgb);
    
    return result;
}

__global__ void unpackRGB32Kernel(cudaTextureObject_t texObj, glm::vec3* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float4 pixel = tex2D<float4>(texObj, x, y);
        output[idx] = glm::vec3(pixel.x, pixel.y, pixel.z);
    }
}

__global__ void unpackRGB10A2PackedKernel(uint32_t* data, glm::vec3* output, int width, int height) {
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
