#pragma once

#ifdef ENABLE_VULKAN

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <exception>
#include "common/TextureExtractor.h"
#include <windows.h>
#include <wrl/client.h>

#ifdef _WIN32
using Microsoft::WRL::ComPtr;
#endif

class TextureExtractorVulkan : public TextureExtractor {
public:
    TextureExtractorVulkan() = default;
    ~TextureExtractorVulkan() = default;
    ExtractorAPI GetAPIType() override { return ExtractorAPI::VULKAN; };

private:
    // Vulkan objects
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkInstance instance = VK_NULL_HANDLE;
    VkDeviceMemory textureMemory = VK_NULL_HANDLE;
    VkImage textureImage = VK_NULL_HANDLE;
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    
    // CUDA objects
    uint32_t cudaDeviceID = 0;
    
    // External memory handles
    VkExternalMemoryHandleTypeFlagBits externalMemoryHandleType;
    #ifdef _WIN32
    HANDLE externalMemoryHandle = nullptr;
    #else
    int externalMemoryHandle = -1;
    #endif

    // shared interface    
    bool initialize(std::string& resourceName, int textureWidth, int textureHeight) override;
    bool importTextureToCuda() override;
    std::vector<glm::vec3> extractTextureData() override;
    void cleanup() override;

    // Vulkan helper functions
    bool initVulkanDevice();
    bool convertVulkanTextureToLinearBuffer(VkDevice device, VkImage textureImage, int width, int height, 
                                            cudaExternalMemory_t& extMemory, void** devicePtr);
    bool findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& memoryTypeIndex);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, 
                      VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height);
};

#endif