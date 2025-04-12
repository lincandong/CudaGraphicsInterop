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
#ifdef _WIN32
#include <windows.h>
#include <wrl/client.h>
#endif

class TextureExtractorVulkan : public TextureExtractor {
public:
    TextureExtractorVulkan(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes);
    ~TextureExtractorVulkan(){ cleanup(); };
    ExtractorAPI GetAPIType() override { return ExtractorAPI::VULKAN; };

    bool importTextureToCuda() override;
private:
    // instance data
    VkDeviceMemory externalMemory = VK_NULL_HANDLE;

    // singleton data
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkInstance instance = VK_NULL_HANDLE;
    int cudaDeviceID = 0;

    // shared interface    
    bool initialize() override;
    void cleanup() override;
};

#endif