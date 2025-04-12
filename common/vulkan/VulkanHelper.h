#pragma once
#ifndef VULKAN_HELPER_H
#define VULKAN_HELPER_H

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan_core.h>
#endif

#include <string>

void printPhysicalDeviceName(VkPhysicalDevice physicalDevice);
VkResult vkAllocateMemoryVerbose(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDeviceMemory* pMemory);
bool createVkSingleton(const std::string& applicationName, const std::string& engineName, bool enableDebug, VkInstance& instance, VkPhysicalDevice& physicalDevice, VkDevice& device);
uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);
#endif // VULKAN_HELPER_H