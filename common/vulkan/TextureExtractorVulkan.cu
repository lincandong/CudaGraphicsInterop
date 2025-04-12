#include "TextureExtractorVulkan.h"

#ifdef ENABLE_VULKAN

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "VulkanHelper.h"
#include "HandleHelper.h"

TextureExtractorVulkan::TextureExtractorVulkan(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes) : TextureExtractor(resName, textureFormat, textureWidth, textureHeight, textureBytes) 
{
    
}

bool TextureExtractorVulkan::initialize() {
    std::string applicationName("TextureExtrator");
    std::string engineName("TextureExtratorVulkan");

    if (!createVkSingleton(applicationName, engineName, false, instance, physicalDevice, device))
    {
        std::cerr << "Failed to initialize vulkan device" << std::endl;
        return false;
    }

    // Initialize CUDA device that matches the Vulkan physical device
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    
    // Simply use the first nvidia device
    cudaDeviceID = -1;
    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        if (strstr(deviceProp.name, "NVIDIA") != NULL ||
            strstr(deviceProp.name, "nvidia") != NULL ||
            strstr(deviceProp.name, "Nvidia") != NULL)
        {            
            cudaDeviceID = i;
            break;
        }
    }
    if (cudaDeviceID == -1)
    {
        std::cerr << "Failed to find nvidia cuda device" << std::endl;
        return false;
    }

    cudaSetDevice(cudaDeviceID);

    return true;
}

VkResult ImportExternalMemory(
    VkDevice VulkanDevice,
    VkPhysicalDevice VulkanPhysicalDevice,
    const std::string& ResName,
    VkDeviceSize MemorySize,
    VkDeviceMemory* OutDeviceMemory
)
{
    // Allocate memory for the imported resource
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = MemorySize;
    
    // Get memory type index that supports external memory
    uint32_t memoryTypeIndex = findMemoryType(
        VulkanPhysicalDevice,
        0xFFFFFFFF,  // All memory types
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
    allocInfo.memoryTypeIndex = memoryTypeIndex; // Replace with actual memory type index
    
#ifdef _WIN32
    // Windows implementation using Win32 handles
    
    // Setup import info for Windows
    VkImportMemoryWin32HandleInfoKHR importInfo = {};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR;

    importInfo.handle = nullptr;
    const auto wResourceName = std::wstring(ResName.begin(), ResName.end());
    importInfo.name = wResourceName.c_str();
    
    // Chain the import info to allocation info
    importInfo.pNext = nullptr;
    allocInfo.pNext = &importInfo;
    
    // Allocate the memory
    return vkAllocateMemoryVerbose(VulkanDevice, VulkanPhysicalDevice, &allocInfo, nullptr, OutDeviceMemory);
        
#else
    // Linux implementation using file descriptors
    std::cout << "opening shared memory: " <<  ResName << std::endl;
    int local_fd = ReceiveLinuxFD(ResName);
    if (local_fd == -1)
    {
        std::cerr << "failed to receive linux fd" << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    // Setup import info for Linux
    std::cout << "importing shared fd: " << local_fd << std::endl;
    std::cout << "Memory Size: " << MemorySize << std::endl;
    VkImportMemoryFdInfoKHR importInfo = {};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    importInfo.fd = local_fd;
    
    // Chain the import info to allocation info
    importInfo.pNext = nullptr;
    allocInfo.pNext = &importInfo;
    
    // Allocate the memory
    // result = vkAllocateMemory(VulkanDevice, &allocInfo, nullptr, OutDeviceMemory);
    VkResult result = vkAllocateMemoryVerbose(VulkanDevice, VulkanPhysicalDevice, &allocInfo, nullptr, OutDeviceMemory);
    if (result != VK_SUCCESS)
    {        
        std::cerr << "failed to receive import external vk memory, error code: " << result << std::endl;
        close(local_fd);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    close(local_fd);
    return result;
#endif
}

bool TextureExtractorVulkan::importTextureToCuda() 
{
    // todo: should be called only once for all extractor instances
    if (!initialized && !(initialized = initialize()))
    {
        std::cerr << "device not initialized yet" << std::endl;
        return false;
    }
    
    if (ImportExternalMemory(device, physicalDevice, resourceName, totalBytes, &externalMemory) != VK_SUCCESS)
    {
        std::cerr << "Failed to import external memory" << std::endl;
        return false;
    }
    
    // Set up CUDA external memory import descriptor
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
#ifdef _WIN32
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    // externalMemoryHandleDesc.handle.win32.handle = externalMemoryHandle;
    const auto wResourceName = std::wstring(resourceName.begin(), resourceName.end());
    externalMemoryHandleDesc.handle.win32.name = wResourceName.c_str();
#else
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = GetVKMemHandle(device, externalMemory);
#endif
    externalMemoryHandleDesc.size = totalBytes;
    externalMemoryHandleDesc.flags = cudaExternalMemoryDedicated;
    
    // Import the external memory to CUDA
    cudaExternalMemory_t cudaExtMemory;
    CUDA_CHECK(cudaImportExternalMemory(&cudaExtMemory, &externalMemoryHandleDesc));
    
    // Map the buffer to get a CUDA device pointer
    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = totalBytes;
    bufferDesc.flags = 0;
    
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devicePtr, cudaExtMemory, &bufferDesc));

    return true;
}

void TextureExtractorVulkan::cleanup() {
   // Clean up CUDA resources
   if (devicePtr) {
       cudaFree(devicePtr);
       devicePtr = nullptr;
   }
   
   if (device != VK_NULL_HANDLE) {
       vkDestroyDevice(device, nullptr);
       device = VK_NULL_HANDLE;
   }
   if (instance != VK_NULL_HANDLE) {
       vkDestroyInstance(instance, nullptr);
       instance = VK_NULL_HANDLE;
   }
}

#endif // ENABLE_VULKAN