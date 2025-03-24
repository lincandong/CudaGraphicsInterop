#include "TextureExtractorVulkan.h"

#ifdef ENABLE_VULKAN

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vulkan/vulkan_core.h>

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#define VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
#else
#include <vulkan/vulkan_core.h>
#define VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
#endif

bool TextureExtractorVulkan::initialize(std::string& resName, int textureWidth, int textureHeight) {
    resourceName = resName;
    width = textureWidth;
    height = textureHeight;
    
    // Initialize Vulkan
    if (!initVulkanDevice()) {
        std::cerr << "Failed to initialize Vulkan device" << std::endl;
        return false;
    }
    
    return importTextureToCuda();
}

bool TextureExtractorVulkan::initVulkanDevice() {
    // Create Vulkan instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "TextureExtractor";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;
    
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // Add required extensions for external memory
    std::vector<const char*> extensions = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };
    
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();
    
    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }
    
    // Select physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    
    if (deviceCount == 0) {
        std::cerr << "Failed to find GPUs with Vulkan support" << std::endl;
        return false;
    }
    
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    
    // Simply use the first device for this example
    physicalDevice = devices[0];
    
    // Find a graphics queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            graphicsQueueFamilyIndex = i;
            break;
        }
    }
    
    if (graphicsQueueFamilyIndex == UINT32_MAX) {
        std::cerr << "Failed to find a graphics queue family" << std::endl;
        return false;
    }
    
    // Create logical device with required extensions
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    
    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
#ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME
#endif
    };
    
    VkPhysicalDeviceFeatures deviceFeatures = {};
    
    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device" << std::endl;
        return false;
    }
    
    // Get graphics queue
    vkGetDeviceQueue(device, graphicsQueueFamilyIndex, 0, &graphicsQueue);
    
    // Create command pool
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    
    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool" << std::endl;
        return false;
    }
    
    // Initialize CUDA device that matches the Vulkan physical device
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    
    for (int i = 0; i < cudaDeviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        // In a real application, would need to match the Vulkan device with the CUDA device
        // This is simplified for example purposes
        cudaDeviceID = i;
        cudaSetDevice(cudaDeviceID);
        break;
    }
    
    externalMemoryHandleType = VULKAN_EXTERNAL_MEMORY_HANDLE_TYPE;
    
    return true;
}

bool TextureExtractorVulkan::importTextureToCuda() {
    // Create an image for the texture
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    
    // Add external memory flag
    VkExternalMemoryImageCreateInfo externalMemoryImageInfo = {};
    externalMemoryImageInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    externalMemoryImageInfo.handleTypes = externalMemoryHandleType;
    imageInfo.pNext = &externalMemoryImageInfo;
    
    if (vkCreateImage(device, &imageInfo, nullptr, &textureImage) != VK_SUCCESS) {
        std::cerr << "Failed to create texture image" << std::endl;
        return false;
    }
    
    // Allocate memory for the image
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, textureImage, &memRequirements);
    
    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = externalMemoryHandleType;
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.pNext = &exportAllocInfo;
    
    uint32_t memoryTypeIndex;
    if (!findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memoryTypeIndex)) {
        std::cerr << "Failed to find suitable memory type for texture" << std::endl;
        return false;
    }
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &textureMemory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate texture memory" << std::endl;
        return false;
    }
    
    vkBindImageMemory(device, textureImage, textureMemory, 0);
    
    // Get exportable handle
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR getWin32HandleInfo = {};
    getWin32HandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getWin32HandleInfo.memory = textureMemory;
    getWin32HandleInfo.handleType = externalMemoryHandleType;
    
    auto fpGetMemoryWin32Handle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!fpGetMemoryWin32Handle) {
        std::cerr << "Failed to get vkGetMemoryWin32HandleKHR function" << std::endl;
        return false;
    }
    
    if (fpGetMemoryWin32Handle(device, &getWin32HandleInfo, &externalMemoryHandle) != VK_SUCCESS) {
        std::cerr << "Failed to get Win32 handle for memory" << std::endl;
        return false;
    }
#else
    VkMemoryGetFdInfoKHR getFdInfo = {};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = textureMemory;
    getFdInfo.handleType = externalMemoryHandleType;
    
    auto fpGetMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!fpGetMemoryFd) {
        std::cerr << "Failed to get vkGetMemoryFdKHR function" << std::endl;
        return false;
    }
    
    if (fpGetMemoryFd(device, &getFdInfo, &externalMemoryHandle) != VK_SUCCESS) {
        std::cerr << "Failed to get file descriptor for memory" << std::endl;
        return false;
    }
#endif
    
    // Transition image layout for copying data
    transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_UNORM, 
                          VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    
    // Here you would populate the texture with data from your resource
    // This is application-specific and depends on how you're loading your textures
    
    // Create a staging buffer for CPU->GPU transfers
    createBuffer(width * height * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer, stagingBufferMemory);
    
    // Map staging buffer memory and populate with texture data
    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, width * height * 4, 0, &data);
    
    // Here you would load your texture data into the mapped memory
    // For example:
    // loadTextureData(resourceName, data, width, height);
    
    vkUnmapMemory(device, stagingBufferMemory);
    
    // Copy staging buffer to the image
    VkCommandBufferAllocateInfo allocInfo2 = {};
    allocInfo2.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo2.commandPool = commandPool;
    allocInfo2.commandBufferCount = 1;
    
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo2, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    
    vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, textureImage, 
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    
    // Transition image to transfer source for later copying to buffer
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = textureImage;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    
    vkCmdPipelineBarrier(commandBuffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    
    // Import Vulkan memory to CUDA
    cudaExternalMemory_t cudaExtMemory;
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    
#ifdef _WIN32
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    externalMemoryHandleDesc.handle.win32.handle = externalMemoryHandle;
#else
    externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    externalMemoryHandleDesc.handle.fd = externalMemoryHandle;
#endif
    
    externalMemoryHandleDesc.size = memRequirements.size;
    
    if (cudaImportExternalMemory(&cudaExtMemory, &externalMemoryHandleDesc) != cudaSuccess) {
        std::cerr << "Failed to import external memory to CUDA" << std::endl;
        return false;
    }
    
    // Map the external memory to a CUDA-accessible pointer
    void* cudaDevicePtr = nullptr;
    bool result = convertVulkanTextureToLinearBuffer(device, textureImage, width, height, cudaExtMemory, &cudaDevicePtr);
    
    if (!result) {
        std::cerr << "Failed to convert Vulkan texture to linear buffer" << std::endl;
        return false;
    }
    
    // Store the CUDA device pointer for later use
    devicePtr = cudaDevicePtr;
    
    return true;
}

bool TextureExtractorVulkan::convertVulkanTextureToLinearBuffer(VkDevice device, VkImage textureImage, 
                                                              int width, int height,
                                                              cudaExternalMemory_t& extMemory, 
                                                              void** devicePtr) {
    // Create a linear buffer to receive image data
    VkBuffer linearBuffer;
    VkDeviceMemory linearBufferMemory;
    
    // Create buffer with external memory flags
    VkExternalMemoryBufferCreateInfo extBufferCreateInfo = {};
    extBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufferCreateInfo.handleTypes = externalMemoryHandleType;
    
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = width * height * 4;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.pNext = &extBufferCreateInfo;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &linearBuffer) != VK_SUCCESS) {
        std::cerr << "Failed to create linear buffer" << std::endl;
        return false;
    }
    
    // Allocate memory for the buffer
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, linearBuffer, &memRequirements);
    
    VkExportMemoryAllocateInfo exportAllocInfo = {};
    exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportAllocInfo.handleTypes = externalMemoryHandleType;
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.pNext = &exportAllocInfo;
    
    uint32_t memoryTypeIndex;
    if (!findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memoryTypeIndex)) {
        std::cerr << "Failed to find suitable memory type for linear buffer" << std::endl;
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &linearBufferMemory) != VK_SUCCESS) {
        std::cerr << "Failed to allocate linear buffer memory" << std::endl;
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    
    vkBindBufferMemory(device, linearBuffer, linearBufferMemory, 0);
    
    // Copy image data to the linear buffer
    copyImageToBuffer(textureImage, linearBuffer, width, height);
    
    // Get exportable handle for the linear buffer memory
    HANDLE bufferMemoryHandle = nullptr;
    
#ifdef _WIN32
    VkMemoryGetWin32HandleInfoKHR getWin32HandleInfo = {};
    getWin32HandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    getWin32HandleInfo.memory = linearBufferMemory;
    getWin32HandleInfo.handleType = externalMemoryHandleType;
    
    auto fpGetMemoryWin32Handle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!fpGetMemoryWin32Handle) {
        std::cerr << "Failed to get vkGetMemoryWin32HandleKHR function" << std::endl;
        vkFreeMemory(device, linearBufferMemory, nullptr);
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    
    if (fpGetMemoryWin32Handle(device, &getWin32HandleInfo, &bufferMemoryHandle) != VK_SUCCESS) {
        std::cerr << "Failed to get Win32 handle for buffer memory" << std::endl;
        vkFreeMemory(device, linearBufferMemory, nullptr);
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
#else
    int bufferMemoryFd = -1;
    VkMemoryGetFdInfoKHR getFdInfo = {};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = linearBufferMemory;
    getFdInfo.handleType = externalMemoryHandleType;
    
    auto fpGetMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!fpGetMemoryFd) {
        std::cerr << "Failed to get vkGetMemoryFdKHR function" << std::endl;
        vkFreeMemory(device, linearBufferMemory, nullptr);
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    
    if (fpGetMemoryFd(device, &getFdInfo, &bufferMemoryFd) != VK_SUCCESS) {
        std::cerr << "Failed to get file descriptor for buffer memory" << std::endl;
        vkFreeMemory(device, linearBufferMemory, nullptr);
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    bufferMemoryHandle = (HANDLE)(intptr_t)bufferMemoryFd;
#endif
    
    // Import the buffer memory to CUDA
    cudaExternalMemoryHandleDesc cudaExtMemDesc = {};
    
#ifdef _WIN32
    cudaExtMemDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cudaExtMemDesc.handle.win32.handle = bufferMemoryHandle;
#else
    cudaExtMemDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    cudaExtMemDesc.handle.fd = (int)(intptr_t)bufferMemoryHandle;
#endif
    
    cudaExtMemDesc.size = memRequirements.size;
    
    if (cudaImportExternalMemory(&extMemory, &cudaExtMemDesc) != cudaSuccess) {
        std::cerr << "Failed to import buffer memory to CUDA" << std::endl;
        
#ifdef _WIN32
        CloseHandle(bufferMemoryHandle);
#else
        close((int)(intptr_t)bufferMemoryHandle);
#endif
        
        vkFreeMemory(device, linearBufferMemory, nullptr);
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    
    // Map the external memory to a CUDA buffer
    cudaExternalMemoryBufferDesc bufferDesc = {};
    bufferDesc.offset = 0;
    bufferDesc.size = width * height * 4;
    
    if (cudaExternalMemoryGetMappedBuffer(devicePtr, extMemory, &bufferDesc) != cudaSuccess) {
        std::cerr << "Failed to map external memory to CUDA buffer" << std::endl;
        cudaDestroyExternalMemory(extMemory);
        
        vkFreeMemory(device, linearBufferMemory, nullptr);
        vkDestroyBuffer(device, linearBuffer, nullptr);
        return false;
    }
    
    // We can free the Vulkan resources after importing to CUDA
    // Note: This is application-dependent; you might want to keep them around
    vkFreeMemory(device, linearBufferMemory, nullptr);
    vkDestroyBuffer(device, linearBuffer, nullptr);
    
    return true;
}

std::vector<glm::vec3> TextureExtractorVulkan::extractTextureData() {
    std::vector<glm::vec3> result(width * height);
    
    // Allocate CUDA output buffer
    glm::vec3* d_outputBuffer;
    cudaMalloc(&d_outputBuffer, width * height * sizeof(glm::vec3));
    
    cudaArray_t cuArray;
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102);
    cudaMallocArray(&cuArray, &channelDesc, width, height);
    CUDA_CHECK(cudaMemcpyToArray(cuArray, 0, 0, devicePtr, width*height*sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    // Create CUDA surface object from the texture memory
    cudaResourceDesc resDesc = {};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

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
    unpackRGB10A2Kernel<<<blocks, threads>>>(texObj, d_outputBuffer, width, height);
  
    // Check for kernel errors
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_outputBuffer);
        cudaDestroyTextureObject(texObj);
        return result;
    }
    
    // Copy result back to host
    cudaMemcpy(result.data(), d_outputBuffer, width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    
    // Cleanup CUDA resources
    cudaFree(d_outputBuffer);
    
    return result;
}

bool TextureExtractorVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t& memoryTypeIndex) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            memoryTypeIndex = i;
            return true;
        }
    }
    
    return false;
}

void TextureExtractorVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, 
                                         VkMemoryPropertyFlags properties, 
                                         VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer!");
    }
    
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    
    uint32_t memoryTypeIndex;
    if (!findMemoryType(memRequirements.memoryTypeBits, properties, memoryTypeIndex)) {
        throw std::runtime_error("Failed to find suitable memory type!");
    }
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    
    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory!");
    }
    
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void TextureExtractorVulkan::transitionImageLayout(VkImage image, VkFormat format, 
                                                  VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer;
    
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        throw std::invalid_argument("Unsupported layout transition!");
    }
    
    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void TextureExtractorVulkan::copyImageToBuffer(VkImage image, VkBuffer buffer, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer;
    
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    
    vkCmdCopyImageToBuffer(
        commandBuffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        buffer,
        1,
        &region
    );
    
    vkEndCommandBuffer(commandBuffer);
    
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
   vkQueueWaitIdle(graphicsQueue);
   
   vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void TextureExtractorVulkan::cleanup() {
   // Clean up CUDA resources
   if (devicePtr) {
       cudaFree(devicePtr);
       devicePtr = nullptr;
   }
   
   // Clean up Vulkan resources
   if (stagingBuffer != VK_NULL_HANDLE) {
       vkDestroyBuffer(device, stagingBuffer, nullptr);
       stagingBuffer = VK_NULL_HANDLE;
   }
   
   if (stagingBufferMemory != VK_NULL_HANDLE) {
       vkFreeMemory(device, stagingBufferMemory, nullptr);
       stagingBufferMemory = VK_NULL_HANDLE;
   }
   
   if (textureImage != VK_NULL_HANDLE) {
       vkDestroyImage(device, textureImage, nullptr);
       textureImage = VK_NULL_HANDLE;
   }
   
   if (textureMemory != VK_NULL_HANDLE) {
       vkFreeMemory(device, textureMemory, nullptr);
       textureMemory = VK_NULL_HANDLE;
   }
   
   if (commandPool != VK_NULL_HANDLE) {
       vkDestroyCommandPool(device, commandPool, nullptr);
       commandPool = VK_NULL_HANDLE;
   }
   
   if (device != VK_NULL_HANDLE) {
       vkDestroyDevice(device, nullptr);
       device = VK_NULL_HANDLE;
   }
   
   if (instance != VK_NULL_HANDLE) {
       vkDestroyInstance(instance, nullptr);
       instance = VK_NULL_HANDLE;
   }
   
   // Reset handle
#ifdef _WIN32
   if (externalMemoryHandle != nullptr) {
       CloseHandle(externalMemoryHandle);
       externalMemoryHandle = nullptr;
   }
#else
   if (externalMemoryHandle != -1) {
       close(externalMemoryHandle);
       externalMemoryHandle = -1;
   }
#endif
}

#endif // ENABLE_VULKAN