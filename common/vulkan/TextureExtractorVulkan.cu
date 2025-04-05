#include "TextureExtractorVulkan.h"

#ifdef ENABLE_VULKAN

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#else
#include <vulkan/vulkan_core.h>
#endif

TextureExtractorVulkan::TextureExtractorVulkan(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes) : TextureExtractor(resName, textureFormat, textureWidth, textureHeight, textureBytes) 
{
    
}

bool TextureExtractorVulkan::initialize() {
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
#ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
#else
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME
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

    return true;
}


// Function to find a memory type that supports external memory
uint32_t FindMemoryType(
    VkPhysicalDevice physicalDevice,
    uint32_t typeFilter,
    VkMemoryPropertyFlags properties) 
{
    // Get memory properties of the physical device
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    // Check each memory type
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        // First check if this memory type satisfies our basic requirements
        if ((typeFilter & (1 << i)) && 
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type for external memory");
}

#ifdef _WIN32
bool OpenWindowsFileAndReadContent(const std::string& ResName, void* dst, size_t size) 
{
    std::wstring wResName(ResName.begin(), ResName.end());

    HANDLE namedMapping = OpenFileMappingW(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        wResName.c_str());
    if (namedMapping == NULL) {
        std::cerr << "Failed to open file: " << ResName << std::endl;
        return false;
    }

    // read handle value from the file
    // Map the file into memory and read the handle value
    LPVOID pView = MapViewOfFile(
        namedMapping,            // Handle
        FILE_MAP_READ,           // Access mode
        0, 0,                    // Offset
        sizeof(HANDLE));         // Size
    
    if (!pView) {
        CloseHandle(namedMapping);
        std::cerr << "Failed to map file: " << ResName << std::endl;
        return false;
    }
    
    // Store the handle value in the mapping
    memcpy(dst, pView, size);
    UnmapViewOfFile(pView);
    CloseHandle(namedMapping);
    
    return true;
}
#endif

VkResult ImportExternalMemory(
    VkDevice VulkanDevice,
    VkPhysicalDevice VulkanPhysicalDevice,
    const std::string& ResName,
    VkDeviceSize MemorySize,
    VkDeviceMemory* OutDeviceMemory,
    void*& externalMemoryHandle)
{
    externalMemoryHandle = nullptr;
    VkResult result = VK_SUCCESS;
    // Allocate memory for the imported resource
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = MemorySize;
    
    // Get memory type index that supports external memory
    uint32_t memoryTypeIndex = FindMemoryType(
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

    // todo: we now use texture name to export/import resource, should consider delete the following code
    if (false)
    {
        // read handle value from the file
        HANDLE remoteHandle;
        if (!OpenWindowsFileAndReadContent(ResName, &remoteHandle, sizeof(HANDLE)) ) {
            std::cerr << "Failed to open shared resource file: " << ResName << std::endl;
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        DWORD exportingProcessId = 67840;
        // if (!OpenWindowsFileAndReadContent(ResName, &exportingProcessId, sizeof(DWORD)) ) {
        //     std::cerr << "Failed to open source process id: " << ResName << std::endl;
        //     return VK_ERROR_INITIALIZATION_FAILED;
        // }

        // Map remote handle to local handle
        // First get the process handle of the exporting process
        HANDLE exportingProcessHandle = OpenProcess(PROCESS_DUP_HANDLE, FALSE, exportingProcessId);
        // Then duplicate the actual Vulkan memory handle
        HANDLE localHandle = NULL;
        DuplicateHandle(
            exportingProcessHandle,    // Source process (handle to Process A)
            remoteHandle,              // The handle in Process A that we want to duplicate
            GetCurrentProcess(),       // Destination process (current process, i.e., Process B)
            &localHandle,              // Output: the new handle in Process B
            0,
            FALSE,
            DUPLICATE_SAME_ACCESS
        );
        if (localHandle == NULL) {
            std::cerr << "Failed to duplicate handle from process: " << ResName << std::endl;
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        externalMemoryHandle = localHandle;
        
        importInfo.handle = externalMemoryHandle;
        importInfo.name = nullptr;
    }

    importInfo.handle = nullptr;
    const auto wResourceName = std::wstring(ResName.begin(), ResName.end());
    importInfo.name = wResourceName.c_str();
    
    // Chain the import info to allocation info
    importInfo.pNext = nullptr;
    allocInfo.pNext = &importInfo;
    
    // Allocate the memory
    result = vkAllocateMemory(VulkanDevice, &allocInfo, nullptr, OutDeviceMemory);
    
    // // Close the handle after allocation
    // CloseHandle(resourceHandle);
    
#else
    // Linux implementation using file descriptors
    
    // Open the resource file
    int fd = open(ResName.c_str(), O_RDWR);
    if (fd == -1) {
        std::cerr << "Failed to open shared resource: " << ResName << std::endl;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    
    // Setup import info for Linux
    VkImportMemoryFdInfoKHR importInfo = {};
    importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR;
    importInfo.fd = fd;
    
    // Chain the import info to allocation info
    allocInfo.pNext = &importInfo;
    
    // Allocate the memory
    result = vkAllocateMemory(VulkanDevice, &allocInfo, nullptr, OutDeviceMemory);
    
    // Note: The fd ownership is transferred to Vulkan in this case
    // Do not close it explicitly after the import
    
#endif

    return result;
}

bool TextureExtractorVulkan::importTextureToCuda() 
{
    initialized = initialize(); // todo: should be called only once for all instances

    if (!initialized)
    {
        std::cerr << "device not initialized yet" << std::endl;
        return false;
    }

    if (ImportExternalMemory(device, physicalDevice, resourceName, totalBytes, &externalMemory, externalMemoryHandle) != VK_SUCCESS)
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
    externalMemoryHandleDesc.handle.fd = externalMemoryHandle;
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
   
   // Clean up Vulkan resources
   if (externalMemory != VK_NULL_HANDLE) {
       vkFreeMemory(device, externalMemory, nullptr);
       externalMemory = VK_NULL_HANDLE;
   }
   if (device != VK_NULL_HANDLE) {
       vkDestroyDevice(device, nullptr);
       device = VK_NULL_HANDLE;
   }
   if (instance != VK_NULL_HANDLE) {
       vkDestroyInstance(instance, nullptr);
       instance = VK_NULL_HANDLE;
   }

    // Clean up external memory handle
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