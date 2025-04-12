#include "VulkanHelper.h"
#include <vector>
#include <string>
#include <iostream>
#include <string.h>

void printPhysicalDeviceName(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    
    std::cout << "Physical Device Name: " << deviceProperties.deviceName << std::endl;
    std::cout << "Vendor ID: 0x" << std::hex << deviceProperties.vendorID << std::dec << std::endl;
    std::cout << "Device ID: 0x" << std::hex << deviceProperties.deviceID << std::dec << std::endl;
    std::cout << "API Version: " 
              << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_MINOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_PATCH(deviceProperties.apiVersion) << std::endl;
    std::cout << "Driver Version: 0x" << std::hex << deviceProperties.driverVersion << std::dec << std::endl;
    
    // Print device type
    std::cout << "Device Type: ";
    switch (deviceProperties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            std::cout << "Integrated GPU";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            std::cout << "Discrete GPU";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            std::cout << "Virtual GPU";
            break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            std::cout << "CPU";
            break;
        default:
            std::cout << "Other";
    }
    std::cout << std::endl;
};


// Helper function to translate VkObjectType to string
std::string translateVkObjectTypeToString(VkObjectType type) {
    switch (type) {
        case VK_OBJECT_TYPE_INSTANCE: return "VK_OBJECT_TYPE_INSTANCE";
        case VK_OBJECT_TYPE_PHYSICAL_DEVICE: return "VK_OBJECT_TYPE_PHYSICAL_DEVICE";
        case VK_OBJECT_TYPE_DEVICE: return "VK_OBJECT_TYPE_DEVICE";
        case VK_OBJECT_TYPE_QUEUE: return "VK_OBJECT_TYPE_QUEUE";
        case VK_OBJECT_TYPE_SEMAPHORE: return "VK_OBJECT_TYPE_SEMAPHORE";
        case VK_OBJECT_TYPE_COMMAND_BUFFER: return "VK_OBJECT_TYPE_COMMAND_BUFFER";
        case VK_OBJECT_TYPE_FENCE: return "VK_OBJECT_TYPE_FENCE";
        case VK_OBJECT_TYPE_DEVICE_MEMORY: return "VK_OBJECT_TYPE_DEVICE_MEMORY";
        case VK_OBJECT_TYPE_BUFFER: return "VK_OBJECT_TYPE_BUFFER";
        case VK_OBJECT_TYPE_IMAGE: return "VK_OBJECT_TYPE_IMAGE";
        case VK_OBJECT_TYPE_EVENT: return "VK_OBJECT_TYPE_EVENT";
        case VK_OBJECT_TYPE_QUERY_POOL: return "VK_OBJECT_TYPE_QUERY_POOL";
        case VK_OBJECT_TYPE_BUFFER_VIEW: return "VK_OBJECT_TYPE_BUFFER_VIEW";
        case VK_OBJECT_TYPE_IMAGE_VIEW: return "VK_OBJECT_TYPE_IMAGE_VIEW";
        case VK_OBJECT_TYPE_SHADER_MODULE: return "VK_OBJECT_TYPE_SHADER_MODULE";
        case VK_OBJECT_TYPE_PIPELINE_CACHE: return "VK_OBJECT_TYPE_PIPELINE_CACHE";
        case VK_OBJECT_TYPE_PIPELINE_LAYOUT: return "VK_OBJECT_TYPE_PIPELINE_LAYOUT";
        case VK_OBJECT_TYPE_RENDER_PASS: return "VK_OBJECT_TYPE_RENDER_PASS";
        case VK_OBJECT_TYPE_PIPELINE: return "VK_OBJECT_TYPE_PIPELINE";
        case VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT: return "VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT";
        case VK_OBJECT_TYPE_SAMPLER: return "VK_OBJECT_TYPE_SAMPLER";
        case VK_OBJECT_TYPE_DESCRIPTOR_POOL: return "VK_OBJECT_TYPE_DESCRIPTOR_POOL";
        case VK_OBJECT_TYPE_DESCRIPTOR_SET: return "VK_OBJECT_TYPE_DESCRIPTOR_SET";
        case VK_OBJECT_TYPE_FRAMEBUFFER: return "VK_OBJECT_TYPE_FRAMEBUFFER";
        case VK_OBJECT_TYPE_COMMAND_POOL: return "VK_OBJECT_TYPE_COMMAND_POOL";
        // Add more cases as needed
        default: return "UNKNOWN_OBJECT_TYPE";
    }
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    
    // Choose output stream based on severity
    std::ostream& outputStream = 
        (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) ? std::cerr : std::cout;
    
    // Create prefix based on severity
    std::string prefix;
    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        prefix = "VERBOSE: ";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        prefix = "INFO: ";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        prefix = "WARNING: ";
    } else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        prefix = "ERROR: ";
    }
    
    // Add message type information
    if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) {
        prefix += "[General] ";
    } else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT) {
        prefix += "[Validation] ";
    } else if (messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT) {
        prefix += "[Performance] ";
    }
    
    // Output message with additional information
    outputStream << prefix << "Message ID: " << pCallbackData->messageIdNumber 
                 << " [" << pCallbackData->pMessageIdName << "]\n"
                 << "Message: " << pCallbackData->pMessage << std::endl;
    
    // Print objects involved in the message if any
    if (pCallbackData->objectCount > 0) {
        outputStream << "Objects involved:" << std::endl;
        for (uint32_t i = 0; i < pCallbackData->objectCount; i++) {
            const auto& obj = pCallbackData->pObjects[i];
            outputStream << "  - Type: " << translateVkObjectTypeToString(obj.objectType)
                         << ", Handle: " << obj.objectHandle;
            
            if (obj.pObjectName) {
                outputStream << ", Name: \"" << obj.pObjectName << "\"";
            }
            outputStream << std::endl;
        }
    }
    
    // Return false to indicate the application should not terminate
    return VK_FALSE;
}

VkDeviceSize getActualBufferSize(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize requestedSize, VkBufferUsageFlags usage, VkExternalMemoryHandleTypeFlagBits externalHandleType) {
    // Create a temporary buffer with the requested size
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = requestedSize;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // Add external memory info if needed
    VkExternalMemoryBufferCreateInfo extBufferInfo = {};
    if (externalHandleType != 0) {
        extBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        extBufferInfo.handleTypes = externalHandleType;
        extBufferInfo.pNext = NULL;
        bufferInfo.pNext = &extBufferInfo;
    }

    // Create the temporary buffer
    VkBuffer tempBuffer;
    VkResult result = vkCreateBuffer(device, &bufferInfo, NULL, &tempBuffer);
    if (result != VK_SUCCESS) {
        printf("Failed to create temporary buffer: %d\n", result);
        return 0; // Return 0 to indicate error
    }

    // Query the memory requirements
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, tempBuffer, &memRequirements);
    
    // Clean up the temporary buffer
    vkDestroyBuffer(device, tempBuffer, NULL);
    
    // Return the actual required size
    return memRequirements.size;
}

VkResult vkAllocateMemoryVerbose(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDeviceMemory* pMemory) {
    
    // Print device memory properties first
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    
    printf("------- Vulkan Memory Allocation Diagnostics -------\n");
    printf("Attempting to allocate %llu bytes\n", (unsigned long long)pAllocateInfo->allocationSize);
    printf("Using memory type index: %u\n", pAllocateInfo->memoryTypeIndex);
    
    // Print info about selected memory type
    if (pAllocateInfo->memoryTypeIndex < memProps.memoryTypeCount) {
        uint32_t heapIndex = memProps.memoryTypes[pAllocateInfo->memoryTypeIndex].heapIndex;
        VkMemoryPropertyFlags propFlags = memProps.memoryTypes[pAllocateInfo->memoryTypeIndex].propertyFlags;
        
        printf("Memory type properties: 0x%x\n", propFlags);
        printf("  - Device Local: %s\n", (propFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "Yes" : "No");
        printf("  - Host Visible: %s\n", (propFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "Yes" : "No");
        printf("  - Host Coherent: %s\n", (propFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "Yes" : "No");
        printf("  - Host Cached: %s\n", (propFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "Yes" : "No");
        printf("  - Lazily Allocated: %s\n", (propFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? "Yes" : "No");
        
        printf("Memory heap index: %u\n", heapIndex);
        printf("Memory heap size: %llu bytes\n", (unsigned long long)memProps.memoryHeaps[heapIndex].size);
    }
    
    // Check for external memory info
    const VkExportMemoryAllocateInfo* exportInfo = NULL;
    const VkImportMemoryFdInfoKHR* importInfo = NULL;
    const VkMemoryDedicatedAllocateInfo* dedicatedInfo = NULL;
    
    const void* pNext = pAllocateInfo->pNext;
    while (pNext != NULL) {
        const VkBaseOutStructure* header = (const VkBaseOutStructure*)pNext;
        
        if (header->sType == VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO) {
            exportInfo = (const VkExportMemoryAllocateInfo*)pNext;
            printf("Export memory info found, handle types: 0x%x\n", exportInfo->handleTypes);
        } else if (header->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR) {
            importInfo = (const VkImportMemoryFdInfoKHR*)pNext;
            printf("Import memory info found, handle type: 0x%x, fd: %d\n", 
                   importInfo->handleType, importInfo->fd);
        } else if (header->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO) {
            dedicatedInfo = (const VkMemoryDedicatedAllocateInfo*)pNext;
            printf("Dedicated allocation info found\n");
            printf("  - For buffer: %p\n", (void*)dedicatedInfo->buffer);
            printf("  - For image: %p\n", (void*)dedicatedInfo->image);
        }
        
        pNext = header->pNext;
    }
    
    // Perform actual allocation
    printf("Calling vkAllocateMemory...\n");
    VkResult result = vkAllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    
    // Print result
    printf("vkAllocateMemory result: %d\n", result);
    switch (result) {
        case VK_SUCCESS:
            printf("  - Success!\n");
            break;
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            printf("  - Out of host memory!\n");
            break;
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            printf("  - Out of device memory!\n");
            break;
        case VK_ERROR_TOO_MANY_OBJECTS:
            printf("  - Too many objects!\n");
            break;
        case VK_ERROR_INVALID_EXTERNAL_HANDLE:
            printf("  - Invalid external handle!\n");
            break;
        default:
            printf("  - Unknown error!\n");
            break;
    }
    
    printf("-------------------------------------------\n");
    return result;
}

bool createVkSingleton(const std::string& applicationName, const std::string& engineName, bool enableDebug, VkInstance& instance, VkPhysicalDevice& physicalDevice,  VkDevice& device)
{
    // Create Vulkan instance
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = applicationName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = engineName.c_str();
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;
    
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    std::vector<const char*> instanceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
    };

    // Specify validation layers and debug Utils
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    if (enableDebug)
    {
        // 1. check if validation layer is available
        // Check available layers
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        // Print available layers
        std::cout << "Available Vulkan layers:" << std::endl;
        for (const auto& layer : availableLayers) {
            std::cout << "  " << layer.layerName << std::endl;
        }
        // Verify specifically if validation layer exists
        bool validationLayerFound = false;
        for (const auto& layer : availableLayers) {
            if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                validationLayerFound = true;
                break;
            }
        }
        std::cout << "Validation layer is " 
                << (validationLayerFound ? "available" : "NOT available") << std::endl;
        if (!validationLayerFound)
        {
            return false;
        }
    
        // 2. enable validation layer
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
        // Optionally set up debug messenger
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = 
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = 
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback; // You need to implement this function
        // Chain it to the instance creation
        debugCreateInfo.pNext = nullptr;
        createInfo.pNext = &debugCreateInfo;

        // 3. enable debugUtils instance extension
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        // Check if debug utils extension is available
        bool debugUtilsFound = false;
        for (const auto& extension : extensions) {
            if (strcmp(extension.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0) {
                debugUtilsFound = true;
                break;
            }
        }
        std::cout << "Debug utils extension is " 
                << (debugUtilsFound ? "available" : "NOT available") << std::endl;
        if (!debugUtilsFound)
        {
            return false;
        }
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);      
    }
    
    // Add required extensions for external memory
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();
    
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
    
    // Simply use the first nvidia device
    int deviceIndex = -1;
    for (int i = 0; i < deviceCount; ++i)
    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
        // std::cout << "Device ID: " << i << std::endl;
        // printPhysicalDeviceName(physicalDevice);
        if (strstr(deviceProperties.deviceName, "NVIDIA") != NULL ||
            strstr(deviceProperties.deviceName, "nvidia") != NULL ||
            strstr(deviceProperties.deviceName, "Nvidia") != NULL)
        {            
            deviceIndex = i;
            break;
        }
    }
    if (deviceIndex == -1)
    {
        std::cerr << "Failed to find nvidia gpu" << std::endl;
        return false;
    }
    physicalDevice = devices[deviceIndex];
        
    // Find a graphics queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    
    // check if gpu supports graphics
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

    return true;
}

// Function to find a memory type that supports external memory
uint32_t findMemoryType(
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