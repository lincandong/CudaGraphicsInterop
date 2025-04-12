#include "TextureExtractorD3D12.h"

#ifdef ENABLE_D3D12
#include "d3d12_helper.h"

bool ReadbackD3D12Texture(ID3D12Device* device, ID3D12Resource* sharedResource, int textureWidth, int textureHeight, std::vector<glm::vec3>& result) {
    HRESULT hr;
    
    // Get resource description
    D3D12_RESOURCE_DESC srcDesc = sharedResource->GetDesc();
    
    // Calculate total bytes for the buffer and get footprint information
    UINT64 totalBytes = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT numRows;
    UINT64 rowSizeInBytes;
    device->GetCopyableFootprints(&srcDesc, 0, 1, 0, &footprint, &numRows, &rowSizeInBytes, &totalBytes);
    
    printf("Texture footprint - rows: %u, rowSize: %llu, total: %llu\n", numRows, rowSizeInBytes, totalBytes);
    
    D3D12_RESOURCE_DESC textureDesc = {};
    memset(&textureDesc, 0, sizeof(textureDesc));
    textureDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    textureDesc.Alignment = 0;
    textureDesc.Width = totalBytes;
    textureDesc.Height = 1;
    textureDesc.DepthOrArraySize = 1;
    textureDesc.MipLevels = 1;
    textureDesc.Format = DXGI_FORMAT_UNKNOWN;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    textureDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    // Resource sharing properties (needed for cross-API sharing)
    D3D12_HEAP_PROPERTIES heapProps = {};
    memset(&heapProps, 0, sizeof(heapProps));
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;

    // Create the readback buffer
    ComPtr<ID3D12Resource> readbackBuffer;
    D3D_CHECK(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &textureDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&readbackBuffer)));
    
    // Create command objects for the copy operation
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    D3D_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
    
    ComPtr<ID3D12GraphicsCommandList> commandList;
    D3D_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, 
                                 commandAllocator.Get(), nullptr, 
                                 IID_PPV_ARGS(&commandList)));
    
    // Transition source texture to COPY_SOURCE state
    D3D12_RESOURCE_BARRIER barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    barrier.Transition.pResource = sharedResource;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON; // Adjust if needed
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(1, &barrier);
    
    // Copy the texture to the readback buffer
    D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
    srcLocation.pResource = sharedResource;
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLocation.SubresourceIndex = 0;
    
    D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
    dstLocation.pResource = readbackBuffer.Get();
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLocation.PlacedFootprint = footprint;
    
    commandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);
    
    // Transition texture back to COMMON state
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    commandList->ResourceBarrier(1, &barrier);
    
    // Close command list
    D3D_CHECK(commandList->Close());
    
    // Execute command list
    ComPtr<ID3D12CommandQueue> commandQueue;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    D3D_CHECK(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
    
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(1, ppCommandLists);
    
    // Wait for completion
    ComPtr<ID3D12Fence> fence;
    D3D_CHECK(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    
    D3D_CHECK(commandQueue->Signal(fence.Get(), 1));
    
    if (fence->GetCompletedValue() < 1) {
        HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (event == nullptr) {
            std::cerr << "Failed to create fence event" << std::endl;
            return false;
        }
        
        hr = fence->SetEventOnCompletion(1, event);
        if (FAILED(hr)) {
            std::cerr << "Failed to set fence event, error: 0x" << std::hex << hr << std::dec << std::endl;
            CloseHandle(event);
            return false;
        }
        
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
    }
    
    // Map the readback buffer to get CPU access
    void* pData;
    D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(totalBytes) };
    hr = readbackBuffer->Map(0, &readRange, &pData);
    if (FAILED(hr)) {
        std::cerr << "Failed to map readback buffer, error: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }

    result.resize(textureWidth * textureHeight);
    for (int y = 0; y < textureHeight; ++y)
    {
        for (int x = 0; x < textureWidth; ++x)
        {
            int index = y * textureWidth + x;
            uint32_t* pixel = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(pData) + y * footprint.Footprint.RowPitch + x * sizeof(uint32_t));
            result[index] = glm::vec3(
                ((*pixel >> 0) & 0x3FF) / 1023.0f,
                ((*pixel >> 10) & 0x3FF) / 1023.0f,
                ((*pixel >> 20) & 0x3FF) / 1023.0f
            );
        }
    }

    // Unmap the readback buffer
    D3D12_RANGE emptyRange = { 0, 0 };
    readbackBuffer->Unmap(0, &emptyRange);
    
    return true;
}

bool TextureExtractorD3D12::ConvertD3D12TextureToLinearBuffer(ID3D12Device* device, ID3D12Resource* sharedResource, int width, int height, cudaExternalMemory_t& extMemory, void** devicePtr) {
    HRESULT hr;
    
    // Get resource description
    D3D12_RESOURCE_DESC srcDesc = sharedResource->GetDesc();
    
    // Get the D3D12 resource description
    UINT64 totalBytes = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT numRows;
    UINT64 rowSizeInBytes;
    device->GetCopyableFootprints(&srcDesc, 0, 1, 0, &footprint, &numRows, &rowSizeInBytes, &totalBytes);
    printf("Num Rows: %llu (expected : %llu)\n", numRows, height);
    printf("Row Size: %llu bytes (expected : %llu bytes)\n", rowSizeInBytes, width * sizeof(uint32_t));
    printf("Row pitch: %llu bytes (expected width: %llu bytes)\n", rowSizeInBytes, width * sizeof(uint32_t));
    printf("Total Bytes: %llu bytes (expected width: %llu bytes)\n", totalBytes, width * height * sizeof(uint32_t));
    printf("Creating staging buffer with %llu bytes\n", totalBytes);
    
    D3D12_RESOURCE_DESC bufferDesc = {};
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Alignment = 0;
    bufferDesc.Width = totalBytes;
    bufferDesc.Height = 1;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.SampleDesc.Quality = 0;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE; // Important for CUDA access
    
    // Create a staging buffer for our linear copy
    D3D12_HEAP_PROPERTIES heapProps = {};
    memset(&heapProps, 0, sizeof(heapProps));
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT; // GPU-accessible memory
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN; // No CPU access
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    heapProps.CreationNodeMask = 1;
    heapProps.VisibleNodeMask = 1;
    
    // Create the staging buffer
    ComPtr<ID3D12Resource> stagingBuffer;
    D3D_CHECK(device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_SHARED, // Make it sharable
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST, // Start as copy destination
        nullptr,
        IID_PPV_ARGS(&stagingBuffer)));

    // Create command allocator
    ComPtr<ID3D12CommandAllocator> commandAllocator;
    D3D_CHECK(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
    
    // Create command list
    ComPtr<ID3D12GraphicsCommandList> commandList;
    D3D_CHECK(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, 
                                 commandAllocator.Get(), nullptr, 
                                 IID_PPV_ARGS(&commandList)));
    
    // Transition source texture to COPY_SOURCE
    D3D12_RESOURCE_BARRIER srcBarrier = {};
    srcBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    srcBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    srcBarrier.Transition.pResource = sharedResource;
    srcBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON; // Adjust if needed
    srcBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    srcBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(1, &srcBarrier);
    
    // Copy the texture to our staging buffer
    D3D12_TEXTURE_COPY_LOCATION srcLocation = {};
    srcLocation.pResource = sharedResource;
    srcLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLocation.SubresourceIndex = 0;
    
    D3D12_TEXTURE_COPY_LOCATION dstLocation = {};
    dstLocation.pResource = stagingBuffer.Get();
    dstLocation.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLocation.PlacedFootprint = footprint;
    
    commandList->CopyTextureRegion(&dstLocation, 0, 0, 0, &srcLocation, nullptr);
    
    // Transition staging buffer to COMMON for CUDA access
    D3D12_RESOURCE_BARRIER dstBarrier = {};
    dstBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    dstBarrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
    dstBarrier.Transition.pResource = stagingBuffer.Get();
    dstBarrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    dstBarrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    dstBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    commandList->ResourceBarrier(1, &dstBarrier);
    
    // Close the command list
    hr = commandList->Close();
    if (FAILED(hr)) {
        std::cerr << "Failed to close command list, error: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }
    
    // Create command queue
    ComPtr<ID3D12CommandQueue> commandQueue;
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    queueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.NodeMask = 0;
    
    hr = device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue));
    if (FAILED(hr)) {
        std::cerr << "Failed to create command queue, error: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }
    
    // Execute command list
    ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
    commandQueue->ExecuteCommandLists(1, ppCommandLists);
    
    // Create fence for synchronization
    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        std::cerr << "Failed to create fence, error: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }
    
    // Signal the fence
    hr = commandQueue->Signal(fence.Get(), 1);
    if (FAILED(hr)) {
        std::cerr << "Failed to signal fence, error: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }
    
    // Wait for the fence value
    if (fence->GetCompletedValue() < 1) {
        HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        if (event == nullptr) {
            std::cerr << "Failed to create fence event" << std::endl;
            return false;
        }
        
        hr = fence->SetEventOnCompletion(1, event);
        if (FAILED(hr)) {
            std::cerr << "Failed to set fence event, error: 0x" << std::hex << hr << std::dec << std::endl;
            CloseHandle(event);
            return false;
        }
        
        WaitForSingleObject(event, INFINITE);
        CloseHandle(event);
    }
    
    // Create a shared handle for the buffer
    HANDLE sharedBufferHandle = nullptr;
    hr = device->CreateSharedHandle(
        stagingBuffer.Get(),
        nullptr,
        GENERIC_ALL,
        nullptr,
        &sharedBufferHandle);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create shared handle for staging buffer, error: 0x" << std::hex << hr << std::dec << std::endl;
        return false;
    }
    
    // Import the buffer into CUDA
    cudaExternalMemoryHandleDesc extMemDesc = {};
    memset(&extMemDesc, 0, sizeof(extMemDesc));
    extMemDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    extMemDesc.handle.win32.handle = sharedBufferHandle;
    extMemDesc.size = totalBytes;
    extMemDesc.flags = cudaExternalMemoryDedicated;
    
    // Import the external memory
    cudaError_t cudaStatus = cudaImportExternalMemory(&extMemory, &extMemDesc);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to import external memory: " << cudaGetErrorString(cudaStatus) << std::endl;
        CloseHandle(sharedBufferHandle);
        return false;
    }
    
    // Get a device pointer to the memory
    cudaExternalMemoryBufferDesc bufferMemDesc = {};
    bufferMemDesc.offset = 0;
    bufferMemDesc.size = totalBytes;
    bufferMemDesc.flags = 0;
    
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(devicePtr, extMemory, &bufferMemDesc));
    
    this->sharedHandle = sharedBufferHandle;
    this->stagingBuffer = stagingBuffer; // Keep reference to prevent release
    
    return true;
}

TextureExtractorD3D12::TextureExtractorD3D12(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes) : TextureExtractor(resName, textureFormat, textureWidth, textureHeight, textureBytes) 
{

}

_Use_decl_annotations_ void GetHardwareAdapter(
    IDXGIFactory2* pFactory, IDXGIAdapter1** ppAdapter) {
  ComPtr<IDXGIAdapter1> adapter;
  *ppAdapter = nullptr;

  for (UINT adapterIndex = 0;
       DXGI_ERROR_NOT_FOUND != pFactory->EnumAdapters1(adapterIndex, &adapter);
       ++adapterIndex) {
    DXGI_ADAPTER_DESC1 desc;
    adapter->GetDesc1(&desc);

    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
      // Don't select the Basic Render Driver adapter.
      // If you want a software adapter, pass in "/warp" on the command line.
      continue;
    }

    // Check to see if the adapter supports Direct3D 12, but don't create the
    // actual device yet.
    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                    _uuidof(ID3D12Device), nullptr))) {
      break;
    }
  }

  *ppAdapter = adapter.Detach();
}

bool CheckTextureLayoutSupport(ID3D12Device* pDevice)
{
    // Check if the device is valid
    if (!pDevice) {
        std::cerr << "Error: Device is null" << std::endl;
        return false;
    }

    // Define the formats to check (common formats you might use)
    DXGI_FORMAT formats[] = {
        DXGI_FORMAT_R10G10B10A2_UNORM
    };

    // Query format support for each layout
    bool supportsUnknown = false;
    bool supportsRowMajor = false;
    bool supports64KBSwizzle = false;
    bool supports64KBUnkownSwizzle = false;

    for (auto format : formats) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT formatSupport = { format };
        
        // Query basic format support
        HRESULT hr = pDevice->CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            &formatSupport,
            sizeof(formatSupport));
            
        if (SUCCEEDED(hr)) {
            // Check if format supports texture2D
            if (formatSupport.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) {
                // Create a resource description for a 2D texture
                D3D12_RESOURCE_DESC desc = {};
                desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
                desc.Width = 256;
                desc.Height = 256;
                desc.DepthOrArraySize = 1;
                desc.MipLevels = 1;
                desc.Format = format;
                desc.SampleDesc.Count = 1;
                desc.SampleDesc.Quality = 0;
                desc.Flags = D3D12_RESOURCE_FLAG_NONE;
                
                // Check ROW_MAJOR layout
                desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                
                // Use CreateCommittedResource to see if the layout is supported
                // We just check the return value and don't actually create the resource
                ComPtr<ID3D12Resource> resource;
                D3D12_HEAP_PROPERTIES heapProps = {
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                    D3D12_MEMORY_POOL_UNKNOWN,
                    0, 0
                };
                
                HRESULT hrRowMajor = pDevice->CreateCommittedResource(
                    &heapProps,
                    D3D12_HEAP_FLAG_SHARED,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&resource)
                );
                if (SUCCEEDED(hrRowMajor)) {
                    supportsRowMajor = true;
                    resource.Reset(); // Release the resource
                    std::cout << "D3D12_TEXTURE_LAYOUT_ROW_MAJOR is supported for format " 
                            << format << std::endl;
                }
                
                // Check 64KB_STANDARD_SWIZZLE layout
                desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE;
                HRESULT hrSwizzle = pDevice->CreateCommittedResource(
                    &heapProps,
                    D3D12_HEAP_FLAG_SHARED,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&resource)
                );
                if (SUCCEEDED(hrSwizzle)) {
                    supports64KBSwizzle = true;
                    resource.Reset(); // Release the resource
                    std::cout << "D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE is supported for format " 
                            << format << std::endl;
                }

                // Check 64KB_STANDARD_SWIZZLE layout
                desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
                HRESULT hrUnknown = pDevice->CreateCommittedResource(
                    &heapProps,
                    D3D12_HEAP_FLAG_SHARED,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&resource)
                );
                if (SUCCEEDED(hrUnknown)) {
                    supportsUnknown = true;
                    resource.Reset(); // Release the resource
                    std::cout << "D3D12_TEXTURE_LAYOUT_UNKNOWN is supported for format " 
                            << format << std::endl;
                }

                // Check 64KB_STANDARD_SWIZZLE layout
                desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
                HRESULT hrUnknownSwizzle = pDevice->CreateCommittedResource(
                    &heapProps,
                    D3D12_HEAP_FLAG_SHARED,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    nullptr,
                    IID_PPV_ARGS(&resource)
                );
                if (SUCCEEDED(hrUnknownSwizzle)) {
                    supports64KBUnkownSwizzle = true;
                    resource.Reset(); // Release the resource
                    std::cout << "D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE is supported for format " 
                            << format << std::endl;
                }
            }
        }
    }

    // Check overall support
    if (!supportsUnknown) {
        std::cerr << "D3D12_TEXTURE_LAYOUT_UNKNOWN is not supported" << std::endl;
    }
    if (!supportsRowMajor) {
        std::cerr << "D3D12_TEXTURE_LAYOUT_ROW_MAJOR is not supported" << std::endl;
    }
    if (!supports64KBSwizzle) {
        std::cerr << "D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE is not supported" << std::endl;
    }
    if (!supports64KBUnkownSwizzle) {
        std::cerr << "D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE is not supported" << std::endl;
    }
}

bool TextureExtractorD3D12::initialize() {
    UINT dxgiFactoryFlags = 0;

    // Enable debug layer in debug mode
    ComPtr<ID3D12Debug> debugController;
    D3D_CHECK(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)));
    debugController->EnableDebugLayer();
    // Enable additional debug layers.
    dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    std::cout <<"Enable D3D12 Debug Layer"<<std::endl;

    ComPtr<IDXGIFactory4> factory;
    D3D_CHECK(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));
    const bool g_useWarpDevice = false;
    if (g_useWarpDevice) {
        ComPtr<IDXGIAdapter> warpAdapter;
        D3D_CHECK(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));

        DXGI_ADAPTER_DESC desc;
		warpAdapter->GetDesc(&desc);
        D3D_CHECK(D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&d3dDevice)));
        printf("Using warp adapter: %ws\n", desc.Description);
    } else {
        ComPtr<IDXGIAdapter1> hardwareAdapter;
        GetHardwareAdapter(factory.Get(), &hardwareAdapter);

        D3D_CHECK(D3D12CreateDevice(hardwareAdapter.Get(),
                                        D3D_FEATURE_LEVEL_11_0,
                                        IID_PPV_ARGS(&d3dDevice)));
        DXGI_ADAPTER_DESC1 desc;
        hardwareAdapter->GetDesc1(&desc);
        printf("Using warp adapter: %ws\n", desc.Description);
        // m_dx12deviceluid = desc.AdapterLuid;
    }
    // // Create D3D12 device directly
    // D3D_CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&d3dDevice)));

    CheckTextureLayoutSupport(d3dDevice.Get());

    // Init cuda device properties
    UINT devId = 0;
    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, devId));
    m_cudaDeviceID = devId;
    m_nodeMask = devProp.luidDeviceNodeMask;
    cudaDevProp = devProp;

    return true;
}

struct TextureCopyContext
{
    TextureCopyContext(ID3D12Device* device, ID3D12Resource* sharedResource)
    {
        // Get resource description
        D3D12_RESOURCE_DESC srcDesc = sharedResource->GetDesc();
        
        // Calculate total bytes for the buffer and get footprint information
        device->GetCopyableFootprints(&srcDesc, 0, 1, 0, &footprint, &numRows, &rowSizeInBytes, &totalBytes);
    }
    UINT64 totalBytes = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
    UINT numRows;
    UINT64 rowSizeInBytes;
};

bool TextureExtractorD3D12::importTextureToCuda() 
{
    if (!initialized)
    {
        std::cerr << "device not initialized yet" << std::endl;
        return false;
    }

    const auto wResourceName = std::wstring(resourceName.begin(), resourceName.end());
    D3D_CHECK(d3dDevice->OpenSharedHandleByName(
        wResourceName.c_str(),  // The name you passed when creating the handle
        GENERIC_ALL,            // Access rights
        &sharedHandle));

    // Log the result
    std::cout << "Opening as ID3D12Resource: 0x" << std::hex << sharedHandle << std::dec << std::endl;

    // Open the shared resource
    ComPtr<ID3D12Resource> sharedResource;
    D3D_CHECK(d3dDevice->OpenSharedHandle(sharedHandle, IID_PPV_ARGS(&sharedResource)));

    // // Readback the texture data for reference
    // static std::vector<glm::vec3> m_textureData;
    // m_textureData.clear();
    // ReadbackD3D12Texture(d3dDevice.Get(), sharedResource.Get(), width, height, m_textureData);

    // // Convert the texture to a linear buffer
    // if (!ConvertD3D12TextureToLinearBuffer(d3dDevice.Get(), sharedResource.Get(), width, height, extMemory, &devicePtr))
    // {
    //     return false;
    // }

    // Setup external memory descriptor
    cudaExternalMemoryHandleDesc extMemDesc = {};
    memset(&extMemDesc, 0, sizeof(extMemDesc));
    extMemDesc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    extMemDesc.handle.win32.handle = sharedHandle;
    extMemDesc.size = totalBytes;
    extMemDesc.flags = cudaExternalMemoryDedicated;    
    CUDA_CHECK(cudaImportExternalMemory(&extMemory, &extMemDesc));
    
    // Get direct access to the texture memory
    cudaExternalMemoryBufferDesc bufferDesc = {};
    memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.offset = 0;
    bufferDesc.size = totalBytes;
    bufferDesc.flags = 0;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&devicePtr, extMemory, &bufferDesc));

    // // Get mapped mipmapped array
    // cudaExternalMemoryMipmappedArrayDesc mipmappedArrayDesc = {};
    // mipmappedArrayDesc.offset = 0;
    // mipmappedArrayDesc.formatDesc = cudaCreateChannelDesc(10, 10, 10, 2, cudaChannelFormatKindUnsignedNormalized1010102);
    // mipmappedArrayDesc.extent = make_cudaExtent(width, height, 1);
    // mipmappedArrayDesc.numLevels = 1;
    // mipmappedArrayDesc.flags = 0;
    // CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mipArray, extMemory, &mipmappedArrayDesc));

    return true;
}

void TextureExtractorD3D12::cleanup() {
    std::cout << "Cleaning up TextureExtractorD3D12" << std::endl;

    if (sharedHandle) {
        CloseHandle(sharedHandle);
        sharedHandle = nullptr;
    }

    if (stagingBuffer) {
        stagingBuffer.Reset();
        stagingBuffer = nullptr;
    }

    if (devicePtr) {
        cudaFree(devicePtr);
        devicePtr = nullptr;
    }

    if (cudaResource) {
        cudaFreeArray(cudaResource);
        cudaResource = nullptr;
    }
    
    if (extMemory) {
        cudaDestroyExternalMemory(extMemory);
        extMemory = nullptr;
    }
    
    d3dDevice = nullptr;
}

#endif