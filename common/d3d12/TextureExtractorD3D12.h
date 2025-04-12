#pragma once

#ifdef ENABLE_D3D12

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <wrl/client.h>
#include <memory>
#include <exception>
#include "common/TextureExtractor.h"

using Microsoft::WRL::ComPtr;

class TextureExtractorD3D12 : public TextureExtractor {
public:
    TextureExtractorD3D12(std::string& resName, TextureFormat textureFormat, int textureWidth, int textureHeight, int textureBytes);
    ~TextureExtractorD3D12(){ cleanup(); };
    ExtractorAPI GetAPIType() override { return ExtractorAPI::D3D12; };

    bool importTextureToCuda() override;
private:
    // instance data
    HANDLE sharedHandle = nullptr;
    ComPtr<ID3D12Resource> stagingBuffer = nullptr; // not necessarily used

    // singleton data
    ComPtr<ID3D12Device> d3dDevice;
    UINT m_cudaDeviceID;
    UINT m_nodeMask;
    bool initialize() override;
    void cleanDevice();

    // shared interface    
    void cleanup() override;

    // d3d12 helper functions
    bool ConvertD3D12TextureToLinearBuffer(ID3D12Device* device, ID3D12Resource* sharedResource, int width, int height, cudaExternalMemory_t& extMemory, void** devicePtr);
};
#endif