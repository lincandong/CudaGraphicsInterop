#pragma once
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <wrl/client.h>
#include <memory>
#include <exception>
#include "common/TextureExtractor.h"

using Microsoft::WRL::ComPtr;

class TextureExtractorD3D12 : public TextureExtractor {
public:
    TextureExtractorD3D12() = default;
    ~TextureExtractorD3D12() = default;
    ExtractorAPI GetAPIType() override { return ExtractorAPI::D3D12; };

private:
    // D3D12 objects
    ComPtr<ID3D12Device> d3dDevice;
    HANDLE sharedHandle = nullptr;
    ComPtr<ID3D12Resource> stagingBuffer = nullptr; // not necessarily used

    // CUDA objects
    UINT m_cudaDeviceID;
    UINT m_nodeMask;

    // shared interface    
    bool initialize(std::string& resourceName, int textureWidth, int textureHeight) override;
    bool importTextureToCuda(std::string& resourceName) override;
    std::vector<glm::vec3> extractTextureData() override;
    void cleanup() override;

    // d3d12 helper functions
    bool initD3D12Device();
    bool ConvertD3D12TextureToLinearBuffer(ID3D12Device* device, ID3D12Resource* sharedResource, int width, int height, cudaExternalMemory_t& extMemory, void** devicePtr);
};