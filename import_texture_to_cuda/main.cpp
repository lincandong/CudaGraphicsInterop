#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <common/Utility.hpp>

#include <chrono>
#include <thread>

#ifdef ENABLE_D3D12
#include "common/d3d12/TextureExtractorD3D12.h"
#endif

#ifdef ENABLE_VULKAN
#include "common/vulkan/TextureExtractorVulkan.h"
#endif

#if ENABLE_D3D12 || ENABLE_VULKAN
int main() {
    try {
        // Hardcoded parameters
        int width = 1227;
        int height = 642;
        int totalBytes = 12648448;
        TextureFormat format = TextureFormat::RGBA32;
        std::string resourceName = "MyTestD3DTexture3";
        
        // Extract texture data
#ifdef ENABLE_D3D12
        TextureExtractorD3D12 extractor(resourceName, format, width, height, totalBytes);
        std::string outputFile = resourceName + "_d3d12.ppm";
#elif ENABLE_VULKAN
        TextureExtractorVulkan extractor(resourceName, format, width, height, totalBytes);
        std::string outputFile = resourceName + "_vulkan.ppm";
#endif

        // import graphics buffer to cuda
        if (!extractor.importTextureToCuda()) {
            std::cerr << "Failed to initialize texture extraction" << std::endl;
            return 1;
        }

        // test if texture could be auto updated
        bool loop = true;
        do
        {
            std::cout << "Extracting texture data..." << std::endl;
            std::vector<glm::vec3> textureData = extractor.extractTextureData();
            if (textureData.empty() ) {
                std::cerr << "Failed to extract texture data" << std::endl;
                return 1;
            }
            // save the texture data to a file        
            Utility::SavePPM("out/import_texture_to_cuda/" + outputFile, textureData, width, height);
            std::cout << "success, texture is saved to : " << outputFile << std::endl;

            if (loop)
            {
                std::cout << "Sleeping for 5 seconds to loop..." << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }
        }
        while(loop);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
#endif