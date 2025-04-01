#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <common/Utility.hpp>

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
        int width = 1510;
        int height = 726;
        int totalBytes = 4390912;
        TextureFormat format = TextureFormat::RGB10A2;
        std::string resourceName = "MyTestD3DTexture3";
        
        // Extract texture data
#ifdef ENABLE_D3D12
        TextureExtractorD3D12 extractor(resourceName, format, width, height, totalBytes);
        std::string outputFile = resourceName + "_d3d12.ppm";
#elif ENABLE_VULKAN
        TextureExtractorVulkan extractor(resourceName, format, width, height, totalBytes);
        std::string outputFile = resourceName + "_vulkan.ppm";
#endif

        // // initialize the texture extractor
        // if (!extractor.initDevice()) {
        //     std::cerr << "Failed to initialize texture extraction" << std::endl;
        //     return 1;
        // }

        // import graphics buffer to cuda
        if (!extractor.importTextureToCuda()) {
            std::cerr << "Failed to initialize texture extraction" << std::endl;
            return 1;
        }

        // extract texture data from cuda buffer
        std::vector<glm::vec3> textureData = extractor.extractTextureData();

        if (textureData.empty() ) {
            std::cerr << "Failed to extract texture data" << std::endl;
            return 1;
        }

        // save the texture data to a file        
        Utility::SavePPM("out/import_texture_to_cuda/" + outputFile, textureData, width, height);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
#endif