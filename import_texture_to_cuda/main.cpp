#include <glm/glm.hpp>
#include <vector>
#include <iostream>
#include <common/Utility.hpp>
#include "common/d3d12/TextureExtractorD3D12.h"


int main() {
    try {
        // Hardcoded parameters
        int width = 1510;
        int height = 726;
        std::string resourceName = "MyTestD3DTexture2";
        
        // Set CUDA device
        cudaSetDevice(0);
        
        // Extract texture data
        TextureExtractorD3D12 extractor;
        std::vector<glm::vec3> textureData = extractor.extract(resourceName, width, height);
        
        if (textureData.empty()) {
            std::cerr << "Failed to extract texture data" << std::endl;
            return 1;
        }

        // save the texture data to a file        
        Utility::SavePPM("out/import_texture_to_cuda/" + resourceName + "_d3d12.ppm", textureData, width, height);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}