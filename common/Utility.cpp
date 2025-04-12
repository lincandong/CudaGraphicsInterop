#include <cuda.h>
#include <cuda_runtime.h>
#include "Utility.hpp"
#include "CudaPortable.hpp"
#include <iostream>
#include <cstring>

FUNC_QUALIFIER inline float clamp(const float& lo, const float& hi, const float& v) {
    return glm::max(lo, glm::min(hi, v));
}

void Utility::SavePPM(const std::string& path, const std::vector<glm::vec3>& frameBuffer, int width, int height)
{
    // save the final render to file
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open file '" << path << "'. Error: " << std::strerror(errno) << std::endl;

    }

    (void)fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (auto i = 0; i < height * width; ++i) {
        static unsigned char color[3];
        // color[0] = (unsigned char)(255 * glm::pow(clamp(0, 1, frameBuffer[i].x), 0.6f));
        // color[1] = (unsigned char)(255 * glm::pow(clamp(0, 1, frameBuffer[i].y), 0.6f));
        // color[2] = (unsigned char)(255 * glm::pow(clamp(0, 1, frameBuffer[i].z), 0.6f));
        color[0] = (unsigned char)(255 * clamp(0, 1, frameBuffer[i].x));
        color[1] = (unsigned char)(255 * clamp(0, 1, frameBuffer[i].y));
        color[2] = (unsigned char)(255 * clamp(0, 1, frameBuffer[i].z));
        fwrite(color, 1, 3, fp);
    }
    fclose(fp);
}

void Utility::UpdateProgress(float progress)
{
    int barWidth = 70;

    std::cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
};