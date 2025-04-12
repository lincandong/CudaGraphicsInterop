#include <vector>
#include <string>
#include <glm/glm.hpp>

class Utility
{
public:
    static void SavePPM(const std::string& path, const std::vector<glm::vec3>& frameBuffer, int width, int height);
    static void UpdateProgress(float progress);
};