// qwen-image implemented with ncnn library

#pragma once

#include <vector>

#include "models.h"

namespace qwenimage {
struct VisionFeatures
{
    std::vector<float> image;
    std::vector<float> deep0;
    std::vector<float> deep1;
    std::vector<float> deep2;
    int tokens = 0;
};

class QwenVisionEncoder
{
public:
    QwenVisionEncoder(const ncnn::Net& net, const RuntimeConfig& config)
        : net_(net), config_(config) {}

    bool encode(const std::vector<float>& patch_values, const std::vector<float>& position_values, const std::vector<float>& cos, const std::vector<float>& sin, int patch_tokens, VisionFeatures& output) const;

private:
    const ncnn::Net& net_;
    const RuntimeConfig& config_;
};
}

