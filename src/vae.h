// qwen-image implemented with ncnn library

#pragma once
#include <vector>
#include "models.h"

namespace qwenimage {
class QwenVaeEncoder
{
public:
    QwenVaeEncoder(const ncnn::Net& net, const RuntimeConfig& config)
        : net_(net), config_(config) {}
    bool encode(const std::vector<float>& rgba, int width, int height, std::vector<float>& packed, int tile_width = 0, int tile_height = 0) const;
private:
    const ncnn::Net& net_;
    const RuntimeConfig& config_;
};

class QwenVaeDecoder
{
public:
    QwenVaeDecoder(const ncnn::Net& net, const RuntimeConfig& config)
        : net_(net), config_(config) {}
    bool decode(const std::vector<float>& packed, int width, int height, std::vector<float>& rgba, int tile_width = 0, int tile_height = 0) const;
private:
    const ncnn::Net& net_;
    const RuntimeConfig& config_;
};
}
