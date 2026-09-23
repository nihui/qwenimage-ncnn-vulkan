// qwen-image implemented with ncnn library

#pragma once
#include <cstdint>
#include "models.h"

namespace qwenimage {

// decoder tile size in pixels, available memory in bytes
bool get_optimal_vae_tile_size(int width, int height, uint64_t available_memory, int& tile_width, int& tile_height);

class QwenVaeEncoder
{
public:
    QwenVaeEncoder(const ncnn::Net& net, const RuntimeConfig& config)
        : net_(net), config_(config) {}
    bool encode(const ncnn::Mat& rgba, ncnn::Mat& packed, int tile_width = 0, int tile_height = 0) const;
private:
    const ncnn::Net& net_;
    const RuntimeConfig& config_;
};

class QwenVaeDecoder
{
public:
    QwenVaeDecoder(const ncnn::Net& net, const RuntimeConfig& config)
        : net_(net), config_(config) {}
    bool decode(const ncnn::Mat& packed, int width, int height, ncnn::Mat& rgba, int tile_width = 0, int tile_height = 0) const;
private:
    const ncnn::Net& net_;
    const RuntimeConfig& config_;
};
}
