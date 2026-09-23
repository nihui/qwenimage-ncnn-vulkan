// qwen-image implemented with ncnn library

#pragma once

#include "models.h"

namespace qwenimage {
struct VisionFeatures
{
    ncnn::Mat image;
    ncnn::Mat deep0;
    ncnn::Mat deep1;
    ncnn::Mat deep2;
    int tokens = 0;
};

class QwenVisionEncoder
{
public:
    QwenVisionEncoder(const ncnn::Net& net, const RuntimeConfig& config)
        : net_(net), config_(config) {}

    bool encode(const ncnn::Mat& patch_values, const ncnn::Mat& position_values, const ncnn::Mat& cos, const ncnn::Mat& sin, int patch_tokens, VisionFeatures& output) const;

private:
    const ncnn::Net& net_;
    const RuntimeConfig& config_;
};
}

