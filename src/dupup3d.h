// qwen-image implemented with ncnn library

#pragma once

#include "layer.h"
#include "net.h"

namespace qwenimage {

class DupUp3D : public ncnn::Layer
{
public:
    DupUp3D();

    int load_param(const ncnn::ParamDict& pd) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const override;

#if NCNN_VULKAN
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const override;
#endif

private:
    int out_channels_;
    int factor_t_;
    int factor_s_;
    int first_chunk_;

#if NCNN_VULKAN
    ncnn::Pipeline* pipeline_;
#endif
};

void register_dupup3d_layer(ncnn::Net& net);

} // namespace qwenimage

