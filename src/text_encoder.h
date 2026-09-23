// qwen-image implemented with ncnn library

#pragma once
#include <vector>

#include "models.h"

namespace qwenimage {
struct TextEncoderConfig
{
    int max_input_tokens = 0;
    int drop_system_tokens = 0;
    bool dynamic_sequence = false;
    bool multimodal_graph = false;
};

class QwenTextEncoder
{
public:
    QwenTextEncoder(const QwenModelSet& models, const RuntimeConfig& config)
        : models_(models), config_(config) {}
    bool encode(const ncnn::Mat& input_ids, int valid_input_tokens, const TextEncoderConfig& text_config, ncnn::Mat& output, int& valid_output_tokens) const;
    bool encode_edit(const ncnn::Mat& input_ids, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& attention_mask, const ncnn::Mat& image_embeds, const ncnn::Mat& image_mask, const std::vector<ncnn::Mat>& deepstack, ncnn::Mat& output) const;
private:
    const QwenModelSet& models_;
    const RuntimeConfig& config_;
};
}
