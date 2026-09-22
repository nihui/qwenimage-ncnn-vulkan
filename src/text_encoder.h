// qwen-image implemented with ncnn library

#pragma once
#include <string>
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
    bool encode(const std::vector<int>& input_ids, int valid_input_tokens, const TextEncoderConfig& text_config, std::vector<float>& output, int& valid_output_tokens) const;
    bool encode_edit(const std::vector<int>& input_ids, const std::vector<float>& cos, const std::vector<float>& sin, const std::vector<float>& attention_mask, const std::vector<float>& image_embeds, const std::vector<float>& image_mask, const std::vector<std::vector<float>>& deepstack, std::vector<float>& output) const;
private:
    const QwenModelSet& models_;
    const RuntimeConfig& config_;
};
}
