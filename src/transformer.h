// qwen-image implemented with ncnn library

#pragma once
#include <vector>
#include "models.h"

namespace qwenimage {
struct TransformerImageShape
{
    int height = 0;
    int width = 0;
};

class QwenTransformer
{
public:
    QwenTransformer(const QwenModelSet& models, const RuntimeConfig& config, int text_tokens, int image_tokens)
        : models_(models), config_(config), text_tokens_(text_tokens), image_tokens_(image_tokens), edit_ready_(false) {}

    static bool make_rope(int text_tokens, int valid_text_tokens, int latent_height, int latent_width, ncnn::Mat& cos, ncnn::Mat& sin);
    static bool make_attention_mask(int text_tokens, int valid_text_tokens, int image_tokens, ncnn::Mat& mask);
    bool run(const ncnn::Mat& latents, const ncnn::Mat& text, float timestep, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, ncnn::Mat& noise) const;

    // prepare the fixed image layout once before the denoising loop
    bool prepare_edit(const ncnn::Mat& condition_latents, const ncnn::Mat& text, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes);
    bool run_edit(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise);

private:
    bool run_input(const ncnn::Mat& latents, const ncnn::Mat& text, float timestep, ncnn::Mat& hidden, ncnn::Mat& modulation, ncnn::Mat& temb, int type = 1) const;
    bool run_blocks(ncnn::Mat hidden, const ncnn::Mat& modulation, const ncnn::Mat& temb, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, ncnn::Mat& noise) const;

#if NCNN_VULKAN
    bool run_input_vulkan(const ncnn::Mat& latents, const ncnn::Mat& text, float timestep, ncnn::VkMat& hidden, ncnn::Mat& modulation, ncnn::Mat& temb) const;
    bool run_blocks_vulkan(ncnn::VkMat hidden, const ncnn::Mat& modulation, const ncnn::Mat& temb, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, ncnn::Mat& noise) const;
#endif

    const QwenModelSet& models_;
    const RuntimeConfig& config_;
    int text_tokens_;
    int image_tokens_;
    bool edit_ready_;
    ncnn::Mat edit_latents_;
    ncnn::Mat edit_text_;
    ncnn::Mat edit_cos_;
    ncnn::Mat edit_sin_;
    ncnn::Mat edit_mask_;
    std::vector<unsigned char> target_mask_;
    std::vector<int> joint_rows_;
};
}
