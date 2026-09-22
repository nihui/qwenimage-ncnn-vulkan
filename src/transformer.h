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
    QwenTransformer(const QwenModelSet& models, const RuntimeConfig& config, int text_tokens, int image_tokens, int latent_height, int latent_width)
        : models_(models), config_(config), text_tokens_(text_tokens),
          image_tokens_(image_tokens), latent_height_(latent_height),
          latent_width_(latent_width) {}

    static void make_rope(int text_tokens, int valid_text_tokens, int latent_height, int latent_width, std::vector<float>& cos, std::vector<float>& sin);
    static void make_attention_mask(int text_tokens, int valid_text_tokens, int image_tokens, std::vector<float>& mask);
    bool run(const std::vector<float>& latents, const std::vector<float>& text, float timestep, const std::vector<float>& cos, const std::vector<float>& sin, const std::vector<float>& mask, std::vector<float>& noise) const;

    // Run the image-conditioned layout used by Qwen-Image 2.1.  The
    // condition shapes precede the target shape; text_image_slots marks the
    // image-pad slots in the post-system-drop text sequence.
    bool run_edit(const std::vector<float>& condition_latents, const std::vector<float>& latents, const std::vector<float>& text, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes, float timestep, std::vector<float>& noise) const;

private:
    const QwenModelSet& models_;
    const RuntimeConfig& config_;
    int text_tokens_;
    int image_tokens_;
    int latent_height_;
    int latent_width_;
};
}

