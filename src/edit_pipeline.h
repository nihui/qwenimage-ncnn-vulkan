// qwen-image implemented with ncnn library

#pragma once

#include <cstdint>
#include <string>

#include "models.h"

namespace qwenimage {
struct EditPreparedImage
{
    int width = 0;
    int height = 0;
    std::vector<float> rgba;
};

struct EditRequest
{
    // File-backed fields are kept for the internal --request compatibility
    // path.  Native --image requests fill the in-memory fields below.
    bool native_inputs = false;
    std::vector<int> input_ids_values;
    std::vector<float> text_cos_values;
    std::vector<float> text_sin_values;
    std::vector<float> text_attention_values;
    std::vector<float> image_mask_values;
    std::vector<float> vision_patch_values;
    std::vector<float> vision_pos_values;
    std::vector<float> vision_cos_values;
    std::vector<float> vision_sin_values;
    std::vector<EditPreparedImage> prepared_images;

    std::string output;
    std::string negative_prompt;
    bool has_negative_prompt = false;
    float guidance_scale = 1.f;
    std::vector<int> negative_input_ids_values;
    std::vector<float> negative_text_cos_values;
    std::vector<float> negative_text_sin_values;
    std::vector<float> negative_text_attention_values;
    std::vector<float> negative_image_mask_values;
    std::string input_ids;
    std::string text_cos;
    std::string text_sin;
    std::string text_attention;
    std::string image_mask;
    std::string vision_patch;
    std::string vision_pos;
    std::string vision_cos;
    std::string vision_sin;
    // The singular vae_rgba/condition_* fields are kept for old request files.
    // New requests use one VAE tensor and size pair per reference image.
    std::string vae_rgba;
    std::vector<std::string> vae_rgba_files;
    std::vector<int> condition_widths;
    std::vector<int> condition_heights;
    int vision_patch_tokens = 0;
    int condition_width = 0;
    int condition_height = 0;
    int width = 0;
    int height = 0;
    int drop_system_tokens = 0;
    int steps = 40;
    uint64_t seed = 42;
};

struct EditTimings
{
    double vision_ms = 0.0;
    double text_encoder_ms = 0.0;
    double vae_encoder_ms = 0.0;
    double transformer_ms = 0.0;
    double vae_decoder_ms = 0.0;
    double total_ms = 0.0;
};

class QwenImageEditPipeline
{
public:
    bool load(const std::string& model_dir, RuntimeConfig config, std::string* error = nullptr);
    bool generate(const EditRequest& request, EditTimings* timings = nullptr);
private:
    std::string model_dir_;
    RuntimeConfig config_;
    ModelPaths paths_;
    QwenModelSet models_;
    bool loaded_ = false;
};
}

