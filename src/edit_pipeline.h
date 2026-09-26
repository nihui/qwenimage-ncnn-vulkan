// qwen-image implemented with ncnn library

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "models.h"

namespace qwenimage {
struct EditPreparedImage
{
    int width = 0;
    int height = 0;
    ncnn::Mat rgba;
};

struct EditRequest
{
    // file-backed fields are kept for legacy requests
    // native image requests fill the in-memory fields below
    bool native_inputs = false;
    ncnn::Mat input_ids_values;
    ncnn::Mat text_cos_values;
    ncnn::Mat text_sin_values;
    ncnn::Mat text_attention_values;
    ncnn::Mat image_mask_values;
    ncnn::Mat vision_patch_values;
    ncnn::Mat vision_pos_values;
    ncnn::Mat vision_cos_values;
    ncnn::Mat vision_sin_values;
    std::vector<EditPreparedImage> prepared_images;

    std::string output;
    std::string negative_prompt;
    bool has_negative_prompt = false;
    float guidance_scale = 1.f;
    ncnn::Mat negative_input_ids_values;
    ncnn::Mat negative_text_cos_values;
    ncnn::Mat negative_text_sin_values;
    ncnn::Mat negative_text_attention_values;
    ncnn::Mat negative_image_mask_values;
    std::string input_ids;
    std::string text_cos;
    std::string text_sin;
    std::string text_attention;
    std::string image_mask;
    std::string vision_patch;
    std::string vision_pos;
    std::string vision_cos;
    std::string vision_sin;
    // the singular vae_rgba/condition_* fields are kept for old request files
    // new requests use one vae tensor and size pair per reference image
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
    bool steps_explicit = false;
    std::string lora_path;
    float lora_scale = 1.f;
    std::string control_image_path;
    std::string controlnet_path;
    float control_scale = 1.f;
    int batch = 1;
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

