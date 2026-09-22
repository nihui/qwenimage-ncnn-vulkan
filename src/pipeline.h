// qwen-image implemented with ncnn library

#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "models.h"
#include "text_encoder.h"

namespace qwenimage {
struct GenerateRequest
{
    std::string prompt;
    std::string negative_prompt;
    bool has_negative_prompt = false;
    float guidance_scale = 1.f;
    std::string output;
    std::string rng_mat_path;
    std::string dump_prefix;
    int width = 0;
    int height = 0;
    int steps = 40;
    uint64_t seed = 42;
};
struct GenerateTimings
{
    double text_encoder_ms = 0.0;
    double transformer_ms = 0.0;
    double vae_decoder_ms = 0.0;
    double total_ms = 0.0;
};
class QwenImagePipeline
{
public:
    bool load(const std::string& model_dir, RuntimeConfig config, std::string* error = nullptr);
    bool generate(const GenerateRequest& request, GenerateTimings* timings = nullptr);
    const RuntimeConfig& config() const { return config_; }
private:
    std::string model_dir_;
    RuntimeConfig config_;
    ModelPaths paths_;
    QwenModelSet models_;
    TextEncoderConfig text_config_;
    int width_ = 0;
    int height_ = 0;
    int latent_height_ = 0;
    int latent_width_ = 0;
    bool static_shape_ = true;
    int text_tokens_ = 0;
    int drop_system_tokens_ = 0;
    int pad_token_id_ = 151643;
    bool loaded_ = false;
};
}
