// qwen-image implemented with ncnn library

#pragma once
#include <memory>
#include <string>

#include "net.h"
#include "runtime.h"

namespace qwenimage {
struct ModelPaths
{
    ModelFiles text_encoder;
    ModelFiles text_encoder_edit;
    ModelFiles vision_encoder;
    ModelFiles vae_encoder;
    ModelFiles vae_decoder;
    ModelFiles transformer_input;
    ModelFiles transformer_blocks;
    ModelFiles transformer_output;
};

ModelPaths make_model_paths(const std::string& model_dir);
bool get_transformer_weight_size(const ModelPaths& paths, uint64_t& bytes, const std::string& controlnet_param = std::string(), const std::string& lora_path = std::string());
bool validate_model_paths(const ModelPaths& paths, std::string* error = nullptr);
bool validate_edit_model_paths(const ModelPaths& paths, std::string* error = nullptr);

struct QwenModelSet
{
    ~QwenModelSet();

    // Keep only the currently active pipeline stage resident.  A Qwen
    // generation uses tens of gigabytes of weights, so keeping every graph
    // alive for the whole request defeats ncnn's allocator reclamation.
    std::unique_ptr<ncnn::Net> text_encoder;
    std::unique_ptr<ncnn::Net> vision_encoder;
    std::unique_ptr<ncnn::Net> vae_encoder;
    std::unique_ptr<ncnn::Net> vae_decoder;
    std::unique_ptr<ncnn::Net> transformer_input;
    std::unique_ptr<ncnn::Net> transformer_blocks;
    std::unique_ptr<ncnn::Net> transformer_output;
    std::unique_ptr<ncnn::Net> transformer_controlnet;
    std::unique_ptr<TransformerLoRA> transformer_lora;

    bool load_text_encoder(const ModelPaths& paths, const RuntimeConfig& config, bool edit_mode = false);
    bool load_vision_encoder(const ModelPaths& paths, const RuntimeConfig& config);
    bool load_vae_encoder(const ModelPaths& paths, const RuntimeConfig& config);
    bool load_vae_decoder(const ModelPaths& paths, const RuntimeConfig& config);
    bool load_transformer(const ModelPaths& paths, const RuntimeConfig& config, const std::string& lora_path = std::string(), float lora_scale = 1.f, const std::string& controlnet_path = std::string(), float control_scale = 1.f);

    void unload_text_encoder();
    void unload_vision_encoder();
    void unload_vae_encoder();
    void unload_vae_decoder();
    void unload_transformer();
    void clear_transformer_workspace() const;
    void unload_all();

#if NCNN_VULKAN
private:
    std::unique_ptr<ncnn::VkBlobAllocator> text_encoder_blob_vkallocator;
    std::unique_ptr<ncnn::VkStagingAllocator> text_encoder_staging_vkallocator;
    std::unique_ptr<ncnn::VkBlobAllocator> vision_encoder_blob_vkallocator;
    std::unique_ptr<ncnn::VkStagingAllocator> vision_encoder_staging_vkallocator;
    std::unique_ptr<ncnn::VkBlobAllocator> transformer_blob_vkallocator;
    std::unique_ptr<ncnn::VkStagingAllocator> transformer_staging_vkallocator;
#endif
};
}
