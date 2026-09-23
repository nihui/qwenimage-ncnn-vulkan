// qwen-image implemented with ncnn library

#pragma once

#include <string>

#include "mat.h"
#include "net.h"

namespace qwenimage {

enum class ModelStage
{
    TokenizerEmbedding,
    VisionPatch,
    VisionEncoder,
    TextEncoder,
    VaeEncoder,
    Transformer,
    VaeDecoder,
};

struct ModelFiles
{
    std::string param;
    std::string bin;
};

struct RuntimeConfig
{
    bool use_vulkan_compute = false;
    int vulkan_device_index = 0;
    int num_threads = 8;

    bool use_fp16_storage = false;
    bool use_fp16_packed = false;
    bool use_fp16_arithmetic = false;
    bool use_bf16_storage = true;
    bool use_bf16_packed = true;
    bool use_packing_layout = true;
    // -1 selects automatically from the currently available Vulkan heap,
    // 0 disables host-backed weights, and 1 forces low-VRAM mode.
    int low_vram = -1;
    bool use_weights_in_host_memory = false;
    bool use_local_pool_allocator = true;
    bool use_winograd_convolution = true;

    // Optional safetensors LoRA applied to the Transformer at load time.
    // The runtime keeps the adapter factorized and executes two extra Gemms
    // per targeted projection, so no dense 3072x3072 delta is materialized.
    std::string transformer_lora_path;
    float transformer_lora_scale = 1.f;

    bool verbose = false;
    bool profile = false;
};

RuntimeConfig normalize_runtime_config(RuntimeConfig config = {});

// Resolve automatic low-VRAM mode for the requested output size.  The
// decision is kept in RuntimeConfig so every lazily loaded stage gets the
// same ncnn memory policy.
void configure_auto_low_vram(RuntimeConfig& config, int width, int height);

ncnn::Option make_ncnn_option(const RuntimeConfig& config, ModelStage stage);

bool load_net(ncnn::Net& net, const ModelFiles& files, const RuntimeConfig& config, ModelStage stage);

} // namespace qwenimage
