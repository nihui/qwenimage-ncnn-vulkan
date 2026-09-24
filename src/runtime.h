// qwen-image implemented with ncnn library

#pragma once

#include <cstdint>
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
    // -1 selects automatically from the generation's vulkan heap budget
    // 0 disables host-backed weights and 1 forces low-vram mode
    int low_vram = -1;
    // capture once per generation; standalone stages may query their own budget
    uint64_t gpu_memory_budget = UINT64_MAX;
    bool use_weights_in_host_memory = false;
    bool use_kvcache_in_host_memory = false;
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

// capture the budget before loading any model stage
bool initialize_memory_budget(RuntimeConfig& config);

// resolve the transformer policy once the actual prefix lengths are available
bool configure_auto_low_vram(RuntimeConfig& config, int width, int height, int prefix_tokens, int negative_prefix_tokens, uint64_t transformer_weights);

#if NCNN_VULKAN
bool has_separate_host_heap(const ncnn::VulkanDevice* vkdev);
uint64_t get_gpu_memory_budget(const RuntimeConfig& config, const ncnn::VulkanDevice* vkdev);
#endif

ncnn::Option make_ncnn_option(const RuntimeConfig& config, ModelStage stage);

bool load_net(ncnn::Net& net, const ModelFiles& files, const RuntimeConfig& config, ModelStage stage);

} // namespace qwenimage
