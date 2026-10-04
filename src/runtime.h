// qwen-image implemented with ncnn library

#pragma once

#include <cstdint>
#include <string>

#include "mat.h"
#include "net.h"
#include "lora.h"

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

// Storage precision of the intermediate blobs and of the resident weights.
enum class StoragePrecision
{
    // bf16 where the device has native bf16 storage, otherwise fp16, and plain
    // fp32 only when the device has neither; see resolve_storage_precision()
    Auto,
    // 16-bit with an 8-bit exponent: the wide activation range of the text
    // encoder and the large DiT residuals cannot overflow it.  The historical
    // default of this project.
    Bf16,
    // 16-bit with a 10-bit mantissa: twice the precision of bf16 for the same
    // footprint.  Only a 5-bit exponent though, so the operators that reduce or
    // exponentiate keep an fp32 accumulator (see the shader patches).
    Fp16,
    // no 16-bit conversion at all; roughly doubles the resident footprint
    Fp32,
};

struct RuntimeConfig
{
    bool use_vulkan_compute = false;
    int vulkan_device_index = 0;
    int num_threads = 8;

    StoragePrecision storage_precision = StoragePrecision::Auto;
    // derived from storage_precision once the device is known; see
    // resolve_storage_precision()
    bool use_fp16_storage = false;
    bool use_fp16_packed = false;
    bool use_fp16_arithmetic = false;
    bool use_bf16_storage = true;
    bool use_bf16_packed = true;
    bool use_packing_layout = true;
    // true when neither fp16 nor bf16 storage is selected, i.e. plain fp32 blobs
    bool use_fp32_storage() const { return !use_fp16_storage && !use_bf16_storage; }
    // bytes-per-element factor relative to the 2-byte bf16/fp16 model files and
    // 16-bit storage.  Plain fp32 keeps 4 bytes, so every resident weight/activation
    // estimate derived from a .bin file size must be scaled by this.
    int storage_bytes_scale() const { return use_fp32_storage() ? 2 : 1; }
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

// "auto", "bf16", "fp16", "fp32" - the canonical lowercase name of a precision
const char* storage_precision_name(StoragePrecision precision);

// parse a --precision argument value; returns false for an unknown name
bool parse_storage_precision(const char* text, StoragePrecision& precision);

// capture the budget before loading any model stage
bool initialize_memory_budget(RuntimeConfig& config);

// resolve the transformer policy once the actual prefix lengths are available
bool configure_auto_low_vram(RuntimeConfig& config, int width, int height, int prefix_tokens, int negative_prefix_tokens, uint64_t transformer_weights, bool use_prefix_cache = true);

#if NCNN_VULKAN
// Resolve StoragePrecision::Auto against the device capabilities and set the
// ncnn storage/arithmetic flags accordingly.  Needs the physical device, so it
// runs after create_gpu_instance(); a null device (cpu inference) keeps the
// plain bf16 defaults.
bool resolve_storage_precision(RuntimeConfig& config, const ncnn::VulkanDevice* vkdev);
bool has_separate_host_heap(const ncnn::VulkanDevice* vkdev);
uint64_t get_gpu_memory_budget(const RuntimeConfig& config, const ncnn::VulkanDevice* vkdev);
#endif

ncnn::Option make_ncnn_option(const RuntimeConfig& config, ModelStage stage);

bool load_net(ncnn::Net& net, const ModelFiles& files, const RuntimeConfig& config, ModelStage stage, TransformerLoRA* lora = nullptr, TransformerPart part = TransformerPart::Input);

} // namespace qwenimage
