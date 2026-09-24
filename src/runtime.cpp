// qwen-image implemented with ncnn library

#include "runtime.h"
#include "gpu.h"
#include "dupup3d.h"
#include "avgdown3d.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <climits>
#include <filesystem>
#include <limits>

namespace qwenimage {

RuntimeConfig normalize_runtime_config(RuntimeConfig config)
{
    return config;
}

#if NCNN_VULKAN
bool has_separate_host_heap(const ncnn::VulkanDevice* vkdev)
{
    const VkPhysicalDeviceMemoryProperties& properties = vkdev->info.physicalDeviceMemoryProperties();
    for (uint32_t i = 0; i < properties.memoryTypeCount; i++)
    {
        const VkMemoryType& type = properties.memoryTypes[i];
        if ((type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && !(properties.memoryHeaps[type.heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
            return true;
    }
    return false;
}
#endif

namespace {
bool select_transformer_memory(RuntimeConfig& config, uint64_t budget, bool separate_host_heap, uint64_t weights, uint64_t prefix_cache, uint64_t upload_cache, uint64_t workspace, uint64_t& selected)
{
    config.use_weights_in_host_memory = config.low_vram > 0;
    config.use_kvcache_in_host_memory = false;
    auto required = [&](bool host_weights, bool host_cache) {
        // host allocations on a shared heap still consume the same memory budget
        const uint64_t device_weights = host_weights && separate_host_heap ? 0 : weights;
        const uint64_t cache = host_cache ? upload_cache + (separate_host_heap ? 0 : prefix_cache) : prefix_cache;
        return device_weights + cache + workspace;
    };
    selected = required(config.use_weights_in_host_memory, false);
    if (selected <= budget)
        return true;

    const bool host_weights_fit = config.low_vram != 0 && required(true, false) <= budget;
    const bool host_cache_fit = required(config.use_weights_in_host_memory, true) <= budget;
    if (host_weights_fit && (!host_cache_fit || weights < prefix_cache))
        config.use_weights_in_host_memory = true;
    else
    {
        if (!host_cache_fit && config.low_vram != 0)
            config.use_weights_in_host_memory = true;
        config.use_kvcache_in_host_memory = true;
    }
    selected = required(config.use_weights_in_host_memory, config.use_kvcache_in_host_memory);
    return selected <= budget;
}
}

bool configure_auto_low_vram(RuntimeConfig& config, int width, int height, int prefix_tokens, int negative_prefix_tokens, uint64_t transformer_weights)
{
    config.use_weights_in_host_memory = config.low_vram > 0;
    config.use_kvcache_in_host_memory = false;
    if (width <= 0 || height <= 0 || width % 16 || height % 16 || prefix_tokens <= 0 || negative_prefix_tokens < 0)
        return false;
    const uint64_t target = (uint64_t)(width / 16) * (height / 16);
    const uint64_t prefix = std::max(prefix_tokens, negative_prefix_tokens);
    if (target + prefix > INT_MAX)
    {
        fprintf(stderr, "transformer sequence is too large\n");
        return false;
    }
#if NCNN_VULKAN
    if (!config.use_vulkan_compute || ncnn::get_gpu_count() == 0)
        return true;

    int device = config.vulkan_device_index;
    if (device < 0 || device >= ncnn::get_gpu_count())
        device = ncnn::get_default_gpu_index();
    const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(device);
    const uint64_t mib = 1024 * 1024;
    const uint64_t budget = (uint64_t)vkdev->get_heap_budget() * mib;
    const uint64_t both_prefixes = (uint64_t)prefix_tokens + negative_prefix_tokens;
    const uint64_t branches = negative_prefix_tokens > 0 ? 2 : 1;
    const uint64_t hidden_row = 4096 * 2;
    const uint64_t cache_row = hidden_row * 2;
    const uint64_t prefix_cache = both_prefixes * cache_row * 32;

    // bf16 one-block workspace envelope calibrated against t2i and edit
    // include the larger prefill/decode mask and both cfg scratch cache pools
    const uint64_t activations = std::max(prefix, target) * hidden_row * 24;
    const uint64_t mask = std::max(prefix * prefix, target * (prefix + target)) * 2;
    const uint64_t decode_cache = (both_prefixes + branches * target) * cache_row * 3 / 2;
    const uint64_t workspace = activations + mask + decode_cache + 384 * mib;
    // each cfg branch retains one reusable upload allocation when spilling kv
    const uint64_t upload_cache = both_prefixes * cache_row;
    if (transformer_weights > std::numeric_limits<uint64_t>::max() - workspace - prefix_cache - upload_cache)
        return false;
    const uint64_t estimate = transformer_weights + prefix_cache + workspace;
    uint64_t selected = 0;
    const bool fits = select_transformer_memory(config, budget, has_separate_host_heap(vkdev), transformer_weights, prefix_cache, upload_cache, workspace, selected);
    auto megabytes = [&](uint64_t bytes) { return (unsigned long long)(bytes / mib + (bytes % mib != 0)); };
    fprintf(stderr, "low_vram = %d (heap=%llu MB, estimated transformer=%llu MB, prefix cache=%llu MB)\n", config.use_weights_in_host_memory ? 1 : 0, megabytes(budget), megabytes(estimate), megabytes(prefix_cache));
    fprintf(stderr, "transformer prefix cache = %s (estimated working memory=%llu MB)\n", config.use_kvcache_in_host_memory ? "host" : "gpu", megabytes(selected));
    if (!fits)
    {
        fprintf(stderr, "insufficient Vulkan memory for transformer: estimated %llu MB, budget %llu MB; reduce output size or reference image count/size\n", megabytes(selected), megabytes(budget));
        return false;
    }
#endif
    return true;
}

ncnn::Option make_ncnn_option(const RuntimeConfig& config, ModelStage stage)
{
    ncnn::Option option;
    option.num_threads = config.num_threads;
    // Use Vulkan for every stage when requested.  The pipeline unloads each
    // stage before loading the next one, so the GPU path does not require
    // keeping text, transformer, and VAE graphs resident together.
    option.use_vulkan_compute = config.use_vulkan_compute;
    option.vulkan_device_index = config.vulkan_device_index;
    option.use_fp16_storage = false;
    option.use_fp16_packed = false;
    option.use_fp16_arithmetic = false;
    option.use_int8_inference = false;
    option.use_fp16_uniform = false;
    option.use_int8_uniform = false;
    option.use_bf16_storage = config.use_bf16_storage;
    option.use_bf16_packed = config.use_bf16_packed;
    option.use_packing_layout = config.use_packing_layout;
    option.use_local_pool_allocator = config.use_local_pool_allocator;
    option.use_weights_in_host_memory = config.use_weights_in_host_memory;
    option.use_mapped_model_loading = true;
    option.use_winograd_convolution = config.use_winograd_convolution;
    option.use_cooperative_matrix = true;
    option.use_shader_local_memory = true;
    option.use_subgroup_ops = true;

    if (stage == ModelStage::VaeEncoder || stage == ModelStage::VaeDecoder)
    {
        // Keep host-backed weights in automatic low-VRAM mode.  Winograd is
        // disabled for the VAE because it creates a large temporary workspace.
        if (!config.use_weights_in_host_memory)
            option.use_weights_in_host_memory = false;
        option.use_winograd_convolution = false;
    }

    return option;
}

bool load_net(ncnn::Net& net, const ModelFiles& files, const RuntimeConfig& config, ModelStage stage)
{
    net.opt = make_ncnn_option(config, stage);
#if NCNN_VULKAN
    if (config.use_vulkan_compute && stage != ModelStage::Transformer)
    {
        if (ncnn::get_gpu_count() == 0)
            return false;
        int device = config.vulkan_device_index;
        if (device < 0 || device >= ncnn::get_gpu_count())
            device = ncnn::get_default_gpu_index();
        const ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(device);
        if (!vkdev)
            return false;
        std::error_code error;
        const uint64_t weights = std::filesystem::file_size(files.bin, error);
        if (error)
            return false;
        const uint64_t mib = 1024 * 1024;
        const uint64_t reserve = 384 * mib;
        const uint64_t budget = (uint64_t)vkdev->get_heap_budget() * mib;
        if (config.low_vram < 0 && (weights > budget || reserve > budget - weights))
            net.opt.use_weights_in_host_memory = true;
        const uint64_t resident_weights = net.opt.use_weights_in_host_memory && has_separate_host_heap(vkdev) ? 0 : weights;
        if (resident_weights > budget || reserve > budget - resident_weights)
        {
            fprintf(stderr, "insufficient Vulkan memory to load %s (budget=%llu MB)\n", files.bin.c_str(), (unsigned long long)(budget / mib));
            return false;
        }
    }
#endif
    if (stage == ModelStage::VaeDecoder)
        register_dupup3d_layer(net);
    if (stage == ModelStage::VaeEncoder)
        register_avgdown3d_layer(net);
    net.opt.lightmode = true;
    if (net.load_param(files.param.c_str()) != 0)
    {
        fprintf(stderr, "failed to load ncnn param %s\n", files.param.c_str());
        return false;
    }
    if (net.load_model(files.bin.c_str()) != 0)
    {
        fprintf(stderr, "failed to load ncnn model %s\n", files.bin.c_str());
        return false;
    }
    return true;
}

} // namespace qwenimage
