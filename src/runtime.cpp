// qwen-image implemented with ncnn library

#include "runtime.h"
#include "gpu.h"
#include "dupup3d.h"
#include "avgdown3d.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>

namespace qwenimage {

RuntimeConfig normalize_runtime_config(RuntimeConfig config)
{
    return config;
}

void configure_auto_low_vram(RuntimeConfig& config, int width, int height)
{
    bool low_vram = config.low_vram > 0;
    uint32_t heap_budget = 0;
    uint32_t estimated_transformer = 0;
    if (config.use_vulkan_compute && ncnn::get_gpu_count() > 0)
    {
        int device = config.vulkan_device_index;
        if (device < 0 || device >= ncnn::get_gpu_count())
            device = ncnn::get_default_gpu_index();
        heap_budget = ncnn::get_gpu_device(device)->get_heap_budget();
        const uint64_t pixels = (uint64_t)width * height;
        const uint32_t megapixels =
            (uint32_t)((pixels + 1024u * 1024u - 1) / (1024u * 1024u));
        estimated_transformer = 13000u + megapixels * 1000u;
        if (config.low_vram < 0)
            low_vram = heap_budget < estimated_transformer;
    }
    else if (config.low_vram < 0)
    {
        low_vram = false;
    }

    // Keep the effective memory policy synchronized for both automatic and
    // explicitly selected modes.  This also makes low_vram=0 really disable
    // host-backed weights after a previous request used low-VRAM mode.
    config.use_weights_in_host_memory = low_vram;

    fprintf(stderr, "low_vram = %d", low_vram ? 1 : 0);
    if (heap_budget != 0)
        fprintf(stderr, " (heap=%u MB, estimated transformer=%u MB)",
                heap_budget, estimated_transformer);
    fprintf(stderr, "\n");
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
