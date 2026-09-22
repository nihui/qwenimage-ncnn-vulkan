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
        estimated_transformer = 13000u
            + (uint32_t)((uint64_t)width * height / (1024 * 1024)) * 1000u;
        if (config.low_vram < 0)
            low_vram = heap_budget < estimated_transformer;
    }
    else if (config.low_vram < 0)
    {
        low_vram = false;
    }

    // In automatic mode this assignment is repeated for dynamic resolutions,
    // allowing a later request to use the appropriate policy as well.
    if (config.low_vram < 0 || low_vram)
        config.use_weights_in_host_memory = low_vram;

    fprintf(stderr, "low_vram = %d", low_vram ? 1 : 0);
    if (heap_budget != 0)
        fprintf(stderr, " (heap=%u MB, estimated transformer=%u MB)",
                heap_budget, estimated_transformer);
    fprintf(stderr, "\n");
}


static int align_up_16(int value)
{
    return ((value + 15) / 16) * 16;
}

void get_optimal_vae_tile_size(int width, int height, const RuntimeConfig& config, int& tile_width, int& tile_height)
{
    tile_width = width;
    tile_height = height;
    if (width <= 0 || height <= 0 || width % 16 || height % 16
        || !config.use_vulkan_compute || ncnn::get_gpu_count() <= 0)
        return;

    int device = config.vulkan_device_index;
    if (device < 0 || device >= ncnn::get_gpu_count())
        device = ncnn::get_default_gpu_index();
    const uint32_t heap_budget = ncnn::get_gpu_device(device)->get_heap_budget();
    if (heap_budget == 0)
        return;

    // Match the zimage VAE policy: reserve an activation area proportional
    // to the reported Vulkan heap budget.  Qwen's VAE has a 16x spatial
    // reduction, so all candidate tiles are aligned to 16 pixels.
    uint64_t max_tile_area =
        (uint64_t)heap_budget * 1024u * 1024u / 6000u;
    if (max_tile_area < 16u * 16u)
        max_tile_area = 16u * 16u;
    if ((uint64_t)width * height <= max_tile_area)
        return;

    const double input_ratio = (double)width / (double)height;
    double best_score = -1.0;
    int best_width = 16;
    int best_height = 16;
    const int max_nx = width / 16;
    const int max_ny = height / 16;

    for (int nx = 1; nx <= max_nx; nx++)
    {
        int candidate_width = align_up_16((width + nx - 1) / nx);
        if (candidate_width > width)
            candidate_width = width;
        for (int ny = 1; ny <= max_ny; ny++)
        {
            int candidate_height = align_up_16((height + ny - 1) / ny);
            if (candidate_height > height)
                candidate_height = height;
            if ((uint64_t)candidate_width * candidate_height
                > max_tile_area)
                continue;

            const double tile_ratio =
                (double)candidate_width / (double)candidate_height;
            const double ratio_score =
                tile_ratio < input_ratio
                    ? tile_ratio / input_ratio
                    : input_ratio / tile_ratio;

            const int actual_nx =
                (width + candidate_width - 1) / candidate_width;
            const int actual_ny =
                (height + candidate_height - 1) / candidate_height;
            const int last_width =
                width - (actual_nx - 1) * candidate_width;
            const int last_height =
                height - (actual_ny - 1) * candidate_height;
            const double utilization =
                ((double)std::max(1, last_width) / candidate_width)
                * ((double)std::max(1, last_height) / candidate_height);
            const double area_score =
                (double)((uint64_t)candidate_width * candidate_height)
                / (double)max_tile_area;
            const double score = 0.45 * utilization
                               + 0.35 * ratio_score
                               + 0.20 * area_score;
            if (score > best_score)
            {
                best_score = score;
                best_width = candidate_width;
                best_height = candidate_height;
            }
        }
    }

    tile_width = best_width;
    tile_height = best_height;
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

ncnn::Mat clone_fp32(const ncnn::Mat& source, const RuntimeConfig& config)
{
    ncnn::Option option = make_ncnn_option(config, ModelStage::Transformer);
    ncnn::Mat pack1;
    if (source.elempack == 1)
        pack1 = source.clone();
    else
        ncnn::convert_packing(source, pack1, 1, option);
    if (pack1.empty())
        return ncnn::Mat();
    if (pack1.elembits() == 32)
        return pack1;
    if (pack1.elembits() == 16)
    {
        ncnn::Mat fp32;
        ncnn::cast_bfloat16_to_float32(pack1, fp32, option);
        return fp32;
    }
    return ncnn::Mat();
}

} // namespace qwenimage
