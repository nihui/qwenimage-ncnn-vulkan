// qwen-image implemented with ncnn library

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "gpu.h"
#include "edit_pipeline.h"
#include "edit_input.h"
#include "image_io.h"
#include "pipeline.h"

using namespace qwenimage;

namespace {

static bool parse_int_option(const char* text, int& value)
{
    if (!text || !*text)
        return false;
    errno = 0;
    char* end = NULL;
    const long parsed = strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != 0 || parsed < INT_MIN || parsed > INT_MAX)
        return false;
    value = (int)parsed;
    return true;
}

static bool parse_seed_option(const char* text, uint64_t& value)
{
    if (!text || !*text || *text == '-')
        return false;
    errno = 0;
    char* end = NULL;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != 0 || parsed > UINT64_MAX)
        return false;
    value = (uint64_t)parsed;
    return true;
}

static bool parse_image_size(const char* text, int& width, int& height)
{
    if (text == NULL)
        return false;

    const char* comma = strchr(text, ',');
    if (comma == NULL || comma == text || comma[1] == 0)
        return false;

    char* end = NULL;
    const long w = strtol(text, &end, 10);
    if (end != comma || w <= 0 || w > 65535)
        return false;

    const long h = strtol(comma + 1, &end, 10);
    if (*end != 0 || h <= 0 || h > 65535)
        return false;

    width = (int)w;
    height = (int)h;
    return true;
}

static void print_help()
{
    fprintf(stdout, "Usage: qwenimage-ncnn-vulkan -p prompt -o outfile [options]...\n\n");
    fprintf(stdout, "  -h                   show this help\n");
    fprintf(stdout, "  -p prompt            prompt\n");
    fprintf(stdout, "  -n negative-prompt   negative prompt (optional)\n");
    fprintf(stdout, "  -w guidance-scale    true CFG scale (default=1.0)\n");
    fprintf(stdout, "  -o output-path       output image path (default=out.png)\n");
    fprintf(stdout, "  -i input-image       reference image for editing (repeat 1 to 10 times)\n");
    fprintf(stdout, "  -s image-size        image resolution (default=1024,1024)\n");
    fprintf(stdout, "  -l steps             denoise steps (default=40)\n");
    fprintf(stdout, "  -r random-seed       random seed (default=42)\n");
    fprintf(stdout, "  -m model-path        qwen-image model path (default=models/qwenimage21)\n");
    fprintf(stdout, "  -g gpu-id            GPU device to use (-1=cpu, default=auto)\n");
    fprintf(stdout, "  -b batch-size        batched generation (default=1)\n");
    fprintf(stdout, "  --lora path          safetensors LoRA or Qwen Fun Acc adapter (optional)\n");
    fprintf(stdout, "  --lora-scale value   LoRA strength (default=1.0)\n");
    fprintf(stdout, "  -c control-image     ControlNet condition image (optional)\n");
    fprintf(stdout, "  --controlnet path    ControlNet ncnn param file (optional)\n");
    fprintf(stdout, "  --control-scale val  ControlNet strength (default=1.0)\n");
}

static void print_saved_paths(const std::string& output, int batch)
{
    if (batch <= 1)
    {
        fprintf(stdout, "saved %s\n", output.c_str());
        return;
    }
    const std::string first = make_batch_output_path(output, 0, batch);
    const std::string last = make_batch_output_path(output, batch - 1, batch);
    fprintf(stdout, "saved %s through %s\n", first.c_str(), last.c_str());
}

static void print_edit_timings(const EditRequest& request, const EditTimings& timings)
{
    fprintf(stdout, "vision encoder: %g ms\n", timings.vision_ms);
    fprintf(stdout, "text encoder: %g ms\n", timings.text_encoder_ms);
    fprintf(stdout, "vae encoder: %g ms\n", timings.vae_encoder_ms);
    fprintf(stdout, "transformer: %g ms\n", timings.transformer_ms);
    fprintf(stdout, "vae decoder: %g ms\n", timings.vae_decoder_ms);
    fprintf(stdout, "total: %g ms\n", timings.total_ms);
    print_saved_paths(request.output, request.batch);
}

} // namespace

int main(int argc, char** argv)
{
    std::string model_dir;
    std::vector<std::string> image_paths;
    GenerateRequest request;
    request.output = "out.png";
    RuntimeConfig config;
    const int gpu_id_auto = 233;
    int gpu_id = gpu_id_auto;

    std::string lora_path;
    float lora_scale = 1.f;
    std::string control_image_path;
    std::string controlnet_path;
    float control_scale = 1.f;
    auto get_value = [&](int& index, const char* option) -> const char* {
        if (index + 1 >= argc)
        {
            fprintf(stderr, "missing value for %s\n", option);
            return NULL;
        }
        return argv[++index];
    };
    for (int i = 1; i < argc; i++)
    {
        const char* arg = argv[i];
        if (strcmp(arg, "--lora") == 0)
        {
            const char* value = get_value(i, arg);
            if (!value) return 2;
            lora_path = value;
            continue;
        }
        if (strcmp(arg, "--lora-scale") == 0)
        {
            const char* value = get_value(i, arg);
            if (!value) return 2;
            char* end = NULL;
            lora_scale = strtof(value, &end);
            if (end == value || *end != 0 || !std::isfinite(lora_scale))
            {
                fprintf(stderr, "invalid LoRA scale: %s\n", value);
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "--controlnet") == 0)
        {
            const char* value = get_value(i, arg);
            if (!value) return 2;
            controlnet_path = value;
            continue;
        }
        if (strcmp(arg, "--control-scale") == 0)
        {
            const char* value = get_value(i, arg);
            if (!value) return 2;
            char* end = NULL;
            control_scale = strtof(value, &end);
            if (end == value || *end != 0 || !std::isfinite(control_scale))
            {
                fprintf(stderr, "invalid ControlNet scale: %s\n", value);
                return 2;
            }
            continue;
        }
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0)
        {
            print_help();
            return 0;
        }
        if (arg[0] != '-' || arg[1] == 0 || arg[1] == '-')
        {
            fprintf(stderr, "unknown option: %s\n", arg);
            print_help();
            return 2;
        }

        const char option[3] = {'-', arg[1], 0};
        const char* optarg = arg[2] ? arg + 2 : NULL;
        if (optarg && optarg[0] == '=') optarg++;
        if (arg[1] != 'h' && !optarg)
        {
            optarg = get_value(i, option);
            if (!optarg) return 2;
        }
        switch (arg[1])
        {
        case 'p':
            request.prompt = optarg;
            break;
        case 'n':
            request.negative_prompt = optarg;
            request.has_negative_prompt = true;
            break;
        case 'w':
        {
            char* end = NULL;
            const float scale = strtof(optarg, &end);
            if (end == optarg || *end != 0 || !std::isfinite(scale) || !(scale >= 0.f))
            {
                fprintf(stderr, "invalid guidance-scale: %s\n", optarg);
                return 2;
            }
            request.guidance_scale = scale;
            break;
        }
        case 'o':
            request.output = optarg;
            break;
        case 'i':
            image_paths.emplace_back(optarg);
            break;
        case 'c':
            control_image_path = optarg;
            break;
        case 's':
            if (!parse_image_size(optarg, request.width, request.height))
            {
                fprintf(stderr, "invalid image size: %s (expected width,height)\n", optarg);
                return 2;
            }
            break;
        case 'l':
            if (!parse_int_option(optarg, request.steps) || request.steps <= 0)
            {
                fprintf(stderr, "invalid steps: %s\n", optarg);
                return 2;
            }
            request.steps_explicit = true;
            break;
        case 'r':
            if (!parse_seed_option(optarg, request.seed))
            {
                fprintf(stderr, "invalid random-seed: %s\n", optarg);
                return 2;
            }
            break;
        case 'm':
            model_dir = optarg;
            break;
        case 'g':
            if (!parse_int_option(optarg, gpu_id) || gpu_id < -1)
            {
                fprintf(stderr, "invalid gpu-id: %s\n", optarg);
                return 2;
            }
            break;
        case 'b':
            if (!parse_int_option(optarg, request.batch) || request.batch <= 0)
            {
                fprintf(stderr, "invalid batch-size: %s\n", optarg);
                return 2;
            }
            break;
        case 'h':
            print_help();
            return 0;
        default:
            fprintf(stderr, "unknown option: %s\n", option);
            print_help();
            return 2;
        }
    }
    request.lora_path = lora_path;
    request.lora_scale = lora_scale;
    request.control_image_path = control_image_path;
    request.controlnet_path = controlnet_path;
    request.control_scale = control_scale;
    if (control_image_path.empty() != controlnet_path.empty())
    {
        fprintf(stderr, "-c control-image and --controlnet must be specified together\n");
        return 2;
    }
    if (!lora_path.empty())
    {
        TransformerLoRA adapter(lora_path, lora_scale);
        if (!adapter.valid())
        {
            fprintf(stderr, "failed to load LoRA %s: %s\n", lora_path.c_str(), adapter.error().c_str());
            return 2;
        }
        if (adapter.has_pdd_output())
        {
            if (request.steps_explicit && request.steps != adapter.required_steps())
            {
                fprintf(stderr, "this PDD LoRA requires exactly %d steps\n", adapter.required_steps());
                return 2;
            }
            request.steps = adapter.required_steps();
        }
    }

    if (model_dir.empty())
    {
        const std::vector<std::string> candidates = {
            "models/qwenimage21",
            "qwenimage21",
        };
        for (const std::string& candidate : candidates)
        {
            if (std::filesystem::is_directory(candidate))
            {
                model_dir = candidate;
                break;
            }
        }
        if (model_dir.empty())
            model_dir = "models/qwenimage21";
    }

    const bool image_edit = !image_paths.empty();
    if (request.prompt.empty())
    {
        fprintf(stderr, "provide -p prompt\n");
        print_help();
        return 2;
    }
    if (image_edit && image_paths.size() > 10)
    {
        fprintf(stderr, "at most 10 input images are supported\n");
        return 2;
    }

    const bool use_vulkan = gpu_id >= 0;
    if (use_vulkan)
    {
        if (ncnn::create_gpu_instance() != 0)
        {
            fprintf(stderr, "failed to create Vulkan instance\n");
            ncnn::destroy_gpu_instance();
            return 1;
        }
        if (gpu_id == gpu_id_auto)
            gpu_id = ncnn::get_default_gpu_index();
        if (gpu_id < 0)
        {
            fprintf(stderr, "no Vulkan GPU found; use -g -1 for CPU inference\n");
            ncnn::destroy_gpu_instance();
            return 1;
        }
        config.vulkan_device_index = gpu_id;
    }
    config.use_vulkan_compute = use_vulkan;

    const int width = request.width > 0 ? request.width : 1024;
    const int height = request.height > 0 ? request.height : 1024;
    // use the longest output side as the reference-image resolution
    const int condition_resolution = std::max(width, height);
    fprintf(stderr, "prompt = %s\n", request.prompt.c_str());
    fprintf(stderr, "negative-prompt = %s\n", request.negative_prompt.c_str());
    fprintf(stderr, "output-path = %s\n", request.output.c_str());
    for (const std::string& image_path : image_paths)
        fprintf(stderr, "input-image = %s\n", image_path.c_str());
    fprintf(stderr, "model = %s\n", model_dir.c_str());
    fprintf(stderr, "image-size = %d x %d\n", width, height);
    fprintf(stderr, "steps = %d\n", request.steps);
    fprintf(stderr, "seed = %llu\n", (unsigned long long)request.seed);
    fprintf(stderr, "gpu-id = %d\n", gpu_id);
    fprintf(stderr, "batch = %d\n", request.batch);
    fprintf(stderr, "guidance-scale = %g\n", request.guidance_scale);
    if (!lora_path.empty()) fprintf(stderr, "lora = %s (scale=%g)\n", lora_path.c_str(), lora_scale);
    if (!control_image_path.empty()) fprintf(stderr, "control-image = %s\n", control_image_path.c_str());
    if (!controlnet_path.empty()) fprintf(stderr, "controlnet = %s (scale=%g)\n", controlnet_path.c_str(), control_scale);

    int ret = 0;
    if (image_edit)
    {
        EditRequest edit_request;
        std::string error;
        if (!prepare_native_edit_request(model_dir, image_paths, request.prompt, request.negative_prompt, request.guidance_scale > 1.f && request.has_negative_prompt, condition_resolution, width, height, request.steps, request.seed, request.output, edit_request, &error))
        {
            fprintf(stderr, "image preparation failed: %s\n", error.c_str());
            ret = 1;
        }
        else
        {
            edit_request.guidance_scale = request.guidance_scale;
            edit_request.steps_explicit = request.steps_explicit;
            edit_request.lora_path = request.lora_path;
            edit_request.lora_scale = request.lora_scale;
            edit_request.control_image_path = request.control_image_path;
            edit_request.controlnet_path = request.controlnet_path;
            edit_request.control_scale = request.control_scale;
            edit_request.batch = request.batch;

            QwenImageEditPipeline pipeline;
            EditTimings timings;
            if (!pipeline.load(model_dir, config, &error))
            {
                fprintf(stderr, "load failed: %s\n", error.c_str());
                ret = 1;
            }
            else if (!pipeline.generate(edit_request, &timings))
            {
                fprintf(stderr, "image editing failed\n");
                ret = 1;
            }
            else
            {
                print_edit_timings(edit_request, timings);
            }
        }
    }
    else
    {
        QwenImagePipeline pipeline;
        std::string error;
        GenerateTimings timings;
        if (!pipeline.load(model_dir, config, &error))
        {
            fprintf(stderr, "load failed: %s\n", error.c_str());
            ret = 1;
        }
        else if (!pipeline.generate(request, &timings))
        {
            fprintf(stderr, "text-to-image generation failed\n");
            ret = 1;
        }
        else
        {
            fprintf(stdout, "text encoder: %g ms\n", timings.text_encoder_ms);
            fprintf(stdout, "transformer: %g ms\n", timings.transformer_ms);
            fprintf(stdout, "vae decoder: %g ms\n", timings.vae_decoder_ms);
            fprintf(stdout, "total: %g ms\n", timings.total_ms);
            print_saved_paths(request.output, request.batch);
        }
    }

    if (use_vulkan)
        ncnn::destroy_gpu_instance();

    return ret;
}
