// qwen-image implemented with ncnn library

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <filesystem>
#include <string>
#include <vector>

#if _WIN32
static char* optarg = NULL;
static int optind = 1;
static int getopt(int argc, char* const argv[], const char* optstring)
{
    if (optind >= argc || argv[optind][0] != '-')
        return -1;

    char opt = argv[optind][1];
    const char* p = strchr(optstring, opt);
    if (p == NULL)
        return '?';

    optarg = NULL;

    if (p[1] == ':')
    {
        optind++;
        if (optind >= argc)
            return '?';

        optarg = argv[optind];
    }

    optind++;

    return opt;
}
#else // _WIN32
#include <unistd.h> // getopt()
#endif // _WIN32

#include "gpu.h"
#include "edit_pipeline.h"
#include "edit_input.h"
#include "image_io.h"
#include "pipeline.h"

using namespace qwenimage;

namespace {

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
    const int condition_resolution = 1024;
    GenerateRequest request;
    request.output = "out.png";
    RuntimeConfig config;
    const int gpu_id_auto = 233;
    int gpu_id = gpu_id_auto;

    int opt;
    while ((opt = getopt(argc, argv, "p:n:w:o:i:s:l:r:m:g:b:h")) != -1)
    {
        switch (opt)
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
            if (end == optarg || *end != 0 || !(scale >= 0.f))
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
        case 's':
            if (!parse_image_size(optarg, request.width, request.height))
            {
                fprintf(stderr, "invalid image size: %s (expected width,height)\n", optarg);
                return 2;
            }
            break;
        case 'l':
            request.steps = atoi(optarg);
            break;
        case 'r':
            request.seed = (uint64_t)strtoull(optarg, NULL, 10);
            break;
        case 'm':
            model_dir = optarg;
            break;
        case 'g':
            gpu_id = atoi(optarg);
            if (gpu_id < -1)
            {
                fprintf(stderr, "invalid gpu-id: %s\n", optarg);
                return 2;
            }
            break;
        case 'b':
            request.batch = atoi(optarg);
            if (request.batch <= 0)
            {
                fprintf(stderr, "invalid batch-size: %s\n", optarg);
                return 2;
            }
            break;
        case 'h':
            print_help();
            return 0;
        default:
            print_help();
            return 2;
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
            return 1;
        }
        if (gpu_id == gpu_id_auto)
            gpu_id = ncnn::get_default_gpu_index();
        if (gpu_id < 0)
        {
            fprintf(stderr, "no Vulkan GPU found; use -g -1 for CPU inference\n");
            return 1;
        }
        config.vulkan_device_index = gpu_id;
    }
    config.use_vulkan_compute = use_vulkan;

    if (image_edit)
    {
        const int width = request.width > 0 ? request.width : 1024;
        const int height = request.height > 0 ? request.height : 1024;
        EditRequest edit_request;
        std::string error;
        if (!prepare_native_edit_request(model_dir, image_paths, request.prompt, request.negative_prompt, request.guidance_scale > 1.f && request.has_negative_prompt, condition_resolution, width, height, request.steps, request.seed, request.output, edit_request, &error))
        {
            fprintf(stderr, "image preparation failed: %s\n", error.c_str());
            return 1;
        }
        edit_request.guidance_scale = request.guidance_scale;
        edit_request.batch = request.batch;

        QwenImageEditPipeline pipeline;
        if (!pipeline.load(model_dir, config, &error))
        {
            fprintf(stderr, "load failed: %s\n", error.c_str());
            return 1;
        }
        EditTimings timings;
        if (!pipeline.generate(edit_request, &timings))
        {
            fprintf(stderr, "image editing failed\n");
            return 1;
        }
        print_edit_timings(edit_request, timings);
        return 0;
    }

    QwenImagePipeline pipeline;
    std::string error;
    if (!pipeline.load(model_dir, config, &error))
    {
        fprintf(stderr, "load failed: %s\n", error.c_str());
        return 1;
    }

    GenerateTimings timings;
    if (!pipeline.generate(request, &timings))
    {
        fprintf(stderr, "text-to-image generation failed\n");
        return 1;
    }
    fprintf(stdout, "text encoder: %g ms\n", timings.text_encoder_ms);
    fprintf(stdout, "transformer: %g ms\n", timings.transformer_ms);
    fprintf(stdout, "vae decoder: %g ms\n", timings.vae_decoder_ms);
    fprintf(stdout, "total: %g ms\n", timings.total_ms);
    print_saved_paths(request.output, request.batch);
    return 0;
}
