// qwen-image implemented with ncnn library

#include "pipeline.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdio>

#include "image_io.h"
#include "scheduler.h"
#include "text_encoder.h"
#include "tokenizer.h"
#include "transformer.h"
#include "vae.h"

namespace qwenimage {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int kLatentDim = 64;
constexpr int kTextDim = 4096;

double ms_since(const Clock::time_point& begin)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
bool read_f32(const std::string& path, int width, int height, ncnn::Mat& data)
{
    data.create(width, height);
    if (data.empty())
        return false;
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    const size_t count = (size_t)width * height;
    const bool ok = fread(data.data, sizeof(float), count, fp) == count && fgetc(fp) == EOF;
    return fclose(fp) == 0 && ok;
}
bool write_f32(const std::string& path, const ncnn::Mat& data)
{
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp)
        return false;
    bool ok = true;
    const size_t count = (size_t)data.w * data.h * data.d;
    for (int c = 0; c < data.c; c++)
        ok = fwrite(data.channel(c), sizeof(float), count, fp) == count && ok;
    return fclose(fp) == 0 && ok;
}
bool make_tokenizer(const std::string& root, QwenBpeTokenizer& tokenizer)
{
    SpecialTokensConfig spec;
    spec.eos_token = "<|im_end|>";
    spec.pad_token = "<|endoftext|>";
    tokenizer = QwenBpeTokenizer::LoadFromFiles(root + "/processor/vocab.txt", root + "/processor/merges.txt", spec, false, false, true);
    if (tokenizer.vocab_size() == 0) return false;
    const char* specials[] = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
        "<|video_pad|>", "<|vision_pad|>",
    };
    for (const char* token : specials)
        tokenizer.AddAdditionalSpecialToken(token, true);
    return true;
}
}
bool QwenImagePipeline::load(const std::string& model_dir, RuntimeConfig config, std::string* error)
{
    models_.unload_all();
    loaded_ = false;
    config_ = normalize_runtime_config(config);
    model_dir_ = model_dir;
    paths_ = make_model_paths(model_dir);
    if (!validate_model_paths(paths_, error))
        return false;

    // Qwen-Image-2.1 has one dynamic ncnn package.  The runtime graph
    // contract is intentionally kept in code so the model folder contains
    // only the weights and the tokenizer data it actually consumes.
    constexpr int kModelWidth = 1024;
    constexpr int kModelHeight = 1024;
    constexpr int kTextCapacity = 512;
    constexpr int kDropSystemTokens = 14;
    width_ = kModelWidth;
    height_ = kModelHeight;
    static_shape_ = false;
    latent_width_ = width_ / 16;
    latent_height_ = height_ / 16;
    text_tokens_ = kTextCapacity - kDropSystemTokens;
    text_config_.max_input_tokens = kTextCapacity;
    text_config_.dynamic_sequence = true;
    text_config_.multimodal_graph = true;
    drop_system_tokens_ = kDropSystemTokens;
    text_config_.drop_system_tokens = drop_system_tokens_;
    pad_token_id_ = 151643;

    configure_auto_low_vram(config_, width_, height_);
    loaded_ = true;
    return true;
}
bool QwenImagePipeline::generate(const GenerateRequest& request, GenerateTimings* timings)
{
    if (!loaded_ || request.prompt.empty() || request.steps <= 0
        || request.batch <= 0)
        return false;
    const int width = request.width > 0 ? request.width : width_;
    const int height = request.height > 0 ? request.height : height_;
    if (width <= 0 || height <= 0 || width % 16 || height % 16)
    {
        fprintf(stderr, "requested size must be positive multiples of 16: %dx%d\n", width, height);
        return false;
    }
    if (static_shape_ && (width != width_ || height != height_))
    {
        fprintf(stderr, "this static export is for %dx%d, requested %dx%d\n", width_, height_, width, height);
        return false;
    }
    const int latent_width = width / 16;
    const int latent_height = height / 16;
    configure_auto_low_vram(config_, width, height);
    models_.unload_all();

    if (request.batch > 1)
    {
        const std::string first = make_batch_output_path(request.output, 0, request.batch);
        const std::string second = make_batch_output_path(request.output, 1, request.batch);
        const std::string third = make_batch_output_path(request.output, 2, request.batch);
        fprintf(stderr, "batch generation enabled. output-path will be %s %s %s ...\n",
                first.c_str(), second.c_str(), third.c_str());
    }

    QwenBpeTokenizer tokenizer = QwenBpeTokenizer::LoadFromFiles(model_dir_ + "/processor/vocab.txt", model_dir_ + "/processor/merges.txt", SpecialTokensConfig(), false, false, true);
    if (tokenizer.vocab_size() == 0) return false;
    const char* specials[] = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
        "<|video_pad|>", "<|vision_pad|>",
    };
    for (const char* token : specials)
        tokenizer.AddAdditionalSpecialToken(token, true);

    auto format_prompt = [](const std::string& value) {
        return std::string("<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n") + value + "<|im_end|>\n<|im_start|>assistant\n";
    };

    auto encode_prompt = [&](const std::string& value, ncnn::Mat& embeds, int& valid_tokens) -> bool
    {
        const std::string text = format_prompt(value);
        const std::vector<int> actual_ids = tokenizer.encode(text);
        if (actual_ids.empty())
        {
            fprintf(stderr, "prompt tokenization failed\n");
            return false;
        }
        if (!text_config_.dynamic_sequence && actual_ids.size() > (size_t)text_config_.max_input_tokens)
        {
            fprintf(stderr, "prompt is longer than the exported text capacity\n");
            return false;
        }

        ncnn::Mat encoder_ids(text_config_.dynamic_sequence ? (int)actual_ids.size() : text_config_.max_input_tokens, 1);
        if (encoder_ids.empty())
            return false;
        int* ids = encoder_ids;
        std::copy(actual_ids.begin(), actual_ids.end(), ids);
        std::fill(ids + actual_ids.size(), ids + encoder_ids.w, pad_token_id_);

        QwenTextEncoder text_encoder(models_, config_);
        return text_encoder.encode(encoder_ids, (int)actual_ids.size(), text_config_, embeds, valid_tokens);
    };

    const bool has_negative_prompt = request.has_negative_prompt || !request.negative_prompt.empty();
    const bool do_true_cfg = request.guidance_scale > 1.f && has_negative_prompt;
    if (request.guidance_scale > 1.f && !has_negative_prompt)
        fprintf(stderr, "true_cfg_scale requires a negative prompt; CFG disabled\n");
    if (has_negative_prompt && !do_true_cfg)
        fprintf(stderr, "negative prompt ignored because true_cfg_scale <= 1\n");
    if (do_true_cfg)
        fprintf(stderr, "true_cfg_scale = %g\n", request.guidance_scale);

    if (!models_.load_text_encoder(paths_, config_))
    {
        fprintf(stderr, "pipeline text encoder graph load failed\n");
        return false;
    }
    ncnn::Mat text_embeds;
    ncnn::Mat negative_text_embeds;
    int valid_output_tokens = 0;
    int negative_valid_output_tokens = 0;
    const Clock::time_point text_begin = Clock::now();
    bool text_ok = encode_prompt(request.prompt.empty() ? " " : request.prompt, text_embeds, valid_output_tokens);
    if (text_ok && do_true_cfg)
        text_ok = encode_prompt(request.negative_prompt, negative_text_embeds, negative_valid_output_tokens);
    models_.unload_text_encoder();
    if (!text_ok)
    {
        fprintf(stderr, "pipeline text encoding failed\n");
        return false;
    }
    if (timings) timings->text_encoder_ms = ms_since(text_begin);
    if (!request.dump_prefix.empty())
    {
        write_f32(request.dump_prefix + ".text.f32", text_embeds);
        if (do_true_cfg)
            write_f32(request.dump_prefix + ".negative_text.f32",
                      negative_text_embeds);
    }

    const int image_tokens = latent_height * latent_width;
    const int transformer_text_tokens = text_config_.dynamic_sequence
        ? valid_output_tokens : text_tokens_;
    const int negative_transformer_text_tokens = do_true_cfg
        ? (text_config_.dynamic_sequence ? negative_valid_output_tokens
                                         : text_tokens_)
        : transformer_text_tokens;
    if (transformer_text_tokens <= 0
        || ((size_t)text_embeds.w * text_embeds.h) !=
           (size_t)transformer_text_tokens * kTextDim
        || (do_true_cfg
            && (negative_transformer_text_tokens <= 0
                || ((size_t)negative_text_embeds.w * negative_text_embeds.h) !=
                   (size_t)negative_transformer_text_tokens * kTextDim)))
    {
        fprintf(stderr,
                "text encoder output capacity failed: pos_tokens=%d "
                "pos_size=%zu neg_tokens=%d neg_size=%zu\n",
                transformer_text_tokens, ((size_t)text_embeds.w * text_embeds.h),
                negative_transformer_text_tokens, ((size_t)negative_text_embeds.w * negative_text_embeds.h));
        return false;
    }

    ncnn::Mat initial_latents;
    if (!request.rng_mat_path.empty())
    {
        if (!read_f32(request.rng_mat_path, kLatentDim, image_tokens, initial_latents))
        {
            fprintf(stderr, "failed to read rng mat %s expected %zu floats\n",
                    request.rng_mat_path.c_str(),
                    (size_t)image_tokens * kLatentDim);
            return false;
        }
    }

    ncnn::Mat cos;
    ncnn::Mat sin;
    ncnn::Mat mask;
    if (!QwenTransformer::make_rope(transformer_text_tokens, transformer_text_tokens, latent_height, latent_width, cos, sin) || !QwenTransformer::make_attention_mask(transformer_text_tokens, transformer_text_tokens, image_tokens, mask))
        return false;

    ncnn::Mat negative_cos;
    ncnn::Mat negative_sin;
    ncnn::Mat negative_mask;
    if (do_true_cfg)
    {
        if (!QwenTransformer::make_rope(negative_transformer_text_tokens, negative_transformer_text_tokens, latent_height, latent_width, negative_cos, negative_sin) || !QwenTransformer::make_attention_mask(negative_transformer_text_tokens, negative_transformer_text_tokens, image_tokens, negative_mask))
            return false;
    }
    if (!models_.load_transformer(paths_, config_))
    {
        fprintf(stderr, "pipeline transformer graph load failed\n");
        return false;
    }

    const std::vector<float> sigmas = QwenScheduler::make_sigmas(request.steps, image_tokens, false);
    std::vector<ncnn::Mat> batch_latents(request.batch);
    const Clock::time_point transformer_begin = Clock::now();
    bool transformer_ok = true;
    for (int b = 0; b < request.batch && transformer_ok; b++)
    {
        ncnn::Mat latents;
        if (!initial_latents.empty())
        {
            latents = initial_latents.clone();
            if (latents.empty())
                return false;
        }
        else
        {
            latents.create(kLatentDim, image_tokens);
            if (latents.empty())
                return false;
            std::mt19937 gen(static_cast<unsigned int>(request.seed + (uint64_t)b));
            std::normal_distribution<float> dist(0.f, 1.f);
            float* values = latents;
            for (size_t i = 0; i < (size_t)image_tokens * kLatentDim; i++)
                values[i] = dist(gen);
        }

        const std::string dump_prefix = request.dump_prefix.empty()
            ? std::string()
            : request.dump_prefix + (request.batch > 1 ? "-" + std::to_string(b) : "");
        if (!dump_prefix.empty())
            write_f32(dump_prefix + ".rng_initial.f32", latents);

        QwenTransformer positive_transformer(models_, config_, transformer_text_tokens, image_tokens);
        QwenTransformer negative_transformer(models_, config_, negative_transformer_text_tokens, image_tokens);
        for (int z = 0; z < request.steps; z++)
        {
            ncnn::Mat noise;
            if (!positive_transformer.run(latents, text_embeds, sigmas[z], cos, sin, mask, noise))
            {
                fprintf(stderr, "pipeline transformer failed at step %d of image %d\n", z, b);
                transformer_ok = false;
                break;
            }
            if (do_true_cfg)
            {
                ncnn::Mat negative_noise;
                if (!negative_transformer.run(latents, negative_text_embeds, sigmas[z], negative_cos, negative_sin, negative_mask, negative_noise) || negative_noise.w != noise.w || negative_noise.h != noise.h)
                {
                    fprintf(stderr, "pipeline negative transformer failed at step %d of image %d\n", z, b);
                    transformer_ok = false;
                    break;
                }
                for (size_t i = 0; i < (size_t)noise.w * noise.h; i++)
                    noise[i] = negative_noise[i] + request.guidance_scale * (noise[i] - negative_noise[i]);
            }

            const float dt = sigmas[z + 1] - sigmas[z];
            for (size_t i = 0; i < (size_t)latents.w * latents.h; i++)
                latents[i] += dt * noise[i];
            if (!dump_prefix.empty())
            {
                write_f32(dump_prefix + ".noise_step_" + std::to_string(z) + ".f32", noise);
                if (z == 0)
                {
                    write_f32(dump_prefix + ".initial_latent.f32", latents);
                    write_f32(dump_prefix + ".noise.f32", noise);
                }
            }
            if (request.batch > 1)
            {
                fprintf(stderr, "step %d/%d of image %d/%d done\n", z + 1, request.steps, b + 1, request.batch);
            }
            else
            {
                fprintf(stderr, "step %d/%d done\n", z + 1, request.steps);
            }
        }
        if (transformer_ok)
        {
            if (!dump_prefix.empty())
                write_f32(dump_prefix + ".final_latent.f32", latents);
            batch_latents[b] = std::move(latents);
        }
    }
    models_.unload_transformer();
    if (!transformer_ok)
        return false;
    if (timings) timings->transformer_ms = ms_since(transformer_begin);

    if (!models_.load_vae_decoder(paths_, config_))
    {
        fprintf(stderr, "pipeline VAE decoder graph load failed\n");
        return false;
    }
    double vae_decoder_ms = 0.0;
    bool vae_ok = true;
    {
        QwenVaeDecoder decoder(*models_.vae_decoder, config_);
        for (int b = 0; b < request.batch; b++)
        {
            ncnn::Mat image;
            const Clock::time_point decode_begin = Clock::now();
            if (!decoder.decode(batch_latents[b], width, height, image))
            {
                fprintf(stderr, "pipeline VAE decoder failed for image %d\n", b);
                vae_ok = false;
                break;
            }
            vae_decoder_ms += ms_since(decode_begin);

            const std::string dump_prefix = request.dump_prefix.empty()
                ? std::string()
                : request.dump_prefix + (request.batch > 1 ? "-" + std::to_string(b) : "");
            if (!dump_prefix.empty())
                write_f32(dump_prefix + ".image.f32", image);
            if (!request.output.empty())
            {
                const std::string output_path = make_batch_output_path(request.output, b, request.batch);
                if (!save_rgba_float_png(output_path, image))
                {
                    fprintf(stderr, "failed to save output %s\n", output_path.c_str());
                    vae_ok = false;
                    break;
                }
            }
            if (request.batch > 1)
                fprintf(stderr, "vae of image %d/%d done\n", b + 1, request.batch);
            else
                fprintf(stderr, "vae done\n");
        }
    }
    models_.unload_vae_decoder();
    if (!vae_ok)
        return false;
    if (timings) timings->vae_decoder_ms = vae_decoder_ms;
    if (timings) timings->total_ms = ms_since(text_begin);
    return true;
}
}
