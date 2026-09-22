// qwen-image implemented with ncnn library

#include "edit_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <random>
#include <cstdio>
#include <iterator>
#include <vector>

#include "image_io.h"
#include "scheduler.h"
#include "text_encoder.h"
#include "transformer.h"
#include "vae.h"
#include "vision.h"

namespace qwenimage {
namespace {
using Clock = std::chrono::steady_clock;
constexpr int kHidden = 4096;
constexpr int kLatent = 64;
constexpr int kTextRope = 128;


double elapsed_ms(const Clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}


bool read_bytes(const std::string& path, std::vector<char>& data)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size < 0) return false;
    stream.seekg(0, std::ios::beg);
    data.resize((size_t)size);
    stream.read(data.data(), size);
    return stream.good() || stream.eof();
}

bool read_f32_all(const std::string& path, std::vector<float>& data)
{
    std::vector<char> bytes;
    if (!read_bytes(path, bytes) || bytes.size() % sizeof(float))
        return false;
    data.resize(bytes.size() / sizeof(float));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return true;
}

bool read_i32_all(const std::string& path, std::vector<int>& data)
{
    std::vector<char> bytes;
    if (!read_bytes(path, bytes) || bytes.size() % sizeof(int))
        return false;
    data.resize(bytes.size() / sizeof(int));
    std::memcpy(data.data(), bytes.data(), bytes.size());
    return true;
}

void drop_rows(const std::vector<float>& source, int rows, int drop, int dim, std::vector<float>& target)
{
    const int kept = rows - drop;
    target.resize((size_t)kept * dim);
    std::memcpy(target.data(),
                source.data() + (size_t)drop * dim,
                target.size() * sizeof(float));
}

bool make_multimodal_feature_rows(const VisionFeatures& vision, const std::vector<float>& full_mask, int tokens, std::vector<float>& image_embeds, std::vector<std::vector<float>>& deepstack, std::vector<unsigned char>& slot_mask)
{
    if (full_mask.size() != (size_t)tokens
        || vision.tokens <= 0
        || vision.image.size() != (size_t)vision.tokens * kHidden
        || vision.deep0.size() != vision.image.size()
        || vision.deep1.size() != vision.image.size()
        || vision.deep2.size() != vision.image.size())
        return false;
    image_embeds.assign((size_t)tokens * kHidden, 0.f);
    deepstack.assign(3, std::vector<float>((size_t)tokens * kHidden, 0.f));
    slot_mask.assign(tokens, 0);
    int row = 0;
    for (int i = 0; i < tokens; i++)
        if (full_mask[i] > 0.5f)
        {
            if (row >= vision.tokens)
                return false;
            slot_mask[i] = 1;
            std::memcpy(image_embeds.data() + (size_t)i * kHidden,
                        vision.image.data() + (size_t)row * kHidden,
                        (size_t)kHidden * sizeof(float));
            for (int level = 0; level < 3; level++)
            {
                const std::vector<float>& source =
                    level == 0 ? vision.deep0
                    : level == 1 ? vision.deep1 : vision.deep2;
                std::memcpy(deepstack[level].data() + (size_t)i * kHidden,
                            source.data() + (size_t)row * kHidden,
                            (size_t)kHidden * sizeof(float));
            }
            row++;
        }
    return row == vision.tokens;
}
}

bool QwenImageEditPipeline::load(const std::string& model_dir, RuntimeConfig config, std::string* error)
{
    models_.unload_all();
    loaded_ = false;
    model_dir_ = model_dir;
    config_ = normalize_runtime_config(config);
    paths_ = make_model_paths(model_dir);
    if (!validate_edit_model_paths(paths_, error))
        return false;
    loaded_ = true;
    return true;
}

bool QwenImageEditPipeline::generate(const EditRequest& request, EditTimings* timings)
{
    if (!loaded_ || request.output.empty() || request.steps <= 0
        || request.batch <= 0 || request.width <= 0 || request.height <= 0
        || request.width % 32 || request.height % 32
        || request.drop_system_tokens < 0)
        return false;

    configure_auto_low_vram(config_, request.width, request.height);
    models_.unload_all();

    if (request.batch > 1)
    {
        const std::string first = make_batch_output_path(request.output, 0, request.batch);
        const std::string second = make_batch_output_path(request.output, 1, request.batch);
        const std::string third = make_batch_output_path(request.output, 2, request.batch);
        fprintf(stderr, "batch generation enabled. output-path will be %s %s %s ...\n",
                first.c_str(), second.c_str(), third.c_str());
    }

    // Qwen-Image-2.1 accepts one flat set of up to ten condition images.  Keep
    // the old singular request fields as a compatibility fallback.
    const bool native_inputs = request.native_inputs;
    const bool has_negative_prompt = request.has_negative_prompt || !request.negative_prompt.empty();
    const bool do_true_cfg = request.guidance_scale > 1.f && has_negative_prompt;
    if (do_true_cfg && !native_inputs)
    {
        fprintf(stderr, "negative prompt requires native image input preparation\n");
        return false;
    }
    if (request.guidance_scale > 1.f && !has_negative_prompt)
        fprintf(stderr, "true_cfg_scale requires a negative prompt; CFG disabled\n");
    if (has_negative_prompt && !do_true_cfg)
        fprintf(stderr, "negative prompt ignored because true_cfg_scale <= 1\n");
    if (do_true_cfg)
        fprintf(stderr, "true_cfg_scale = %g\n", request.guidance_scale);

    std::vector<std::string> vae_files = request.vae_rgba_files;
    if (vae_files.empty() && !request.vae_rgba.empty())
        vae_files.push_back(request.vae_rgba);
    std::vector<int> condition_widths = request.condition_widths;
    std::vector<int> condition_heights = request.condition_heights;
    if (condition_widths.empty() && request.condition_width > 0)
        condition_widths.push_back(request.condition_width);
    if (condition_heights.empty() && request.condition_height > 0)
        condition_heights.push_back(request.condition_height);
    const size_t image_count = native_inputs
        ? request.prepared_images.size() : vae_files.size();
    if (image_count == 0 || image_count > 10
        || condition_widths.size() != image_count
        || condition_heights.size() != image_count)
        return false;
    if (native_inputs
        && (request.input_ids_values.empty()
            || request.text_cos_values.empty()
            || request.text_sin_values.empty()
            || request.text_attention_values.empty()
            || request.image_mask_values.empty()
            || request.vision_patch_values.empty()
            || request.vision_pos_values.empty()
            || request.vision_cos_values.empty()
            || request.vision_sin_values.empty()))
        return false;
    for (size_t i = 0; i < image_count; i++)
        if (condition_widths[i] <= 0 || condition_heights[i] <= 0
            || condition_widths[i] % 16 || condition_heights[i] % 16)
            return false;

    std::vector<int> ids;
    std::vector<int> negative_ids;
    std::vector<float> text_cos, text_sin, text_attention, full_image_mask;
    std::vector<float> negative_text_cos, negative_text_sin;
    std::vector<float> negative_text_attention, negative_image_mask;
    std::vector<float> patch, position, vision_cos, vision_sin;
    std::vector<std::vector<float>> rgba_images(image_count);
    if (native_inputs)
    {
        ids = request.input_ids_values;
        text_cos = request.text_cos_values;
        text_sin = request.text_sin_values;
        text_attention = request.text_attention_values;
        full_image_mask = request.image_mask_values;
        patch = request.vision_patch_values;
        position = request.vision_pos_values;
        vision_cos = request.vision_cos_values;
        vision_sin = request.vision_sin_values;
        if (do_true_cfg)
        {
            negative_ids = request.negative_input_ids_values;
            negative_text_cos = request.negative_text_cos_values;
            negative_text_sin = request.negative_text_sin_values;
            negative_text_attention = request.negative_text_attention_values;
            negative_image_mask = request.negative_image_mask_values;
        }
        for (size_t i = 0; i < image_count; i++)
        {
            rgba_images[i] = request.prepared_images[i].rgba;
            if (rgba_images[i].size() != (size_t)4 * condition_widths[i] * condition_heights[i])
                return false;
        }
    }
    else
    {
        if (!read_i32_all(request.input_ids, ids)
            || !read_f32_all(request.text_cos, text_cos)
            || !read_f32_all(request.text_sin, text_sin)
            || !read_f32_all(request.text_attention, text_attention)
            || !read_f32_all(request.image_mask, full_image_mask)
            || !read_f32_all(request.vision_patch, patch)
            || !read_f32_all(request.vision_pos, position)
            || !read_f32_all(request.vision_cos, vision_cos)
            || !read_f32_all(request.vision_sin, vision_sin))
            return false;
        for (size_t i = 0; i < image_count; i++)
            if (!read_f32_all(vae_files[i], rgba_images[i])
                || rgba_images[i].size() != (size_t)4 * condition_widths[i]
                                          * condition_heights[i])
                return false;
    }

    const int full_tokens = (int)ids.size();
    const int negative_full_tokens = (int)negative_ids.size();
    if (full_tokens <= request.drop_system_tokens
        || text_cos.size() != (size_t)full_tokens * kTextRope
        || text_sin.size() != text_cos.size()
        || text_attention.size() != (size_t)full_tokens * full_tokens
        || full_image_mask.size() != (size_t)full_tokens
        || request.vision_patch_tokens <= 0
        || patch.size() != (size_t)request.vision_patch_tokens * 1536
        || position.size() != (size_t)request.vision_patch_tokens * 1152
        || vision_cos.size() != (size_t)request.vision_patch_tokens * 72
        || vision_sin.size() != vision_cos.size())
        return false;
    if (do_true_cfg
        && (negative_full_tokens <= request.drop_system_tokens
            || negative_text_cos.size() != (size_t)negative_full_tokens * kTextRope
            || negative_text_sin.size() != negative_text_cos.size()
            || negative_text_attention.size() != (size_t)negative_full_tokens * negative_full_tokens
            || negative_image_mask.size() != (size_t)negative_full_tokens))
        return false;

    if (!models_.load_vision_encoder(paths_, config_))
    {
        fprintf(stderr, "edit vision encoder graph load failed\n");
        return false;
    }
    VisionFeatures vision_features;
    const Clock::time_point vision_begin = Clock::now();
    bool vision_ok = false;
    {
        QwenVisionEncoder vision(*models_.vision_encoder, config_);
        vision_ok = vision.encode(patch, position, vision_cos, vision_sin, request.vision_patch_tokens, vision_features);
    }
    models_.unload_vision_encoder();
    if (!vision_ok)
    {
        fprintf(stderr, "edit vision encoder failed\n");
        return false;
    }
    if (timings) timings->vision_ms = elapsed_ms(vision_begin);

    std::vector<float> image_embeds;
    std::vector<std::vector<float>> deepstack;
    std::vector<unsigned char> full_slots;
    if (!make_multimodal_feature_rows(vision_features, full_image_mask, full_tokens, image_embeds, deepstack, full_slots))
        return false;

    std::vector<float> negative_image_embeds;
    std::vector<std::vector<float>> negative_deepstack;
    std::vector<unsigned char> negative_full_slots;
    if (do_true_cfg
        && !make_multimodal_feature_rows(vision_features, negative_image_mask, negative_full_tokens, negative_image_embeds, negative_deepstack, negative_full_slots))
        return false;

    if (!models_.load_text_encoder(paths_, config_, true))
    {
        fprintf(stderr, "edit text encoder graph load failed\n");
        return false;
    }
    std::vector<float> full_text;
    std::vector<float> negative_full_text;
    const Clock::time_point text_begin = Clock::now();
    bool text_ok = false;
    {
        QwenTextEncoder text_encoder(models_, config_);
        text_ok = text_encoder.encode_edit(ids, text_cos, text_sin, text_attention, image_embeds, full_image_mask, deepstack, full_text);
        if (text_ok && do_true_cfg)
            text_ok = text_encoder.encode_edit(negative_ids, negative_text_cos, negative_text_sin, negative_text_attention, negative_image_embeds, negative_image_mask, negative_deepstack, negative_full_text);
    }
    models_.unload_text_encoder();
    if (!text_ok)
    {
        fprintf(stderr, "edit text encoder failed\n");
        return false;
    }
    if (timings) timings->text_encoder_ms = elapsed_ms(text_begin);

    const int drop = request.drop_system_tokens;
    const int text_tokens = full_tokens - drop;
    std::vector<float> text;
    drop_rows(full_text, full_tokens, drop, kHidden, text);
    std::vector<unsigned char> text_slots(text_tokens);
    for (int i = 0; i < text_tokens; i++)
        text_slots[i] = full_slots[drop + i];

    const int negative_text_tokens = do_true_cfg
        ? negative_full_tokens - drop : text_tokens;
    std::vector<float> negative_text;
    std::vector<unsigned char> negative_text_slots;
    if (do_true_cfg)
    {
        drop_rows(negative_full_text, negative_full_tokens, drop, kHidden, negative_text);
        negative_text_slots.resize(negative_text_tokens);
        for (int i = 0; i < negative_text_tokens; i++)
            negative_text_slots[i] = negative_full_slots[drop + i];
    }

    if (!models_.load_vae_encoder(paths_, config_))
    {
        fprintf(stderr, "edit VAE encoder graph load failed\n");
        return false;
    }
    std::vector<float> condition_latents;
    std::vector<TransformerImageShape> image_shapes;
    const Clock::time_point vae_encode_begin = Clock::now();
    bool vae_encode_ok = true;
    {
        QwenVaeEncoder encoder(*models_.vae_encoder, config_);
        for (size_t i = 0; i < rgba_images.size(); i++)
        {
            std::vector<float> image_latents;
            if (!encoder.encode(rgba_images[i], condition_widths[i], condition_heights[i], image_latents))
            {
                fprintf(stderr, "edit VAE encoder failed for reference image %zu\n", i);
                vae_encode_ok = false;
                break;
            }
            const int image_h = condition_heights[i] / 16;
            const int image_w = condition_widths[i] / 16;
            if (image_latents.size() != (size_t)image_h * image_w * kLatent)
            {
                vae_encode_ok = false;
                break;
            }
            condition_latents.insert(condition_latents.end(),
                                     image_latents.begin(), image_latents.end());
            image_shapes.push_back({image_h, image_w});
        }
    }
    models_.unload_vae_encoder();
    if (!vae_encode_ok)
        return false;
    if (timings) timings->vae_encoder_ms = elapsed_ms(vae_encode_begin);

    const int target_h = request.height / 16;
    const int target_w = request.width / 16;
    const int target_tokens = target_h * target_w;
    int expected_image_slots = 0;
    for (const TransformerImageShape& shape : image_shapes)
        expected_image_slots += shape.height * shape.width / 4;
    if ((int)std::count(text_slots.begin(), text_slots.end(), 1)
        != expected_image_slots)
        return false;

    image_shapes.push_back({target_h, target_w});
    if (!models_.load_transformer(paths_, config_))
    {
        fprintf(stderr, "edit transformer graph load failed\n");
        return false;
    }
    const std::vector<float> sigmas = QwenScheduler::make_sigmas(request.steps, target_tokens, false);
    std::vector<std::vector<float>> batch_latents(request.batch);
    const Clock::time_point transformer_begin = Clock::now();
    bool transformer_ok = true;
    for (int b = 0; b < request.batch && transformer_ok; b++)
    {
        std::vector<float> target_latents((size_t)target_tokens * kLatent);
        std::mt19937 gen(static_cast<unsigned int>(request.seed + (uint64_t)b));
        std::normal_distribution<float> dist(0.f, 1.f);
        for (float& value : target_latents)
            value = dist(gen);

        QwenTransformer positive_transformer(models_, config_, text_tokens, target_tokens, target_h, target_w);
        QwenTransformer negative_transformer(models_, config_, negative_text_tokens, target_tokens, target_h, target_w);
        for (int z = 0; z < request.steps; z++)
        {
            std::vector<float> noise;
            if (!positive_transformer.run_edit(condition_latents, target_latents, text, text_slots, image_shapes, sigmas[z], noise))
            {
                fprintf(stderr, "edit transformer failed at step %d of image %d\n", z, b);
                transformer_ok = false;
                break;
            }
            if (do_true_cfg)
            {
                std::vector<float> negative_noise;
                if (!negative_transformer.run_edit(condition_latents, target_latents, negative_text, negative_text_slots, image_shapes, sigmas[z], negative_noise) || negative_noise.size() != noise.size())
                {
                    fprintf(stderr, "edit negative transformer failed at step %d of image %d\n", z, b);
                    transformer_ok = false;
                    break;
                }
                for (size_t i = 0; i < noise.size(); i++)
                    noise[i] = negative_noise[i] + request.guidance_scale * (noise[i] - negative_noise[i]);
            }

            const float dt = sigmas[z + 1] - sigmas[z];
            for (size_t i = 0; i < target_latents.size(); i++)
                target_latents[i] += dt * noise[i];
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
            batch_latents[b] = std::move(target_latents);
    }
    models_.unload_transformer();
    if (!transformer_ok)
        return false;
    if (timings) timings->transformer_ms = elapsed_ms(transformer_begin);

    if (!models_.load_vae_decoder(paths_, config_))
    {
        fprintf(stderr, "edit VAE decoder graph load failed\n");
        return false;
    }
    double vae_decoder_ms = 0.0;
    bool vae_decode_ok = true;
    {
        QwenVaeDecoder decoder(*models_.vae_decoder, config_);
        for (int b = 0; b < request.batch; b++)
        {
            std::vector<float> image;
            const Clock::time_point decode_begin = Clock::now();
            if (!decoder.decode(batch_latents[b], request.width, request.height, image))
            {
                fprintf(stderr, "edit VAE decoder failed for image %d\n", b);
                vae_decode_ok = false;
                break;
            }
            vae_decoder_ms += elapsed_ms(decode_begin);
            const std::string output_path = make_batch_output_path(request.output, b, request.batch);
            if (!save_rgba_float_png(output_path, image, request.width, request.height))
            {
                fprintf(stderr, "failed to save output %s\n", output_path.c_str());
                vae_decode_ok = false;
                break;
            }
            if (request.batch > 1)
                fprintf(stderr, "vae of image %d/%d done\n", b + 1, request.batch);
            else
                fprintf(stderr, "vae done\n");
        }
    }
    models_.unload_vae_decoder();
    if (!vae_decode_ok)
        return false;
    if (timings) timings->vae_decoder_ms = vae_decoder_ms;
    if (timings) timings->total_ms = elapsed_ms(vision_begin);
    return true;
}
}

