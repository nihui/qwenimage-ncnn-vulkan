// qwen-image implemented with ncnn library

#include "edit_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <climits>
#include <filesystem>
#include <random>
#include <cstdio>
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
constexpr int kTextRope = 64;


double elapsed_ms(const Clock::time_point& start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}


bool valid_matrix(const ncnn::Mat& value, int width, int height)
{
    return !value.empty() && value.refcount && value.dims == 2 && value.w == width && value.h == height && value.elempack == 1 && value.elembits() == 32;
}

bool read_tensor(const std::string& path, int width, int height, ncnn::Mat& data)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    if (fseek(fp, 0, SEEK_END) != 0)
    {
        fclose(fp);
        return false;
    }
    const long bytes = ftell(fp);
    if (bytes <= 0 || bytes % sizeof(float) || height <= 0 || fseek(fp, 0, SEEK_SET) != 0)
    {
        fclose(fp);
        return false;
    }
    if (width == 0)
        width = (int)(bytes / sizeof(float) / height);
    if ((size_t)bytes != (size_t)width * height * sizeof(float))
    {
        fclose(fp);
        return false;
    }
    data.create(width, height);
    const bool ok = !data.empty() && fread(data.data, 1, bytes, fp) == (size_t)bytes;
    return fclose(fp) == 0 && ok;
}

bool read_rope(const std::string& path, int width, int height, bool channels, ncnn::Mat& data)
{
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    if (width <= 0 || height <= 0 || fseek(fp, 0, SEEK_END) != 0 || ftell(fp) != (long)((size_t)width * height * 2 * sizeof(float)) || fseek(fp, 0, SEEK_SET) != 0)
    {
        fclose(fp);
        return false;
    }
    if (channels)
        data.create(width, height, 1);
    else
        data.create(width, height);
    bool ok = !data.empty();
    for (int i = 0; i < height && ok; i++)
    {
        ok = fread(data.row(i), sizeof(float), width, fp) == (size_t)width;
        ok = fseek(fp, (long)width * sizeof(float), SEEK_CUR) == 0 && ok;
    }
    ok = fgetc(fp) == EOF && ok;
    return fclose(fp) == 0 && ok;
}

bool drop_rows(const ncnn::Mat& source, int drop, ncnn::Mat& target)
{
    if (source.empty() || source.dims != 2 || drop < 0 || drop >= source.h)
        return false;
    ncnn::copy_cut_border(source, target, drop, 0, 0, 0);
    return !target.empty() && target.refcount;
}

bool make_multimodal_feature_rows(const VisionFeatures& vision, const ncnn::Mat& full_mask, int tokens, ncnn::Mat& image_embeds, std::vector<ncnn::Mat>& deepstack, std::vector<unsigned char>& slot_mask)
{
    if (!valid_matrix(full_mask, 1, tokens) || vision.tokens <= 0 || !valid_matrix(vision.image, kHidden, vision.tokens) || !valid_matrix(vision.deep0, kHidden, vision.tokens) || !valid_matrix(vision.deep1, kHidden, vision.tokens) || !valid_matrix(vision.deep2, kHidden, vision.tokens))
        return false;
    image_embeds.create(kHidden, tokens);
    if (image_embeds.empty())
        return false;
    image_embeds.fill(0.f);
    deepstack.resize(3);
    for (ncnn::Mat& value : deepstack)
    {
        value.create(kHidden, tokens);
        if (value.empty())
            return false;
        value.fill(0.f);
    }
    slot_mask.assign(tokens, 0);
    int row = 0;
    for (int i = 0; i < tokens; i++)
        if (full_mask[i] > 0.5f)
        {
            if (row >= vision.tokens)
                return false;
            slot_mask[i] = 1;
            std::memcpy(image_embeds.row(i), vision.image.row(row), (size_t)kHidden * sizeof(float));
            for (int level = 0; level < 3; level++)
            {
                const ncnn::Mat& source = level == 0 ? vision.deep0 : level == 1 ? vision.deep1 : vision.deep2;
                std::memcpy(deepstack[level].row(i), source.row(row), (size_t)kHidden * sizeof(float));
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

    models_.unload_all();
    if (!initialize_memory_budget(config_))
        return false;
    const uint64_t target_token_count = (uint64_t)(request.width / 16) * (request.height / 16);
    if (target_token_count > INT_MAX)
        return false;

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

    ncnn::Mat ids;
    ncnn::Mat negative_ids;
    ncnn::Mat text_cos, text_sin, text_attention, full_image_mask;
    ncnn::Mat negative_text_cos, negative_text_sin;
    ncnn::Mat negative_text_attention, negative_image_mask;
    ncnn::Mat patch, position, vision_cos, vision_sin;
    std::vector<ncnn::Mat> rgba_images(image_count);
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
            if (rgba_images[i].w != condition_widths[i] || rgba_images[i].h != condition_heights[i])
                return false;
        }
    }
    else
    {
        if (!read_tensor(request.input_ids, 0, 1, ids)
            || !read_rope(request.text_cos, kTextRope, ids.w, true, text_cos)
            || !read_rope(request.text_sin, kTextRope, ids.w, true, text_sin)
            || !read_tensor(request.text_attention, ids.w, ids.w, text_attention)
            || !read_tensor(request.image_mask, 1, ids.w, full_image_mask)
            || !read_tensor(request.vision_patch, 1536, request.vision_patch_tokens, patch)
            || !read_tensor(request.vision_pos, 1152, request.vision_patch_tokens, position)
            || !read_rope(request.vision_cos, 36, request.vision_patch_tokens, false, vision_cos)
            || !read_rope(request.vision_sin, 36, request.vision_patch_tokens, false, vision_sin))
            return false;
        for (size_t i = 0; i < image_count; i++)
        {
            ncnn::Mat rgba;
            if (!read_tensor(vae_files[i], condition_widths[i], condition_heights[i] * 4, rgba))
                return false;
            rgba_images[i] = rgba.reshape(condition_widths[i], condition_heights[i], 4);
            if (rgba_images[i].empty())
                return false;
        }
    }

    const int full_tokens = ids.w;
    const int negative_full_tokens = negative_ids.w;
    if (full_tokens <= request.drop_system_tokens || !valid_matrix(ids, full_tokens, 1)
        || text_cos.dims != 3 || text_cos.w != kTextRope || text_cos.h != full_tokens || text_cos.c != 1
        || text_sin.dims != 3 || text_sin.w != kTextRope || text_sin.h != full_tokens || text_sin.c != 1
        || !valid_matrix(text_attention, full_tokens, full_tokens) || !valid_matrix(full_image_mask, 1, full_tokens)
        || request.vision_patch_tokens <= 0 || !valid_matrix(patch, 1536, request.vision_patch_tokens)
        || !valid_matrix(position, 1152, request.vision_patch_tokens) || !valid_matrix(vision_cos, 36, request.vision_patch_tokens) || !valid_matrix(vision_sin, 36, request.vision_patch_tokens))
        return false;
    if (do_true_cfg && (negative_full_tokens <= request.drop_system_tokens || !valid_matrix(negative_ids, negative_full_tokens, 1)
        || negative_text_cos.dims != 3 || negative_text_cos.w != kTextRope || negative_text_cos.h != negative_full_tokens || negative_text_cos.c != 1
        || negative_text_sin.dims != 3 || negative_text_sin.w != kTextRope || negative_text_sin.h != negative_full_tokens || negative_text_sin.c != 1
        || !valid_matrix(negative_text_attention, negative_full_tokens, negative_full_tokens) || !valid_matrix(negative_image_mask, 1, negative_full_tokens)))
        return false;

    uint64_t condition_token_count = 0;
    for (size_t i = 0; i < image_count; i++)
        condition_token_count += (uint64_t)(condition_widths[i] / 16) * (condition_heights[i] / 16);
    if (condition_token_count > (uint64_t)INT_MAX - std::max(full_tokens, negative_full_tokens))
        return false;
    const int condition_tokens = (int)condition_token_count;
    int prefix_tokens = condition_tokens;
    for (int i = request.drop_system_tokens; i < full_tokens; i++)
        if (full_image_mask[i] == 0.f)
            prefix_tokens++;
    int negative_prefix_tokens = 0;
    if (do_true_cfg)
    {
        negative_prefix_tokens = condition_tokens;
        for (int i = request.drop_system_tokens; i < negative_full_tokens; i++)
            if (negative_image_mask[i] == 0.f)
                negative_prefix_tokens++;
    }
    uint64_t transformer_weights = 0;
    if (!get_transformer_weight_size(paths_, transformer_weights) || !configure_auto_low_vram(config_, request.width, request.height, prefix_tokens, negative_prefix_tokens, transformer_weights))
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

    ncnn::Mat image_embeds;
    std::vector<ncnn::Mat> deepstack;
    std::vector<unsigned char> full_slots;
    if (!make_multimodal_feature_rows(vision_features, full_image_mask, full_tokens, image_embeds, deepstack, full_slots))
        return false;

    ncnn::Mat negative_image_embeds;
    std::vector<ncnn::Mat> negative_deepstack;
    std::vector<unsigned char> negative_full_slots;
    if (do_true_cfg
        && !make_multimodal_feature_rows(vision_features, negative_image_mask, negative_full_tokens, negative_image_embeds, negative_deepstack, negative_full_slots))
        return false;

    if (!models_.load_text_encoder(paths_, config_, true))
    {
        fprintf(stderr, "edit text encoder graph load failed\n");
        return false;
    }
    ncnn::Mat full_text;
    ncnn::Mat negative_full_text;
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
    ncnn::Mat text;
    if (!drop_rows(full_text, drop, text))
        return false;
    std::vector<unsigned char> text_slots(text_tokens);
    for (int i = 0; i < text_tokens; i++)
        text_slots[i] = full_slots[drop + i];

    const int negative_text_tokens = do_true_cfg
        ? negative_full_tokens - drop : text_tokens;
    ncnn::Mat negative_text;
    std::vector<unsigned char> negative_text_slots;
    if (do_true_cfg)
    {
        if (!drop_rows(negative_full_text, drop, negative_text))
            return false;
        negative_text_slots.resize(negative_text_tokens);
        for (int i = 0; i < negative_text_tokens; i++)
            negative_text_slots[i] = negative_full_slots[drop + i];
    }

    if (!models_.load_vae_encoder(paths_, config_))
    {
        fprintf(stderr, "edit VAE encoder graph load failed\n");
        return false;
    }
    ncnn::Mat condition_latents(kLatent, condition_tokens);
    if (condition_latents.empty())
    {
        models_.unload_vae_encoder();
        return false;
    }
    int condition_offset = 0;
    std::vector<TransformerImageShape> image_shapes;
    const Clock::time_point vae_encode_begin = Clock::now();
    bool vae_encode_ok = true;
    {
        QwenVaeEncoder encoder(*models_.vae_encoder, config_);
        for (size_t i = 0; i < rgba_images.size(); i++)
        {
            ncnn::Mat image_latents;
            if (!encoder.encode(rgba_images[i], image_latents))
            {
                fprintf(stderr, "edit VAE encoder failed for reference image %zu\n", i);
                vae_encode_ok = false;
                break;
            }
            const int image_h = condition_heights[i] / 16;
            const int image_w = condition_widths[i] / 16;
            if (!valid_matrix(image_latents, kLatent, image_h * image_w))
            {
                vae_encode_ok = false;
                break;
            }
            std::memcpy(condition_latents.row(condition_offset), image_latents.data, (size_t)image_h * image_w * kLatent * sizeof(float));
            condition_offset += image_h * image_w;
            image_shapes.push_back({image_h, image_w});
        }
    }
    models_.unload_vae_encoder();
    if (!vae_encode_ok)
        return false;
    if (timings) timings->vae_encoder_ms = elapsed_ms(vae_encode_begin);

    const int target_h = request.height / 16;
    const int target_w = request.width / 16;
    const int target_tokens = (int)target_token_count;
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
    std::vector<ncnn::Mat> batch_latents(request.batch);
    const Clock::time_point transformer_begin = Clock::now();
    bool transformer_ok = true;
    {
        QwenTransformer positive_transformer(models_, config_, text_tokens, target_tokens);
        QwenTransformer negative_transformer(models_, config_, negative_text_tokens, target_tokens);
        transformer_ok = positive_transformer.prepare_edit(condition_latents, text, text_slots, image_shapes, sigmas[0]);
        if (!transformer_ok)
            fprintf(stderr, "edit positive transformer prefix prefill failed\n");
        if (transformer_ok && do_true_cfg)
        {
            transformer_ok = negative_transformer.prepare_edit(condition_latents, negative_text, negative_text_slots, image_shapes, sigmas[0]);
            if (!transformer_ok)
                fprintf(stderr, "edit negative transformer prefix prefill failed\n");
        }
        for (int b = 0; b < request.batch && transformer_ok; b++)
        {
            ncnn::Mat target_latents(kLatent, target_tokens);
            if (target_latents.empty())
            {
                transformer_ok = false;
                break;
            }
            std::mt19937 gen(static_cast<unsigned int>(request.seed + (uint64_t)b));
            std::normal_distribution<float> dist(0.f, 1.f);
            float* values = target_latents;
            for (size_t i = 0; i < (size_t)target_tokens * kLatent; i++)
                values[i] = dist(gen);

            for (int z = 0; z < request.steps; z++)
            {
                ncnn::Mat noise;
                if (!positive_transformer.run_edit(target_latents, sigmas[z], noise))
                {
                    fprintf(stderr, "edit transformer failed at step %d of image %d\n", z, b);
                    transformer_ok = false;
                    break;
                }
                if (do_true_cfg)
                {
                    ncnn::Mat negative_noise;
                    if (!negative_transformer.run_edit(target_latents, sigmas[z], negative_noise) || negative_noise.w != noise.w || negative_noise.h != noise.h)
                    {
                        fprintf(stderr, "edit negative transformer failed at step %d of image %d\n", z, b);
                        transformer_ok = false;
                        break;
                    }
                    for (size_t i = 0; i < (size_t)noise.w * noise.h; i++)
                        noise[i] = negative_noise[i] + request.guidance_scale * (noise[i] - negative_noise[i]);
                }

                const float dt = sigmas[z + 1] - sigmas[z];
                for (size_t i = 0; i < (size_t)target_latents.w * target_latents.h; i++)
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
            ncnn::Mat image;
            const Clock::time_point decode_begin = Clock::now();
            if (!decoder.decode(batch_latents[b], request.width, request.height, image))
            {
                fprintf(stderr, "edit VAE decoder failed for image %d\n", b);
                vae_decode_ok = false;
                break;
            }
            vae_decoder_ms += elapsed_ms(decode_begin);
            const std::string output_path = make_batch_output_path(request.output, b, request.batch);
            if (!save_rgba_float_png(output_path, image))
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

