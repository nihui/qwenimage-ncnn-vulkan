// qwen-image implemented with ncnn library

#include "transformer.h"

#include <cmath>
#include <cstring>
#include <cstdio>

namespace qwenimage {
namespace {
constexpr int kLatentDim = 64;
constexpr int kHiddenDim = 4096;
constexpr int kRopeDim = 64;
constexpr float kTheta = 10000.f;

ncnn::Mat float_mat(int w, int h, const std::vector<float>& data)
{
    return ncnn::Mat(w, h, (void*)data.data()).clone();
}
bool expand_condition_rows(const ncnn::Mat& source, int text_tokens, int image_tokens, int dim, std::vector<float>& expanded)
{
    if (source.empty() || source.elempack != 1 || source.w != dim
        || source.h != 2 || source.c != 1 || source.elembits() != 32)
        return false;
    const float* rows = static_cast<const float*>(source.data);
    expanded.resize((size_t)(text_tokens + image_tokens) * dim);
    for (int i = 0; i < text_tokens; i++)
        std::memcpy(expanded.data() + (size_t)i * dim, rows + dim, (size_t)dim * sizeof(float));
    for (int i = 0; i < image_tokens; i++)
        std::memcpy(expanded.data() + (size_t)(text_tokens + i) * dim, rows, (size_t)dim * sizeof(float));
    return true;
}
void append_axis(double position, int dim, std::vector<float>& cos, std::vector<float>& sin)
{
    for (int i = 0; i < dim / 2; i++)
    {
        const float inv = 1.f / std::pow(kTheta, (float)(2 * i) / dim);
        const float angle = static_cast<float>(position) * inv;
        cos.push_back(std::cos(angle));
        sin.push_back(std::sin(angle));
    }
}
}
void QwenTransformer::make_rope(int text_tokens, int valid_text_tokens, int latent_height, int latent_width, std::vector<float>& cos, std::vector<float>& sin)
{
    const int image_tokens = latent_height * latent_width;
    cos.clear();
    sin.clear();
    cos.reserve((size_t)(text_tokens + image_tokens) * kRopeDim);
    sin.reserve(cos.capacity());
    for (int i = 0; i < text_tokens; i++)
    {
        append_axis(i, 16, cos, sin);
        append_axis(i, 56, cos, sin);
        append_axis(i, 56, cos, sin);
    }
    const double frame = valid_text_tokens;
    std::vector<double> hpos;
    std::vector<double> wpos;
    for (int h = -(latent_height - latent_height / 2);
         h < latent_height / 2; h++)
        hpos.push_back(h);
    for (int w = -(latent_width - latent_width / 2);
         w < latent_width / 2; w++)
        wpos.push_back(w);
    for (int y = 0; y < latent_height; y++)
        for (int x = 0; x < latent_width; x++)
        {
            append_axis(frame, 16, cos, sin);
            append_axis(hpos[y], 56, cos, sin);
            append_axis(wpos[x], 56, cos, sin);
        }
}
void QwenTransformer::make_attention_mask(int text_tokens, int valid_text_tokens, int image_tokens, std::vector<float>& mask)
{
    const int sequence = text_tokens + image_tokens;
    mask.assign((size_t)sequence * sequence, -1.0e30f);
    for (int q = 0; q < sequence; q++)
        for (int k = 0; k < sequence; k++)
        {
            bool allowed;
            if (q < text_tokens)
                allowed = k <= q && k < valid_text_tokens;
            else
                allowed = (k < valid_text_tokens) || k >= text_tokens;
            if (allowed)
                mask[(size_t)q * sequence + k] = 0.f;
        }
}
bool QwenTransformer::run(const std::vector<float>& latents, const std::vector<float>& text, float timestep, const std::vector<float>& cos, const std::vector<float>& sin, const std::vector<float>& mask, std::vector<float>& noise) const
{
    if (latents.size() != (size_t)image_tokens_ * kLatentDim
        || text.size() != (size_t)text_tokens_ * kHiddenDim
        || cos.size() != (size_t)(text_tokens_ + image_tokens_) * kRopeDim
        || sin.size() != cos.size()
        || mask.size() != (size_t)(text_tokens_ + image_tokens_) *
                          (text_tokens_ + image_tokens_))
    {
        fprintf(stderr, "transformer input vector size failed: latents=%zu text=%zu cos=%zu sin=%zu mask=%zu\n", latents.size(), text.size(), cos.size(), sin.size(), mask.size());
        return false;
    }

    ncnn::Extractor input = models_.transformer_input->create_extractor();
    set_extractor_light_mode(input);
    ncnn::Mat in_latents = float_mat(kLatentDim, image_tokens_, latents);
    ncnn::Mat in_text = float_mat(kHiddenDim, text_tokens_, text);
    ncnn::Mat in_timestep(1);
    in_timestep[0] = timestep;
    if (input.input("in0", in_latents) != 0
        || input.input("in1", in_text) != 0
        || input.input("in2", in_timestep) != 0)
    {
        fprintf(stderr, "transformer input graph input failed\n");
        return false;
    }
    ncnn::Mat raw_hidden, raw_modulation, raw_temb;
    if (input.extract("out0", raw_hidden) != 0
        || input.extract("out1", raw_modulation) != 0
        || input.extract("out2", raw_temb) != 0)
    {
        fprintf(stderr, "transformer input graph extraction failed\n");
        return false;
    }
    ncnn::Mat hidden = clone_fp32(raw_hidden, config_);
    ncnn::Mat modulation = clone_fp32(raw_modulation, config_);
    ncnn::Mat temb = clone_fp32(raw_temb, config_);
    const int sequence = text_tokens_ + image_tokens_;
    std::vector<float> expanded_modulation;
    std::vector<float> expanded_temb;
    if (!expand_condition_rows(modulation, text_tokens_, image_tokens_, 16384, expanded_modulation) || !expand_condition_rows(temb, text_tokens_, image_tokens_, kHiddenDim, expanded_temb))
    {
        fprintf(stderr, "transformer condition shape failed: modulation %dx%dx%d, temb %dx%dx%d\n", modulation.w, modulation.h, modulation.c, temb.w, temb.h, temb.c);
        return false;
    }
    if (hidden.empty() || modulation.empty() || temb.empty()
        || hidden.elempack != 1 || hidden.w != kHiddenDim
        || hidden.h != sequence)
    {
        fprintf(stderr, "transformer hidden shape failed: %dx%dx%d\n",
                hidden.w, hidden.h, hidden.c);
        return false;
    }

    ncnn::Mat in_cos = float_mat(kRopeDim, sequence, cos);
    ncnn::Mat in_sin = float_mat(kRopeDim, sequence, sin);
    ncnn::Mat in_mask = float_mat(sequence, sequence, mask);
    ncnn::Mat in_modulation = float_mat(16384, sequence, expanded_modulation);
    ncnn::Mat in_temb = float_mat(kHiddenDim, sequence, expanded_temb);
    for (const auto& net : models_.transformer_blocks)
    {
        ncnn::Extractor block = net->create_extractor();
        set_extractor_light_mode(block);
        if (block.input("in0", hidden) != 0
            || block.input("in1", in_modulation) != 0
            || block.input("in2", in_cos) != 0
            || block.input("in3", in_sin) != 0
            || block.input("in4", in_mask) != 0)
        {
            fprintf(stderr, "transformer block input failed\n");
            return false;
        }
        ncnn::Mat raw;
        if (block.extract("out0", raw) != 0)
        {
            fprintf(stderr, "transformer block extraction failed\n");
            return false;
        }
        hidden = clone_fp32(raw, config_);
        if (hidden.empty() || hidden.elempack != 1
            || hidden.w != kHiddenDim || hidden.h != sequence)
        {
            fprintf(stderr,
                    "transformer block output shape failed: %dx%dx%d\n", hidden.w, hidden.h, hidden.c);
            return false;
        }
    }

    ncnn::Extractor output = models_.transformer_output->create_extractor();
    set_extractor_light_mode(output);
    if (output.input("in0", hidden) != 0
        || output.input("in1", in_temb) != 0)
    {
        fprintf(stderr, "transformer output input failed\n");
        return false;
    }
    ncnn::Mat raw_noise;
    if (output.extract("out0", raw_noise) != 0)
    {
        fprintf(stderr, "transformer output extraction failed\n");
        return false;
    }
    ncnn::Mat result = clone_fp32(raw_noise, config_);
    if (result.empty() || result.elempack != 1
        || result.w != kLatentDim || result.h < image_tokens_)
    {
        fprintf(stderr, "transformer output shape failed: %dx%dx%d\n",
                result.w, result.h, result.c);
        return false;
    }
    const float* source = static_cast<const float*>(result.data);
    const int first_image_row = result.h - image_tokens_;
    noise.resize((size_t)image_tokens_ * kLatentDim);
    for (int y = 0; y < image_tokens_; y++)
        std::memcpy(noise.data() + (size_t)y * kLatentDim, source + (size_t)(first_image_row + y) * kLatentDim, (size_t)kLatentDim * sizeof(float));
    return true;
}
namespace {
bool build_edit_layout(int text_tokens, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes, int target_tokens, std::vector<int>& image_ids, std::vector<unsigned char>& target_mask, int& condition_tokens)
{
    if (text_tokens <= 0
        || text_image_slots.size() != (size_t)text_tokens
        || image_shapes.size() < 2)
        return false;
    const int condition_count = (int)image_shapes.size() - 1;
    int expected_slots = 0;
    condition_tokens = 0;
    for (int i = 0; i < condition_count; i++)
    {
        const int h = image_shapes[i].height;
        const int w = image_shapes[i].width;
        if (h <= 0 || w <= 0 || (h * w) % 4 != 0)
            return false;
        expected_slots += h * w / 4;
        condition_tokens += h * w;
    }
    const int target_h = image_shapes.back().height;
    const int target_w = image_shapes.back().width;
    if (target_h <= 0 || target_w <= 0
        || target_h * target_w != target_tokens
        || target_tokens % 4)
        return false;

    int actual_slots = 0;
    for (unsigned char value : text_image_slots)
        actual_slots += value ? 1 : 0;
    if (actual_slots != expected_slots)
        return false;

    image_ids.clear();
    target_mask.clear();
    image_ids.reserve((size_t)text_tokens + target_tokens);
    target_mask.reserve((size_t)text_tokens + target_tokens);
    int block = 0;
    int used_in_block = 0;
    for (int i = 0; i < text_tokens; i++)
    {
        if (!text_image_slots[i])
        {
            image_ids.push_back(-1);
            target_mask.push_back(0);
            continue;
        }
        while (block < condition_count
               && used_in_block >= image_shapes[block].height
                                      * image_shapes[block].width / 4)
        {
            block++;
            used_in_block = 0;
        }
        if (block >= condition_count)
            return false;
        for (int j = 0; j < 4; j++)
        {
            image_ids.push_back(block);
            target_mask.push_back(0);
        }
        used_in_block++;
    }
    int counted = 0;
    for (int id : image_ids)
        if (id >= 0) counted++;
    if (counted != condition_tokens)
        return false;
    const int target_block = condition_count;
    for (int i = 0; i < target_tokens; i++)
    {
        image_ids.push_back(target_block);
        target_mask.push_back(1);
    }
    return true;
}

bool make_edit_rope(const std::vector<int>& image_ids, const std::vector<TransformerImageShape>& image_shapes, std::vector<float>& cos, std::vector<float>& sin)
{
    const int sequence = (int)image_ids.size();
    const int block_count = (int)image_shapes.size();
    std::vector<int> frame(sequence, 0);
    std::vector<int> height(sequence, 0);
    std::vector<int> width(sequence, 0);
    int cursor = 0;
    int position = 0;
    for (int block = 0; block < block_count; block++)
    {
        int begin = -1;
        for (int i = cursor; i < sequence; i++)
            if (image_ids[i] == block)
            {
                begin = i;
                break;
            }
        if (begin < 0)
            return false;
        for (int i = cursor; i < begin; i++)
            frame[i] = position++;
        const int h = image_shapes[block].height;
        const int w = image_shapes[block].width;
        const int count = h * w;
        if (begin + count > sequence)
            return false;
        int row = begin;
        for (int y = -(h - h / 2); y < h / 2; y++)
            for (int x = 0; x < w; x++, row++)
            {
                frame[row] = position;
                height[row] = y;
            }
        row = begin;
        for (int y = 0; y < h; y++)
            for (int x = -(w - w / 2); x < w / 2; x++, row++)
                width[row] = x;
        cursor = begin + count;
        position += std::max(h, w);
    }
    for (int i = cursor; i < sequence; i++)
        frame[i] = position++;
    for (int i = 0; i < sequence; i++)
        if (image_ids[i] < 0)
        {
            height[i] = frame[i];
            width[i] = frame[i];
        }

    cos.clear();
    sin.clear();
    cos.reserve((size_t)sequence * kRopeDim);
    sin.reserve(cos.capacity());
    for (int i = 0; i < sequence; i++)
    {
        append_axis(frame[i], 16, cos, sin);
        append_axis(height[i], 56, cos, sin);
        append_axis(width[i], 56, cos, sin);
    }
    return true;
}

bool make_edit_attention_mask(const std::vector<int>& image_ids, std::vector<float>& mask)
{
    const int sequence = (int)image_ids.size();
    mask.assign((size_t)sequence * sequence, -1.0e30f);
    for (int q = 0; q < sequence; q++)
        for (int k = 0; k < sequence; k++)
        {
            const bool same_image = image_ids[q] >= 0
                                  && image_ids[q] == image_ids[k];
            if (k <= q || same_image)
                mask[(size_t)q * sequence + k] = 0.f;
        }
    return true;
}

bool expand_edit_rows(const ncnn::Mat& source, const std::vector<unsigned char>& target_mask, int dim, std::vector<float>& expanded)
{
    if (source.empty() || source.elempack != 1 || source.w != dim
        || source.h != 2 || source.c != 1 || source.elembits() != 32
        || target_mask.empty())
        return false;
    const float* rows = static_cast<const float*>(source.data);
    expanded.resize((size_t)target_mask.size() * dim);
    for (size_t i = 0; i < target_mask.size(); i++)
    {
        const int source_row = target_mask[i] ? 0 : 1;
        std::memcpy(expanded.data() + i * dim, rows + (size_t)source_row * dim, (size_t)dim * sizeof(float));
    }
    return true;
}

bool make_joint_hidden(const ncnn::Mat& source, int text_tokens, const std::vector<unsigned char>& slots, int latent_tokens, std::vector<float>& joint)
{
    if (source.empty() || source.elempack != 1 || source.w != kHiddenDim
        || source.h != text_tokens + latent_tokens || source.c != 1
        || source.elembits() != 32 || slots.size() != (size_t)text_tokens)
        return false;
    const float* rows = static_cast<const float*>(source.data);
    joint.clear();
    joint.reserve((size_t)(text_tokens + latent_tokens) * kHiddenDim);
    int image_row = text_tokens;
    for (int i = 0; i < text_tokens; i++)
    {
        if (!slots[i])
        {
            const float* row = rows + (size_t)i * kHiddenDim;
            joint.insert(joint.end(), row, row + kHiddenDim);
        }
        else
        {
            for (int j = 0; j < 4; j++)
            {
                if (image_row >= source.h)
                    return false;
                const float* row = rows + (size_t)image_row++ * kHiddenDim;
                joint.insert(joint.end(), row, row + kHiddenDim);
            }
        }
    }
    while (image_row < source.h)
    {
        const float* row = rows + (size_t)image_row++ * kHiddenDim;
        joint.insert(joint.end(), row, row + kHiddenDim);
    }
    return true;
}
}

bool QwenTransformer::run_edit(const std::vector<float>& condition_latents, const std::vector<float>& latents, const std::vector<float>& text, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes, float timestep, std::vector<float>& noise) const
{
    const int text_tokens = (int)text_image_slots.size();
    const int target_tokens = (int)latents.size() / kLatentDim;
    if (text_tokens <= 0 || target_tokens <= 0
        || text.size() != (size_t)text_tokens * kHiddenDim
        || latents.size() != (size_t)target_tokens * kLatentDim
        || condition_latents.empty()
        || condition_latents.size() % kLatentDim)
        return false;
    const int condition_tokens = (int)condition_latents.size() / kLatentDim;

    std::vector<int> image_ids;
    std::vector<unsigned char> target_mask;
    int layout_condition_tokens = 0;
    if (!build_edit_layout(text_tokens, text_image_slots, image_shapes, target_tokens, image_ids, target_mask, layout_condition_tokens) || layout_condition_tokens != condition_tokens)
    {
        fprintf(stderr, "edit transformer image layout failed\n");
        return false;
    }
    const int sequence = (int)image_ids.size();
    std::vector<float> cos;
    std::vector<float> sin;
    std::vector<float> attention_mask;
    if (!make_edit_rope(image_ids, image_shapes, cos, sin)
        || !make_edit_attention_mask(image_ids, attention_mask))
        return false;
    if (cos.size() != (size_t)sequence * kRopeDim
        || sin.size() != cos.size()
        || attention_mask.size() != (size_t)sequence * sequence)
        return false;

    std::vector<float> all_latents;
    all_latents.reserve(condition_latents.size() + latents.size());
    all_latents.insert(all_latents.end(),
                       condition_latents.begin(), condition_latents.end());
    all_latents.insert(all_latents.end(), latents.begin(), latents.end());

    ncnn::Extractor input = models_.transformer_input->create_extractor();
    set_extractor_light_mode(input);
    ncnn::Mat in_latents = float_mat(kLatentDim, condition_tokens + target_tokens, all_latents);
    ncnn::Mat in_text = float_mat(kHiddenDim, text_tokens, text);
    ncnn::Mat in_timestep(1);
    in_timestep[0] = timestep;
    if (input.input("in0", in_latents) != 0
        || input.input("in1", in_text) != 0
        || input.input("in2", in_timestep) != 0)
        return false;

    ncnn::Mat raw_hidden, raw_modulation, raw_temb;
    if (input.extract("out0", raw_hidden) != 0
        || input.extract("out1", raw_modulation) != 0
        || input.extract("out2", raw_temb) != 0)
        return false;
    ncnn::Mat hidden = clone_fp32(raw_hidden, config_);
    ncnn::Mat modulation = clone_fp32(raw_modulation, config_);
    ncnn::Mat temb = clone_fp32(raw_temb, config_);
    std::vector<float> joint_hidden;
    std::vector<float> expanded_modulation;
    std::vector<float> expanded_temb;
    if (!make_joint_hidden(hidden, text_tokens, text_image_slots, condition_tokens + target_tokens, joint_hidden) || !expand_edit_rows(modulation, target_mask, 16384, expanded_modulation) || !expand_edit_rows(temb, target_mask, kHiddenDim, expanded_temb))
        return false;

    ncnn::Mat in_hidden = float_mat(kHiddenDim, sequence, joint_hidden);
    ncnn::Mat in_cos = float_mat(kRopeDim, sequence, cos);
    ncnn::Mat in_sin = float_mat(kRopeDim, sequence, sin);
    ncnn::Mat in_mask = float_mat(sequence, sequence, attention_mask);
    ncnn::Mat in_modulation = float_mat(16384, sequence, expanded_modulation);
    ncnn::Mat in_temb = float_mat(kHiddenDim, sequence, expanded_temb);

    for (const auto& net : models_.transformer_blocks)
    {
        ncnn::Extractor block = net->create_extractor();
        set_extractor_light_mode(block);
        if (block.input("in0", in_hidden) != 0
            || block.input("in1", in_modulation) != 0
            || block.input("in2", in_cos) != 0
            || block.input("in3", in_sin) != 0
            || block.input("in4", in_mask) != 0)
            return false;
        ncnn::Mat raw;
        if (block.extract("out0", raw) != 0)
            return false;
        in_hidden = clone_fp32(raw, config_);
        if (in_hidden.empty() || in_hidden.elempack != 1
            || in_hidden.w != kHiddenDim || in_hidden.h != sequence)
            return false;
    }

    ncnn::Extractor output = models_.transformer_output->create_extractor();
    set_extractor_light_mode(output);
    if (output.input("in0", in_hidden) != 0
        || output.input("in1", in_temb) != 0)
        return false;
    ncnn::Mat raw_noise;
    if (output.extract("out0", raw_noise) != 0)
        return false;
    ncnn::Mat result = clone_fp32(raw_noise, config_);
    if (result.empty() || result.elempack != 1
        || result.w != kLatentDim || result.h < target_tokens)
        return false;
    const float* source = static_cast<const float*>(result.data);
    const int first_target = result.h - target_tokens;
    noise.resize((size_t)target_tokens * kLatentDim);
    for (int y = 0; y < target_tokens; y++)
        std::memcpy(noise.data() + (size_t)y * kLatentDim, source + (size_t)(first_target + y) * kLatentDim, (size_t)kLatentDim * sizeof(float));
    return true;
}


}
