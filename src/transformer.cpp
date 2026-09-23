// qwen-image implemented with ncnn library

#include "transformer.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace qwenimage {
namespace {
constexpr int kLatentDim = 64;
constexpr int kHiddenDim = 4096;
constexpr int kRopeDim = 64;
constexpr float kTheta = 10000.f;

bool is_matrix(const ncnn::Mat& value, int width, int height)
{
    return !value.empty() && value.dims == 2 && value.w == width && value.h == height && value.elempack == 1 && value.elembits() == 32 && value.refcount;
}

bool valid_hidden(const ncnn::Mat& value, int sequence)
{
    return !value.empty() && value.refcount && value.dims == 2 && value.w == kHiddenDim && value.h * value.elempack == sequence && (value.elembits() == 16 || value.elembits() == 32);
}

void make_axis(double position, int dim, float* cos, float* sin)
{
    for (int i = 0; i < dim / 2; i++)
    {
        const float inv = 1.f / std::pow(kTheta, (float)(2 * i) / dim);
        const float angle = static_cast<float>(position) * inv;
        cos[i] = std::cos(angle);
        sin[i] = std::sin(angle);
    }
}

bool expand_condition_rows(const ncnn::Mat& source, int text_tokens, int image_tokens, int dim, ncnn::Mat& expanded)
{
    if (!is_matrix(source, dim, 2))
        return false;
    expanded.create(dim, text_tokens + image_tokens);
    if (expanded.empty())
        return false;
    for (int i = 0; i < text_tokens; i++)
        std::memcpy(expanded.row(i), source.row(1), (size_t)dim * sizeof(float));
    for (int i = 0; i < image_tokens; i++)
        std::memcpy(expanded.row(text_tokens + i), source.row(0), (size_t)dim * sizeof(float));
    return true;
}

bool expand_edit_rows(const ncnn::Mat& source, const std::vector<unsigned char>& target_mask, int dim, ncnn::Mat& expanded)
{
    if (!is_matrix(source, dim, 2) || target_mask.empty())
        return false;
    expanded.create(dim, (int)target_mask.size());
    if (expanded.empty())
        return false;
    for (size_t i = 0; i < target_mask.size(); i++)
        std::memcpy(expanded.row((int)i), source.row(target_mask[i] ? 0 : 1), (size_t)dim * sizeof(float));
    return true;
}

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

bool make_edit_rope(const std::vector<int>& image_ids, const std::vector<TransformerImageShape>& image_shapes, ncnn::Mat& cos, ncnn::Mat& sin)
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

    cos.create(kRopeDim, sequence);
    sin.create(kRopeDim, sequence);
    if (cos.empty() || sin.empty())
        return false;
    for (int i = 0; i < sequence; i++)
    {
        make_axis(frame[i], 16, cos.row(i), sin.row(i));
        make_axis(height[i], 56, cos.row(i) + 8, sin.row(i) + 8);
        make_axis(width[i], 56, cos.row(i) + 36, sin.row(i) + 36);
    }
    return true;
}

bool make_edit_attention_mask(const std::vector<int>& image_ids, ncnn::Mat& mask)
{
    const int sequence = (int)image_ids.size();
    mask.create(sequence, sequence);
    if (mask.empty())
        return false;
    mask.fill(-1.0e30f);
    for (int q = 0; q < sequence; q++)
    {
        float* row = mask.row(q);
        for (int k = 0; k < sequence; k++)
        {
            const bool same_image = image_ids[q] >= 0 && image_ids[q] == image_ids[k];
            if (k <= q || same_image)
                row[k] = 0.f;
        }
    }
    return true;
}
}

bool QwenTransformer::make_rope(int text_tokens, int valid_text_tokens, int latent_height, int latent_width, ncnn::Mat& cos, ncnn::Mat& sin)
{
    if (text_tokens <= 0 || latent_height <= 0 || latent_width <= 0)
        return false;
    const int sequence = text_tokens + latent_height * latent_width;
    cos.create(kRopeDim, sequence);
    sin.create(kRopeDim, sequence);
    if (cos.empty() || sin.empty())
        return false;
    for (int i = 0; i < text_tokens; i++)
    {
        make_axis(i, 16, cos.row(i), sin.row(i));
        make_axis(i, 56, cos.row(i) + 8, sin.row(i) + 8);
        make_axis(i, 56, cos.row(i) + 36, sin.row(i) + 36);
    }
    for (int y = 0; y < latent_height; y++)
        for (int x = 0; x < latent_width; x++)
        {
            const int row = text_tokens + y * latent_width + x;
            make_axis(valid_text_tokens, 16, cos.row(row), sin.row(row));
            make_axis(y - (latent_height - latent_height / 2), 56, cos.row(row) + 8, sin.row(row) + 8);
            make_axis(x - (latent_width - latent_width / 2), 56, cos.row(row) + 36, sin.row(row) + 36);
        }
    return true;
}

bool QwenTransformer::make_attention_mask(int text_tokens, int valid_text_tokens, int image_tokens, ncnn::Mat& mask)
{
    const int sequence = text_tokens + image_tokens;
    if (text_tokens <= 0 || valid_text_tokens < 0 || valid_text_tokens > text_tokens || image_tokens <= 0)
        return false;
    mask.create(sequence, sequence);
    if (mask.empty())
        return false;
    mask.fill(-1.0e30f);
    for (int q = 0; q < sequence; q++)
    {
        float* row = mask.row(q);
        for (int k = 0; k < sequence; k++)
        {
            const bool allowed = q < text_tokens ? k <= q && k < valid_text_tokens : k < valid_text_tokens || k >= text_tokens;
            if (allowed)
                row[k] = 0.f;
        }
    }
    return true;
}

bool QwenTransformer::run_input(const ncnn::Mat& latents, const ncnn::Mat& text, float timestep, ncnn::Mat& hidden, ncnn::Mat& modulation, ncnn::Mat& temb, int type) const
{
    if (!models_.transformer_input)
        return false;
    ncnn::Mat in_timestep(1);
    if (in_timestep.empty())
        return false;
    in_timestep[0] = timestep;
    ncnn::Extractor input = models_.transformer_input->create_extractor();
    if (input.input("in0", latents) != 0 || input.input("in1", text) != 0 || input.input("in2", in_timestep) != 0)
        return false;
    if (input.extract("out0", hidden, type) != 0 || input.extract("out1", modulation) != 0 || input.extract("out2", temb) != 0)
        return false;
    return valid_hidden(hidden, text.h + latents.h);
}

bool QwenTransformer::run_blocks(ncnn::Mat hidden, const ncnn::Mat& modulation, const ncnn::Mat& temb, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, ncnn::Mat& noise) const
{
    if (!models_.transformer_output)
        return false;
#if NCNN_VULKAN
    if (config_.use_vulkan_compute)
    {
        const ncnn::Net& net = *models_.transformer_output;
        ncnn::VkCompute cmd(net.vulkan_device());
        ncnn::VkMat hidden_gpu;
        cmd.record_upload(hidden, hidden_gpu, net.opt);
        if (hidden_gpu.empty() || cmd.submit_and_wait() != 0)
            return false;
        return run_blocks_vulkan(hidden_gpu, modulation, temb, cos, sin, mask, noise);
    }
#endif
    const int sequence = hidden.h * hidden.elempack;
    for (const auto& net : models_.transformer_blocks)
    {
        ncnn::Extractor block = net->create_extractor();
        if (block.input("in0", hidden) != 0 || block.input("in1", modulation) != 0 || block.input("in2", cos) != 0 || block.input("in3", sin) != 0 || block.input("in4", mask) != 0)
            return false;
        if (block.extract("out0", hidden, 1) != 0 || !valid_hidden(hidden, sequence))
            return false;
    }
    if (!models_.transformer_output)
        return false;
    ncnn::Extractor output = models_.transformer_output->create_extractor();
    if (output.input("in0", hidden) != 0 || output.input("in1", temb) != 0)
        return false;
    ncnn::Mat result;
    if (output.extract("out0", result) != 0 || !is_matrix(result, kLatentDim, sequence) || result.h < image_tokens_)
        return false;
    ncnn::Mat selected(kLatentDim, image_tokens_);
    if (selected.empty())
        return false;
    std::memcpy(selected.data, result.row(result.h - image_tokens_), (size_t)image_tokens_ * kLatentDim * sizeof(float));
    noise = selected;
    return true;
}

#if NCNN_VULKAN
bool QwenTransformer::run_input_vulkan(const ncnn::Mat& latents, const ncnn::Mat& text, float timestep, ncnn::VkMat& hidden, ncnn::Mat& modulation, ncnn::Mat& temb) const
{
    if (!models_.transformer_input)
        return false;
    const ncnn::Net& net = *models_.transformer_input;
    ncnn::Mat in_timestep(1);
    if (in_timestep.empty())
        return false;
    in_timestep[0] = timestep;
    ncnn::Extractor input = net.create_extractor();
    if (input.input("in0", latents) != 0 || input.input("in1", text) != 0 || input.input("in2", in_timestep) != 0)
        return false;
    // only the condition rows are needed on the cpu
    if (input.extract("out1", modulation) != 0 || input.extract("out2", temb) != 0)
        return false;
    ncnn::VkCompute cmd(net.vulkan_device());
    if (input.extract("out0", hidden, cmd) != 0 || cmd.submit_and_wait() != 0)
        return false;
    return !hidden.empty() && hidden.dims == 2 && hidden.w == kHiddenDim && hidden.h * hidden.elempack == text.h + latents.h;
}

bool QwenTransformer::run_blocks_vulkan(ncnn::VkMat hidden, const ncnn::Mat& modulation, const ncnn::Mat& temb, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, ncnn::Mat& noise) const
{
    const int sequence = hidden.h * hidden.elempack;
    for (const auto& net : models_.transformer_blocks)
    {
        ncnn::Extractor block = net->create_extractor();
        ncnn::VkCompute cmd(net->vulkan_device());
        if (block.input("in0", hidden) != 0 || block.input("in1", modulation) != 0 || block.input("in2", cos) != 0 || block.input("in3", sin) != 0 || block.input("in4", mask) != 0)
            return false;
        ncnn::VkMat result;
        if (block.extract("out0", result, cmd) != 0 || cmd.submit_and_wait() != 0)
            return false;
        if (result.empty() || result.dims != 2 || result.w != kHiddenDim || result.h * result.elempack != sequence)
            return false;
        hidden = result;
    }
    if (!models_.transformer_output)
        return false;
    ncnn::Extractor output = models_.transformer_output->create_extractor();
    if (output.input("in0", hidden) != 0 || output.input("in1", temb) != 0)
        return false;
    ncnn::Mat result;
    if (output.extract("out0", result) != 0 || !is_matrix(result, kLatentDim, sequence) || result.h < image_tokens_)
        return false;
    ncnn::Mat selected(kLatentDim, image_tokens_);
    if (selected.empty())
        return false;
    std::memcpy(selected.data, result.row(result.h - image_tokens_), (size_t)image_tokens_ * kLatentDim * sizeof(float));
    noise = selected;
    return true;
}
#endif

bool QwenTransformer::run(const ncnn::Mat& latents, const ncnn::Mat& text, float timestep, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, ncnn::Mat& noise) const
{
    const int sequence = text_tokens_ + image_tokens_;
    if (!is_matrix(latents, kLatentDim, image_tokens_) || !is_matrix(text, kHiddenDim, text_tokens_) || !is_matrix(cos, kRopeDim, sequence) || !is_matrix(sin, kRopeDim, sequence) || !is_matrix(mask, sequence, sequence))
        return false;
#if NCNN_VULKAN
    if (config_.use_vulkan_compute)
    {
        ncnn::VkMat hidden;
        ncnn::Mat modulation, temb;
        if (!run_input_vulkan(latents, text, timestep, hidden, modulation, temb))
            return false;
        ncnn::Mat expanded_modulation, expanded_temb;
        if (!expand_condition_rows(modulation, text_tokens_, image_tokens_, 16384, expanded_modulation) || !expand_condition_rows(temb, text_tokens_, image_tokens_, kHiddenDim, expanded_temb))
            return false;
        return run_blocks_vulkan(hidden, expanded_modulation, expanded_temb, cos, sin, mask, noise);
    }
#endif
    ncnn::Mat hidden, modulation, temb;
    if (!run_input(latents, text, timestep, hidden, modulation, temb))
        return false;
    ncnn::Mat expanded_modulation, expanded_temb;
    if (!expand_condition_rows(modulation, text_tokens_, image_tokens_, 16384, expanded_modulation) || !expand_condition_rows(temb, text_tokens_, image_tokens_, kHiddenDim, expanded_temb))
        return false;
    return run_blocks(hidden, expanded_modulation, expanded_temb, cos, sin, mask, noise);
}

bool QwenTransformer::prepare_edit(const ncnn::Mat& condition_latents, const ncnn::Mat& text, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes)
{
    edit_ready_ = false;
    if (!is_matrix(condition_latents, kLatentDim, condition_latents.h) || !is_matrix(text, kHiddenDim, text_tokens_))
        return false;
    int condition_tokens = 0;
    std::vector<int> image_ids;
    if (!build_edit_layout(text_tokens_, text_image_slots, image_shapes, image_tokens_, image_ids, target_mask_, condition_tokens) || condition_tokens != condition_latents.h)
        return false;
    const int sequence = (int)image_ids.size();
    if (!make_edit_rope(image_ids, image_shapes, edit_cos_, edit_sin_) || !make_edit_attention_mask(image_ids, edit_mask_))
        return false;
    joint_rows_.clear();
    joint_rows_.reserve(sequence);
    int image_row = text_tokens_;
    for (int i = 0; i < text_tokens_; i++)
    {
        if (!text_image_slots[i])
            joint_rows_.push_back(i);
        else
            for (int j = 0; j < 4; j++)
                joint_rows_.push_back(image_row++);
    }
    while (image_row < text_tokens_ + condition_tokens + image_tokens_)
        joint_rows_.push_back(image_row++);
    if ((int)joint_rows_.size() != sequence)
        return false;
    edit_latents_.create(kLatentDim, condition_tokens + image_tokens_);
    if (edit_latents_.empty())
        return false;
    std::memcpy(edit_latents_.data, condition_latents.data, (size_t)condition_tokens * kLatentDim * sizeof(float));
    edit_text_ = text;
    edit_ready_ = true;
    return true;
}

bool QwenTransformer::run_edit(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise)
{
    if (!edit_ready_ || !is_matrix(latents, kLatentDim, image_tokens_))
        return false;
    std::memcpy(edit_latents_.row(edit_latents_.h - image_tokens_), latents.data, (size_t)image_tokens_ * kLatentDim * sizeof(float));
    ncnn::Mat hidden, modulation, temb;
    if (!run_input(edit_latents_, edit_text_, timestep, hidden, modulation, temb, 0))
        return false;
    ncnn::Mat joint(kHiddenDim, (int)joint_rows_.size());
    if (joint.empty())
        return false;
    for (int i = 0; i < joint.h; i++)
        std::memcpy(joint.row(i), hidden.row(joint_rows_[i]), (size_t)kHiddenDim * sizeof(float));
    hidden.release();
    ncnn::Mat expanded_modulation, expanded_temb;
    if (!expand_edit_rows(modulation, target_mask_, 16384, expanded_modulation) || !expand_edit_rows(temb, target_mask_, kHiddenDim, expanded_temb))
        return false;
    return run_blocks(joint, expanded_modulation, expanded_temb, edit_cos_, edit_sin_, edit_mask_, noise);
}
}
