// qwen-image implemented with ncnn library

#include "transformer.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdio>
#include <string>
#include <memory>

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace qwenimage {
namespace {
constexpr int kLatentDim = 64;
constexpr int kHiddenDim = 4096;
constexpr int kRopeDim = 64;
constexpr int kModulationDim = 16384;
constexpr int kTheta = 10000;
constexpr int kTransformerBlocks = 32;

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
        const float inv = 1.f / std::pow((float)kTheta, (float)(2 * i) / dim);
        const float angle = static_cast<float>(position) * inv;
        cos[i] = std::cos(angle);
        sin[i] = std::sin(angle);
    }
}

bool make_joint_modulation(const ncnn::Mat& source, int prefix_rows, int target_rows, std::array<ncnn::Mat, 4>& expanded)
{
    const int rows = prefix_rows + target_rows;
    if (source.dims != 2 || source.w != kModulationDim || source.h != 2 || source.elempack != 1 || source.elembits() != 32 || prefix_rows <= 0 || target_rows <= 0)
        return false;
    for (int part = 0; part < 4; part++)
    {
        expanded[part].create(kHiddenDim, rows);
        if (expanded[part].empty())
            return false;
        for (int row = 0; row < prefix_rows; row++)
            std::memcpy(expanded[part].row(row), source.row(1) + (size_t)part * kHiddenDim, (size_t)kHiddenDim * sizeof(float));
        for (int row = 0; row < target_rows; row++)
            std::memcpy(expanded[part].row(prefix_rows + row), source.row(0) + (size_t)part * kHiddenDim, (size_t)kHiddenDim * sizeof(float));
    }
    return true;
}

bool make_joint_hidden(const ncnn::Mat& prefix, const ncnn::Mat& target, ncnn::Mat& joint)
{
    if (!valid_hidden(prefix, prefix.h) || !valid_hidden(target, target.h)
        || prefix.elembits() != 32 || target.elembits() != 32 || prefix.elempack != 1 || target.elempack != 1)
        return false;
    joint.create(kHiddenDim, prefix.h + target.h);
    if (joint.empty())
        return false;
    for (int i = 0; i < prefix.h; i++)
        std::memcpy(joint.row(i), prefix.row(i), (size_t)kHiddenDim * sizeof(float));
    for (int i = 0; i < target.h; i++)
        std::memcpy(joint.row(prefix.h + i), target.row(i), (size_t)kHiddenDim * sizeof(float));
    return true;
}

bool make_modulation_rows(const ncnn::Mat& source, int source_row, int rows, std::array<ncnn::Mat, 4>& expanded)
{
    if (source.dims != 2 || source.w != kModulationDim || source.h != 2 || source.elempack != 1 || source.elembits() != 32 || source_row < 0 || source_row >= source.h || rows <= 0)
        return false;
    for (int part = 0; part < 4; part++)
    {
        expanded[part].create(kHiddenDim, rows);
        if (expanded[part].empty())
            return false;
        for (int row = 0; row < rows; row++)
            std::memcpy(expanded[part].row(row), source.row(source_row) + (size_t)part * kHiddenDim, (size_t)kHiddenDim * sizeof(float));
    }
    return true;
}

bool expand_temb_row(const ncnn::Mat& source, int source_row, int rows, ncnn::Mat& expanded)
{
    if (source.dims != 2 || source.w != kHiddenDim || source.h != 2 || source.elempack != 1 || source.elembits() != 32 || source_row < 0 || source_row >= source.h || rows <= 0)
        return false;
    expanded.create(kHiddenDim, rows);
    if (expanded.empty())
        return false;
    for (int row = 0; row < rows; row++)
        std::memcpy(expanded.row(row), source.row(source_row), (size_t)kHiddenDim * sizeof(float));
    return true;
}

bool copy_rows(const ncnn::Mat& source, int first_row, int rows, int width, ncnn::Mat& output)
{
    if (source.dims != 2 || source.w != width || source.elempack != 1 || source.elembits() != 32 || first_row < 0 || rows <= 0 || first_row + rows > source.h)
        return false;
    output.create(width, rows);
    if (output.empty())
        return false;
    for (int row = 0; row < rows; row++)
        std::memcpy(output.row(row), source.row(first_row + row), (size_t)width * sizeof(float));
    return true;
}

bool crop_square_prefix(const ncnn::Mat& source, int prefix_tokens, ncnn::Mat& output)
{
    if (source.dims != 2 || source.w != source.h || prefix_tokens <= 0 || prefix_tokens > source.h || source.elempack != 1 || source.elembits() != 32)
        return false;
    output.create(prefix_tokens, prefix_tokens);
    if (output.empty())
        return false;
    for (int row = 0; row < prefix_tokens; row++)
        std::memcpy(output.row(row), source.row(row), (size_t)prefix_tokens * sizeof(float));
    return true;
}

bool make_target_mask(int prefix_tokens, int target_tokens, ncnn::Mat& mask)
{
    if (prefix_tokens <= 0 || target_tokens <= 0)
        return false;
    mask.create(prefix_tokens + target_tokens, target_tokens);
    if (mask.empty())
        return false;
    mask.fill(0.f);
    return true;
}

bool build_edit_layout(int text_tokens, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes, int target_tokens, std::vector<int>& image_ids, std::vector<unsigned char>& target_mask, int& condition_tokens)
{
    if (text_tokens <= 0 || text_image_slots.size() != (size_t)text_tokens || image_shapes.size() < 2)
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
    if (target_h <= 0 || target_w <= 0 || target_h * target_w != target_tokens || target_tokens % 4)
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
        while (block < condition_count && used_in_block >= image_shapes[block].height * image_shapes[block].width / 4)
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

bool QwenTransformer::initialize_block_blobs()
{
    if (!models_.transformer_blocks)
        return false;
    const std::vector<ncnn::Blob>& blobs = models_.transformer_blocks->blobs();
    auto find_blob = [&](const std::string& name) {
        for (size_t i = 0; i < blobs.size(); i++)
            if (blobs[i].name == name)
                return (int)i;
        return -1;
    };
    for (int block = 0; block < kTransformerBlocks; block++)
    {
        char prefix[8];
        std::snprintf(prefix, sizeof(prefix), "b%02d", block);
        const std::string p = prefix;
        BlockBlobs& b = block_blobs_[block];
        b.residual = find_blob(p + "_1");
        b.normalized = find_blob(p + "_2");
        b.modulation[0] = find_blob(p + "_10");
        b.modulation[1] = find_blob(p + "_11");
        b.modulation[2] = find_blob(p + "_86");
        b.modulation[3] = find_blob(p + "_87");
        b.cos_q = find_blob(p + "_rotary_cos_q");
        b.cos_k = find_blob(p + "_rotary_cos_k");
        b.sin_q = find_blob(p + "_rotary_sin_q");
        b.sin_k = find_blob(p + "_rotary_sin_k");
        b.mask = find_blob(p + "_in4");
        b.output = find_blob(block == kTransformerBlocks - 1 ? "out0" : p + "_out");
        b.cache_k_in = find_blob(p + "_cache_k_in");
        b.cache_v_in = find_blob(p + "_cache_v_in");
        b.cache_k_out = find_blob(p + "_cache_k_out");
        b.cache_v_out = find_blob(p + "_cache_v_out");
        const bool controlnet = models_.transformer_controlnet != nullptr;
        if (b.residual < 0 || b.normalized < 0 || b.output < 0 || b.mask < 0
            || b.cos_q < 0 || b.cos_k < 0 || b.sin_q < 0 || b.sin_k < 0
            || (!controlnet && (b.cache_k_in < 0 || b.cache_v_in < 0 || b.cache_k_out < 0 || b.cache_v_out < 0))
            || b.modulation[0] < 0 || b.modulation[1] < 0 || b.modulation[2] < 0 || b.modulation[3] < 0)
        {
            fprintf(stderr, "transformer block %02d blob map invalid: residual=%d normalized=%d mod=%d,%d,%d,%d rope=%d,%d,%d,%d mask=%d output=%d cache=%d,%d,%d,%d\n",
                    block, b.residual, b.normalized, b.modulation[0], b.modulation[1], b.modulation[2], b.modulation[3], b.cos_q, b.cos_k, b.sin_q, b.sin_k, b.mask, b.output, b.cache_k_in, b.cache_v_in, b.cache_k_out, b.cache_v_out);
            return false;
        }
    }
    if (models_.transformer_controlnet)
    {
        const std::vector<ncnn::Blob>& control_blobs = models_.transformer_controlnet->blobs();
        auto find_control_blob = [&](const std::string& name) {
            for (size_t i = 0; i < control_blobs.size(); i++)
                if (control_blobs[i].name == name)
                    return (int)i;
            return -1;
        };
        control_net_blobs_.image_input = find_control_blob("control_context");
        control_net_blobs_.image_output = find_control_blob("control_image_features");
        control_net_blobs_.before_input = find_control_blob("control_joint");
        control_net_blobs_.before_output = find_control_blob("control_before_output");
        control_net_blobs_.rope_cos = find_control_blob("control_rope_cos");
        control_net_blobs_.rope_sin = find_control_blob("control_rope_sin");
        control_net_blobs_.mask = find_control_blob("control_mask");
        const char* const mod_suffixes[4] = {"10", "11", "86", "87"};
        for (int i = 0; i < 16; i++)
        {
            char prefix[16];
            char name[64];
            std::snprintf(prefix, sizeof(prefix), "c%02d", i);
            control_net_blobs_.hidden[i] = find_control_blob(std::string(prefix) + "_hidden");
            control_net_blobs_.output[i] = find_control_blob(std::string(prefix) + "_state");
            std::snprintf(name, sizeof(name), "control_after_%02d_output", i);
            control_net_blobs_.after_output[i] = find_control_blob(name);
            for (int j = 0; j < 4; j++)
                control_net_blobs_.modulation[i][j] = find_control_blob(std::string(prefix) + "_" + mod_suffixes[j]);
        }
        if (control_net_blobs_.image_input < 0 || control_net_blobs_.image_output < 0
            || control_net_blobs_.before_input < 0 || control_net_blobs_.before_output < 0
            || control_net_blobs_.rope_cos < 0 || control_net_blobs_.rope_sin < 0
            || control_net_blobs_.mask < 0)
            return false;
        for (int i = 0; i < 16; i++)
        {
            if (control_net_blobs_.hidden[i] < 0 || control_net_blobs_.output[i] < 0
                || control_net_blobs_.after_output[i] < 0)
                return false;
            for (int j = 0; j < 4; j++)
                if (control_net_blobs_.modulation[i][j] < 0)
                    return false;
        }

        control_blobs_.before_input = find_blob("control_before_input");
        control_blobs_.base_input = find_blob("control_base_input");
        control_blobs_.add_before_output = find_blob("control_add_before_output");
        for (int i = 0; i < 16; i++)
        {
            char name[64];
            std::snprintf(name, sizeof(name), "control_after_%02d_input", i);
            control_blobs_.after_input[i] = find_blob(name);
            std::snprintf(name, sizeof(name), "control_add_%02d_output", i);
            control_blobs_.add_output[i] = find_blob(name);
        }
        if (control_blobs_.before_input < 0 || control_blobs_.base_input < 0
            || control_blobs_.add_before_output < 0)
            return false;
        for (int i = 0; i < 16; i++)
            if (control_blobs_.after_input[i] < 0 || control_blobs_.add_output[i] < 0)
                return false;
    }
    return true;
}

bool QwenTransformer::project_text(const ncnn::Mat& text, ncnn::Mat& projected, int output_type) const
{
    if (!models_.transformer_input || !is_matrix(text, kHiddenDim, text_tokens_))
        return false;
    ncnn::Extractor ex = models_.transformer_input->create_extractor();
    if (ex.input("in1", text) != 0 || ex.extract("9", projected, output_type) != 0)
        return false;
    return valid_hidden(projected, text_tokens_);
}

bool QwenTransformer::project_latents(const ncnn::Mat& latents, ncnn::Mat& projected, int output_type) const
{
    if (!models_.transformer_input || !is_matrix(latents, kLatentDim, latents.h))
        return false;
    ncnn::Extractor ex = models_.transformer_input->create_extractor();
    if (ex.input("in0", latents) != 0 || ex.extract("5", projected, output_type) != 0)
        return false;
    return valid_hidden(projected, latents.h);
}

#if NCNN_VULKAN
bool QwenTransformer::project_latents_vulkan(const ncnn::Mat& latents, ncnn::VkMat& projected) const
{
    if (!models_.transformer_input || !is_matrix(latents, kLatentDim, latents.h))
        return false;
    const ncnn::Net& net = *models_.transformer_input;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input("in0", latents) != 0)
        return false;
    ncnn::VkCompute cmd(net.vulkan_device());
    return ex.extract("5", projected, cmd) == 0 && cmd.submit_and_wait() == 0 && !projected.empty() && projected.dims == 2 && projected.w == kHiddenDim && projected.h * projected.elempack == latents.h;
}
#endif

bool QwenTransformer::run_time_condition(float timestep, ncnn::Mat& modulation, ncnn::Mat& temb) const
{
    if (!models_.transformer_input)
        return false;
    ncnn::Mat time(1);
    if (time.empty())
        return false;
    time[0] = timestep;
    ncnn::Extractor ex = models_.transformer_input->create_extractor();
    if (ex.input("in2", time) != 0 || ex.extract("out1", modulation) != 0 || ex.extract("out2", temb) != 0)
        return false;
    return is_matrix(modulation, kModulationDim, 2) && is_matrix(temb, kHiddenDim, 2);
}

bool QwenTransformer::prepare_prefix(const ncnn::Mat& hidden, const ncnn::Mat& modulation, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask)
{
    prefix_ready_ = false;
    std::array<ncnn::Mat, 4> prefix_modulation;
    if (!models_.transformer_blocks || prefix_tokens_ <= 0 || !valid_hidden(hidden, prefix_tokens_)
        || !is_matrix(cos, kRopeDim, prefix_tokens_) || !is_matrix(sin, kRopeDim, prefix_tokens_)
        || !is_matrix(mask, prefix_tokens_, prefix_tokens_)
        || !make_modulation_rows(modulation, 1, prefix_tokens_, prefix_modulation))
        return false;
    if (!initialize_block_blobs())
    {
        fprintf(stderr, "transformer block internal blob metadata is incomplete\n");
        return false;
    }

    prefix_cache_.clear();
    prefix_cache_.resize(kTransformerBlocks);
    prefix_kvcache_allocator_.reset(new ncnn::PoolAllocator());
    decode_kvcache_allocator_.reset(new ncnn::PoolAllocator());
#if NCNN_VULKAN
    prefix_kvcache_vkallocator_.reset();
    decode_kvcache_vkallocator_.reset();
#endif

#if NCNN_VULKAN
    if (config_.use_vulkan_compute)
    {
        const ncnn::Net& net = *models_.transformer_blocks;
        prefix_kvcache_vkallocator_.reset(new ncnn::VkBlobAllocator(net.vulkan_device()));
        decode_kvcache_vkallocator_.reset(new ncnn::VkBlobAllocator(net.vulkan_device()));
        ncnn::VkMat hidden_vk;
        {
            ncnn::VkCompute upload_cmd(net.vulkan_device());
            upload_cmd.record_upload(hidden, hidden_vk, net.opt);
            if (hidden_vk.empty() || upload_cmd.submit_and_wait() != 0)
                return false;
        }
        ncnn::VkMat final_hidden;
        if (!run_blocks_vulkan(hidden_vk, prefix_modulation, cos, sin, mask, true, final_hidden))
            return false;
    }
    else
#endif
    {
        ncnn::Mat final_hidden;
        if (!run_blocks_cpu(hidden, prefix_modulation, cos, sin, mask, true, final_hidden))
            return false;
    }
    // prefill views are gone; persistent kv uses separate allocators
    models_.clear_transformer_workspace();
    prefix_ready_ = true;
    return true;
}

bool QwenTransformer::prepare_controlnet(const ncnn::Mat& control_context, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask)
{
    control_before_.release();
#if NCNN_VULKAN
    control_before_vk_.release();
#endif
    if (!models_.transformer_controlnet
        || !is_matrix(control_context, 129, image_tokens_)
        || !is_matrix(cos, kRopeDim, joint_tokens_) || !is_matrix(sin, kRopeDim, joint_tokens_)
        || !is_matrix(mask, joint_tokens_, joint_tokens_)
        || !valid_hidden(prefix_hidden_, prefix_tokens_) || prefix_hidden_.elembits() != 32)
        return false;
    if (!initialize_block_blobs())
    {
        fprintf(stderr, "ControlNet graph blob map is incomplete\n");
        return false;
    }

    ncnn::Mat image_features;
#if NCNN_VULKAN
    const bool use_vulkan = config_.use_vulkan_compute;
    if (use_vulkan)
    {
        const ncnn::Net& controlnet = *models_.transformer_controlnet;
        ncnn::VkCompute image_cmd(controlnet.vulkan_device());
        ncnn::VkMat context_vk;
        ncnn::VkMat image_features_vk;
        image_cmd.record_upload(control_context, context_vk, controlnet.opt);
        ncnn::Extractor image_ex = controlnet.create_extractor();
        if (context_vk.empty() || image_ex.input(control_net_blobs_.image_input, context_vk) != 0
            || image_ex.extract(control_net_blobs_.image_output, image_features_vk, image_cmd) != 0
            || image_cmd.submit_and_wait() != 0 || image_features_vk.empty())
            return false;

        ncnn::VkCompute download_cmd(controlnet.vulkan_device());
        download_cmd.record_download(image_features_vk, image_features, controlnet.opt);
        if (image_features.empty() || download_cmd.submit_and_wait() != 0)
            return false;
        if (image_features.elempack != 1)
        {
            ncnn::Mat unpacked;
            ncnn::convert_packing(image_features, unpacked, 1, controlnet.opt);
            if (unpacked.empty())
                return false;
            image_features = unpacked;
        }
    }
    else
#endif
    {
        ncnn::Extractor image_ex = models_.transformer_controlnet->create_extractor();
        if (image_ex.input(control_net_blobs_.image_input, control_context) != 0
            || image_ex.extract(control_net_blobs_.image_output, image_features, 0) != 0)
            return false;
    }
    if (image_features.empty() || image_features.dims != 2 || image_features.w != kHiddenDim
        || image_features.h != image_tokens_ || image_features.elempack != 1
        || (image_features.elembits() != 16 && image_features.elembits() != 32))
    {
        fprintf(stderr, "ControlNet image projection failed\n");
        return false;
    }

    ncnn::Mat control_joint;
    control_joint.create(kHiddenDim, joint_tokens_, image_features.elemsize, 1);
    if (control_joint.empty())
        return false;
    std::memset(control_joint.data, 0, control_joint.total() * control_joint.elemsize);
    const size_t row_bytes = (size_t)kHiddenDim * image_features.elemsize;
    for (int i = 0; i < image_tokens_; i++)
        std::memcpy(control_joint.row(target_row_begin_ + i), image_features.row(i), row_bytes);
    image_features.release();

#if NCNN_VULKAN
    if (use_vulkan)
    {
        const ncnn::Net& controlnet = *models_.transformer_controlnet;
        ncnn::VkCompute before_cmd(controlnet.vulkan_device());
        ncnn::VkMat control_joint_vk;
        before_cmd.record_upload(control_joint, control_joint_vk, controlnet.opt);
        ncnn::Extractor before_ex = controlnet.create_extractor();
        if (control_joint_vk.empty()
            || before_ex.input(control_net_blobs_.before_input, control_joint_vk) != 0
            || before_ex.extract(control_net_blobs_.before_output, control_before_vk_, before_cmd) != 0
            || before_cmd.submit_and_wait() != 0 || control_before_vk_.empty()
            || control_before_vk_.dims != 2 || control_before_vk_.w != kHiddenDim
            || control_before_vk_.h * control_before_vk_.elempack != joint_tokens_)
            return false;
    }
    else
#endif
    {
        ncnn::Extractor before_ex = models_.transformer_controlnet->create_extractor();
        if (before_ex.input(control_net_blobs_.before_input, control_joint) != 0
            || before_ex.extract(control_net_blobs_.before_output, control_before_, 0) != 0)
            return false;
    }
    control_joint.release();
#if NCNN_VULKAN
    if (!use_vulkan)
#endif
    {
        if (!valid_hidden(control_before_, joint_tokens_))
        {
            fprintf(stderr, "ControlNet condition projection failed\n");
            return false;
        }
    }

    joint_cos_ = cos;
    joint_sin_ = sin;
    joint_mask_ = mask;
#if NCNN_VULKAN
    if (use_vulkan && !upload_control_static())
        return false;
#endif
    control_ready_ = true;
    return true;
}

#if NCNN_VULKAN
bool QwenTransformer::upload_control_static()
{
    const ncnn::Net& net = *models_.transformer_blocks;
    ncnn::VkCompute cmd(net.vulkan_device());
    cmd.record_upload(joint_cos_, joint_cos_vk_, net.opt);
    cmd.record_upload(joint_sin_, joint_sin_vk_, net.opt);
    ncnn::Mat mask_storage;
    if (net.opt.use_bf16_storage)
        ncnn::cast_float32_to_bfloat16(joint_mask_, mask_storage, net.opt);
    else
        mask_storage = joint_mask_;
    if (control_before_vk_.empty() || joint_cos_vk_.empty() || joint_sin_vk_.empty() || mask_storage.empty())
        return false;
    cmd.record_clone(mask_storage, joint_mask_vk_, net.opt);
    return !joint_mask_vk_.empty() && cmd.submit_and_wait() == 0;
}
#endif

bool QwenTransformer::prepare_text_to_image(const ncnn::Mat& text, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, float first_timestep, const ncnn::Mat& control_context)
{
    edit_ready_ = false;
    prefix_ready_ = false;
    control_ready_ = false;
    const int sequence = text_tokens_ + image_tokens_;
    if (!is_matrix(text, kHiddenDim, text_tokens_) || !is_matrix(cos, kRopeDim, sequence)
        || !is_matrix(sin, kRopeDim, sequence) || !is_matrix(mask, sequence, sequence))
        return false;

    const bool controlnet = models_.transformer_controlnet != nullptr;
    if (controlnet)
    {
        prefix_tokens_ = text_tokens_;
        target_row_begin_ = prefix_tokens_;
        joint_tokens_ = sequence;
        if (!project_text(text, prefix_hidden_, 0))
            return false;
        joint_cos_ = cos;
        joint_sin_ = sin;
        joint_mask_ = mask;
        if (!prepare_controlnet(control_context, joint_cos_, joint_sin_, joint_mask_))
            return false;
        prefix_ready_ = true;
        return true;
    }

    ncnn::Mat prefix_hidden;
    ncnn::Mat modulation;
    ncnn::Mat temb;
    ncnn::Mat prefix_cos;
    ncnn::Mat prefix_sin;
    ncnn::Mat prefix_mask;
    if (!project_text(text, prefix_hidden) || !run_time_condition(first_timestep, modulation, temb)
        || !copy_rows(cos, 0, text_tokens_, kRopeDim, prefix_cos)
        || !copy_rows(sin, 0, text_tokens_, kRopeDim, prefix_sin)
        || !crop_square_prefix(mask, text_tokens_, prefix_mask)
        || !copy_rows(cos, text_tokens_, image_tokens_, kRopeDim, target_cos_)
        || !copy_rows(sin, text_tokens_, image_tokens_, kRopeDim, target_sin_)
        || !make_target_mask(text_tokens_, image_tokens_, target_mask_))
        return false;

    prefix_tokens_ = text_tokens_;
    if (!prepare_prefix(prefix_hidden, modulation, prefix_cos, prefix_sin, prefix_mask))
        return false;
    return true;
}

bool QwenTransformer::run_blocks_cpu(ncnn::Mat hidden, const std::array<ncnn::Mat, 4>& modulation, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, bool prefill, ncnn::Mat& output)
{
    const int sequence = hidden.h * hidden.elempack;
    if (!valid_hidden(hidden, sequence) || !is_matrix(cos, kRopeDim, sequence) || !is_matrix(sin, kRopeDim, sequence)
        || mask.dims != 2 || mask.h != sequence || mask.elempack != 1 || mask.elembits() != 32)
        return false;
    for (int block = 0; block < kTransformerBlocks; block++)
    {
        const BlockBlobs& b = block_blobs_[block];
        ncnn::Extractor ex = models_.transformer_blocks->create_extractor();
        ex.set_kvcache_allocator(prefill ? prefix_kvcache_allocator_.get() : decode_kvcache_allocator_.get());
        ex.set_kvcache_max_seqlen_hint(prefill ? prefix_tokens_ : prefix_tokens_ + image_tokens_);
        if (ex.input(b.residual, hidden) != 0 || ex.input(b.normalized, hidden) != 0)
            return false;
        for (int i = 0; i < 4; i++)
            if (!is_matrix(modulation[i], kHiddenDim, sequence) || ex.input(b.modulation[i], modulation[i]) != 0)
                return false;
        if (ex.input(b.cos_q, cos) != 0 || ex.input(b.cos_k, cos) != 0 || ex.input(b.sin_q, sin) != 0 || ex.input(b.sin_k, sin) != 0 || ex.input(b.mask, mask) != 0)
            return false;
        ncnn::Mat empty;
        const ncnn::Mat& key_cache = prefill ? empty : prefix_cache_[block].key;
        const ncnn::Mat& value_cache = prefill ? empty : prefix_cache_[block].value;
        if (ex.input(b.cache_k_in, key_cache) != 0 || ex.input(b.cache_v_in, value_cache) != 0)
            return false;
        ncnn::Mat next;
        if (ex.extract(b.output, next, 1) != 0 || !valid_hidden(next, sequence))
            return false;
        if (prefill)
        {
            if (ex.extract(b.cache_k_out, prefix_cache_[block].key, 1) != 0 || ex.extract(b.cache_v_out, prefix_cache_[block].value, 1) != 0
                || prefix_cache_[block].key.empty() || prefix_cache_[block].value.empty()
                || prefix_cache_[block].key.h != prefix_tokens_ || prefix_cache_[block].value.h != prefix_tokens_)
                return false;
        }
        hidden = next;
    }
    output = hidden;
    return true;
}

bool QwenTransformer::run_control_net_block_cpu(int block, const ncnn::Mat& hidden,
                                                       const std::array<ncnn::Mat, 4>& modulation,
                                                       const ncnn::Mat& cos, const ncnn::Mat& sin,
                                                       const ncnn::Mat& mask, ncnn::Mat& output,
                                                       ncnn::Mat& hint)
{
    if (block < 0 || block >= 16 || !models_.transformer_controlnet)
        return false;
    const ncnn::Net& net = *models_.transformer_controlnet;
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input(control_net_blobs_.hidden[block], hidden) != 0)
        return false;
    for (int i = 0; i < 4; i++)
        if (ex.input(control_net_blobs_.modulation[block][i], modulation[i]) != 0)
            return false;
    if (ex.input(control_net_blobs_.rope_cos, cos) != 0
        || ex.input(control_net_blobs_.rope_sin, sin) != 0
        || ex.input(control_net_blobs_.mask, mask) != 0
        || ex.extract(control_net_blobs_.output[block], output, 0) != 0
        || ex.extract(control_net_blobs_.after_output[block], hint, 0) != 0)
        return false;
    return valid_hidden(output, joint_tokens_) && valid_hidden(hint, joint_tokens_);
}

bool QwenTransformer::run_control_blocks_cpu(const ncnn::Mat& hidden_input,
                                              const std::array<ncnn::Mat, 4>& modulation,
                                              const ncnn::Mat& cos, const ncnn::Mat& sin,
                                              const ncnn::Mat& mask, ncnn::Mat& output)
{
    if (!control_ready_ || !valid_hidden(hidden_input, joint_tokens_) || hidden_input.elembits() != 32
        || !is_matrix(cos, kRopeDim, joint_tokens_) || !is_matrix(sin, kRopeDim, joint_tokens_)
        || !is_matrix(mask, joint_tokens_, joint_tokens_))
        return false;
    for (int i = 0; i < 4; i++)
        if (!is_matrix(modulation[i], kHiddenDim, joint_tokens_))
            return false;

    const ncnn::Net& net = *models_.transformer_blocks;
    auto run_base_block = [&](int block, const ncnn::Mat& hidden, ncnn::Mat& next) {
        const BlockBlobs& b = block_blobs_[block];
        ncnn::Extractor ex = net.create_extractor();
        if (ex.input(b.residual, hidden) != 0 || ex.input(b.normalized, hidden) != 0)
            return false;
        for (int i = 0; i < 4; i++)
            if (ex.input(b.modulation[i], modulation[i]) != 0)
                return false;
        if (ex.input(b.cos_q, cos) != 0 || ex.input(b.cos_k, cos) != 0
            || ex.input(b.sin_q, sin) != 0 || ex.input(b.sin_k, sin) != 0
            || ex.input(b.mask, mask) != 0)
            return false;
        return ex.extract(b.output, next, 0) == 0
            && valid_hidden(next, joint_tokens_) && next.elembits() == 32;
    };

    ncnn::Extractor before = net.create_extractor();
    ncnn::Mat hidden = hidden_input;
    ncnn::Mat control;
    if (before.input(control_blobs_.before_input, control_before_) != 0
        || before.input(control_blobs_.base_input, hidden) != 0
        || before.extract(control_blobs_.add_before_output, control, 0) != 0
        || !valid_hidden(control, joint_tokens_))
        return false;

    for (int i = 0; i < 16; i++)
    {
        ncnn::Mat control_next;
        ncnn::Mat hint;
        ncnn::Mat base_next;
        if (!run_control_net_block_cpu(i, control, modulation, cos, sin, mask, control_next, hint)
            || !run_base_block(i * 2, hidden, base_next))
            return false;

        ncnn::Extractor add = net.create_extractor();
        ncnn::Mat added;
        if (add.input(block_blobs_[i * 2].output, base_next) != 0
            || add.input(control_blobs_.after_input[i], hint) != 0
            || add.extract(control_blobs_.add_output[i], added, 0) != 0
            || !valid_hidden(added, joint_tokens_))
            return false;
        hidden = added;
        control = control_next;

        ncnn::Mat odd_next;
        if (!run_base_block(i * 2 + 1, hidden, odd_next))
        {
            fprintf(stderr, "base transformer block %d inference failed\n", i * 2 + 1);
            return false;
        }
        hidden = odd_next;
    }
    output = hidden;
    return true;
}

#if NCNN_VULKAN
bool QwenTransformer::run_control_net_block_vulkan(int block, const ncnn::VkMat& hidden,
                                                     const std::array<ncnn::VkMat, 4>& modulation,
                                                     const ncnn::VkMat& cos, const ncnn::VkMat& sin,
                                                     const ncnn::VkMat& mask, ncnn::VkMat& output,
                                                     ncnn::VkMat& hint)
{
    if (block < 0 || block >= 16 || !models_.transformer_controlnet || hidden.empty())
        return false;
    const ncnn::Net& net = *models_.transformer_controlnet;
    ncnn::VkCompute cmd(net.vulkan_device());
    ncnn::Extractor ex = net.create_extractor();
    if (ex.input(control_net_blobs_.hidden[block], hidden) != 0)
    {
        fprintf(stderr, "ControlNet block %d hidden input failed\n", block);
        return false;
    }
    for (int i = 0; i < 4; i++)
        if (ex.input(control_net_blobs_.modulation[block][i], modulation[i]) != 0)
        {
            fprintf(stderr, "ControlNet block %d modulation %d input failed\n", block, i);
            return false;
        }
    if (ex.input(control_net_blobs_.rope_cos, cos) != 0
        || ex.input(control_net_blobs_.rope_sin, sin) != 0
        || ex.input(control_net_blobs_.mask, mask) != 0)
    {
        fprintf(stderr, "ControlNet block %d rope/mask input failed\n", block);
        return false;
    }
    const int state_ret = ex.extract(control_net_blobs_.output[block], output, cmd);
    if (state_ret != 0)
    {
        fprintf(stderr, "ControlNet block %d state extraction failed: %d\n", block, state_ret);
        return false;
    }
    const int hint_ret = ex.extract(control_net_blobs_.after_output[block], hint, cmd);
    if (hint_ret != 0)
    {
        fprintf(stderr, "ControlNet block %d hint extraction failed: %d\n", block, hint_ret);
        return false;
    }
    const int submit_ret = cmd.submit_and_wait();
    if (submit_ret != 0)
    {
        fprintf(stderr, "ControlNet block %d command submit failed: %d\n", block, submit_ret);
        return false;
    }
    return !output.empty() && !hint.empty()
        && output.dims == 2 && output.w == kHiddenDim
        && output.h * output.elempack == joint_tokens_
        && hint.dims == 2 && hint.w == kHiddenDim
        && hint.h * hint.elempack == joint_tokens_;
}

bool QwenTransformer::run_control_blocks_vulkan(const ncnn::VkMat& hidden_input,
                                                 const std::array<ncnn::VkMat, 4>& modulation,
                                                 const ncnn::VkMat& cos, const ncnn::VkMat& sin,
                                                 const ncnn::VkMat& mask, ncnn::VkMat& output)
{
    if (!control_ready_ || hidden_input.empty() || hidden_input.dims != 2
        || hidden_input.w != kHiddenDim || hidden_input.h * hidden_input.elempack != joint_tokens_
        || cos.empty() || sin.empty() || mask.empty())
        return false;
    for (int i = 0; i < 4; i++)
        if (modulation[i].empty())
            return false;

    const ncnn::Net& net = *models_.transformer_blocks;
    auto run_base_block = [&](int block, const ncnn::VkMat& hidden, ncnn::VkMat& next) {
        const BlockBlobs& b = block_blobs_[block];
        ncnn::VkCompute cmd(net.vulkan_device());
        ncnn::Extractor ex = net.create_extractor();
        if (ex.input(b.residual, hidden) != 0 || ex.input(b.normalized, hidden) != 0)
            return false;
        for (int i = 0; i < 4; i++)
            if (ex.input(b.modulation[i], modulation[i]) != 0)
                return false;
        if (ex.input(b.cos_q, cos) != 0 || ex.input(b.cos_k, cos) != 0
            || ex.input(b.sin_q, sin) != 0 || ex.input(b.sin_k, sin) != 0
            || ex.input(b.mask, mask) != 0)
            return false;
        return ex.extract(b.output, next, cmd) == 0 && cmd.submit_and_wait() == 0
            && !next.empty() && next.dims == 2 && next.w == kHiddenDim
            && next.h * next.elempack == joint_tokens_;
    };

    ncnn::VkCompute before_cmd(net.vulkan_device());
    ncnn::Extractor before = net.create_extractor();
    ncnn::VkMat hidden = hidden_input;
    ncnn::VkMat control;
    if (before.input(control_blobs_.before_input, control_before_vk_) != 0
        || before.input(control_blobs_.base_input, hidden) != 0
        || before.extract(control_blobs_.add_before_output, control, before_cmd) != 0
        || before_cmd.submit_and_wait() != 0 || control.empty())
        return false;

    for (int i = 0; i < 16; i++)
    {
        ncnn::VkMat control_next;
        ncnn::VkMat hint;
        ncnn::VkMat base_next;
        if (!run_control_net_block_vulkan(i, control, modulation, cos, sin, mask,
                                          control_next, hint))
        {
            fprintf(stderr, "ControlNet block %d inference failed\n", i);
            return false;
        }
        if (!run_base_block(i * 2, hidden, base_next))
        {
            fprintf(stderr, "base transformer block %d inference failed\n", i * 2);
            return false;
        }
        ncnn::VkCompute add_cmd(net.vulkan_device());
        ncnn::Extractor add = net.create_extractor();
        ncnn::VkMat added;
        if (add.input(block_blobs_[i * 2].output, base_next) != 0
            || add.input(control_blobs_.after_input[i], hint) != 0
            || add.extract(control_blobs_.add_output[i], added, add_cmd) != 0
            || add_cmd.submit_and_wait() != 0 || added.empty())
        {
            fprintf(stderr, "ControlNet hint %d injection failed\n", i);
            return false;
        }
        hidden = added;
        control = control_next;

        ncnn::VkMat odd_next;
        if (!run_base_block(i * 2 + 1, hidden, odd_next))
        {
            fprintf(stderr, "base transformer block %d inference failed\n", i * 2 + 1);
            return false;
        }
        hidden = odd_next;
    }
    output = hidden;
    return true;
}


#endif

#if NCNN_VULKAN
bool QwenTransformer::download_cache(const ncnn::VkMat& source, ncnn::Mat& data, CacheLayout& layout, ncnn::VkCompute& cmd, const ncnn::Option& opt)
{
    if (source.empty() || source.dims != 3 || source.n != 1 || source.offset != 0 || source.elempack != 1 || source.total() > INT_MAX || source.buffer_capacity() != source.total() * source.elemsize)
        return false;
    layout.w = source.w;
    layout.h = source.h;
    layout.c = source.c;
    layout.cstep = source.cstep;

    // spill the opaque cache including its capacity and channel padding
    // clone copies bytes without casting or changing the packing
    ncnn::VkMat raw = source;
    raw.dims = 1;
    raw.w = (int)source.total();
    raw.h = raw.d = raw.c = 1;
    raw.cstep = raw.w;
    // check staging allocation before the buffer-to-host clone recurses
    if (raw.allocator->mappable)
    {
        if (!raw.data->mapped_ptr)
            return false;
        cmd.record_clone(raw, data, opt);
    }
    else
    {
        if (!opt.staging_vkallocator || !opt.staging_vkallocator->mappable)
            return false;
        ncnn::Option staging_opt = opt;
        staging_opt.blob_vkallocator = opt.staging_vkallocator;
        ncnn::VkMat staging;
        cmd.record_clone(raw, staging, staging_opt);
        if (staging.empty() || !staging.data->mapped_ptr)
            return false;
        cmd.record_clone(staging, data, opt);
    }
    return !data.empty();
}

bool QwenTransformer::upload_cache(const ncnn::Mat& data, const CacheLayout& layout, ncnn::VkMat& cache, ncnn::VkCompute& cmd, const ncnn::Option& opt)
{
    if (data.empty() || data.dims != 1 || layout.w <= 0 || layout.h <= 0 || layout.c <= 0 || layout.cstep < (size_t)layout.w * layout.h || data.total() != layout.cstep * layout.c)
        return false;
    cmd.record_clone(data, cache, opt);
    if (cache.empty())
        return false;
    cache.dims = 3;
    cache.w = layout.w;
    cache.h = layout.h;
    cache.d = 1;
    cache.c = layout.c;
    cache.cstep = layout.cstep;
    return true;
}

bool QwenTransformer::run_blocks_vulkan(ncnn::VkMat hidden, const std::array<ncnn::Mat, 4>& modulation, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, bool prefill, ncnn::VkMat& output)
{
    const ncnn::Net& net = *models_.transformer_blocks;
    const int sequence = hidden.h * hidden.elempack;
    if (hidden.empty() || hidden.dims != 2 || hidden.w != kHiddenDim || sequence <= 0
        || !is_matrix(cos, kRopeDim, sequence) || !is_matrix(sin, kRopeDim, sequence)
        || mask.dims != 2 || mask.h != sequence || mask.elempack != 1 || mask.elembits() != 32)
        return false;

    std::array<ncnn::VkMat, 4> modulation_vk;
    ncnn::VkMat cos_vk;
    ncnn::VkMat sin_vk;
    ncnn::VkMat mask_vk;
    {
        ncnn::VkCompute upload_cmd(net.vulkan_device());
        for (int i = 0; i < 4; i++)
        {
            if (!is_matrix(modulation[i], kHiddenDim, sequence))
                return false;
            upload_cmd.record_upload(modulation[i], modulation_vk[i], net.opt);
            if (modulation_vk[i].empty())
                return false;
        }
        upload_cmd.record_upload(cos, cos_vk, net.opt);
        upload_cmd.record_upload(sin, sin_vk, net.opt);
        // sdpa consumes pack1 masks; avoid retaining a second packed copy
        ncnn::Mat mask_storage;
        if (net.opt.use_bf16_storage)
            ncnn::cast_float32_to_bfloat16(mask, mask_storage, net.opt);
        else
            mask_storage = mask;
        if (mask_storage.empty())
            return false;
        upload_cmd.record_clone(mask_storage, mask_vk, net.opt);
        if (cos_vk.empty() || sin_vk.empty() || mask_vk.empty() || upload_cmd.submit_and_wait() != 0)
            return false;
    }
    // release large prefill transfer buffers before retaining host kv caches
    if (prefill && config_.use_kvcache_in_host_memory && net.opt.staging_vkallocator)
        net.opt.staging_vkallocator->clear();

    ncnn::Option cache_opt = net.opt;
    cache_opt.blob_allocator = prefix_kvcache_allocator_.get();
    cache_opt.blob_vkallocator = prefix_kvcache_vkallocator_.get();

    // release each block's temporary activations after its command completes
    for (int block = 0; block < kTransformerBlocks; block++)
    {
        const BlockBlobs& b = block_blobs_[block];
        LayerCache& cache = prefix_cache_[block];
        ncnn::VkCompute cmd(net.vulkan_device());
        ncnn::Extractor ex = net.create_extractor();
        ex.set_kvcache_vkallocator(prefill ? prefix_kvcache_vkallocator_.get() : decode_kvcache_vkallocator_.get());
        ex.set_kvcache_max_seqlen_hint(prefill ? prefix_tokens_ : prefix_tokens_ + image_tokens_);
        if (ex.input(b.residual, hidden) != 0 || ex.input(b.normalized, hidden) != 0)
            return false;
        for (int i = 0; i < 4; i++)
            if (ex.input(b.modulation[i], modulation_vk[i]) != 0)
                return false;
        if (ex.input(b.cos_q, cos_vk) != 0 || ex.input(b.cos_k, cos_vk) != 0 || ex.input(b.sin_q, sin_vk) != 0 || ex.input(b.sin_k, sin_vk) != 0 || ex.input(b.mask, mask_vk) != 0)
            return false;

        ncnn::VkMat key_cache;
        ncnn::VkMat value_cache;
        if (!prefill)
        {
            if (config_.use_kvcache_in_host_memory)
            {
                if (!upload_cache(cache.key, cache.key_layout, key_cache, cmd, cache_opt) || !upload_cache(cache.value, cache.value_layout, value_cache, cmd, cache_opt))
                    return false;
            }
            else
            {
                key_cache = cache.key_vk;
                value_cache = cache.value_vk;
            }
        }
        if (ex.input(b.cache_k_in, key_cache) != 0 || ex.input(b.cache_v_in, value_cache) != 0)
            return false;
        ncnn::VkMat next;
        if (ex.extract(b.output, next, cmd) != 0 || next.empty() || next.dims != 2 || next.w != kHiddenDim || next.h * next.elempack != sequence)
            return false;
        if (prefill)
        {
            if (ex.extract(b.cache_k_out, key_cache, cmd) != 0 || ex.extract(b.cache_v_out, value_cache, cmd) != 0 || key_cache.empty() || value_cache.empty() || key_cache.h != prefix_tokens_ || value_cache.h != prefix_tokens_)
                return false;
            if (config_.use_kvcache_in_host_memory)
            {
                if (!download_cache(key_cache, cache.key, cache.key_layout, cmd, cache_opt) || !download_cache(value_cache, cache.value, cache.value_layout, cmd, cache_opt))
                    return false;
            }
            else
            {
                cache.key_vk = key_cache;
                cache.value_vk = value_cache;
            }
        }
        if (cmd.submit_and_wait() != 0)
            return false;
        hidden = next;
    }
    output = hidden;
    return true;
}
#endif

bool QwenTransformer::run(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise)
{
    return !edit_ready_ && run_target(latents, timestep, noise);
}

bool QwenTransformer::run_edit(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise)
{
    return edit_ready_ && run_target(latents, timestep, noise);
}

bool QwenTransformer::run_target(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise)
{
    if (!prefix_ready_ || !is_matrix(latents, kLatentDim, image_tokens_))
        return false;
    if (models_.transformer_controlnet)
    {
        ncnn::Mat modulation;
        ncnn::Mat temb;
        if (!run_time_condition(timestep, modulation, temb))
            return false;
        return run_control_target(latents, modulation, temb, noise);
    }
    ncnn::Mat modulation;
    ncnn::Mat temb;
    std::array<ncnn::Mat, 4> target_modulation;
    ncnn::Mat target_temb;
    if (!run_time_condition(timestep, modulation, temb)
        || !make_modulation_rows(modulation, 0, image_tokens_, target_modulation)
        || !expand_temb_row(temb, 0, image_tokens_, target_temb))
        return false;

#if NCNN_VULKAN
    if (config_.use_vulkan_compute)
    {
        ncnn::VkMat hidden;
        if (!project_latents_vulkan(latents, hidden))
            return false;
        ncnn::VkMat final_hidden;
        if (!run_blocks_vulkan(hidden, target_modulation, target_cos_, target_sin_, target_mask_, false, final_hidden))
            return false;
        ncnn::Extractor output = models_.transformer_output->create_extractor();
        if (output.input("in0", final_hidden) != 0 || output.input("in1", target_temb) != 0)
            return false;
        return output.extract("out0", noise) == 0 && is_matrix(noise, kLatentDim, image_tokens_);
    }
#endif
    ncnn::Mat hidden;
    if (!project_latents(latents, hidden))
        return false;
    ncnn::Mat final_hidden;
    if (!run_blocks_cpu(hidden, target_modulation, target_cos_, target_sin_, target_mask_, false, final_hidden))
        return false;
    ncnn::Extractor output = models_.transformer_output->create_extractor();
    if (output.input("in0", final_hidden) != 0 || output.input("in1", target_temb) != 0)
        return false;
    return output.extract("out0", noise) == 0 && is_matrix(noise, kLatentDim, image_tokens_);
}

bool QwenTransformer::run_control_target(const ncnn::Mat& latents, const ncnn::Mat& timestep_modulation, const ncnn::Mat& temb, ncnn::Mat& noise)
{
    if (!control_ready_ || !valid_hidden(prefix_hidden_, prefix_tokens_)
        || !is_matrix(timestep_modulation, kModulationDim, 2)
        || !is_matrix(temb, kHiddenDim, 2))
        return false;
    std::array<ncnn::Mat, 4> modulation;
    ncnn::Mat target_temb;
    ncnn::Mat target_hidden;
    ncnn::Mat joint_hidden;
    if (!make_joint_modulation(timestep_modulation, prefix_tokens_, image_tokens_, modulation)
        || !expand_temb_row(temb, 0, image_tokens_, target_temb)
        || !project_latents(latents, target_hidden, 0)
        || !make_joint_hidden(prefix_hidden_, target_hidden, joint_hidden))
        return false;

    ncnn::Mat final_hidden;
#if NCNN_VULKAN
    if (config_.use_vulkan_compute)
    {
        const ncnn::Net& net = *models_.transformer_blocks;
        ncnn::VkMat hidden_vk;
        std::array<ncnn::VkMat, 4> modulation_vk;
        {
            ncnn::VkCompute upload_cmd(net.vulkan_device());
            upload_cmd.record_upload(joint_hidden, hidden_vk, net.opt);
            for (int i = 0; i < 4; i++)
                upload_cmd.record_upload(modulation[i], modulation_vk[i], net.opt);
            if (hidden_vk.empty())
                return false;
            for (int i = 0; i < 4; i++)
                if (modulation_vk[i].empty())
                    return false;
            if (upload_cmd.submit_and_wait() != 0)
                return false;
        }
        ncnn::VkMat final_hidden_vk;
        if (!run_control_blocks_vulkan(hidden_vk, modulation_vk, joint_cos_vk_, joint_sin_vk_, joint_mask_vk_, final_hidden_vk))
            return false;
        ncnn::VkCompute download_cmd(net.vulkan_device());
        download_cmd.record_download(final_hidden_vk, final_hidden, net.opt);
        if (final_hidden.empty() || download_cmd.submit_and_wait() != 0)
            return false;
        if (final_hidden.elempack != 1)
        {
            ncnn::Mat unpacked;
            ncnn::convert_packing(final_hidden, unpacked, 1, net.opt);
            if (unpacked.empty())
                return false;
            final_hidden = unpacked;
        }
        if (final_hidden.elembits() == 16)
        {
            ncnn::Mat fp32;
            ncnn::cast_bfloat16_to_float32(final_hidden, fp32, net.opt);
            if (fp32.empty())
                return false;
            final_hidden = fp32;
        }
        if (!valid_hidden(final_hidden, joint_tokens_) || final_hidden.elembits() != 32)
            return false;
    }
    else
#endif
    {
        if (!run_control_blocks_cpu(joint_hidden, modulation, joint_cos_, joint_sin_, joint_mask_, final_hidden))
            return false;
    }

    ncnn::Mat target_output;
    if (!copy_rows(final_hidden, target_row_begin_, image_tokens_, kHiddenDim, target_output))
        return false;
    ncnn::Extractor output = models_.transformer_output->create_extractor();
    if (output.input("in0", target_output) != 0 || output.input("in1", target_temb) != 0)
        return false;
    return output.extract("out0", noise) == 0 && is_matrix(noise, kLatentDim, image_tokens_);
}

bool QwenTransformer::prepare_edit(const ncnn::Mat& condition_latents, const ncnn::Mat& text, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes, float first_timestep, const ncnn::Mat& control_context)
{
    edit_ready_ = false;
    prefix_ready_ = false;
    control_ready_ = false;
    if (!models_.transformer_input || !is_matrix(condition_latents, kLatentDim, condition_latents.h) || !is_matrix(text, kHiddenDim, text_tokens_))
        return false;
    int condition_tokens = 0;
    std::vector<int> image_ids;
    std::vector<unsigned char> target_mask;
    if (!build_edit_layout(text_tokens_, text_image_slots, image_shapes, image_tokens_, image_ids, target_mask, condition_tokens) || condition_tokens != condition_latents.h)
        return false;
    const int sequence = (int)image_ids.size();
    const int prefix_tokens = sequence - image_tokens_;
    ncnn::Mat edit_cos;
    ncnn::Mat edit_sin;
    ncnn::Mat edit_mask;
    if (prefix_tokens <= 0 || !make_edit_rope(image_ids, image_shapes, edit_cos, edit_sin) || !make_edit_attention_mask(image_ids, edit_mask))
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

    ncnn::Mat combined_hidden;
    {
        ncnn::Extractor input = models_.transformer_input->create_extractor();
        if (input.input("in0", condition_latents) != 0 || input.input("in1", text) != 0 || input.extract("out0", combined_hidden) != 0 || !is_matrix(combined_hidden, kHiddenDim, text_tokens_ + condition_tokens))
            return false;
    }
    ncnn::Mat prefix_hidden(kHiddenDim, prefix_tokens);
    if (prefix_hidden.empty())
        return false;
    for (int row = 0; row < prefix_tokens; row++)
    {
        if (joint_rows_[row] < 0 || joint_rows_[row] >= combined_hidden.h)
            return false;
        std::memcpy(prefix_hidden.row(row), combined_hidden.row(joint_rows_[row]), (size_t)kHiddenDim * sizeof(float));
    }

    ncnn::Mat modulation;
    ncnn::Mat temb;
    ncnn::Mat prefix_cos;
    ncnn::Mat prefix_sin;
    ncnn::Mat prefix_mask;
    if (!run_time_condition(first_timestep, modulation, temb)
        || !copy_rows(edit_cos, 0, prefix_tokens, kRopeDim, prefix_cos)
        || !copy_rows(edit_sin, 0, prefix_tokens, kRopeDim, prefix_sin)
        || !crop_square_prefix(edit_mask, prefix_tokens, prefix_mask)
        || !copy_rows(edit_cos, prefix_tokens, image_tokens_, kRopeDim, target_cos_)
        || !copy_rows(edit_sin, prefix_tokens, image_tokens_, kRopeDim, target_sin_)
        || !make_target_mask(prefix_tokens, image_tokens_, target_mask_))
        return false;

    combined_hidden.release();
    prefix_tokens_ = prefix_tokens;
    target_row_begin_ = prefix_tokens;
    joint_tokens_ = sequence;
    if (models_.transformer_controlnet)
    {
        prefix_hidden_ = prefix_hidden;
        joint_cos_ = edit_cos;
        joint_sin_ = edit_sin;
        joint_mask_ = edit_mask;
        edit_mask.release();
        edit_cos.release();
        edit_sin.release();
        if (!prepare_controlnet(control_context, joint_cos_, joint_sin_, joint_mask_))
            return false;
        edit_ready_ = true;
        prefix_ready_ = true;
        return true;
    }
    edit_mask.release();
    edit_cos.release();
    edit_sin.release();
    if (!prepare_prefix(prefix_hidden, modulation, prefix_cos, prefix_sin, prefix_mask))
        return false;
    edit_ready_ = true;
    return true;
}
}
