// qwen-image implemented with ncnn library

#include "text_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

namespace qwenimage {
namespace {
constexpr int kHidden = 4096;
constexpr int kHeadDim = 128;
constexpr int kRopeCacheDim = kHeadDim / 2;

bool valid_mat(const ncnn::Mat& value, int width, int height, int dims = 2)
{
    return !value.empty() && value.refcount && value.dims == dims && value.w == width && value.h == height && value.c == 1 && value.elempack == 1 && value.elemsize == 4u;
}

bool make_text_rope(int length, ncnn::Mat& cos, ncnn::Mat& sin)
{
    cos.create(kRopeCacheDim, length, 1);
    sin.create(kRopeCacheDim, length, 1);
    if (cos.empty() || sin.empty())
        return false;
    for (int pos = 0; pos < length; pos++)
    {
        float* cos_row = cos.row(pos);
        float* sin_row = sin.row(pos);
        for (int i = 0; i < kRopeCacheDim; i++)
        {
            const float inv = 1.f / std::pow(5000000.f, (float)(2 * i) / kHeadDim);
            const float angle = pos * inv;
            cos_row[i] = std::cos(angle);
            sin_row[i] = std::sin(angle);
        }
    }
    return true;
}

ncnn::Mat attention_mask_mat(int length, int valid)
{
    ncnn::Mat mask(length, length);
    if (mask.empty())
        return mask;
    mask.fill(-1.0e30f);
    for (int q = 0; q < length; q++)
    {
        float* row = mask.row(q);
        std::fill(row, row + std::min(q + 1, valid), 0.f);
    }
    return mask;
}
}

bool QwenTextEncoder::encode(const ncnn::Mat& input_ids, int valid_input_tokens, const TextEncoderConfig& text_config, ncnn::Mat& output, int& valid_output_tokens) const
{
    if (!models_.text_encoder)
    {
        fprintf(stderr, "text encoder graph is not loaded\n");
        return false;
    }
    const int input_tokens = input_ids.w;
    if (input_tokens <= 0 || !valid_mat(input_ids, input_tokens, 1)
        || text_config.drop_system_tokens < 0
        || valid_input_tokens < text_config.drop_system_tokens
        || valid_input_tokens > input_tokens)
        return false;
    if (!text_config.dynamic_sequence && (text_config.max_input_tokens <= 0 || input_tokens != text_config.max_input_tokens))
        return false;

    ncnn::Mat in_cos;
    ncnn::Mat in_sin;
    if (!make_text_rope(input_tokens, in_cos, in_sin))
        return false;
    ncnn::Mat in_mask = attention_mask_mat(input_tokens, valid_input_tokens);
    if (in_mask.empty())
        return false;

    ncnn::Extractor extractor = models_.text_encoder->create_extractor();
    if (extractor.input("in0", input_ids) != 0
        || extractor.input("in1", in_cos) != 0
        || extractor.input("in2", in_sin) != 0
        || extractor.input("in3", in_mask) != 0)
    {
        fprintf(stderr, "text encoder input failed\n");
        return false;
    }
    if (text_config.multimodal_graph)
    {
        // the edit graph reduces to text-only encoding with zero image and deepstack features
        ncnn::Mat zero_features(kHidden, input_tokens);
        ncnn::Mat image_mask(1, input_tokens);
        if (zero_features.empty() || image_mask.empty())
            return false;
        zero_features.fill(0.f);
        image_mask.fill(0.f);
        if (extractor.input("edit_image_embeds", zero_features) != 0
            || extractor.input("edit_image_mask", image_mask) != 0
            || extractor.input("edit_deep0", zero_features) != 0
            || extractor.input("edit_deep1", zero_features) != 0
            || extractor.input("edit_deep2", zero_features) != 0)
        {
            fprintf(stderr, "multimodal text encoder input failed\n");
            return false;
        }
    }
    ncnn::Mat hidden;
    if (extractor.extract("out0", hidden) != 0)
    {
        fprintf(stderr, "text encoder extraction failed\n");
        return false;
    }
    if (!valid_mat(hidden, kHidden, input_tokens))
    {
        fprintf(stderr, "text encoder shape failed: %dx%dx%d\n", hidden.w, hidden.h, hidden.c);
        return false;
    }

    valid_output_tokens = valid_input_tokens - text_config.drop_system_tokens;
    const int output_tokens = text_config.dynamic_sequence ? valid_output_tokens : text_config.max_input_tokens - text_config.drop_system_tokens;
    if (output_tokens <= 0)
        return false;
    if (text_config.drop_system_tokens == 0 && valid_output_tokens == input_tokens)
    {
        output = hidden;
        return true;
    }

    // row_range does not retain its source, so cropped outputs need owned storage
    ncnn::Mat cropped(kHidden, output_tokens);
    if (!valid_mat(cropped, kHidden, output_tokens))
        return false;
    if (valid_output_tokens < output_tokens)
        cropped.fill(0.f);
    if (valid_output_tokens > 0)
        std::memcpy(cropped.data, hidden.row(text_config.drop_system_tokens), (size_t)valid_output_tokens * kHidden * sizeof(float));
    output = cropped;
    return true;
}

bool QwenTextEncoder::encode_edit(const ncnn::Mat& input_ids, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& attention_mask, const ncnn::Mat& image_embeds, const ncnn::Mat& image_mask, const std::vector<ncnn::Mat>& deepstack, ncnn::Mat& output) const
{
    if (!models_.text_encoder)
    {
        fprintf(stderr, "text encoder graph is not loaded\n");
        return false;
    }
    const int tokens = input_ids.w;
    if (tokens <= 0 || !valid_mat(input_ids, tokens, 1)
        || !valid_mat(cos, kRopeCacheDim, tokens, 3)
        || !valid_mat(sin, kRopeCacheDim, tokens, 3)
        || !valid_mat(attention_mask, tokens, tokens)
        || !valid_mat(image_embeds, kHidden, tokens)
        || !valid_mat(image_mask, 1, tokens)
        || deepstack.size() != 3)
        return false;
    for (const ncnn::Mat& value : deepstack)
        if (!valid_mat(value, kHidden, tokens))
            return false;

    ncnn::Extractor extractor = models_.text_encoder->create_extractor();
    if (extractor.input("in0", input_ids) != 0
        || extractor.input("in1", cos) != 0
        || extractor.input("in2", sin) != 0
        || extractor.input("in3", attention_mask) != 0
        || extractor.input("edit_image_embeds", image_embeds) != 0
        || extractor.input("edit_image_mask", image_mask) != 0
        || extractor.input("edit_deep0", deepstack[0]) != 0
        || extractor.input("edit_deep1", deepstack[1]) != 0
        || extractor.input("edit_deep2", deepstack[2]) != 0)
    {
        fprintf(stderr, "edit text encoder input failed\n");
        return false;
    }
    ncnn::Mat hidden;
    if (extractor.extract("out0", hidden) != 0)
    {
        fprintf(stderr, "edit text encoder extraction failed\n");
        return false;
    }
    if (!valid_mat(hidden, kHidden, tokens))
    {
        fprintf(stderr, "edit text encoder shape failed: %dx%dx%d\n", hidden.w, hidden.h, hidden.c);
        return false;
    }
    output = hidden;
    return true;
}
}
