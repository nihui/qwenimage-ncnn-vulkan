// qwen-image implemented with ncnn library

#include "text_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>

#include "mat.h"

namespace qwenimage {
namespace {
constexpr int kHidden = 4096;
constexpr int kHeadDim = 128;
constexpr int kRopeCacheDim = kHeadDim / 2;

ncnn::Mat float_mat(int w, int h, const std::vector<float>& values)
{
    return ncnn::Mat(w, h, (void*)values.data()).clone();
}
ncnn::Mat float_mat_channels(int w, int h, int c, const std::vector<float>& values)
{
    return ncnn::Mat(w, h, c, (void*)values.data()).clone();
}
ncnn::Mat int_mat(const std::vector<int>& values)
{
    return ncnn::Mat((int)values.size(), 1, (void*)values.data()).clone();
}
void make_text_rope(int length, std::vector<float>& cos, std::vector<float>& sin)
{
    cos.resize((size_t)length * kRopeCacheDim);
    sin.resize((size_t)length * kRopeCacheDim);
    for (int pos = 0; pos < length; pos++)
        for (int i = 0; i < kRopeCacheDim; i++)
        {
            const float inv = 1.f / std::pow(5000000.f, (float)(2 * i) / kHeadDim);
            const float angle = pos * inv;
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            // RotaryEmbed consumes one cosine/sine value for each pair.
            cos[(size_t)pos * kRopeCacheDim + i] = c;
            sin[(size_t)pos * kRopeCacheDim + i] = s;
        }
}
ncnn::Mat attention_mask_mat(int length, int valid)
{
    std::vector<float> values((size_t)length * length, -1.0e30f);
    for (int q = 0; q < length; q++)
        for (int k = 0; k <= q && k < valid; k++)
            values[(size_t)q * length + k] = 0.f;
    return float_mat(length, length, values);
}
}

bool QwenTextEncoder::encode(const std::vector<int>& input_ids, int valid_input_tokens, const TextEncoderConfig& text_config, std::vector<float>& output, int& valid_output_tokens) const
{
    if (!models_.text_encoder)
    {
        fprintf(stderr, "text encoder graph is not loaded\\n");
        return false;
    }
    const int input_tokens = (int)input_ids.size();
    if (input_tokens <= 0
        || valid_input_tokens < text_config.drop_system_tokens
        || valid_input_tokens > input_tokens)
        return false;
    if (!text_config.dynamic_sequence
        && (text_config.max_input_tokens <= 0
            || input_tokens != text_config.max_input_tokens))
        return false;

    std::vector<float> cos;
    std::vector<float> sin;
    make_text_rope(input_tokens, cos, sin);
    ncnn::Mat in_ids = int_mat(input_ids);
    ncnn::Mat in_cos = float_mat_channels(kRopeCacheDim, input_tokens, 1, cos);
    ncnn::Mat in_sin = float_mat_channels(kRopeCacheDim, input_tokens, 1, sin);
    ncnn::Mat in_mask = attention_mask_mat(input_tokens, valid_input_tokens);

    ncnn::Extractor extractor = models_.text_encoder->create_extractor();
    if (extractor.input("in0", in_ids) != 0
        || extractor.input("in1", in_cos) != 0
        || extractor.input("in2", in_sin) != 0
        || extractor.input("in3", in_mask) != 0)
    {
        fprintf(stderr, "text encoder input failed\n");
        return false;
    }
    if (text_config.multimodal_graph)
    {
        // The edit graph reduces to the text-only graph when the image
        // branch and the three deepstack branches are zero.
        ncnn::Mat image_embeds(kHidden, input_tokens);
        ncnn::Mat image_mask(1, input_tokens);
        ncnn::Mat deep0(kHidden, input_tokens);
        ncnn::Mat deep1(kHidden, input_tokens);
        ncnn::Mat deep2(kHidden, input_tokens);
        image_embeds.fill(0.f);
        image_mask.fill(0.f);
        deep0.fill(0.f);
        deep1.fill(0.f);
        deep2.fill(0.f);
        if (extractor.input("edit_image_embeds", image_embeds) != 0
            || extractor.input("edit_image_mask", image_mask) != 0
            || extractor.input("edit_deep0", deep0) != 0
            || extractor.input("edit_deep1", deep1) != 0
            || extractor.input("edit_deep2", deep2) != 0)
        {
            fprintf(stderr, "multimodal text encoder input failed\n");
            return false;
        }
    }
    ncnn::Mat raw;
    if (extractor.extract("out0", raw) != 0)
    {
        fprintf(stderr, "text encoder extraction failed\n");
        return false;
    }
    ncnn::Mat hidden = clone_fp32(raw, config_);
    if (hidden.empty() || hidden.elempack != 1
        || hidden.w != kHidden || hidden.h != input_tokens)
    {
        fprintf(stderr, "text encoder shape failed: %dx%dx%d\n",
                hidden.w, hidden.h, hidden.c);
        return false;
    }

    valid_output_tokens = valid_input_tokens - text_config.drop_system_tokens;
    const int output_tokens = text_config.dynamic_sequence
        ? valid_output_tokens
        : text_config.max_input_tokens - text_config.drop_system_tokens;
    if (output_tokens <= 0)
        return false;
    output.assign((size_t)output_tokens * kHidden, 0.f);
    const float* source = static_cast<const float*>(hidden.data);
    const size_t copy_tokens = std::min((size_t)valid_output_tokens, (size_t)output_tokens);
    std::memcpy(output.data(),
                source + (size_t)text_config.drop_system_tokens * kHidden,
                copy_tokens * kHidden * sizeof(float));
    return true;
}
bool QwenTextEncoder::encode_edit(const std::vector<int>& input_ids, const std::vector<float>& cos, const std::vector<float>& sin, const std::vector<float>& attention_mask, const std::vector<float>& image_embeds, const std::vector<float>& image_mask, const std::vector<std::vector<float>>& deepstack, std::vector<float>& output) const
{
    if (!models_.text_encoder)
    {
        fprintf(stderr, "text encoder graph is not loaded\\n");
        return false;
    }
    constexpr int kImageHidden = 4096;
    const int tokens = (int)input_ids.size();
    if (tokens <= 0
        || cos.size() != (size_t)tokens * kHeadDim
        || sin.size() != cos.size()
        || attention_mask.size() != (size_t)tokens * tokens
        || image_embeds.size() != (size_t)tokens * kImageHidden
        || image_mask.size() != (size_t)tokens
        || deepstack.size() != 3)
        return false;
    for (const std::vector<float>& value : deepstack)
        if (value.size() != (size_t)tokens * kImageHidden)
            return false;

    ncnn::Mat in_ids = int_mat(input_ids);
    std::vector<float> cos_cache((size_t)tokens * kRopeCacheDim);
    std::vector<float> sin_cache((size_t)tokens * kRopeCacheDim);
    for (int token = 0; token < tokens; token++)
    {
        std::memcpy(cos_cache.data() + (size_t)token * kRopeCacheDim, cos.data() + (size_t)token * kHeadDim, (size_t)kRopeCacheDim * sizeof(float));
        std::memcpy(sin_cache.data() + (size_t)token * kRopeCacheDim, sin.data() + (size_t)token * kHeadDim, (size_t)kRopeCacheDim * sizeof(float));
    }
    ncnn::Mat in_cos = float_mat_channels(kRopeCacheDim, tokens, 1, cos_cache);
    ncnn::Mat in_sin = float_mat_channels(kRopeCacheDim, tokens, 1, sin_cache);
    ncnn::Mat in_mask = float_mat(tokens, tokens, attention_mask);
    ncnn::Mat in_image = float_mat(kImageHidden, tokens, image_embeds);
    ncnn::Mat in_image_mask = float_mat(1, tokens, image_mask);
    ncnn::Mat in_deep0 = float_mat(kImageHidden, tokens, deepstack[0]);
    ncnn::Mat in_deep1 = float_mat(kImageHidden, tokens, deepstack[1]);
    ncnn::Mat in_deep2 = float_mat(kImageHidden, tokens, deepstack[2]);

    ncnn::Extractor extractor = models_.text_encoder->create_extractor();
    if (extractor.input("in0", in_ids) != 0
        || extractor.input("in1", in_cos) != 0
        || extractor.input("in2", in_sin) != 0
        || extractor.input("in3", in_mask) != 0
        || extractor.input("edit_image_embeds", in_image) != 0
        || extractor.input("edit_image_mask", in_image_mask) != 0
        || extractor.input("edit_deep0", in_deep0) != 0
        || extractor.input("edit_deep1", in_deep1) != 0
        || extractor.input("edit_deep2", in_deep2) != 0)
    {
        fprintf(stderr, "edit text encoder input failed\n");
        return false;
    }
    ncnn::Mat raw;
    if (extractor.extract("out0", raw) != 0)
    {
        fprintf(stderr, "edit text encoder extraction failed\n");
        return false;
    }
    ncnn::Mat hidden = clone_fp32(raw, config_);
    if (hidden.empty() || hidden.elempack != 1 || hidden.w != kImageHidden || hidden.h != tokens)
    {
        fprintf(stderr, "edit text encoder shape failed: %dx%dx%d\n", hidden.w, hidden.h, hidden.c);
        return false;
    }
    const float* source = static_cast<const float*>(hidden.data);
    output.assign(source, source + (size_t)tokens * kImageHidden);
    return true;
}


}
