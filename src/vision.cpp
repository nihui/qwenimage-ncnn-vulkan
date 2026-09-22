// qwen-image implemented with ncnn library

#include "vision.h"

#include <cstring>
#include <cstdio>

namespace qwenimage {
namespace {
constexpr int kPatchDim = 1536;
constexpr int kPositionDim = 1152;
constexpr int kRopeDim = 72;
constexpr int kRopeCacheDim = kRopeDim / 2;
constexpr int kOutputDim = 4096;

ncnn::Mat float_mat(int w, int h, const std::vector<float>& values)
{
    return ncnn::Mat(w, h, (void*)values.data()).clone();
}

ncnn::Mat clone_output(const ncnn::Mat& source, const RuntimeConfig& config)
{
    ncnn::Option option = make_ncnn_option(config, ModelStage::VisionEncoder);
    ncnn::Mat pack1;
    if (source.elempack == 1)
        pack1 = source.clone();
    else
        ncnn::convert_packing(source, pack1, 1, option);
    if (pack1.empty())
        return ncnn::Mat();
    if (pack1.elembits() == 32)
        return pack1;
    if (pack1.elembits() == 16)
    {
        ncnn::Mat fp32;
        ncnn::cast_bfloat16_to_float32(pack1, fp32, option);
        return fp32;
    }
    return ncnn::Mat();
}

bool copy_output(const ncnn::Mat& source, const RuntimeConfig& config, std::vector<float>& target, int expected_tokens)
{
    ncnn::Mat value = clone_output(source, config);
    if (value.empty() || value.elempack != 1
        || value.w != kOutputDim || value.h != expected_tokens
        || value.c != 1)
        return false;
    const float* ptr = static_cast<const float*>(value.data);
    target.assign(ptr, ptr + (size_t)value.w * value.h);
    return true;
}
}

bool QwenVisionEncoder::encode(const std::vector<float>& patch_values, const std::vector<float>& position_values, const std::vector<float>& cos, const std::vector<float>& sin, int patch_tokens, VisionFeatures& output) const
{
    if (patch_tokens <= 0
        || patch_values.size() != (size_t)patch_tokens * kPatchDim
        || position_values.size() != (size_t)patch_tokens * kPositionDim
        || cos.size() != (size_t)patch_tokens * kRopeDim
        || sin.size() != cos.size()
        || patch_tokens % 4)
        return false;

    ncnn::Extractor extractor = net_.create_extractor();
    ncnn::Mat patch = float_mat(kPatchDim, patch_tokens, patch_values);
    ncnn::Mat position = float_mat(kPositionDim, patch_tokens, position_values);
    // RotaryEmbed consumes the compact [sequence, head_dim / 2] cache.
    // Qwen's exported cache contains the same cosine/sine half twice; retain
    // one half and let RotaryEmbed reconstruct the paired coordinates.
    std::vector<float> cos_cache((size_t)patch_tokens * kRopeCacheDim);
    std::vector<float> sin_cache((size_t)patch_tokens * kRopeCacheDim);
    for (int token = 0; token < patch_tokens; token++)
    {
        std::memcpy(cos_cache.data() + (size_t)token * kRopeCacheDim, cos.data() + (size_t)token * kRopeDim, (size_t)kRopeCacheDim * sizeof(float));
        std::memcpy(sin_cache.data() + (size_t)token * kRopeCacheDim, sin.data() + (size_t)token * kRopeDim, (size_t)kRopeCacheDim * sizeof(float));
    }
    ncnn::Mat in_cos = float_mat(kRopeCacheDim, patch_tokens, cos_cache);
    ncnn::Mat in_sin = float_mat(kRopeCacheDim, patch_tokens, sin_cache);
    if (extractor.input("in0", patch) != 0
        || extractor.input("in1", position) != 0
        || extractor.input("in2", in_cos) != 0
        || extractor.input("in3", in_sin) != 0)
    {
        fprintf(stderr, "vision encoder input failed\n");
        return false;
    }

    ncnn::Mat raw[4];
    for (int i = 0; i < 4; i++)
    {
        const std::string name = "out" + std::to_string(i);
        if (extractor.extract(name.c_str(), raw[i]) != 0)
        {
            fprintf(stderr, "vision encoder extraction failed for %s\n", name.c_str());
            return false;
        }
    }
    const int merged_tokens = patch_tokens / 4;
    if (!copy_output(raw[0], config_, output.image, merged_tokens)
        || !copy_output(raw[1], config_, output.deep0, merged_tokens)
        || !copy_output(raw[2], config_, output.deep1, merged_tokens)
        || !copy_output(raw[3], config_, output.deep2, merged_tokens))
        return false;
    output.tokens = merged_tokens;
    return true;
}
}

