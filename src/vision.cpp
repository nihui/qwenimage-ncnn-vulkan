// qwen-image implemented with ncnn library

#include "vision.h"

#include <cstdio>

namespace qwenimage {
namespace {
constexpr int kPatchDim = 1536;
constexpr int kPositionDim = 1152;
constexpr int kRopeCacheDim = 36;
constexpr int kOutputDim = 4096;

bool valid_mat(const ncnn::Mat& value, int width, int height)
{
    return !value.empty() && value.refcount && value.dims == 2 && value.w == width && value.h == height && value.elempack == 1 && value.elemsize == 4u;
}
}

bool QwenVisionEncoder::encode(const ncnn::Mat& patch_values, const ncnn::Mat& position_values, const ncnn::Mat& cos, const ncnn::Mat& sin, int patch_tokens, VisionFeatures& output) const
{
    if (patch_tokens <= 0 || patch_tokens % 4
        || !valid_mat(patch_values, kPatchDim, patch_tokens)
        || !valid_mat(position_values, kPositionDim, patch_tokens)
        || !valid_mat(cos, kRopeCacheDim, patch_tokens)
        || !valid_mat(sin, kRopeCacheDim, patch_tokens))
        return false;

    ncnn::Extractor extractor = net_.create_extractor();
    if (extractor.input("in0", patch_values) != 0
        || extractor.input("in1", position_values) != 0
        || extractor.input("in2", cos) != 0
        || extractor.input("in3", sin) != 0)
    {
        fprintf(stderr, "vision encoder input failed\n");
        return false;
    }

    VisionFeatures features;
    features.tokens = patch_tokens / 4;
    const char* names[] = {"out0", "out1", "out2", "out3"};
    ncnn::Mat* outputs[] = {&features.image, &features.deep0, &features.deep1, &features.deep2};
    for (int i = 0; i < 4; i++)
    {
        if (extractor.extract(names[i], *outputs[i]) != 0)
        {
            fprintf(stderr, "vision encoder extraction failed for %s\n", names[i]);
            return false;
        }
        if (!valid_mat(*outputs[i], kOutputDim, features.tokens))
        {
            fprintf(stderr, "vision encoder shape failed for %s: %dx%dx%d\n", names[i], outputs[i]->w, outputs[i]->h, outputs[i]->c);
            return false;
        }
    }
    output = features;
    return true;
}
}
