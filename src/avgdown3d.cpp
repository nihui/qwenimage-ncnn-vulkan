// qwen-image implemented with ncnn library

#include "avgdown3d.h"

#include <cstddef>
#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#include "pipeline.h"
#endif

namespace qwenimage {

AvgDown3D::AvgDown3D()
    : out_channels_(0), factor_t_(1), factor_s_(1)
#if NCNN_VULKAN
    , pipeline_(nullptr)
#endif
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
    support_bf16_storage = false;
    support_fp16_storage = false;
    support_int8_storage = false;
    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
}

int AvgDown3D::load_param(const ncnn::ParamDict& pd)
{
    out_channels_ = pd.get(0, 0);
    factor_t_ = pd.get(1, 1);
    factor_s_ = pd.get(2, 1);
    if (out_channels_ <= 0 || factor_t_ <= 0 || factor_s_ <= 0)
        return -1;
    return 0;
}

int AvgDown3D::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
{
    if ((bottom_blob.dims != 3 && bottom_blob.dims != 4)
        || bottom_blob.elempack != 1 || bottom_blob.elemsize != 4u)
        return -1;

    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.dims == 4 ? bottom_blob.d : 1;
    const int in_c = bottom_blob.c;
    if (in_w <= 0 || in_h <= 0 || in_d <= 0 || in_c <= 0
        || in_h % factor_s_ || in_w % factor_s_)
        return -1;

    const int pad_t = (factor_t_ - in_d % factor_t_) % factor_t_;
    const int padded_d = in_d + pad_t;
    const int out_d = padded_d / factor_t_;
    const int group_size =
        in_c * factor_t_ * factor_s_ * factor_s_ / out_channels_;
    if (group_size <= 0
        || in_c * factor_t_ * factor_s_ * factor_s_
               != out_channels_ * group_size)
        return -1;

    const int out_w = in_w / factor_s_;
    const int out_h = in_h / factor_s_;
    // Keep a singleton temporal dimension when the input graph is 4D.  This
    // matches the exported Qwen VAE branches; a 3D input remains 3D.
    if (bottom_blob.dims == 4 || out_d != 1)
        top_blob.create(out_w, out_h, out_d, out_channels_, 4u, 1,
                        opt.blob_allocator);
    else
        top_blob.create(out_w, out_h, out_channels_, 4u, 1,
                        opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* src = static_cast<const float*>(bottom_blob.data);
    float* dst = static_cast<float*>(top_blob.data);
    const int out_spatial = out_w * out_h;
    for (int oc = 0; oc < out_channels_; oc++)
        for (int ot = 0; ot < out_d; ot++)
            for (int oy = 0; oy < out_h; oy++)
                for (int ox = 0; ox < out_w; ox++)
                {
                    float sum = 0.f;
                    for (int g = 0; g < group_size; g++)
                    {
                        const int e = oc * group_size + g;
                        const int src_c =
                            e / (factor_t_ * factor_s_ * factor_s_);
                        int rem = e % (factor_t_ * factor_s_ * factor_s_);
                        const int sub_t = rem / (factor_s_ * factor_s_);
                        rem %= factor_s_ * factor_s_;
                        const int sub_y = rem / factor_s_;
                        const int sub_x = rem % factor_s_;
                        const int padded_t = ot * factor_t_ + sub_t;
                        if (padded_t < pad_t)
                            continue;
                        const int src_t = padded_t - pad_t;
                        const int src_y = oy * factor_s_ + sub_y;
                        const int src_x = ox * factor_s_ + sub_x;
                        const std::size_t index =
                            static_cast<std::size_t>(src_c) * bottom_blob.cstep
                            + static_cast<std::size_t>(src_t) * in_h * in_w
                            + static_cast<std::size_t>(src_y) * in_w + src_x;
                        sum += src[index];
                    }
                    const std::size_t index =
                        static_cast<std::size_t>(oc) * top_blob.cstep
                        + static_cast<std::size_t>(ot) * out_spatial
                        + static_cast<std::size_t>(oy) * out_w + ox;
                    dst[index] = sum / group_size;
                }
    return 0;
}

#if NCNN_VULKAN

static const char k_avgdown3d_shader[] = R"glsl(
#version 450

layout(binding = 0) readonly buffer bottom_blob
{
    sfp bottom_blob_data[];
};

layout(binding = 1) writeonly buffer top_blob
{
    sfp top_blob_data[];
};

layout(push_constant) uniform parameter
{
    int in_w;
    int in_h;
    int in_d;
    int in_c;
    int in_cstep;
    int out_w;
    int out_h;
    int out_d;
    int out_c;
    int out_cstep;
    int group_size;
    int factor_t;
    int factor_s;
    int pad_t;
} p;

void main()
{
    const int ox = int(gl_GlobalInvocationID.x);
    const int flat_y = int(gl_GlobalInvocationID.y);
    const int oc = int(gl_GlobalInvocationID.z);
    if (ox >= p.out_w || flat_y >= p.out_h * p.out_d || oc >= p.out_c)
        return;

    const int ot = flat_y / p.out_h;
    const int oy = flat_y - ot * p.out_h;
    const int spatial = p.in_w * p.in_h;
    float sum = 0.f;
    for (int g = 0; g < p.group_size; g++)
    {
        const int e = oc * p.group_size + g;
        const int factor_spatial = p.factor_s * p.factor_s;
        const int src_c =
            e / (p.factor_t * factor_spatial);
        int rem = e - src_c * p.factor_t * factor_spatial;
        const int sub_t = rem / factor_spatial;
        rem -= sub_t * factor_spatial;
        const int sub_y = rem / p.factor_s;
        const int sub_x = rem - sub_y * p.factor_s;
        const int padded_t = ot * p.factor_t + sub_t;
        if (padded_t >= p.pad_t)
        {
            const int src_t = padded_t - p.pad_t;
            const int src_y = oy * p.factor_s + sub_y;
            const int src_x = ox * p.factor_s + sub_x;
            const int src_index = src_c * p.in_cstep
                + src_t * spatial + src_y * p.in_w + src_x;
            sum += float(buffer_ld1(bottom_blob_data, src_index));
        }
    }
    const int out_index = oc * p.out_cstep
        + ot * p.out_h * p.out_w + oy * p.out_w + ox;
    buffer_st1(top_blob_data, out_index, sum / float(p.group_size));
}
)glsl";

int AvgDown3D::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute)
        return 0;
    std::vector<uint32_t> spirv;
    const int compile_ret =
        ncnn::compile_spirv_module(k_avgdown3d_shader, opt, spirv);
    if (compile_ret != 0)
        return compile_ret;
    pipeline_ = new ncnn::Pipeline(vkdev);
    pipeline_->set_optimal_local_size_xyz(8, 8, 1);
    const int pipeline_ret = pipeline_->create(
        spirv.data(), spirv.size() * sizeof(uint32_t),
        std::vector<ncnn::vk_specialization_type>());
    if (pipeline_ret != 0)
    {
        delete pipeline_;
        pipeline_ = nullptr;
        return pipeline_ret;
    }
    return 0;
}

int AvgDown3D::destroy_pipeline(const ncnn::Option&)
{
    delete pipeline_;
    pipeline_ = nullptr;
    return 0;
}

int AvgDown3D::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if ((bottom_blob.dims != 3 && bottom_blob.dims != 4)
        || bottom_blob.elempack != 1)
        return -1;

    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.dims == 4 ? bottom_blob.d : 1;
    const int in_c = bottom_blob.c;
    if (in_w <= 0 || in_h <= 0 || in_d <= 0 || in_c <= 0
        || in_h % factor_s_ || in_w % factor_s_)
        return -1;

    const int pad_t = (factor_t_ - in_d % factor_t_) % factor_t_;
    const int out_d = (in_d + pad_t) / factor_t_;
    const int group_size =
        in_c * factor_t_ * factor_s_ * factor_s_ / out_channels_;
    if (group_size <= 0
        || in_c * factor_t_ * factor_s_ * factor_s_
               != out_channels_ * group_size)
        return -1;

    const int out_w = in_w / factor_s_;
    const int out_h = in_h / factor_s_;
    const size_t out_elemsize =
        bottom_blob.elemsize / bottom_blob.elempack;
    if (bottom_blob.dims == 4 || out_d != 1)
        top_blob.create(out_w, out_h, out_d, out_channels_, out_elemsize, 1,
                        opt.blob_vkallocator);
    else
        top_blob.create(out_w, out_h, out_channels_, out_elemsize, 1,
                        opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;
    std::vector<ncnn::vk_constant_type> constants(14);
    constants[0].i = in_w;
    constants[1].i = in_h;
    constants[2].i = in_d;
    constants[3].i = in_c;
    constants[4].i = static_cast<int>(bottom_blob.cstep);
    constants[5].i = out_w;
    constants[6].i = out_h;
    constants[7].i = out_d;
    constants[8].i = out_channels_;
    constants[9].i = static_cast<int>(top_blob.cstep);
    constants[10].i = group_size;
    constants[11].i = factor_t_;
    constants[12].i = factor_s_;
    constants[13].i = pad_t;
    cmd.record_pipeline(pipeline_, bindings, constants, top_blob);
    return 0;
}

#endif // NCNN_VULKAN

static ncnn::Layer* avgdown3d_layer_creator(void*)
{
    return new AvgDown3D;
}

void register_avgdown3d_layer(ncnn::Net& net)
{
    net.register_custom_layer("AvgDown3D", avgdown3d_layer_creator);
}

} // namespace qwenimage
