// qwen-image implemented with ncnn library

#include "dupup3d.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#if NCNN_VULKAN
#include "gpu.h"
#include "pipeline.h"
#endif

namespace qwenimage {

DupUp3D::DupUp3D()
    : out_channels_(0), factor_t_(1), factor_s_(1), first_chunk_(0)
#if NCNN_VULKAN
    , pipeline_(nullptr)
#endif
{
    one_blob_only = true;
    support_inplace = false;

    // The direct implementation deliberately consumes and produces pack1.
    // ncnn will insert layout conversion around this layer when the rest of
    // the VAE is using pack4.
    support_packing = false;
    support_bf16_storage = false;
    support_fp16_storage = false;
    support_int8_storage = false;

    support_vulkan = true;
    support_vulkan_packing = false;
    support_vulkan_any_packing = false;
}

int DupUp3D::load_param(const ncnn::ParamDict& pd)
{
    out_channels_ = pd.get(0, 0);
    factor_t_ = pd.get(1, 1);
    factor_s_ = pd.get(2, 1);
    first_chunk_ = pd.get(3, 0);

    if (out_channels_ <= 0 || factor_t_ <= 0 || factor_s_ <= 0
        || first_chunk_ < 0 || first_chunk_ > 1)
        return -1;

    return 0;
}

int DupUp3D::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
{
    if (bottom_blob.dims != 3 && bottom_blob.dims != 4)
        return -1;
    if (bottom_blob.elempack != 1 || bottom_blob.elemsize != 4u)
        return -1;

    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.dims == 4 ? bottom_blob.d : 1;
    if (in_d != 1)
        return -1;
    const int in_channels = bottom_blob.c;

    const int expanded_channels =
        out_channels_ * factor_t_ * factor_s_ * factor_s_;
    if (expanded_channels % in_channels != 0)
        return -1;
    const int repeats = expanded_channels / in_channels;

    const int out_w = in_w * factor_s_;
    const int out_h = in_h * factor_s_;
    const int out_d = in_d * factor_t_ -
                      (first_chunk_ ? factor_t_ - 1 : 0);
    if (out_d <= 0)
        return -1;

    if (out_d == 1)
        top_blob.create(out_w, out_h, out_channels_, 4u, 1,
                        opt.blob_allocator);
    else
        top_blob.create(out_w, out_h, out_d, out_channels_, 4u, 1,
                        opt.blob_allocator);

    if (top_blob.empty())
        return -100;

    const float* src = static_cast<const float*>(bottom_blob.data);
    float* dst = static_cast<float*>(top_blob.data);

    for (int oc = 0; oc < out_channels_; oc++)
    {
        for (int ot = 0; ot < out_d; ot++)
        {
            for (int oy = 0; oy < out_h; oy++)
            {
                for (int ox = 0; ox < out_w; ox++)
                {
                    const int sub_x = ox % factor_s_;
                    const int sub_y = oy % factor_s_;
                    const int pixel_y = oy / factor_s_;
                    const int pixel_x = ox / factor_s_;
                    const int spatial = in_h * in_w;

                    // The exported ncnn graph represents this repeat as
                    // Tile(repeat_c), which copies the complete channel
                    // block. Reverse its reshape/permute chain directly.
                    const int pixel_channel =
                        oc * factor_s_ * factor_s_
                        + sub_y * factor_s_ + sub_x;
                    std::size_t flat =
                        static_cast<std::size_t>(pixel_channel) * spatial
                        + static_cast<std::size_t>(pixel_y) * in_w + pixel_x;
                    const int crop_channels =
                        first_chunk_ ? (factor_t_ - 1)
                            * out_channels_ * factor_s_ * factor_s_ : 0;
                    flat += static_cast<std::size_t>(crop_channels) * spatial;

                    const std::size_t temporal_stride =
                        static_cast<std::size_t>(out_channels_)
                        * factor_s_ * factor_s_ * spatial;
                    const int graph_temporal =
                        static_cast<int>(flat / temporal_stride);
                    flat %= temporal_stride;
                    const std::size_t channel_stride =
                        static_cast<std::size_t>(factor_s_)
                        * factor_s_ * spatial;
                    const int graph_channel =
                        static_cast<int>(flat / channel_stride);
                    flat %= channel_stride;
                    const int graph_subpixel =
                        static_cast<int>(flat / spatial);
                    const int graph_spatial = static_cast<int>(flat % spatial);

                    const std::size_t permuted_flat =
                        ((static_cast<std::size_t>(graph_channel)
                          * factor_t_ + graph_temporal)
                         * factor_s_ * factor_s_ + graph_subpixel)
                        * spatial + graph_spatial;
                    const int tile_block =
                        static_cast<int>((permuted_flat / spatial) % repeats);
                    const int tile_channel =
                        static_cast<int>(permuted_flat
                                         / (static_cast<std::size_t>(repeats)
                                            * spatial));
                    const std::size_t source_flat =
                        ((static_cast<std::size_t>(tile_block)
                             * in_channels + tile_channel) * spatial
                         + (permuted_flat % spatial))
                        % (static_cast<std::size_t>(in_channels) * spatial);
                    const int src_c =
                        static_cast<int>(source_flat / spatial);
                    const int source_spatial =
                        static_cast<int>(source_flat % spatial);
                    const int src_y = source_spatial / in_w;
                    const int src_x = source_spatial % in_w;

                    const std::size_t src_index =
                        static_cast<std::size_t>(src_c) * bottom_blob.cstep
                        + static_cast<std::size_t>(src_y) * in_w + src_x;
                    const std::size_t dst_index =
                        static_cast<std::size_t>(oc) * top_blob.cstep
                        + static_cast<std::size_t>(ot) * out_h * out_w
                        + static_cast<std::size_t>(oy) * out_w + ox;
                    dst[dst_index] = src[src_index];
                }
            }
        }
    }

    return 0;
}

#if NCNN_VULKAN

static const char k_dupup3d_shader[] = R"glsl(
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
    int repeats;
    int factor_t;
    int factor_s;
    int first_chunk;
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
    const int sub_x = ox - (ox / p.factor_s) * p.factor_s;
    const int sub_y = oy - (oy / p.factor_s) * p.factor_s;
    const int pixel_x = ox / p.factor_s;
    const int pixel_y = oy / p.factor_s;
    const int spatial = p.in_w * p.in_h;

    const int pixel_channel =
        oc * p.factor_s * p.factor_s + sub_y * p.factor_s + sub_x;
    int flat_index = pixel_channel * spatial + pixel_y * p.in_w + pixel_x;
    const int crop_channels = p.first_chunk != 0
        ? (p.factor_t - 1) * p.out_c * p.factor_s * p.factor_s : 0;
    flat_index += crop_channels * spatial;

    const int temporal_stride =
        p.out_c * p.factor_s * p.factor_s * spatial;
    const int graph_temporal = flat_index / temporal_stride;
    flat_index -= graph_temporal * temporal_stride;
    const int channel_stride = p.factor_s * p.factor_s * spatial;
    const int graph_channel = flat_index / channel_stride;
    flat_index -= graph_channel * channel_stride;
    const int graph_subpixel = flat_index / spatial;
    const int graph_spatial = flat_index - graph_subpixel * spatial;

    const int permuted_flat =
        ((graph_channel * p.factor_t + graph_temporal)
         * p.factor_s * p.factor_s + graph_subpixel) * spatial
        + graph_spatial;
    const int tile_block = (permuted_flat / spatial) % p.repeats;
    const int tile_channel =
        permuted_flat / (p.repeats * spatial);
    const int source_flat =
        ((tile_block * p.in_c + tile_channel) * spatial
         + permuted_flat % spatial) % (p.in_c * spatial);
    const int src_c = source_flat / spatial;
    const int source_spatial = source_flat - src_c * spatial;
    const int src_y = source_spatial / p.in_w;
    const int src_x = source_spatial - src_y * p.in_w;

    const int src_index = src_c * p.in_cstep
        + src_y * p.in_w + src_x;
    const int dst_index = oc * p.out_cstep
        + ot * p.out_h * p.out_w + oy * p.out_w + ox;

    const float value = float(buffer_ld1(bottom_blob_data, src_index));
    buffer_st1(top_blob_data, dst_index, value);
}
)glsl";

int DupUp3D::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute)
        return 0;

    std::vector<uint32_t> spirv;
    const int compile_ret =
        ncnn::compile_spirv_module(k_dupup3d_shader, opt, spirv);
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

int DupUp3D::destroy_pipeline(const ncnn::Option&)
{
    delete pipeline_;
    pipeline_ = nullptr;
    return 0;
}

int DupUp3D::forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob, ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if ((bottom_blob.dims != 3 && bottom_blob.dims != 4)
        || bottom_blob.elempack != 1)
        return -1;

    const int in_w = bottom_blob.w;
    const int in_h = bottom_blob.h;
    const int in_d = bottom_blob.dims == 4 ? bottom_blob.d : 1;
    if (in_d != 1)
        return -1;
    const int in_channels = bottom_blob.c;
    const int expanded_channels =
        out_channels_ * factor_t_ * factor_s_ * factor_s_;
    if (expanded_channels % in_channels != 0)
        return -1;

    const int repeats = expanded_channels / in_channels;
    const int out_w = in_w * factor_s_;
    const int out_h = in_h * factor_s_;
    const int out_d = in_d * factor_t_ -
                      (first_chunk_ ? factor_t_ - 1 : 0);
    if (out_d <= 0)
        return -1;

    const size_t out_elemsize = bottom_blob.elemsize / bottom_blob.elempack;
    if (out_d == 1)
        top_blob.create(out_w, out_h, out_channels_, out_elemsize, 1,
                        opt.blob_vkallocator);
    else
        top_blob.create(out_w, out_h, out_d, out_channels_, out_elemsize, 1,
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
    constants[3].i = in_channels;
    constants[4].i = static_cast<int>(bottom_blob.cstep);
    constants[5].i = out_w;
    constants[6].i = out_h;
    constants[7].i = out_d;
    constants[8].i = out_channels_;
    constants[9].i = static_cast<int>(top_blob.cstep);
    constants[10].i = repeats;
    constants[11].i = factor_t_;
    constants[12].i = factor_s_;
    constants[13].i = first_chunk_;

    cmd.record_pipeline(pipeline_, bindings, constants, top_blob);
    return 0;
}

#endif // NCNN_VULKAN

static ncnn::Layer* dupup3d_layer_creator(void*)
{
    return new DupUp3D;
}

void register_dupup3d_layer(ncnn::Net& net)
{
    net.register_custom_layer("DupUp3D", dupup3d_layer_creator);
}

} // namespace qwenimage

