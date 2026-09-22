// qwen-image implemented with ncnn library

#include "vae.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace qwenimage {
namespace {
constexpr int kChannels = 64;
constexpr int kVaeScale = 16;
constexpr int kTilePadLatent = 16;

const float kMean[kChannels] = {
    0.5126f, 0.7721f, -0.0631f, 1.3506f, -0.7855f, -2.1025f, -0.3458f, 1.3722f,
    1.8873f, -1.7177f, -0.651f, 0.2732f, 0.7562f, -0.6163f, -1.0277f, 3.8363f,
    2.021f, 0.0472f, 0.932f, 2.0087f, 2.4954f, -0.1391f, -1.4249f, 1.8464f,
    -0.5236f, 1.2826f, 3.7046f, -1.3035f, 2.7286f, -1.4518f, -1.9036f, -1.9955f,
    -0.0342f, -1.0265f, -0.7636f, 3.0555f, 0.0746f, -3.0751f, -0.1076f, 1.7376f,
    -1.0914f, -1.9435f, -0.2784f, -1.368f, 0.4809f, -0.4433f, 0.3764f, 0.5729f,
    -2.0595f, 1.096f, -1.326f, -2.0211f, -5.0179f, 0.5275f, 4.0162f, 1.8505f,
    0.3026f, 1.9373f, 1.4937f, 0.2632f, 0.5547f, -1.7121f, -0.1562f, 0.0304f
};
const float kStd[kChannels] = {
    3.2001f, 3.2936f, 3.4321f, 3.0091f, 3.1061f, 4.0379f, 4.0705f, 3.791f,
    3.0785f, 3.65f, 3.9308f, 3.0904f, 2.8778f, 3.7675f, 3.732f, 5.0756f,
    3.2864f, 4.0397f, 3.1317f, 4.0443f, 2.9249f, 3.9454f, 3.0988f, 4.2489f,
    3.4896f, 3.8513f, 3.9323f, 3.4719f, 3.7498f, 4.283f, 3.5694f, 4.2467f,
    3.9037f, 3.2947f, 5.077f, 3.5075f, 3.27f, 3.4767f, 2.8063f, 5.1125f,
    3.5327f, 4.7833f, 3.1286f, 4.1819f, 3.8527f, 3.8312f, 3.5605f, 4.3875f,
    3.9624f, 4.0168f, 3.5643f, 4.055f, 5.5614f, 4.2963f, 4.408f, 3.4959f,
    3.8747f, 3.7608f, 3.5735f, 3.149f, 3.7662f, 3.6746f, 3.4563f, 3.8161f
};
ncnn::Mat make_image_mat(int width, int height, int channels, const std::vector<float>& data)
{
    return ncnn::Mat(width, height, channels, (void*)data.data()).clone();
}

ncnn::Mat make_channel_mat(int width, int height, int channels, const std::vector<float>& data)
{
    return ncnn::Mat(width, height, channels, (void*)data.data()).clone();
}

ncnn::Mat clone_output(const ncnn::Mat& source, const RuntimeConfig& config, ModelStage stage)
{
    ncnn::Option option = make_ncnn_option(config, stage);
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

bool crop_mat(const ncnn::Mat& source, int top, int bottom, int left, int right, ncnn::Mat& destination)
{
    if (source.empty() || source.elempack != 1
        || source.elembits() != 32
        || (source.dims != 3 && source.dims != 4))
        return false;

    const int depth = source.dims == 4 ? source.d : 1;
    const int output_width = source.w - left - right;
    const int output_height = source.h - top - bottom;
    if (depth <= 0 || output_width <= 0 || output_height <= 0
        || top < 0 || bottom < 0 || left < 0 || right < 0)
        return false;

    if (source.dims == 4)
        destination.create(output_width, output_height, depth, source.c,
                           4u, 1);
    else
        destination.create(output_width, output_height, source.c, 4u, 1);
    if (destination.empty())
        return false;

    for (int c = 0; c < source.c; c++)
    {
        const ncnn::Mat source_channel = source.channel(c);
        ncnn::Mat destination_channel = destination.channel(c);
        for (int d = 0; d < depth; d++)
        {
            const ncnn::Mat source_plane = source.dims == 4
                ? source_channel.depth(d) : source_channel;
            ncnn::Mat destination_plane = destination.dims == 4
                ? destination_channel.depth(d) : destination_channel;
            for (int y = 0; y < output_height; y++)
            {
                const float* source_row =
                    source_plane.row(top + y) + left;
                float* destination_row = destination_plane.row(y);
                std::memcpy(destination_row, source_row, (size_t)output_width * sizeof(float));
            }
        }
    }
    return true;
}

bool extract_blob(const ncnn::Net& net, const RuntimeConfig& config, ModelStage stage, const ncnn::Mat& input, const char* extra_name, const ncnn::Mat* extra, const char* output_name, ncnn::Mat& output)
{
    ncnn::Extractor extractor = net.create_extractor();
    if (extractor.input("in0", input) != 0)
        return false;
    if (extra_name != nullptr && extra != nullptr
        && extractor.input(extra_name, *extra) != 0)
        return false;

    ncnn::Mat raw;
    if (extractor.extract(output_name, raw) != 0)
        return false;
    output = clone_output(raw, config, stage);
    return !output.empty();
}

void choose_tile_size(int width, int height, const RuntimeConfig& config, int requested_width, int requested_height, int& tile_width, int& tile_height)
{
    if (requested_width > 0 && requested_height > 0)
    {
        tile_width = std::min(width, requested_width);
        tile_height = std::min(height, requested_height);
    }
    else
    {
        get_optimal_vae_tile_size(width, height, config,
                                  tile_width, tile_height);
    }
}

void normalize_latent(const ncnn::Mat& latent, int width, int height, std::vector<float>& packed)
{
    const float* source = static_cast<const float*>(latent.data);
    const int latent_width = width / kVaeScale;
    const int latent_height = height / kVaeScale;
    packed.resize((size_t)latent_width * latent_height * kChannels);
    for (int y = 0; y < latent_height; y++)
        for (int x = 0; x < latent_width; x++)
            for (int c = 0; c < kChannels; c++)
                packed[((size_t)y * latent_width + x) * kChannels + c] =
                    (source[((size_t)c * latent_height + y)
                            * latent_width + x] - kMean[c]) / kStd[c];
}

bool paste_encoder_tile(const ncnn::Mat& encoder_out, int crop_top, int crop_bottom, int crop_left, int crop_right, int output_x, int output_y, int output_width, int output_height, std::vector<float>& packed)
{
    if ((encoder_out.dims != 3
         && (encoder_out.dims != 4 || encoder_out.d != 1))
        || encoder_out.c != kChannels || encoder_out.elempack != 1
        || encoder_out.elembits() != 32)
        return false;
    const int output_tile_width =
        encoder_out.w - crop_left - crop_right;
    const int output_tile_height =
        encoder_out.h - crop_top - crop_bottom;
    if (output_tile_width <= 0 || output_tile_height <= 0
        || output_x < 0 || output_y < 0
        || output_x + output_tile_width > output_width
        || output_y + output_tile_height > output_height)
        return false;

    const float* source = static_cast<const float*>(encoder_out.data);
    for (int y = 0; y < output_tile_height; y++)
        for (int x = 0; x < output_tile_width; x++)
            for (int c = 0; c < kChannels; c++)
            {
                const float value =
                    source[((size_t)c * encoder_out.h + crop_top + y)
                           * encoder_out.w + crop_left + x];
                packed[((size_t)(output_y + y) * output_width
                         + output_x + x) * kChannels + c] =
                    (value - kMean[c]) / kStd[c];
            }
    return true;
}

bool paste_decoder_tile(const ncnn::Mat& decoder_out, int crop_top, int crop_bottom, int crop_left, int crop_right, int output_x, int output_y, int output_width, int output_height, std::vector<float>& rgba)
{
    if (decoder_out.dims != 3 || decoder_out.c != 4
        || decoder_out.elempack != 1
        || decoder_out.elembits() != 32)
        return false;
    const int output_tile_width =
        decoder_out.w - crop_left - crop_right;
    const int output_tile_height =
        decoder_out.h - crop_top - crop_bottom;
    if (output_tile_width <= 0 || output_tile_height <= 0
        || output_x < 0 || output_y < 0
        || output_x + output_tile_width > output_width
        || output_y + output_tile_height > output_height)
        return false;

    const float* source = static_cast<const float*>(decoder_out.data);
    for (int y = 0; y < output_tile_height; y++)
        for (int x = 0; x < output_tile_width; x++)
            for (int c = 0; c < 4; c++)
                rgba[((size_t)c * output_height + output_y + y)
                     * output_width + output_x + x] =
                    source[((size_t)c * decoder_out.h + crop_top + y)
                           * decoder_out.w + crop_left + x];
    return true;
}

bool encode_tiled(const ncnn::Net& net, const RuntimeConfig& config, const std::vector<float>& rgba, int width, int height, int tile_width, int tile_height, std::vector<float>& packed)
{
    const int latent_width = width / kVaeScale;
    const int latent_height = height / kVaeScale;
    const int latent_tile_width = tile_width / kVaeScale;
    const int latent_tile_height = tile_height / kVaeScale;
    if (latent_tile_width <= 0 || latent_tile_height <= 0)
        return false;

    ncnn::Mat image = make_image_mat(width, height, 4, rgba);

    // Blob 296 is the encoder's post-attention bottleneck residual. Extract
    // it at the original resolution once, then run only the local tail of
    // the graph for each tile.
    ncnn::Mat attn;
    if (!extract_blob(net, config, ModelStage::VaeEncoder, image,
                      nullptr, nullptr, "296", attn))
        return false;
    if (attn.dims != 3 || attn.w != latent_width
        || attn.h != latent_height || attn.c != 768)
        return false;

    packed.assign((size_t)latent_width * latent_height * kChannels, 0.f);
    const int image_pad = kTilePadLatent * kVaeScale;
    const int tiles_w = (width + tile_width - 1) / tile_width;
    const int tiles_h = (height + tile_height - 1) / tile_height;

    for (int ty = 0; ty < tiles_h; ty++)
        for (int tx = 0; tx < tiles_w; tx++)
        {
            const int target_start_x = tx * tile_width;
            const int target_start_y = ty * tile_height;
            const int target_end_x = std::min(width,
                                              (tx + 1) * tile_width);
            const int target_end_y = std::min(height,
                                              (ty + 1) * tile_height);
            const int image_start_x = std::max(0,
                                                target_start_x - image_pad);
            const int image_start_y = std::max(0,
                                                target_start_y - image_pad);
            const int image_end_x = std::min(width,
                                              target_end_x + image_pad);
            const int image_end_y = std::min(height,
                                              target_end_y + image_pad);
            const int latent_start_x = image_start_x / kVaeScale;
            const int latent_start_y = image_start_y / kVaeScale;
            const int latent_end_x = image_end_x / kVaeScale;
            const int latent_end_y = image_end_y / kVaeScale;

            ncnn::Mat image_tile;
            ncnn::Mat attn_tile;
            if (!crop_mat(image, image_start_y, height - image_end_y,
                          image_start_x, width - image_end_x, image_tile)
                || !crop_mat(attn, latent_start_y,
                             latent_height - latent_end_y,
                             latent_start_x, latent_width - latent_end_x,
                             attn_tile))
                return false;

            ncnn::Mat encoder_out;
            if (!extract_blob(net, config, ModelStage::VaeEncoder,
                              image_tile, "296", &attn_tile, "out0",
                              encoder_out))
                return false;

            const int actual_crop_top =
                (target_start_y - image_start_y) / kVaeScale;
            const int actual_crop_bottom =
                (image_end_y - target_end_y) / kVaeScale;
            const int actual_crop_left =
                (target_start_x - image_start_x) / kVaeScale;
            const int actual_crop_right =
                (image_end_x - target_end_x) / kVaeScale;
            if (!paste_encoder_tile(
                    encoder_out, actual_crop_top, actual_crop_bottom,
                    actual_crop_left, actual_crop_right,
                    target_start_x / kVaeScale,
                    target_start_y / kVaeScale,
                    latent_width, latent_height, packed))
                return false;
        }
    return true;
}

bool decode_tiled(const ncnn::Net& net, const RuntimeConfig& config, const std::vector<float>& packed, int width, int height, int tile_width, int tile_height, std::vector<float>& rgba)
{
    const int latent_width = width / kVaeScale;
    const int latent_height = height / kVaeScale;
    const int latent_tile_width = tile_width / kVaeScale;
    const int latent_tile_height = tile_height / kVaeScale;
    if (latent_tile_width <= 0 || latent_tile_height <= 0)
        return false;

    std::vector<float> latent_data((size_t)kChannels
                                   * latent_height * latent_width);
    for (int y = 0; y < latent_height; y++)
        for (int x = 0; x < latent_width; x++)
            for (int c = 0; c < kChannels; c++)
                latent_data[((size_t)c * latent_height + y)
                            * latent_width + x] =
                    packed[((size_t)y * latent_width + x) * kChannels + c]
                    * kStd[c] + kMean[c];

    ncnn::Mat latent = make_channel_mat(latent_width, latent_height,
                                        kChannels, latent_data);

    // Blob 45 is the decoder's post-attention bottleneck residual. Keep its
    // original spatial resolution and execute only the local reconstruction
    // tail for each tile.
    ncnn::Mat attn;
    if (!extract_blob(net, config, ModelStage::VaeDecoder, latent,
                      nullptr, nullptr, "45", attn))
        return false;
    if (attn.dims != 3 || attn.w != latent_width
        || attn.h != latent_height || attn.c != 1152)
        return false;

    rgba.assign((size_t)4 * width * height, 0.f);
    const int tiles_w = (latent_width + latent_tile_width - 1)
                        / latent_tile_width;
    const int tiles_h = (latent_height + latent_tile_height - 1)
                        / latent_tile_height;

    for (int ty = 0; ty < tiles_h; ty++)
        for (int tx = 0; tx < tiles_w; tx++)
        {
            const int target_start_x = tx * latent_tile_width;
            const int target_start_y = ty * latent_tile_height;
            const int target_end_x = std::min(latent_width,
                                              (tx + 1) * latent_tile_width);
            const int target_end_y = std::min(latent_height,
                                              (ty + 1) * latent_tile_height);
            const int latent_start_x = std::max(
                0, target_start_x - kTilePadLatent);
            const int latent_start_y = std::max(
                0, target_start_y - kTilePadLatent);
            const int latent_end_x = std::min(
                latent_width, target_end_x + kTilePadLatent);
            const int latent_end_y = std::min(
                latent_height, target_end_y + kTilePadLatent);

            ncnn::Mat latent_tile;
            ncnn::Mat attn_tile;
            if (!crop_mat(latent, latent_start_y,
                          latent_height - latent_end_y,
                          latent_start_x, latent_width - latent_end_x,
                          latent_tile)
                || !crop_mat(attn, latent_start_y,
                             latent_height - latent_end_y,
                             latent_start_x, latent_width - latent_end_x,
                             attn_tile))
                return false;

            ncnn::Mat decoder_out;
            if (!extract_blob(net, config, ModelStage::VaeDecoder,
                              latent_tile, "45", &attn_tile, "out0",
                              decoder_out))
                return false;

            const int actual_crop_top = target_start_y - latent_start_y;
            const int actual_crop_bottom = latent_end_y - target_end_y;
            const int actual_crop_left = target_start_x - latent_start_x;
            const int actual_crop_right = latent_end_x - target_end_x;
            if (!paste_decoder_tile(
                    decoder_out, actual_crop_top * kVaeScale,
                    actual_crop_bottom * kVaeScale,
                    actual_crop_left * kVaeScale,
                    actual_crop_right * kVaeScale,
                    target_start_x * kVaeScale,
                    target_start_y * kVaeScale,
                    width, height, rgba))
                return false;
        }
    return true;
}
}

bool QwenVaeEncoder::encode(const std::vector<float>& rgba, int width, int height, std::vector<float>& packed, int tile_width, int tile_height) const
{
    if (width <= 0 || height <= 0 || width % kVaeScale
        || height % kVaeScale
        || rgba.size() != (size_t)4 * width * height)
        return false;

    choose_tile_size(width, height, config_, tile_width, tile_height, tile_width, tile_height);
    tile_width = std::max(kVaeScale,
                          std::min(width, (tile_width / kVaeScale)
                                             * kVaeScale));
    tile_height = std::max(kVaeScale,
                           std::min(height, (tile_height / kVaeScale)
                                                * kVaeScale));
    if (tile_width >= width && tile_height >= height)
    {
        ncnn::Extractor extractor = net_.create_extractor();
        ncnn::Mat input = make_image_mat(width, height, 4, rgba);
        ncnn::Mat raw;
        if (extractor.input("in0", input) != 0
            || extractor.extract("out0", raw) != 0)
            return false;
        ncnn::Mat latent = clone_output(raw, config_,
                                        ModelStage::VaeEncoder);
        const int latent_width = width / kVaeScale;
        const int latent_height = height / kVaeScale;
        if (latent.empty()
            || (latent.dims != 3
                && (latent.dims != 4 || latent.d != 1))
            || latent.elempack != 1
            || latent.w != latent_width || latent.h != latent_height
            || latent.c != kChannels)
            return false;
        normalize_latent(latent, width, height, packed);
        return true;
    }

    fprintf(stderr, "vae encoder tile size = %d x %d\n",
            tile_width, tile_height);
    return encode_tiled(net_, config_, rgba, width, height,
                        tile_width, tile_height, packed);
}

bool QwenVaeDecoder::decode(const std::vector<float>& packed, int width, int height, std::vector<float>& rgba, int tile_width, int tile_height) const
{
    if (width <= 0 || height <= 0 || width % kVaeScale
        || height % kVaeScale
        || packed.size() != (size_t)(width / kVaeScale)
                                  * (height / kVaeScale) * kChannels)
        return false;

    choose_tile_size(width, height, config_, tile_width, tile_height, tile_width, tile_height);
    tile_width = std::max(kVaeScale,
                          std::min(width, (tile_width / kVaeScale)
                                             * kVaeScale));
    tile_height = std::max(kVaeScale,
                           std::min(height, (tile_height / kVaeScale)
                                                * kVaeScale));
    if (tile_width >= width && tile_height >= height)
    {
        const int latent_width = width / kVaeScale;
        const int latent_height = height / kVaeScale;
        std::vector<float> latent_data((size_t)kChannels
                                       * latent_height * latent_width);
        for (int y = 0; y < latent_height; y++)
            for (int x = 0; x < latent_width; x++)
                for (int c = 0; c < kChannels; c++)
                    latent_data[((size_t)c * latent_height + y)
                                * latent_width + x] =
                        packed[((size_t)y * latent_width + x) * kChannels + c]
                        * kStd[c] + kMean[c];
        ncnn::Extractor extractor = net_.create_extractor();
        ncnn::Mat input = make_channel_mat(latent_width, latent_height,
                                           kChannels, latent_data);
        ncnn::Mat raw;
        if (extractor.input("in0", input) != 0
            || extractor.extract("out0", raw) != 0)
            return false;
        ncnn::Mat image = clone_output(raw, config_,
                                       ModelStage::VaeDecoder);
        if (image.empty() || image.dims != 3 || image.w != width
            || image.h != height || image.c != 4
            || image.elempack != 1)
            return false;
        const float* source = static_cast<const float*>(image.data);
        rgba.assign(source, source + (size_t)4 * width * height);
        return true;
    }

    fprintf(stderr, "vae decoder tile size = %d x %d\n",
            tile_width, tile_height);
    return decode_tiled(net_, config_, packed, width, height,
                        tile_width, tile_height, rgba);
}
}
