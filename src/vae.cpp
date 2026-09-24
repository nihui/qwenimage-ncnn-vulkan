// qwen-image implemented with ncnn library

#include "vae.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>

namespace qwenimage {
namespace {
constexpr int kChannels = 64;
constexpr int kVaeScale = 16;
constexpr int kTilePadLatent = 4;
// blob 211 has receptive field [16x - 61, 16x + 91] after four local downsampling stages
constexpr int kEncoderTilePadLatent = 6;
constexpr uint64_t kVaeReserve = 160ull * 1024 * 1024;
constexpr uint64_t kVaeWorkspaceOverhead = 32ull * 1024 * 1024;

bool vae_global_workspace_fits(int width, int height, uint64_t available_memory)
{
    if (width <= 0 || height <= 0 || width % kVaeScale || height % kVaeScale || available_memory <= kVaeReserve + kVaeWorkspaceOverhead)
        return false;
    const uint64_t tokens = (uint64_t)(width / kVaeScale) * (height / kVaeScale);
    // both wide vae attention heads use a quadratic bf16 matrix
    // measured decoder bottleneck workspace is 150, 908 and 3461 mib at 1024, 2048 and 3072 square
    // this envelope also bounds the encoder global tail and avoids overflowing oversized requests
    return tokens <= (available_memory - kVaeReserve - kVaeWorkspaceOverhead) / (2 * tokens + 24576);
}

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
bool crop_mat(const ncnn::Mat& source, int top, int bottom, int left, int right, ncnn::Mat& destination)
{
    if (source.empty() || source.elempack != 1 || source.elembits() != 32 || source.dims != 3)
        return false;
    if (top < 0 || bottom < 0 || left < 0 || right < 0 || top + bottom >= source.h || left + right >= source.w)
        return false;
    ncnn::copy_cut_border(source, destination, top, bottom, left, right);
    return !destination.empty();
}

#if NCNN_VULKAN
uint64_t get_available_vae_memory(const ncnn::Net& net, bool encoder = false)
{
    const ncnn::VulkanDevice* vkdev = net.vulkan_device();
    const uint64_t budget = (uint64_t)vkdev->get_heap_budget() * 1024 * 1024;
    // heap budget includes this process, so subtract its estimated resident vae weights
    // measured bf16 encoder weights use 154 mib and decoder weights use 494 mib
    // host weights on a shared heap still count against the same budget
    const uint64_t weights = !net.opt.use_weights_in_host_memory || !has_separate_host_heap(vkdev) ? (encoder ? 160ull : 512ull) * 1024 * 1024 : 0;
    return budget > weights ? budget - weights : 0;
}
#endif

class VaeWorkspace
{
public:
    explicit VaeWorkspace(const ncnn::Net& net) : width(0), height(0)
    {
#if NCNN_VULKAN
        if (net.opt.use_vulkan_compute)
        {
            blob.reset(new ncnn::VkBlobAllocator(net.vulkan_device()));
            staging.reset(new ncnn::VkStagingAllocator(net.vulkan_device()));
        }
#endif
    }

    void prepare(int w, int h, const char* output)
    {
        if (width != w || height != h || output_name != output)
            clear();
        width = w;
        height = h;
        output_name = output;
    }

    void clear()
    {
#if NCNN_VULKAN
        if (blob)
        {
            blob->clear();
            staging->clear();
        }
#endif
        width = 0;
        height = 0;
        output_name.clear();
    }

    int width;
    int height;
    std::string output_name;
#if NCNN_VULKAN
    std::unique_ptr<ncnn::VkBlobAllocator> blob;
    std::unique_ptr<ncnn::VkStagingAllocator> staging;
#endif
};

bool extract_blob(const ncnn::Net& net, const RuntimeConfig& config, ModelStage stage, const ncnn::Mat& input, const char* extra_name, const ncnn::Mat* extra, const char* output_name, VaeWorkspace& workspace, ncnn::Mat& output, const char* input_name = "in0")
{
    workspace.prepare(input.w, input.h, output_name);
    ncnn::Extractor extractor = net.create_extractor();
#if NCNN_VULKAN
    if (workspace.blob)
    {
        extractor.set_blob_vkallocator(workspace.blob.get());
        extractor.set_workspace_vkallocator(workspace.blob.get());
        extractor.set_staging_vkallocator(workspace.staging.get());
    }
#endif
    if (extractor.input(input_name, input) != 0 || (extra && extractor.input(extra_name, *extra) != 0))
        return false;
    ncnn::Mat raw;
    if (extractor.extract(output_name, raw) != 0)
        return false;
    output = raw;
    return !output.empty();
}

bool paste_tile(const ncnn::Mat& source, int top, int bottom, int left, int right, int output_x, int output_y, ncnn::Mat& output)
{
    if (source.dims != 3 || source.c != output.c || source.elempack != 1 || source.elembits() != 32)
        return false;
    const int width = source.w - left - right;
    const int height = source.h - top - bottom;
    if (width <= 0 || height <= 0 || top < 0 || bottom < 0 || left < 0 || right < 0 || output_x < 0 || output_y < 0 || output_x + width > output.w || output_y + height > output.h)
        return false;

    for (int c = 0; c < source.c; c++)
    {
        const ncnn::Mat source_channel = source.channel(c);
        ncnn::Mat output_channel = output.channel(c);
        for (int y = 0; y < height; y++)
            std::memcpy(output_channel.row(output_y + y) + output_x, source_channel.row(top + y) + left, (size_t)width * sizeof(float));
    }
    return true;
}

struct VaeTile
{
    int x0;
    int y0;
    int x1;
    int y1;
    int left;
    int top;
    int right;
    int bottom;
};

std::vector<VaeTile> make_tiles(int width, int height, int tile_width, int tile_height, int pad = kTilePadLatent)
{
    std::vector<VaeTile> tiles;
    for (int y = 0; y < height; y += tile_height)
        for (int x = 0; x < width; x += tile_width)
        {
            VaeTile tile;
            tile.x0 = x;
            tile.y0 = y;
            tile.x1 = std::min(width, x + tile_width);
            tile.y1 = std::min(height, y + tile_height);
            tile.left = std::max(0, tile.x0 - pad);
            tile.top = std::max(0, tile.y0 - pad);
            tile.right = std::min(width, tile.x1 + pad);
            tile.bottom = std::min(height, tile.y1 + pad);
            tiles.push_back(tile);
        }
    // group equal crop shapes so their workspace can be reused
    std::stable_sort(tiles.begin(), tiles.end(), [](const VaeTile& a, const VaeTile& b) {
        const int aw = a.right - a.left;
        const int ah = a.bottom - a.top;
        const int bw = b.right - b.left;
        const int bh = b.bottom - b.top;
        if ((int64_t)aw * ah != (int64_t)bw * bh)
            return (int64_t)aw * ah > (int64_t)bw * bh;
        return aw != bw ? aw > bw : ah > bh;
    });
    return tiles;
}

struct TileAxis
{
    int size;
    int count;
    int largest;
    int total;
};

std::vector<TileAxis> tile_axis_candidates(int length, int pad = kTilePadLatent)
{
    std::vector<TileAxis> candidates;
    for (int count = 1; count <= length; count++)
    {
        const int size = (length + count - 1) / count;
        if (!candidates.empty() && candidates.back().size == size)
            continue;
        TileAxis axis = {size, (length + size - 1) / size, 0, 0};
        for (int start = 0; start < length; start += size)
        {
            const int crop = std::min(length, start + size + pad) - std::max(0, start - pad);
            axis.largest = std::max(axis.largest, crop);
            axis.total += crop;
        }
        candidates.push_back(axis);
    }
    return candidates;
}

int largest_vae_tile_crop(int length, int size, int pad)
{
    int largest = 0;
    for (int start = 0; start < length; start += size)
        largest = std::max(largest, std::min(length, start + size + pad) - std::max(0, start - pad));
    return largest;
}

bool vae_tiles_fit(int width, int height, int tile_width, int tile_height, uint64_t available_memory, bool encoder)
{
    if (!vae_global_workspace_fits(width, height, available_memory) || tile_width <= 0 || tile_height <= 0 || tile_width % kVaeScale || tile_height % kVaeScale)
        return false;
    const int pad = encoder ? kEncoderTilePadLatent : kTilePadLatent;
    // explicit tiles need the same actual clipped halo accounting as automatic tiles
    const uint64_t crop_width = largest_vae_tile_crop(width / kVaeScale, tile_width / kVaeScale, pad);
    const uint64_t crop_height = largest_vae_tile_crop(height / kVaeScale, tile_height / kVaeScale, pad);
    const uint64_t pixels = crop_width * crop_height * kVaeScale * kVaeScale;
    return pixels <= (available_memory - kVaeReserve - kVaeWorkspaceOverhead) / (encoder ? 1152 : 4352);
}

bool process_vae_encoder(const ncnn::Net& net, const RuntimeConfig& config, const ncnn::Mat& input, int requested_width, int requested_height, ncnn::Mat& output)
{
    const int width = input.w;
    const int height = input.h;
    const int latent_width = width / kVaeScale;
    const int latent_height = height / kVaeScale;
    const bool automatic = requested_width <= 0 || requested_height <= 0;
    int tile_width = automatic ? width : std::max(kVaeScale, std::min(width, requested_width / kVaeScale * kVaeScale));
    int tile_height = automatic ? height : std::max(kVaeScale, std::min(height, requested_height / kVaeScale * kVaeScale));
    VaeWorkspace workspace(net);
#if NCNN_VULKAN
    if (workspace.blob)
    {
        const uint64_t available_memory = get_available_vae_memory(net, true);
        if (automatic ? !get_optimal_vae_encoder_tile_size(width, height, available_memory, tile_width, tile_height) : !vae_tiles_fit(width, height, tile_width, tile_height, available_memory, true))
        {
            fprintf(stderr, "vae encoder local tiles or global attention cannot fit available gpu memory\n");
            return false;
        }
    }
#endif
    fprintf(stderr, "vae encoder tile size = %d x %d\n", tile_width, tile_height);
    if (!net.opt.use_vulkan_compute && tile_width >= width && tile_height >= height)
        return extract_blob(net, config, ModelStage::VaeEncoder, input, nullptr, nullptr, "out0", workspace, output) && output.dims == 3 && output.w == latent_width && output.h == latent_height && output.c == kChannels;

    // normalize is per pixel across channels, so the frontend is spatially local
    // stitch before the low-resolution residual blocks and global attention, not after blob 296
    ncnn::Mat frontend;
    if (tile_width >= width && tile_height >= height)
    {
        if (!extract_blob(net, config, ModelStage::VaeEncoder, input, nullptr, nullptr, "211", workspace, frontend))
            return false;
    }
    else
    {
        frontend.create(latent_width, latent_height, 768);
        if (frontend.empty())
            return false;
        const std::vector<VaeTile> tiles = make_tiles(latent_width, latent_height, tile_width / kVaeScale, tile_height / kVaeScale, kEncoderTilePadLatent);
        for (const VaeTile& tile : tiles)
        {
            ncnn::Mat input_tile;
            if (!crop_mat(input, tile.top * kVaeScale, height - tile.bottom * kVaeScale, tile.left * kVaeScale, width - tile.right * kVaeScale, input_tile))
                return false;
            ncnn::Mat result;
            if (!extract_blob(net, config, ModelStage::VaeEncoder, input_tile, nullptr, nullptr, "211", workspace, result))
                return false;
            if (result.w != tile.right - tile.left || result.h != tile.bottom - tile.top)
                return false;
            if (!paste_tile(result, tile.y0 - tile.top, tile.bottom - tile.y1, tile.x0 - tile.left, tile.right - tile.x1, tile.x0, tile.y0, frontend))
                return false;
        }
    }
    if (frontend.dims != 3 || frontend.w != latent_width || frontend.h != latent_height || frontend.c != 768)
        return false;
    // release the frontend gpu workspace before allocating the full latent attention workspace
    workspace.clear();
#if NCNN_VULKAN
    if (workspace.blob && !vae_global_workspace_fits(width, height, get_available_vae_memory(net, true)))
    {
        fprintf(stderr, "vae encoder global attention cannot fit available gpu memory\n");
        return false;
    }
#endif
    return extract_blob(net, config, ModelStage::VaeEncoder, frontend, nullptr, nullptr, "out0", workspace, output, "211") && output.dims == 3 && output.w == latent_width && output.h == latent_height && output.c == kChannels;
}

bool process_vae_decoder(const ncnn::Net& net, const RuntimeConfig& config, ModelStage stage, const ncnn::Mat& input, int width, int height, int requested_width, int requested_height, ncnn::Mat& output)
{
    const char* name = "decoder";
    const char* bottleneck = "45";
    const int latent_width = width / kVaeScale;
    const int latent_height = height / kVaeScale;
    const bool automatic = requested_width <= 0 || requested_height <= 0;
    VaeWorkspace workspace(net);
    int tile_width = automatic ? width : std::max(kVaeScale, std::min(width, requested_width / kVaeScale * kVaeScale));
    int tile_height = automatic ? height : std::max(kVaeScale, std::min(height, requested_height / kVaeScale * kVaeScale));
#if NCNN_VULKAN
    if (workspace.blob)
    {
        const uint64_t available_memory = get_available_vae_memory(net);
        if (automatic ? !get_optimal_vae_tile_size(width, height, available_memory, tile_width, tile_height) : !vae_tiles_fit(width, height, tile_width, tile_height, available_memory, false))
        {
            fprintf(stderr, "vae decoder tiles or global attention cannot fit available gpu memory\n");
            return false;
        }
    }
#endif
    if (!net.opt.use_vulkan_compute && (automatic || (requested_width >= width && requested_height >= height)))
    {
        fprintf(stderr, "vae %s tile size = %d x %d\n", name, width, height);
        return extract_blob(net, config, stage, input, nullptr, nullptr, "out0", workspace, output) && output.dims == 3 && output.w == width && output.h == height && output.c == 4;
    }
    // preserve global attention at the original resolution for every tile
    ncnn::Mat attn;
    if (!extract_blob(net, config, stage, input, nullptr, nullptr, bottleneck, workspace, attn))
    {
        fprintf(stderr, "vae %s full-resolution bottleneck failed\n", name);
        return false;
    }
    if (attn.dims != 3 || attn.w != latent_width || attn.h != latent_height || attn.c != 1152)
        return false;
    workspace.clear();

    // refresh the budget after freeing the full-resolution bottleneck workspace
#if NCNN_VULKAN
    if (workspace.blob)
    {
        const uint64_t available_memory = get_available_vae_memory(net);
        if (automatic ? !get_optimal_vae_tile_size(width, height, available_memory, tile_width, tile_height) : !vae_tiles_fit(width, height, tile_width, tile_height, available_memory, false))
        {
            fprintf(stderr, "vae decoder tile cannot fit available gpu memory\n");
            return false;
        }
    }
#endif
    fprintf(stderr, "vae %s tile size = %d x %d\n", name, tile_width, tile_height);
    if (tile_width >= width && tile_height >= height)
        return extract_blob(net, config, stage, input, bottleneck, &attn, "out0", workspace, output) && output.dims == 3 && output.w == width && output.h == height && output.c == 4;

    output.create(width, height, 4);
    if (output.empty())
        return false;
    const std::vector<VaeTile> tiles = make_tiles(latent_width, latent_height, tile_width / kVaeScale, tile_height / kVaeScale);
    for (size_t i = 0; i < tiles.size(); i++)
    {
        const VaeTile& tile = tiles[i];
        ncnn::Mat input_tile;
        ncnn::Mat attn_tile;
        if (!crop_mat(input, tile.top, input.h - tile.bottom, tile.left, input.w - tile.right, input_tile) || !crop_mat(attn, tile.top, latent_height - tile.bottom, tile.left, latent_width - tile.right, attn_tile))
            return false;
        ncnn::Mat result;
        if (!extract_blob(net, config, stage, input_tile, bottleneck, &attn_tile, "out0", workspace, result))
            return false;
        const int output_scale = kVaeScale;
        if (result.w != (tile.right - tile.left) * output_scale || result.h != (tile.bottom - tile.top) * output_scale)
            return false;
        const int top = tile.y0 - tile.top;
        const int bottom = tile.bottom - tile.y1;
        const int left = tile.x0 - tile.left;
        const int right = tile.right - tile.x1;
        if (!paste_tile(result, top * output_scale, bottom * output_scale, left * output_scale, right * output_scale, tile.x0 * output_scale, tile.y0 * output_scale, output))
            return false;
    }
    return true;
}
}

bool get_optimal_vae_tile_size(int width, int height, uint64_t available_memory, int& tile_width, int& tile_height)
{
    tile_width = tile_height = 0;
    if (!vae_global_workspace_fits(width, height, available_memory))
        return false;
    // measured full-pipeline overhead outside the tile workspace is about 130 mib
    // round up to 160 mib for runtime resources and driver variation
    const uint64_t budget = available_memory - kVaeReserve;
    const std::vector<TileAxis> xs = tile_axis_candidates(width / kVaeScale);
    const std::vector<TileAxis> ys = tile_axis_candidates(height / kVaeScale);
    uint64_t best_count = UINT64_MAX;
    uint64_t best_area = UINT64_MAX;
    uint64_t best_memory = UINT64_MAX;
    for (const TileAxis& x : xs)
        for (const TileAxis& y : ys)
        {
            // measured bf16 decoder workspace envelope on rtx 3060 and rx 9060 xt
            // count actual cropped pixels, including clipped four-latent-pixel halos
            const uint64_t pixels = (uint64_t)x.largest * y.largest * kVaeScale * kVaeScale;
            if (pixels > (budget - kVaeWorkspaceOverhead) / 4352)
                continue;
            const uint64_t memory = pixels * 4352 + kVaeWorkspaceOverhead;
            const uint64_t count = (uint64_t)x.count * y.count;
            const uint64_t area = (uint64_t)x.total * y.total;
            if (count > best_count || (count == best_count && area > best_area) || (count == best_count && area == best_area && memory >= best_memory))
                continue;
            best_count = count;
            best_area = area;
            best_memory = memory;
            tile_width = x.size * kVaeScale;
            tile_height = y.size * kVaeScale;
        }
    return best_count != UINT64_MAX;
}

bool get_optimal_vae_encoder_tile_size(int width, int height, uint64_t available_memory, int& tile_width, int& tile_height)
{
    tile_width = tile_height = 0;
    if (!vae_global_workspace_fits(width, height, available_memory))
        return false;
    const uint64_t budget = available_memory - kVaeReserve;
    const std::vector<TileAxis> xs = tile_axis_candidates(width / kVaeScale, kEncoderTilePadLatent);
    const std::vector<TileAxis> ys = tile_axis_candidates(height / kVaeScale, kEncoderTilePadLatent);
    uint64_t best_count = UINT64_MAX;
    uint64_t best_area = UINT64_MAX;
    uint64_t best_memory = UINT64_MAX;
    for (const TileAxis& x : xs)
        for (const TileAxis& y : ys)
        {
            // measured frontend workspace is 281, 1122 and 2487 mib at 512, 1024 and 1536 square
            // bound actual cropped pixels, including the clipped six-latent-pixel halo
            const uint64_t pixels = (uint64_t)x.largest * y.largest * kVaeScale * kVaeScale;
            if (pixels > (budget - kVaeWorkspaceOverhead) / 1152)
                continue;
            const uint64_t memory = pixels * 1152 + kVaeWorkspaceOverhead;
            const uint64_t count = (uint64_t)x.count * y.count;
            const uint64_t area = (uint64_t)x.total * y.total;
            if (count > best_count || (count == best_count && area > best_area) || (count == best_count && area == best_area && memory >= best_memory))
                continue;
            best_count = count;
            best_area = area;
            best_memory = memory;
            tile_width = x.size * kVaeScale;
            tile_height = y.size * kVaeScale;
        }
    return best_count != UINT64_MAX;
}

bool QwenVaeEncoder::encode(const ncnn::Mat& rgba, ncnn::Mat& packed, int tile_width, int tile_height) const
{
    if (rgba.empty() || rgba.dims != 3 || rgba.c != 4 || rgba.elempack != 1 || rgba.elembits() != 32 || rgba.w % kVaeScale || rgba.h % kVaeScale)
        return false;
    ncnn::Mat output;
    if (!process_vae_encoder(net_, config_, rgba, tile_width, tile_height, output))
        return false;
    float norm[kChannels];
    for (int c = 0; c < kChannels; c++)
        norm[c] = 1.f / kStd[c];
    output.substract_mean_normalize(kMean, norm);
    packed.create(kChannels, output.w * output.h);
    if (packed.empty())
        return false;
    for (int c = 0; c < kChannels; c++)
    {
        const float* source = output.channel(c);
        for (int y = 0; y < output.h; y++)
            for (int x = 0; x < output.w; x++)
                packed[((size_t)y * output.w + x) * kChannels + c] = source[(size_t)y * output.w + x];
    }
    return true;
}

bool QwenVaeDecoder::decode(const ncnn::Mat& packed, int width, int height, ncnn::Mat& rgba, int tile_width, int tile_height) const
{
    if (width <= 0 || height <= 0 || width % kVaeScale || height % kVaeScale || packed.empty() || packed.dims != 2 || packed.w != kChannels || packed.h != (int64_t)(width / kVaeScale) * (height / kVaeScale) || packed.elempack != 1 || packed.elembits() != 32)
        return false;
    const int latent_width = width / kVaeScale;
    const int latent_height = height / kVaeScale;
    ncnn::Mat input(latent_width, latent_height, kChannels);
    if (input.empty())
        return false;
    float mean[kChannels];
    for (int c = 0; c < kChannels; c++)
    {
        float* destination = input.channel(c);
        for (int y = 0; y < latent_height; y++)
            for (int x = 0; x < latent_width; x++)
                destination[(size_t)y * latent_width + x] = packed[((size_t)y * latent_width + x) * kChannels + c];
        mean[c] = -kMean[c] / kStd[c];
    }
    input.substract_mean_normalize(mean, kStd);
    return process_vae_decoder(net_, config_, ModelStage::VaeDecoder, input, width, height, tile_width, tile_height, rgba);
}
}
