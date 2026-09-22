// qwen-image implemented with ncnn library

#include "edit_input.h"

#include "edit_pipeline.h"
#include "image_io.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace qwenimage {
namespace {
constexpr char kSystemPrompt[] = "Comprehend and analyze the provided prompt.";
constexpr int kPatchSize = 16;
constexpr int kMergeSize = 2;
constexpr int kTemporalPatchSize = 2;
constexpr int kVisionPatchDim = 1536;
constexpr int kVisionPositionDim = 1152;
constexpr int kVisionRopeDim = 72;
constexpr int kTextRopeDim = 128;
constexpr int kVisionPositionSide = 48;
constexpr int kVisionPositionCount =
    kVisionPositionSide * kVisionPositionSide * kVisionPositionDim;
constexpr int kImagePadId = 151655;

struct AxisTap
{
    int index = 0;
    double weight = 0.0;
};

enum class ResampleFilter
{
    Lanczos,
    Bicubic,
};

double sinc(double x)
{
    if (std::abs(x) < 1.0e-12)
        return 1.0;
    const double pix = 3.14159265358979323846 * x;
    return std::sin(pix) / pix;
}

double filter_value(double x, ResampleFilter filter)
{
    const double distance = std::abs(x);
    if (filter == ResampleFilter::Lanczos)
    {
        if (distance >= 3.0)
            return 0.0;
        return sinc(distance) * sinc(distance / 3.0);
    }

    // PIL's BICUBIC uses the Catmull-Rom cubic (a=-0.5).
    if (distance < 1.0)
        return (1.5 * distance - 2.5) * distance * distance + 1.0;
    if (distance < 2.0)
        return ((-0.5 * distance + 2.5) * distance - 4.0) * distance + 2.0;
    return 0.0;
}

std::vector<AxisTap> make_axis_taps(int source_size, int target_size, int output_index, ResampleFilter filter)
{
    if (source_size <= 1)
        return {{0, 1.0}};

    const double scale = (double)source_size / target_size;
    const double support = (filter == ResampleFilter::Lanczos ? 3.0 : 2.0)
                         * std::max(1.0, scale);
    const double center = (output_index + 0.5) * scale - 0.5;
    const int first = (int)std::ceil(center - support);
    const int last = (int)std::floor(center + support);
    std::vector<AxisTap> taps;
    taps.reserve((size_t)(last - first + 1));
    double sum = 0.0;
    for (int source_index = first; source_index <= last; source_index++)
    {
        const double distance = (source_index - center)
                              / std::max(1.0, scale);
        const double weight = filter_value(distance, filter)
                            / std::max(1.0, scale);
        if (weight == 0.0)
            continue;
        const int clamped = std::max(0, std::min(source_size - 1, source_index));
        bool merged = false;
        for (AxisTap& tap : taps)
            if (tap.index == clamped)
            {
                tap.weight += weight;
                merged = true;
                break;
            }
        if (!merged)
            taps.push_back({clamped, weight});
        sum += weight;
    }
    if (sum == 0.0)
        return {{std::max(0, std::min(source_size - 1, (int)std::lround(center))), 1.0}};
    for (AxisTap& tap : taps)
        tap.weight /= sum;
    return taps;
}

void resize_rgba(const QwenRgbaImage& source, int width, int height, ResampleFilter filter, std::vector<uint8_t>& output)
{
    if (source.width == width && source.height == height)
    {
        output = source.rgba;
        return;
    }

    const int channels = 4;
    std::vector<float> horizontal((size_t)source.height * width * channels, 0.f);
    for (int y = 0; y < source.height; y++)
        for (int x = 0; x < width; x++)
        {
            const std::vector<AxisTap> taps = make_axis_taps(source.width, width, x, filter);
            for (int c = 0; c < channels; c++)
            {
                double value = 0.0;
                for (const AxisTap& tap : taps)
                    value += tap.weight
                           * source.rgba[((size_t)y * source.width
                                          + tap.index) * channels + c];
                horizontal[((size_t)y * width + x) * channels + c] =
                    (float)value;
            }
        }

    std::vector<float> resized((size_t)height * width * channels, 0.f);
    for (int y = 0; y < height; y++)
    {
        const std::vector<AxisTap> taps = make_axis_taps(source.height, height, y, filter);
        for (int x = 0; x < width; x++)
            for (int c = 0; c < channels; c++)
            {
                double value = 0.0;
                for (const AxisTap& tap : taps)
                    value += tap.weight
                           * horizontal[((size_t)tap.index * width + x)
                                        * channels + c];
                resized[((size_t)y * width + x) * channels + c] =
                    (float)value;
            }
    }

    output.resize((size_t)height * width * channels);
    for (size_t i = 0; i < output.size(); i++)
    {
        const float value = std::max(0.f, std::min(255.f, resized[i]));
        output[i] = (uint8_t)std::lround(value);
    }
}

void rgba_to_rgb_white(const std::vector<uint8_t>& rgba, int width, int height, std::vector<uint8_t>& rgb)
{
    rgb.resize((size_t)width * height * 3);
    for (size_t i = 0; i < (size_t)width * height; i++)
    {
        const int alpha = rgba[i * 4 + 3];
        // Match PIL Image.paste(..., mask=alpha) integer compositing.
        for (int c = 0; c < 3; c++)
            rgb[i * 3 + c] = (uint8_t)(
                (rgba[i * 4 + c] * alpha + 255 * (255 - alpha) + 127)
                / 255);
    }
}

void resize_rgb(const std::vector<uint8_t>& source, int source_width, int source_height, int width, int height, ResampleFilter filter, std::vector<uint8_t>& output)
{
    QwenRgbaImage rgba;
    rgba.width = source_width;
    rgba.height = source_height;
    rgba.rgba.resize((size_t)source_width * source_height * 4);
    for (size_t i = 0; i < (size_t)source_width * source_height; i++)
    {
        rgba.rgba[i * 4 + 0] = source[i * 3 + 0];
        rgba.rgba[i * 4 + 1] = source[i * 3 + 1];
        rgba.rgba[i * 4 + 2] = source[i * 3 + 2];
        rgba.rgba[i * 4 + 3] = 255;
    }
    std::vector<uint8_t> resized;
    resize_rgba(rgba, width, height, filter, resized);
    output.resize((size_t)width * height * 3);
    for (size_t i = 0; i < (size_t)width * height; i++)
    {
        output[i * 3 + 0] = resized[i * 4 + 0];
        output[i * 3 + 1] = resized[i * 4 + 1];
        output[i * 3 + 2] = resized[i * 4 + 2];
    }
}

int round_to_multiple(double value, int multiple)
{
    return std::max(multiple, (int)std::floor(value / multiple + 0.5)
                                * multiple);
}

bool condition_dimensions(int width, int height, int target_area, int& condition_width, int& condition_height)
{
    if (width <= 0 || height <= 0 || target_area <= 0)
        return false;
    const double ratio = (double)width / height;
    condition_width = round_to_multiple(std::sqrt((double)target_area * ratio), 32);
    condition_height = round_to_multiple((double)condition_width / ratio, 32);
    return condition_width >= 32 && condition_height >= 32;
}

void smart_resize_dimensions(int source_width, int source_height, int& width, int& height)
{
    constexpr int factor = 32;
    constexpr int min_pixels = 256 * 256;
    constexpr int max_pixels = 4096 * 4096;
    const int rounded_height = std::max(factor, (int)std::floor((double)source_height / factor + 0.5) * factor);
    const int rounded_width = std::max(factor, (int)std::floor((double)source_width / factor + 0.5) * factor);
    int h = rounded_height;
    int w = rounded_width;
    if ((int64_t)h * w > max_pixels)
    {
        const double beta = std::sqrt((double)source_width * source_height / max_pixels);
        h = std::max(factor, (int)std::floor(source_height / beta / factor) * factor);
        w = std::max(factor, (int)std::floor(source_width / beta / factor) * factor);
    }
    else if ((int64_t)h * w < min_pixels)
    {
        const double beta = std::sqrt((double)min_pixels / (source_width * source_height));
        h = std::max(factor, (int)std::ceil(source_height * beta / factor) * factor);
        w = std::max(factor, (int)std::ceil(source_width * beta / factor) * factor);
    }
    width = w;
    height = h;
}

bool read_f32_file(const std::string& path, size_t count, std::vector<float>& output)
{
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return false;
    output.resize(count);
    const size_t actual = std::fread(output.data(), sizeof(float), count, fp);
    std::fclose(fp);
    return actual == count;
}

bool make_position_table(const std::string& path, std::vector<float>& position_table)
{
    return read_f32_file(path, kVisionPositionCount, position_table);
}

void append_vision_image(const std::vector<uint8_t>& rgb, int width, int height, const std::vector<float>& position_table, std::vector<float>& patch_values, std::vector<float>& position_values, std::vector<float>& vision_cos, std::vector<float>& vision_sin, int& patch_tokens)
{
    const int grid_h = height / kPatchSize;
    const int grid_w = width / kPatchSize;
    const int image_tokens = grid_h * grid_w;
    const size_t patch_offset = patch_values.size();
    patch_values.resize(patch_offset
                        + (size_t)image_tokens * kVisionPatchDim);
    position_values.resize(position_values.size()
                           + (size_t)image_tokens * kVisionPositionDim);
    vision_cos.resize(vision_cos.size()
                      + (size_t)image_tokens * kVisionRopeDim);
    vision_sin.resize(vision_sin.size()
                      + (size_t)image_tokens * kVisionRopeDim);

    const double rope_theta = 10000.0;
    const int rope_axis_dim = 36;
    for (int token = 0; token < image_tokens; token++)
    {
        const int block_width = grid_w / kMergeSize;
        const int in_col = token % kMergeSize;
        const int in_row = (token / kMergeSize) % kMergeSize;
        const int block_col = (token / (kMergeSize * kMergeSize))
                            % block_width;
        const int block_row = token / (kMergeSize * kMergeSize * block_width);
        const int row = block_row * kMergeSize + in_row;
        const int col = block_col * kMergeSize + in_col;

        // Patch order is [block-row, block-col, 2x2 merge, channel,
        // temporal duplicate, patch-row, patch-col].
        float* patch = patch_values.data() + patch_offset
                     + (size_t)token * kVisionPatchDim;
        int value_index = 0;
        for (int channel = 0; channel < 3; channel++)
            for (int temporal = 0; temporal < kTemporalPatchSize; temporal++)
                for (int py = 0; py < kPatchSize; py++)
                    for (int px = 0; px < kPatchSize; px++)
                    {
                        const int y = row * kPatchSize + py;
                        const int x = col * kPatchSize + px;
                        const float value =
                            rgb[((size_t)y * width + x) * 3 + channel]
                            / 127.5f - 1.f;
                        patch[value_index++] = value;
                    }

        float* position = position_values.data()
                        + position_values.size() - (size_t)image_tokens
                          * kVisionPositionDim
                        + (size_t)token * kVisionPositionDim;
        for (int c = 0; c < kVisionPositionDim; c++)
        {
            // Bilinear interpolation with align_corners=True from the
            // 48x48 learned table to the current patch grid.
            const double source_y = grid_h <= 1 ? 0.0
                : (double)row * (kVisionPositionSide - 1) / (grid_h - 1);
            const double source_x = grid_w <= 1 ? 0.0
                : (double)col * (kVisionPositionSide - 1) / (grid_w - 1);
            const int y0 = std::max(0, std::min(kVisionPositionSide - 1,
                                                (int)std::floor(source_y)));
            const int x0 = std::max(0, std::min(kVisionPositionSide - 1,
                                                (int)std::floor(source_x)));
            const int y1 = std::min(kVisionPositionSide - 1, y0 + 1);
            const int x1 = std::min(kVisionPositionSide - 1, x0 + 1);
            const float fy = (float)(source_y - std::floor(source_y));
            const float fx = (float)(source_x - std::floor(source_x));
            const float* p00 = position_table.data()
                + ((y0 * kVisionPositionSide + x0)
                   * kVisionPositionDim + c);
            const float* p01 = position_table.data()
                + ((y0 * kVisionPositionSide + x1)
                   * kVisionPositionDim + c);
            const float* p10 = position_table.data()
                + ((y1 * kVisionPositionSide + x0)
                   * kVisionPositionDim + c);
            const float* p11 = position_table.data()
                + ((y1 * kVisionPositionSide + x1)
                   * kVisionPositionDim + c);
            position[c] = (1.f - fy) * ((1.f - fx) * *p00 + fx * *p01)
                        + fy * ((1.f - fx) * *p10 + fx * *p11);
        }

        float* cos = vision_cos.data()
                   + vision_cos.size() - (size_t)image_tokens
                     * kVisionRopeDim + (size_t)token * kVisionRopeDim;
        float* sin = vision_sin.data()
                   + vision_sin.size() - (size_t)image_tokens
                     * kVisionRopeDim + (size_t)token * kVisionRopeDim;
        for (int i = 0; i < 18; i++)
        {
            const double inv = 1.0 / std::pow(
                rope_theta, (double)(2 * i) / rope_axis_dim);
            const double ah = row * inv;
            const double aw = col * inv;
            cos[i] = cos[i + 36] = (float)std::cos(ah);
            sin[i] = sin[i + 36] = (float)std::sin(ah);
            cos[i + 18] = cos[i + 54] = (float)std::cos(aw);
            sin[i + 18] = sin[i + 54] = (float)std::sin(aw);
        }
    }
    patch_tokens += image_tokens;
}

void make_text_positions(const std::vector<int>& ids, const std::vector<std::pair<int, int>>& grids, std::vector<float>& cos, std::vector<float>& sin, int& image_slot_count)
{
    const int tokens = (int)ids.size();
    std::vector<int> pos0(tokens, 0), pos1(tokens, 0), pos2(tokens, 0);
    int current = 0;
    int image_index = 0;
    int cursor = 0;
    image_slot_count = 0;
    while (cursor < tokens)
    {
        if (ids[cursor] != kImagePadId)
        {
            const int begin = cursor;
            while (cursor < tokens && ids[cursor] != kImagePadId)
                cursor++;
            for (int i = begin; i < cursor; i++)
                pos0[i] = pos1[i] = pos2[i] = current + i - begin;
            current += cursor - begin;
            continue;
        }

        const int begin = cursor;
        while (cursor < tokens && ids[cursor] == kImagePadId)
            cursor++;
        const int count = cursor - begin;
        if (image_index >= (int)grids.size())
            continue;
        const int grid_h = grids[image_index].first;
        const int grid_w = grids[image_index].second;
        const int h = grid_h / kMergeSize;
        const int w = grid_w / kMergeSize;
        int row = 0;
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                if (row >= count)
                    break;
                pos0[begin + row] = current;
                pos1[begin + row] = current + y;
                pos2[begin + row] = current + x;
                row++;
            }
        current += std::max(grid_h, grid_w) / kMergeSize;
        image_slot_count += count;
        image_index++;
    }

    cos.resize((size_t)tokens * kTextRopeDim);
    sin.resize(cos.size());
    constexpr int mrope_section[3] = {24, 20, 20};
    for (int token = 0; token < tokens; token++)
        for (int i = 0; i < 64; i++)
        {
            const double inv = 1.0 / std::pow(
                5000000.0, (double)i / 64.0);
            int position = pos0[token];
            if (i >= 1 && i < mrope_section[1] * 3 && i % 3 == 1)
                position = pos1[token];
            else if (i >= 2 && i < mrope_section[2] * 3 && i % 3 == 2)
                position = pos2[token];
            const float c = (float)std::cos(position * inv);
            const float s = (float)std::sin(position * inv);
            cos[(size_t)token * kTextRopeDim + i] = c;
            cos[(size_t)token * kTextRopeDim + i + 64] = c;
            sin[(size_t)token * kTextRopeDim + i] = s;
            sin[(size_t)token * kTextRopeDim + i + 64] = s;
        }
}

void make_causal_mask(int tokens, std::vector<float>& mask)
{
    mask.assign((size_t)tokens * tokens,
                -std::numeric_limits<float>::max());
    for (int q = 0; q < tokens; q++)
        for (int k = 0; k <= q; k++)
            mask[(size_t)q * tokens + k] = 0.f;
}
}

bool prepare_native_edit_request(const std::string& model_dir, const std::vector<std::string>& image_paths, const std::string& prompt, const std::string& negative_prompt, bool has_negative_prompt, int condition_resolution, int width, int height, int steps, unsigned long long seed, const std::string& output, EditRequest& request, std::string* error)
{
    if (image_paths.empty() || image_paths.size() > 10)
    {
        if (error) *error = "provide between one and ten reference images";
        return false;
    }
    if (width <= 0 || height <= 0 || width % 32 || height % 32)
    {
        if (error) *error = "edit output size must be positive multiples of 32";
        return false;
    }
    if (condition_resolution <= 0)
    {
        if (error) *error = "condition resolution must be positive";
        return false;
    }

    std::vector<float> position_table;
    if (!make_position_table(model_dir + "/vision/vision_pos_embed.f32",
                             position_table))
    {
        if (error) *error = "missing vision position embedding table";
        return false;
    }

    SpecialTokensConfig special;
    special.eos_token = "<|im_end|>";
    special.pad_token = "<|endoftext|>";
    QwenBpeTokenizer tokenizer = QwenBpeTokenizer::LoadFromFiles(model_dir + "/processor/vocab.txt", model_dir + "/processor/merges.txt", special, false, false, true);
    if (tokenizer.vocab_size() == 0)
    {
        if (error) *error = "failed to load processor tokenizer";
        return false;
    }
    // The added-token IDs are not present in the base vocabulary.  Register the
    // complete tokenizer_config sequence so IDs after <|im_end|> retain the
    // Qwen3-VL assignments (vision_start=151652, image_pad=151655, ...).
    const char* special_tokens[] = {
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>",
        "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
        "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>",
        "<tool_response>", "</tool_response>", "<think>", "</think>",
    };
    for (const char* token : special_tokens)
        tokenizer.AddAdditionalSpecialToken(token, true);

    std::vector<std::pair<int, int>> vision_grids;
    std::vector<std::vector<uint8_t>> vision_rgb;
    std::vector<EditPreparedImage> prepared_images;
    std::vector<int> condition_widths;
    std::vector<int> condition_heights;
    std::vector<float> patch_values;
    std::vector<float> position_values;
    std::vector<float> vision_cos;
    std::vector<float> vision_sin;
    int patch_tokens = 0;

    for (const std::string& image_path : image_paths)
    {
        QwenRgbaImage source;
        if (!load_rgba_image(image_path, source))
        {
            if (error) *error = "failed to load image: " + image_path;
            return false;
        }

        int condition_width = 0;
        int condition_height = 0;
        if (!condition_dimensions(source.width, source.height, condition_resolution * condition_resolution, condition_width, condition_height))
        {
            if (error) *error = "failed to determine condition image size";
            return false;
        }
        condition_widths.push_back(condition_width);
        condition_heights.push_back(condition_height);

        std::vector<uint8_t> condition_rgba;
        resize_rgba(source, condition_width, condition_height, ResampleFilter::Lanczos, condition_rgba);

        EditPreparedImage prepared;
        prepared.width = condition_width;
        prepared.height = condition_height;
        prepared.rgba.resize((size_t)4 * condition_width * condition_height);
        for (int c = 0; c < 4; c++)
            for (int y = 0; y < condition_height; y++)
                for (int x = 0; x < condition_width; x++)
                {
                    const size_t pixel =
                        ((size_t)y * condition_width + x) * 4 + c;
                    prepared.rgba[(size_t)c * condition_width * condition_height
                                  + (size_t)y * condition_width + x] =
                        condition_rgba[pixel] / 127.5f - 1.f;
                }
        prepared_images.push_back(std::move(prepared));

        int vision_width = 0;
        int vision_height = 0;
        smart_resize_dimensions(condition_width, condition_height, vision_width, vision_height);
        std::vector<uint8_t> white_rgba = condition_rgba;
        std::vector<uint8_t> white_rgb;
        rgba_to_rgb_white(white_rgba, condition_width, condition_height, white_rgb);
        std::vector<uint8_t> resized_rgb;
        resize_rgb(white_rgb, condition_width, condition_height, vision_width, vision_height, ResampleFilter::Bicubic, resized_rgb);
        vision_grids.emplace_back(vision_height / kPatchSize,
                                  vision_width / kPatchSize);
        append_vision_image(resized_rgb, vision_width, vision_height, position_table, patch_values, position_values, vision_cos, vision_sin, patch_tokens);
    }

    std::string marker;
    for (size_t i = 0; i < image_paths.size(); i++)
    {
        if (i != 0)
            marker += " ";
        marker += "<image" + std::to_string(i + 1)
               + "><|vision_start|>";
        const int image_tokens = vision_grids[i].first
                               * vision_grids[i].second
                               / (kMergeSize * kMergeSize);
        for (int n = 0; n < image_tokens; n++)
            marker += "<|image_pad|>";
        marker += "<|vision_end|>";
    }
    const std::string system =
        "<|im_start|>system\n" + std::string(kSystemPrompt)
        + "<|im_end|>\n";
    const std::vector<int> system_ids = tokenizer.encode(system);
    if (system_ids.empty())
    {
        if (error) *error = "failed to tokenize edit system prompt";
        return false;
    }

    auto make_text_inputs = [&](const std::string& value, std::vector<int>& out_ids, std::vector<float>& out_cos, std::vector<float>& out_sin, std::vector<float>& out_attention, std::vector<float>& out_image_mask) -> bool
    {
        const std::string full_text = system + "<|im_start|>user\n" + marker + value + "<|im_end|>\n<|im_start|>assistant\n";
        out_ids = tokenizer.encode(full_text);
        if (out_ids.empty())
        {
            if (error) *error = "failed to tokenize edit prompt";
            return false;
        }

        int image_slot_count = 0;
        make_text_positions(out_ids, vision_grids, out_cos, out_sin, image_slot_count);
        out_image_mask.assign(out_ids.size(), 0.f);
        for (size_t i = 0; i < out_ids.size(); i++)
            out_image_mask[i] = out_ids[i] == kImagePadId ? 1.f : 0.f;
        if (image_slot_count != (int)std::count(
                out_image_mask.begin(), out_image_mask.end(), 1.f))
        {
            if (error) *error = "text image token layout mismatch";
            return false;
        }
        make_causal_mask((int)out_ids.size(), out_attention);
        return true;
    };

    std::vector<int> ids;
    std::vector<float> text_cos;
    std::vector<float> text_sin;
    std::vector<float> text_attention;
    std::vector<float> image_mask;
    if (!make_text_inputs(prompt, ids, text_cos, text_sin, text_attention, image_mask))
        return false;

    std::vector<int> negative_ids;
    std::vector<float> negative_text_cos;
    std::vector<float> negative_text_sin;
    std::vector<float> negative_text_attention;
    std::vector<float> negative_image_mask;
    if (has_negative_prompt
        && !make_text_inputs(negative_prompt, negative_ids, negative_text_cos, negative_text_sin, negative_text_attention, negative_image_mask))
        return false;

    request = EditRequest();
    request.native_inputs = true;
    request.input_ids_values = ids;
    request.text_cos_values = std::move(text_cos);
    request.text_sin_values = std::move(text_sin);
    request.text_attention_values = std::move(text_attention);
    request.image_mask_values = std::move(image_mask);
    request.negative_prompt = negative_prompt;
    request.has_negative_prompt = has_negative_prompt;
    request.negative_input_ids_values = std::move(negative_ids);
    request.negative_text_cos_values = std::move(negative_text_cos);
    request.negative_text_sin_values = std::move(negative_text_sin);
    request.negative_text_attention_values = std::move(negative_text_attention);
    request.negative_image_mask_values = std::move(negative_image_mask);
    request.vision_patch_values = std::move(patch_values);
    request.vision_pos_values = std::move(position_values);
    request.vision_cos_values = std::move(vision_cos);
    request.vision_sin_values = std::move(vision_sin);
    request.prepared_images = std::move(prepared_images);
    request.condition_widths = std::move(condition_widths);
    request.condition_heights = std::move(condition_heights);
    request.vision_patch_tokens = patch_tokens;
    request.output = output;
    request.width = width;
    request.height = height;
    request.drop_system_tokens = (int)system_ids.size();
    request.steps = steps;
    request.seed = seed;
    return true;
}
}
