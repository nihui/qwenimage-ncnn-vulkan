// qwen-image implemented with ncnn library

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwenimage {

struct QwenRgbImage
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

struct QwenRgbaImage
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;
};

bool load_rgb_image(const std::string& path, QwenRgbImage& image);
bool load_rgba_image(const std::string& path, QwenRgbaImage& image);
bool save_rgb_png(const std::string& path, const QwenRgbImage& image);
bool save_float_png(const std::string& path, const std::vector<float>& image, int width, int height);

bool save_rgba_float_png(const std::string& path, const std::vector<float>& image, int width, int height);
} // namespace qwenimage
