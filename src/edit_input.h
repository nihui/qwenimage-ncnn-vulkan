// qwen-image implemented with ncnn library

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwenimage {
struct EditRequest;
struct QwenRgbaImage;

bool resize_rgba_lanczos(const QwenRgbaImage& source, int width, int height, std::vector<uint8_t>& output);

bool prepare_native_edit_request(const std::string& model_dir, const std::vector<std::string>& image_paths, const std::string& prompt, const std::string& negative_prompt, bool has_negative_prompt, int condition_resolution, int width, int height, int steps, unsigned long long seed, const std::string& output, EditRequest& request, std::string* error = nullptr);
}
