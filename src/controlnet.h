// qwen-image implemented with ncnn library

#pragma once

#include <string>

#include "mat.h"
#include "vae.h"

namespace qwenimage {

bool encode_control_context(const QwenVaeEncoder& encoder, const std::string& image_path, int width, int height, ncnn::Mat& context);

} // namespace qwenimage
