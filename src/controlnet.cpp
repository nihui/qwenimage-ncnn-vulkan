// qwen-image implemented with ncnn library

#include "controlnet.h"

#include "edit_input.h"

#include <cstdio>
#include <cstring>
#include <vector>

#include "image_io.h"

namespace qwenimage {

bool encode_control_context(const QwenVaeEncoder& encoder, const std::string& image_path, int width, int height, ncnn::Mat& context)
{
    if (image_path.empty() || width <= 0 || height <= 0 || width % 16 || height % 16)
        return false;

    QwenRgbaImage source;
    if (!load_rgba_image(image_path, source))
    {
        fprintf(stderr, "failed to read ControlNet image %s\n", image_path.c_str());
        return false;
    }
    std::vector<unsigned char> resized;
    const unsigned char* pixels = source.rgba.data();
    if (source.width != width || source.height != height)
    {
        if (!resize_rgba_lanczos(source, width, height, resized))
        {
            fprintf(stderr, "failed to resize ControlNet image %s\n", image_path.c_str());
            return false;
        }
        pixels = resized.data();
    }
    ncnn::Mat rgba = ncnn::Mat::from_pixels(pixels, ncnn::Mat::PIXEL_RGBA, width, height);
    if (rgba.empty())
        return false;
    const float mean[4] = {127.5f, 127.5f, 127.5f, 127.5f};
    const float norm[4] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    rgba.substract_mean_normalize(mean, norm);

    ncnn::Mat latent;
    if (!encoder.encode(rgba, latent) || latent.dims != 2 || latent.w != 64
        || latent.h != (int64_t)(width / 16) * (height / 16)
        || latent.elempack != 1 || latent.elembits() != 32)
    {
        fprintf(stderr, "failed to encode ControlNet image %s\n", image_path.c_str());
        return false;
    }

    context.create(129, latent.h);
    if (context.empty())
        return false;
    context.fill(0.f);
    for (int i = 0; i < latent.h; i++)
        std::memcpy(context.row(i), latent.row(i), 64 * sizeof(float));
    return true;
}

} // namespace qwenimage
