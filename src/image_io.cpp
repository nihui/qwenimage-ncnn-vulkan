// qwen-image implemented with ncnn library

#include "image_io.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if _WIN32
#include <objbase.h>
#include <windows.h>
#endif

#include "jpeg_image.h"
#include "png_image.h"
#include "webp_image.h"
#if _WIN32
#include "wic_image.h"
#endif

namespace qwenimage {

namespace {

std::string lowercase_path(std::string value)
{
    for (char& c : value)
    {
        if (c >= 'A' && c <= 'Z')
            c = (char)(c - 'A' + 'a');
    }
    return value;
}

int read_file_bytes(const std::string& path, std::vector<unsigned char>& bytes)
{
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp)
        return -1;

    if (std::fseek(fp, 0, SEEK_END) != 0)
    {
        std::fclose(fp);
        return -1;
    }
    const long length = std::ftell(fp);
    if (length <= 0 || std::fseek(fp, 0, SEEK_SET) != 0)
    {
        std::fclose(fp);
        return -1;
    }

    bytes.resize((size_t)length);
    const size_t read = std::fread(bytes.data(), 1, bytes.size(), fp);
    std::fclose(fp);
    return read == bytes.size() ? 0 : -1;
}

void assign_rgba(const unsigned char* pixels, int width, int height, int channels, QwenRgbaImage& image, bool bgr)
{
    image.width = width;
    image.height = height;
    image.rgba.resize((size_t)width * height * 4);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            const unsigned char* source =
                pixels + ((size_t)y * width + x) * channels;
            unsigned char* destination =
                image.rgba.data() + ((size_t)y * width + x) * 4;
            if (bgr)
            {
                destination[0] = source[2];
                destination[1] = source[1];
                destination[2] = source[0];
            }
            else
            {
                destination[0] = source[0];
                destination[1] = channels >= 3 ? source[1] : source[0];
                destination[2] = channels >= 3 ? source[2] : source[0];
            }
            destination[3] = channels == 4 ? source[3]
                             : channels == 2 ? source[1] : 255;
        }
}

void assign_rgb(const unsigned char* pixels, int width, int height, int channels, QwenRgbImage& image, bool bgr)
{
    image.width = width;
    image.height = height;
    image.rgb.resize((size_t)width * height * 3);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            const unsigned char* source =
                pixels + ((size_t)y * width + x) * channels;
            unsigned char* destination =
                image.rgb.data() + ((size_t)y * width + x) * 3;
            if (bgr)
            {
                destination[0] = source[2];
                destination[1] = source[1];
                destination[2] = source[0];
            }
            else
            {
                destination[0] = source[0];
                destination[1] = source[1];
                destination[2] = source[2];
            }
        }
}

#if _WIN32
void ensure_wic_initialized()
{
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
}

std::wstring wide_path(const std::string& path)
{
    const int length = MultiByteToWideChar(
        CP_UTF8, 0, path.c_str(), (int)path.size(), nullptr, 0);
    if (length <= 0)
        return std::wstring(path.begin(), path.end());
    std::wstring result((size_t)length, L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, path.c_str(), (int)path.size(), result.data(), length);
    return result;
}

void rgb_to_bgr(const unsigned char* source, int width, int height, int channels, std::vector<unsigned char>& destination)
{
    destination.resize((size_t)width * height * channels);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            const unsigned char* input =
                source + ((size_t)y * width + x) * channels;
            unsigned char* output =
                destination.data() + ((size_t)y * width + x) * channels;
            output[0] = input[2];
            output[1] = input[1];
            output[2] = input[0];
            if (channels == 4)
                output[3] = input[3];
        }
}
#endif

bool save_png(const std::string& path, int width, int height, int channels, const unsigned char* pixels)
{
#if _WIN32
    ensure_wic_initialized();
    std::vector<unsigned char> bgr;
    rgb_to_bgr(pixels, width, height, channels, bgr);
    const std::wstring output = wide_path(path);
    return wic_encode_image(
               output.c_str(), width, height, channels, bgr.data()) != 0;
#else
    return png_save(path.c_str(), width, height, channels, pixels) != 0;
#endif
}

} // namespace

bool load_rgb_image(const std::string& path, QwenRgbImage& image)
{
    const std::string extension = lowercase_path(path.substr(path.find_last_of('.') == std::string::npos ? path.size() : path.find_last_of('.') + 1));

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = nullptr;

#if _WIN32
    if (extension != "webp")
    {
        ensure_wic_initialized();
        const std::wstring input = wide_path(path);
        pixels = wic_decode_image(
            input.c_str(), &width, &height, &channels);
        if (pixels)
        {
            assign_rgb(pixels, width, height, channels, image, true);
            std::free(pixels);
            return true;
        }
    }
#endif

    std::vector<unsigned char> bytes;
    if (read_file_bytes(path, bytes) != 0)
        return false;

    if (extension == "png")
        pixels = png_load(bytes.data(), (int)bytes.size(),
                          &width, &height, &channels);
    else if (extension == "jpg" || extension == "jpeg")
        pixels = jpeg_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);
    else if (extension == "webp")
        pixels = webp_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);

    if (!pixels)
        pixels = png_load(bytes.data(), (int)bytes.size(),
                          &width, &height, &channels);
    if (!pixels)
        pixels = jpeg_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);
    if (!pixels)
        pixels = webp_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);
    if (!pixels)
        return false;

#if _WIN32
    const bool bgr = extension == "webp";
#else
    const bool bgr = false;
#endif
    assign_rgb(pixels, width, height, channels, image, bgr);
    std::free(pixels);
    return true;
}

bool load_rgba_image(const std::string& path, QwenRgbaImage& image)
{
    const std::string extension = lowercase_path(path.substr(path.find_last_of('.') == std::string::npos ? path.size() : path.find_last_of('.') + 1));

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = nullptr;

#if _WIN32
    if (extension != "webp")
    {
        ensure_wic_initialized();
        const std::wstring input = wide_path(path);
        pixels = wic_decode_image(
            input.c_str(), &width, &height, &channels);
        if (pixels)
        {
            assign_rgba(pixels, width, height, channels, image, true);
            std::free(pixels);
            return true;
        }
    }
#endif

    std::vector<unsigned char> bytes;
    if (read_file_bytes(path, bytes) != 0)
        return false;

    if (extension == "png")
        pixels = png_load(bytes.data(), (int)bytes.size(),
                          &width, &height, &channels);
    else if (extension == "jpg" || extension == "jpeg")
        pixels = jpeg_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);
    else if (extension == "webp")
        pixels = webp_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);

    if (!pixels)
        pixels = png_load(bytes.data(), (int)bytes.size(),
                          &width, &height, &channels);
    if (!pixels)
        pixels = jpeg_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);
    if (!pixels)
        pixels = webp_load(bytes.data(), (int)bytes.size(),
                           &width, &height, &channels);
    if (!pixels)
        return false;

#if _WIN32
    const bool bgr = extension == "webp";
#else
    const bool bgr = false;
#endif
    assign_rgba(pixels, width, height, channels, image, bgr);
    std::free(pixels);
    return true;
}

bool save_rgb_png(const std::string& path, const QwenRgbImage& image)
{
    if (image.width <= 0 || image.height <= 0
        || image.rgb.size() != (size_t)image.width * image.height * 3)
        return false;
    return save_png(path, image.width, image.height, 3, image.rgb.data());
}

bool save_float_png(const std::string& path, const std::vector<float>& image, int width, int height)
{
    if (width <= 0 || height <= 0
        || image.size() != (size_t)3 * width * height)
        return false;

    std::vector<unsigned char> pixels((size_t)width * height * 3);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
            {
                const float value = std::max(0.f, std::min(1.f, image[((size_t)c * height + y) * width + x] * .5f + .5f));
                pixels[((size_t)y * width + x) * 3 + c] =
                    (unsigned char)std::lround(value * 255.f);
            }
    return save_png(path, width, height, 3, pixels.data());
}

std::string make_batch_output_path(const std::string& path, int b, int batch)
{
    if (batch <= 1)
        return path;

    const size_t slash = path.find_last_of("/\\");
    const size_t dot = path.find_last_of('.');
    const bool has_extension = dot != std::string::npos
        && (slash == std::string::npos || dot > slash);
    const std::string suffix = "-" + std::to_string(b);
    if (!has_extension)
        return path + suffix;
    return path.substr(0, dot) + suffix + path.substr(dot);
}

bool save_rgba_float_png(const std::string& path, ncnn::Mat& image)
{
    if (image.empty() || image.dims != 3 || image.c != 4 || image.elempack != 1 || image.elembits() != 32)
        return false;
    const int width = image.w;
    const int height = image.h;

    const size_t dot = path.find_last_of('.');
    std::string extension = dot == std::string::npos
        ? std::string() : lowercase_path(path.substr(dot + 1));
    const bool jpeg = extension == "jpg" || extension == "jpeg";
    const int channels = jpeg ? 3 : 4;
    std::vector<unsigned char> pixels((size_t)width * height * channels);
    // to_pixels truncates, so include 0.5 for rounding
    const float mean[4] = {-128.f / 127.5f, -128.f / 127.5f, -128.f / 127.5f, -128.f / 127.5f};
    const float norm[4] = {127.5f, 127.5f, 127.5f, 127.5f};
    image.substract_mean_normalize(mean, norm);
    if (jpeg)
        image.channel_range(0, 3).to_pixels(pixels.data(), ncnn::Mat::PIXEL_RGB);
    else
        image.to_pixels(pixels.data(), ncnn::Mat::PIXEL_RGBA);

    if (jpeg)
    {
#if _WIN32
        const std::wstring output = wide_path(path);
        return jpeg_save(output.c_str(), width, height, 3, pixels.data()) != 0;
#else
        return jpeg_save(path.c_str(), width, height, 3, pixels.data()) != 0;
#endif
    }
    if (extension == "webp")
    {
#if _WIN32
        const std::wstring output = wide_path(path);
        return webp_save(output.c_str(), width, height, 4, pixels.data()) != 0;
#else
        return webp_save(path.c_str(), width, height, 4, pixels.data()) != 0;
#endif
    }
    return save_png(path, width, height, 4, pixels.data());
}

} // namespace qwenimage
