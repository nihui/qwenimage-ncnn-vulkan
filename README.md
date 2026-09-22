# Qwen-Image-2.1 ncnn Vulkan

:exclamation: :exclamation: :exclamation: This software is in the early development stage, it may bite your cat

![CI](https://github.com/nihui/qwenimage-ncnn-vulkan/workflows/CI/badge.svg)
![download](https://img.shields.io/github/downloads/nihui/qwenimage-ncnn-vulkan/total.svg)

ncnn implementation of [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1), with text-to-image, image editing, multi-reference image input and transparent RGBA output.

qwenimage-ncnn-vulkan uses [ncnn project](https://github.com/Tencent/ncnn) as the universal neural network inference framework.

## [Download](https://github.com/nihui/qwenimage-ncnn-vulkan/releases)

Download Windows/Linux/macOS Executable for Intel/AMD/NVIDIA/Apple-Silicon GPU

**https://github.com/nihui/qwenimage-ncnn-vulkan/releases**

This package includes all the binaries required. It is portable, so no CUDA, PyTorch or Python runtime environment is needed :)

### prepare model files

Download the qwenimage21 model folder:

https://huggingface.co/nihui-szyl/qwen-image-ncnn

Put it under the `models/` directory when running from the source tree, or pass its path with `-m`.

```
qwenimage-ncnn-vulkan
models/
    qwenimage21/
        processor/
            vocab.txt
            merges.txt
        text_encoder/
            text_encoder.ncnn.param
            text_encoder.ncnn.bin
        vision/
            vision_encoder.ncnn.param
            vision_encoder.ncnn.bin
            vision_pos_embed.f32
        transformer/
            input.ncnn.param
            input.ncnn.bin
            blocks.ncnn.param
            blocks.ncnn.bin
            output.ncnn.param
            output.ncnn.bin
        vae/
            encoder.ncnn.param
            encoder.ncnn.bin
            decoder.ncnn.param
            decoder.ncnn.bin
```

One model package supports text-to-image and image editing, dynamic output sizes and up to ten reference images.

## About Qwen-Image-2.1

Qwen-Image-2.1: A unified text-to-image generation and image editing model open-sourced by the Qwen team. With a 7B parameter visual component across 32 Single-Stream DiT layers, it balances generation quality, inference efficiency, and versatility.

https://github.com/QwenLM/Qwen-Image-2.1

## Usages

### requirements

- Minimum (Linux / macOS): 16GB RAM, any Vulkan capable GPU

- Minimum (Windows):

  *Due to WDDM limitations: Vulkan applications can only use half of the system RAM.*

  *The following condition shall be met:* **(Half of system RAM) + (GPU memory) >= 16GB**

  Examples of valid combinations:
  - Any amount RAM, 16GB dedicated GPU
  - 16GB RAM, 8GB dedicated GPU
  - 24GB RAM, 4GB dedicated GPU
  - 32GB RAM, any Vulkan capable GPU

- Recommended: 32GB RAM, 16GB dedicated GPU with tensorcore/matrix hardware

- CPU inference is available with `-g -1`

### Example Command

Text to image

```shell
qwenimage-ncnn-vulkan -p "A small red kite over a quiet lake." -o output.png
```

Text to image with a negative prompt

```shell
qwenimage-ncnn-vulkan -p "A small red kite over a quiet lake." -n "blurry image, dull colors" -w 4 -o output.png
```

The -w value maps to Torch true_cfg_scale and defaults to 1.0. The negative prompt participates in CFG only when -w is greater than 1.

Image editing

```shell
qwenimage-ncnn-vulkan -i input.png -p "Change the clothes to a blue jacket." -o output.png
```

Multiple reference images

```shell
qwenimage-ncnn-vulkan -i subject.png -i clothing.png -i background.png -p "Combine these references into one coherent image." -o output.png
```

Transparent image generation

```shell
qwenimage-ncnn-vulkan -p "A glass bottle on a transparent background." -o output.png
```

Set image size, denoise steps, seed and GPU

```shell
qwenimage-ncnn-vulkan -s 1024,1024 -l 40 -r 42 -g 0 -p "A red flower." -o output.png
```

Use `-i` once for each reference image, up to ten images. Text-to-image sizes must be multiples of 16; image-editing sizes must be multiples of 32.

The output is an RGBA PNG. The alpha channel can be used for transparent image generation.

### Full Usages

```console
Usage: qwenimage-ncnn-vulkan -p prompt -o outfile [options]...

  -h                   show this help
  -p prompt            prompt
  -n negative-prompt   negative prompt (optional)
  -w guidance-scale    true CFG scale (default=1.0)
  -o output-path       output image path (default=out.png)
  -i input-image       reference image for editing (repeat 1 to 10 times)
  -s image-size        image resolution (default=1024,1024)
  -l steps             denoise steps (default=40)
  -r random-seed       random seed (default=42)
  -m model-path        qwen-image model path (default=models/qwenimage21)
  -g gpu-id            GPU device to use (-1=cpu, default=auto)
```

If you encounter a crash or error, try upgrading your GPU driver:

- Intel: https://downloadcenter.intel.com/product/80939/Graphics-Drivers
- AMD: https://www.amd.com/en/support
- NVIDIA: https://www.nvidia.com/Download/index.aspx

## Build from Source

1. Clone this project with all submodules

```shell
git clone https://github.com/nihui/qwenimage-ncnn-vulkan.git
cd qwenimage-ncnn-vulkan
git submodule update --init --recursive --depth 1
```

2. Build with CMake

```shell
mkdir build
cd build
cmake ../src
cmake --build . -j 4
```

## Sample Images

TBA

## Original Qwen-Image-2.1 Project

- https://github.com/QwenLM/Qwen-Image-2.1

## Other Open-Source Code Used

- https://github.com/Tencent/ncnn for fast neural network inference on ALL PLATFORMS
- https://github.com/futz12/ncnn_llm for BPE tokenizer
- https://github.com/webmproject/libwebp for encoding and decoding Webp images on ALL PLATFORMS
- https://github.com/libjpeg-turbo/libjpeg-turbo for encoding and decoding JPEG images on ALL PLATFORMS
- https://github.com/pnggroup/libpng for encoding and decoding PNG images on ALL PLATFORMS
- https://github.com/zlib-ng/zlib-ng for encoding and decoding PNG images on ALL PLATFORMS
- https://github.com/tronkko/dirent for listing files in directory on Windows

