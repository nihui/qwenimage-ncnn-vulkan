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

- Low-VRAM mode and VAE encoder/decoder tiling adapt automatically to the available GPU memory at the start of each generation

- CPU inference is available with `-g -1`

Earlier text-to-image memory measurements before prefix KV caching (RTX 3060, BF16, 2 steps):

| Image resolution | Example VAE tile | Peak VRAM (MiB) |
|---|---|---:|
| 512x512 | 256x256 | 531.6 |
| 1024x1024 | 256x256 | 1027.3 |
| 1024x1024 | 512x512 | 1486.5 |
| 2048x2048 | 512x512 | 3411.1 |
| 2048x2048 | 1024x1024 | 4970.6 |

These historical measurements are examples, not current minimum VRAM guarantees. Current memory requirements also depend on prompt length and KV caches. Tile sizes are selected automatically from the current memory budget and may be larger when more memory is available. No manual low-VRAM or tile setting is required.

The peaks are VRAM increments over the usage before the process starts. Allow additional GPU memory for the driver, desktop and other applications when choosing a graphics card.

Image editing also accounts for reference image sizes and negative prompts when choosing its memory policy. When needed, reference KV caches are kept in system RAM and transferred one Transformer block at a time. More reference images require more system RAM and may take longer to process. VAE encoder and decoder tile sizes are selected independently. If the estimated Transformer workspace or full-image VAE attention cannot fit even with offloading and tiling, generation stops with an insufficient-memory error; reduce the output or reference image sizes, or use fewer references.

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

Load a safetensors LoRA, including PEFT `lora_A` / `lora_B` adapters

```shell
qwenimage-ncnn-vulkan --lora qwen-image-style.safetensors --lora-scale 0.8 -p "A red flower." -o output.png
```

Some Qwen Fun Acc adapters use a fixed step schedule. For example, the 4-step adapter requires `-l 4`.

ControlNet generation

```shell
qwenimage-ncnn-vulkan -c control.png --control-scale 1.0 -p "A red flower." -o output.png
```

When `-c` is used, ControlNet loads by default from `<model-path>/controlnet/controlnet.ncnn.param` and its matching `.bin` file; `--controlnet` can override the model path. ControlNet can also be combined with `-i` image editing and a LoRA. The model can be used with Canny, depth, grayscale, HED, lineart, MLSD, pose and scribble condition images.

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

Batch generation

```shell
qwenimage-ncnn-vulkan -b 4 -r 42 -p "A red flower." -o output.png
```

With `-b` greater than one, outputs are saved as `output-0.png`, `output-1.png`, and so on. Each image uses the next seed value. Use `-i` once for each reference image, up to ten images. Text-to-image sizes must be multiples of 16; image-editing sizes must be multiples of 32.

The default output is an RGBA PNG. A jpg or jpeg suffix writes RGB JPEG output, and a webp suffix writes lossless RGBA WebP output. PNG and WebP alpha can be used for transparent image generation.

### Full Usages

```console
Usage: qwenimage-ncnn-vulkan [options]...

  -h                   show this help
  -p prompt            prompt (default=A half-length portrait in the warm light of a convenience store late at night. An East Asian beauty, holding milk, meets your gaze in front of the freezer.)
  -n negative-prompt   negative prompt (optional)
  -w guidance-scale    true CFG scale (default=1.0)
  -o output-path       output image path (default=out.png)
  -i input-image       reference image for editing (repeat 1 to 10 times)
  -c control-image     ControlNet condition image (optional)
  --controlnet path    override default ControlNet model path (optional)
  --control-scale val  ControlNet strength (default=1.0)
  --lora path          LoRA or Qwen Fun Acc safetensors adapter (optional)
  --lora-scale value   LoRA strength (default=1.0)
  -s image-size        image resolution (default=1024,1024)
  -l steps             denoise steps (default=40)
  -r random-seed       random seed (default=rand)
  -m model-path        qwen-image model path (default=models/qwenimage21)
  -g gpu-id            GPU device to use (-1=cpu, default=auto)
  -b batch-size        batched generation (default=1)
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


<table width="100%">
<tr>
<td width="33%">

<details>
<summary>expand for full command</summary>

```qwenimage-ncnn-vulkan.exe -p "A half-length portrait in the warm light of a convenience store late at night. An East Asian beauty, holding milk, meets your gaze in front of the freezer." -r 1527207263```

</details>

![qwenimage](images/beauty.jpg)

</td>
<td width="33%">

<details>
<summary>expand for full command</summary>

```qwenimage-ncnn-vulkan.exe -p "Create a sophisticated contemporary Chinese art exhibition poster, combining modern editorial graphic design with restrained traditional Chinese visual elements. The poster should have a refined, minimal, museum-quality aesthetic. Use a warm off-white textured paper background with subtle natural fibers. Incorporate an abstract ink landscape inspired by Chinese shanshui painting: distant misty mountains, a flowing river, soft black and gray ink washes, and a small accent of muted cinnabar red. The landscape should be elegant and atmospheric rather than highly detailed, leaving enough clean space for typography. Typography is the main focus of the poster. All Chinese text must be clearly legible, correctly written, and faithfully reproduced. Do not alter, misspell, duplicate, substitute, or invent any characters. The main title must read exactly:「山海之间」 Display the four Chinese characters prominently in the upper-middle area of the poster. Use large, elegant Chinese typography with a strong contemporary editorial feeling, inspired by refined Song-style Chinese type or sophisticated modern exhibition typography. The title should be visually dominant but not overly decorative. Below the main title, add the subtitle exactly:「东方艺术与当代视觉展」 Set it in smaller, clean Chinese typography with generous letter spacing and precise alignment. Near the lower section of the poster, include the following information exactly:「2026年10月18日 — 11月30日」「当代艺术馆」「水墨 · 山川 · 光影 · 新境」 Arrange these three lines with clear hierarchy, careful spacing, and professional grid-based alignment. Keep them smaller than the main title but fully readable. Add a very small English subtitle: 'BETWEEN MOUNTAINS AND SEAS' The English text should be secondary and understated, supporting the Chinese typography rather than competing with it. Use a carefully structured editorial layout with generous margins, deliberate negative space, subtle asymmetry, and a strong typographic grid. Integrate the ink landscape naturally around and behind the text without reducing readability. The composition should feel like a professionally designed poster for a major contemporary art museum. Use a restrained palette: warm ivory paper, charcoal black, soft ink gray, and one subtle cinnabar-red accent. The red accent may appear as a small seal-like geometric mark or a restrained graphic element. The final result should look like high-end professional Chinese graphic design photographed or scanned from a real printed exhibition poster: crisp typography, refined paper texture, excellent visual hierarchy, balanced spacing, subtle print texture, and sophisticated art direction. IMPORTANT: Render every Chinese character exactly as specified. Preserve the exact wording and punctuation. Do not generate any additional Chinese or English text beyond the text explicitly requested. Avoid garbled Chinese characters, pseudo-Chinese writing, incorrect characters, repeated text, random letters, excessive decorative calligraphy, crowded layout, cheap commercial advertising style, neon colors, gradients, 3D typography, glossy effects, cyberpunk aesthetics, excessive ornaments, logos, QR codes, watermarks, or unrelated text." -r 385448266```

</details>

![qwenimage](images/poster.jpg)

</td>
<td width="34%" rowspan="2">

<details>
<summary>expand for full command</summary>

```qwenimage-ncnn-vulkan.exe -p "A Yuan dynasty Chinese literati landscape painting in the manner of Ni Zan, traditional monochrome ink on paper, composed as a tall 1:2 vertical hanging scroll. Create a sparse yet visually rich and well-balanced composition, with restrained detail distributed across the full height of the painting rather than leaving excessively large empty areas. In the lower foreground, depict a gently sloping rocky riverbank with layered stones, several tall slender old trees, a few smaller shrubs, and a simple empty pavilion partially hidden among the trees. Add subtle clusters of reeds, low vegetation, exposed roots, and small irregular rocks along the shoreline. The foreground should have enough structure and texture to create a strong visual anchor while remaining elegant and uncluttered. Extend part of the foreground bank slightly toward the center of the composition, and introduce a small secondary sandbar or low rocky islet in the middle distance. Add a few sparse trees or shrubs on this secondary landform to create spatial depth and visual rhythm. The middle ground should contain calm open water, but not be completely empty. Use extremely subtle horizontal ink traces, faint ripples, and barely visible tonal variations to suggest water while preserving the feeling of stillness. Allow some blank paper to remain visible, but avoid an excessively large uninterrupted empty area. Across the water, depict a more developed distant shoreline with low hills, scattered tree silhouettes, and several overlapping mountain forms rendered in pale ink. The distant mountains should remain low, restrained, and atmospheric rather than dramatic, with several softly layered ridgelines creating depth. Use economical and refined brushwork, with pale ink, dry brush, rubbing, restrained hemp-fiber texture strokes, and subtle folding-ribbon brush accents. Preserve natural variations in ink density, dry-brush texture, and flying-white effects. The landscape should contain enough brushwork to feel complete and substantial, while still maintaining the simplicity and restraint of Yuan dynasty literati painting. The overall mood should be tranquil, austere, poetic, cool, refined, and contemplative. Maintain the detached elegance associated with Ni Zan, but avoid making the scene feel barren or unfinished. The painting should feel spacious rather than empty, layered rather than crowded, and quiet rather than desolate. Use negative space deliberately between groups of landscape elements, but distribute visual information throughout the composition. Keep a clear foreground, middle ground, and distant background, with several overlapping layers of shoreline, trees, rocks, water, and mountains. The visual density should be moderate: restrained and minimal, but not extremely sparse. In the upper-right area, add a small vertical Chinese calligraphic inscription written with a traditional brush. The calligraphy should feel natural, elegant, slightly archaic, and hand-written rather than typographic. The inscription must read exactly:「江天寥廓水雲間 獨坐亭中看遠山 萬籟俱寂心自遠 不知身在此塵寰 水竹院落」 Keep the inscription small and understated so that it does not dominate the landscape. Near the inscription, place one or two small traditional Chinese cinnabar seals, naturally integrated into the composition. Use one red-character seal and one white-character seal if possible. The seals should appear hand-stamped, with slightly irregular edges, subtle imperfections, and muted mineral-red cinnabar tones. The final image should resemble a well-preserved Yuan dynasty ink hanging scroll: ancient, restrained, elegant, layered, balanced, subtle, and contemplative. Avoid excessively empty composition, huge uninterrupted blank areas, modern Chinese illustration aesthetics, digital painting, watercolor effects, blue-and-green landscape painting, saturated colors, dramatic mountains, waterfalls, cloud seas, fantasy scenery, crowded figures, elaborate architecture, decorative borders, English text, modern typography, logos, or watermarks." -r 1944099538 -s 1024,2048```

</details>

![qwenimage](images/shanshui.jpg)

</td>
</tr>
<tr>
<td width="33%">

<details>
<summary>expand for full command</summary>

```qwenimage-ncnn-vulkan.exe -p "A highly realistic candid street photograph taken on a rainy evening in a dense East Asian city. A young woman in a dark charcoal coat walks alone along a narrow city sidewalk, holding a transparent umbrella. She is caught naturally in mid-step rather than posing for the camera. Her expression is calm and slightly thoughtful. Her face, skin texture, hair, hands, clothing folds, and proportions should look completely natural and anatomically correct. The street has just been soaked by rain. Wet asphalt and stone pavement reflect warm storefront lights, traffic signals, and subtle red, amber, and cool white highlights. Small puddles create imperfect broken reflections. Raindrops cling to the umbrella, glass windows, parked bicycles, metal railings, and street signs. A few distant pedestrians carrying umbrellas appear farther down the street, softly blurred by atmospheric perspective. The environment should feel lived-in and authentic: small restaurants, convenience stores, apartment entrances, utility boxes, bicycles, subtle signage, window condensation, slightly weathered walls, and ordinary urban details. Avoid an overly clean or staged environment. Use natural mixed lighting from shop windows, street lamps, and distant traffic. The woman's face is softly illuminated by warm reflected storefront light from one side, while the surrounding street remains slightly cool and subdued. Preserve realistic highlight roll-off and shadow detail. No dramatic studio lighting. Shot as professional full-frame street photography with a 50mm lens at approximately f/2.0, eye-level perspective, shallow but realistic depth of field. The woman is in sharp focus while the distant background gradually falls out of focus. Use subtle natural lens characteristics, realistic bokeh, mild high-ISO grain, and a restrained documentary color palette. The composition should feel spontaneous and observational, as if captured during a real evening walk. Slight asymmetry, natural visual clutter, believable perspective, realistic scale, and imperfect everyday details are important. Photorealistic skin, realistic hair strands, physically plausible wet surfaces, accurate reflections, natural fabric texture, realistic glass and metal materials, true-to-life lighting, high dynamic range, fine photographic detail. The final image should look like an authentic high-end documentary photograph captured with a real camera, not a digital painting or AI illustration. Avoid beauty-retouched skin, plastic-looking faces, excessive skin smoothing, exaggerated cinematic lighting, oversaturated neon colors, cyberpunk aesthetics, fantasy elements, artificial symmetry, excessive bokeh, distorted anatomy, malformed hands, duplicated people, floating objects, unrealistic reflections, overly sharp HDR, CGI rendering, 3D render appearance, illustration, anime, watercolor, text overlays, logos, or watermarks." -r 85854995 -s 1280,720```

</details>

![qwenimage](images/street.jpg)

</td>
<td width="33%">

<details>
<summary>expand for full command</summary>

```qwenimage-ncnn-vulkan.exe -p "A highly realistic candid street photograph taken on a rainy evening in a dense East Asian city. A young woman in a dark charcoal coat walks alone along a narrow city sidewalk, holding a transparent umbrella. She is caught naturally in mid-step rather than posing for the camera. Her expression is calm and slightly thoughtful. Her face, skin texture, hair, hands, clothing folds, and proportions should look completely natural and anatomically correct. The street has just been soaked by rain. Wet asphalt and stone pavement reflect warm storefront lights, traffic signals, and subtle red, amber, and cool white highlights. Small puddles create imperfect broken reflections. Raindrops cling to the umbrella, glass windows, parked bicycles, metal railings, and street signs. A few distant pedestrians carrying umbrellas appear farther down the street, softly blurred by atmospheric perspective. The environment should feel lived-in and authentic: small restaurants, convenience stores, apartment entrances, utility boxes, bicycles, subtle signage, window condensation, slightly weathered walls, and ordinary urban details. Avoid an overly clean or staged environment. Use natural mixed lighting from shop windows, street lamps, and distant traffic. The woman's face is softly illuminated by warm reflected storefront light from one side, while the surrounding street remains slightly cool and subdued. Preserve realistic highlight roll-off and shadow detail. No dramatic studio lighting. Shot as professional full-frame street photography with a 50mm lens at approximately f/2.0, eye-level perspective, shallow but realistic depth of field. The woman is in sharp focus while the distant background gradually falls out of focus. Use subtle natural lens characteristics, realistic bokeh, mild high-ISO grain, and a restrained documentary color palette. The composition should feel spontaneous and observational, as if captured during a real evening walk. Slight asymmetry, natural visual clutter, believable perspective, realistic scale, and imperfect everyday details are important. Photorealistic skin, realistic hair strands, physically plausible wet surfaces, accurate reflections, natural fabric texture, realistic glass and metal materials, true-to-life lighting, high dynamic range, fine photographic detail. The final image should look like an authentic high-end documentary photograph captured with a real camera, not a digital painting or AI illustration. Avoid beauty-retouched skin, plastic-looking faces, excessive skin smoothing, exaggerated cinematic lighting, oversaturated neon colors, cyberpunk aesthetics, fantasy elements, artificial symmetry, excessive bokeh, distorted anatomy, malformed hands, duplicated people, floating objects, unrealistic reflections, overly sharp HDR, CGI rendering, 3D render appearance, illustration, anime, watercolor, text overlays, logos, or watermarks." -r 85854995 -s 1280,720```

</details>

![qwenimage](images/street.jpg)

</td>
</tr>
</table>


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

