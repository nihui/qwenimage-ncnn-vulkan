// qwen-image implemented with ncnn library

#include "models.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace qwenimage {
namespace {
bool exists(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    return static_cast<bool>(stream);
}
ModelFiles pair_from_param(const std::filesystem::path& path)
{
    std::filesystem::path bin = path;
    bin.replace_extension(".bin");
    return {path.string(), bin.string()};
}
bool valid_file_pair(const ModelFiles& files)
{
    return exists(files.param) && exists(files.bin);
}
}
ModelPaths make_model_paths(const std::string& model_dir)
{
    ModelPaths paths;
    const std::filesystem::path root(model_dir);
    const auto text = root / "text_encoder";
    const auto text_edit = root / "text_encoder_edit";
    const auto vision = root / "vision";
    const auto vae = root / "vae";
    const auto transformer = root / "transformer";
    paths.text_encoder = {(text / "text_encoder.ncnn.param").string(),
                          (text / "text_encoder.ncnn.bin").string()};
    paths.text_encoder_edit = {
        (text_edit / "text_encoder.ncnn.param").string(),
        (text_edit / "text_encoder.ncnn.bin").string()};
    paths.vision_encoder = {
        (vision / "vision_encoder.ncnn.param").string(),
        (vision / "vision_encoder.ncnn.bin").string()};
    paths.vae_encoder = {(vae / "encoder.ncnn.param").string(),
                         (vae / "encoder.ncnn.bin").string()};
    paths.vae_decoder = {(vae / "decoder.ncnn.param").string(),
                         (vae / "decoder.ncnn.bin").string()};
    paths.transformer_input = {(transformer / "input.ncnn.param").string(),
                               (transformer / "input.ncnn.bin").string()};
    paths.transformer_blocks_merged = {
        (transformer / "blocks.ncnn.param").string(),
        (transformer / "blocks.ncnn.bin").string()};
    paths.transformer_output = {(transformer / "output.ncnn.param").string(),
                                (transformer / "output.ncnn.bin").string()};

    if (std::filesystem::is_directory(transformer))
        for (const auto& entry : std::filesystem::directory_iterator(transformer))
        {
            const std::string name = entry.path().filename().string();
            if (entry.path().extension() == ".param"
                && name.rfind("transformer_block_", 0) == 0)
                paths.transformer_blocks.push_back(pair_from_param(entry.path()));
        }
    std::sort(paths.transformer_blocks.begin(), paths.transformer_blocks.end(), [](const ModelFiles& a, const ModelFiles& b) { return a.param < b.param; });
    return paths;
}
bool validate_model_paths(const ModelPaths& paths, std::string* error)
{
    const bool has_merged_blocks = valid_file_pair(paths.transformer_blocks_merged);
    if (!valid_file_pair(paths.text_encoder)
        || !valid_file_pair(paths.vae_decoder)
        || !valid_file_pair(paths.transformer_input)
        || (!has_merged_blocks && paths.transformer_blocks.empty())
        || !valid_file_pair(paths.transformer_output))
    {
        if (error) *error = "model graph files are incomplete";
        return false;
    }
    if (!has_merged_blocks)
        for (const ModelFiles& files : paths.transformer_blocks)
            if (!valid_file_pair(files))
            {
                if (error) *error = "missing transformer graph: " + files.param;
                return false;
            }
    return true;
}
bool validate_edit_model_paths(const ModelPaths& paths, std::string* error)
{
    // The final unified package stores one multimodal text graph.  Older
    // edit packages may still provide a second text_encoder_edit directory.
    if ((!valid_file_pair(paths.text_encoder_edit)
         && !valid_file_pair(paths.text_encoder))
        || !valid_file_pair(paths.vision_encoder)
        || !valid_file_pair(paths.vae_encoder))
    {
        if (error) *error = "edit model graph files are incomplete";
        return false;
    }
    return validate_model_paths(paths, error);
}

bool QwenModelSet::load_text_encoder(const ModelPaths& paths, const RuntimeConfig& config, bool edit_mode)
{
    unload_text_encoder();
    const ModelFiles& text_files = edit_mode
        && valid_file_pair(paths.text_encoder_edit)
        ? paths.text_encoder_edit : paths.text_encoder;
    text_encoder.reset(new ncnn::Net());
    if (!load_net(*text_encoder, text_files, config, ModelStage::TextEncoder))
    {
        unload_text_encoder();
        return false;
    }
    return true;
}

bool QwenModelSet::load_vision_encoder(const ModelPaths& paths, const RuntimeConfig& config)
{
    unload_vision_encoder();
    vision_encoder.reset(new ncnn::Net());
    if (!load_net(*vision_encoder, paths.vision_encoder, config,
                  ModelStage::VisionEncoder))
    {
        unload_vision_encoder();
        return false;
    }
    return true;
}

bool QwenModelSet::load_vae_encoder(const ModelPaths& paths, const RuntimeConfig& config)
{
    unload_vae_encoder();
    vae_encoder.reset(new ncnn::Net());
    if (!load_net(*vae_encoder, paths.vae_encoder, config,
                  ModelStage::VaeEncoder))
    {
        unload_vae_encoder();
        return false;
    }
    return true;
}

bool QwenModelSet::load_vae_decoder(const ModelPaths& paths, const RuntimeConfig& config)
{
    unload_vae_decoder();
    vae_decoder.reset(new ncnn::Net());
    if (!load_net(*vae_decoder, paths.vae_decoder, config,
                  ModelStage::VaeDecoder))
    {
        unload_vae_decoder();
        return false;
    }
    return true;
}

bool QwenModelSet::load_transformer(const ModelPaths& paths, const RuntimeConfig& config)
{
    unload_transformer();
    transformer_input.reset(new ncnn::Net());
    if (!load_net(*transformer_input, paths.transformer_input, config,
                  ModelStage::Transformer))
    {
        unload_transformer();
        return false;
    }
    transformer_blocks.clear();
    if (valid_file_pair(paths.transformer_blocks_merged))
    {
        transformer_blocks.emplace_back(new ncnn::Net());
        if (!load_net(*transformer_blocks.back(), paths.transformer_blocks_merged,
                      config, ModelStage::Transformer))
        {
            unload_transformer();
            return false;
        }
    }
    else
    {
        transformer_blocks.reserve(paths.transformer_blocks.size());
        for (size_t i = 0; i < paths.transformer_blocks.size(); i++)
        {
            transformer_blocks.emplace_back(new ncnn::Net());
            if (!load_net(*transformer_blocks.back(), paths.transformer_blocks[i],
                          config, ModelStage::Transformer))
            {
                unload_transformer();
                return false;
            }
        }
    }
    transformer_output.reset(new ncnn::Net());
    if (!load_net(*transformer_output, paths.transformer_output, config,
                  ModelStage::Transformer))
    {
        unload_transformer();
        return false;
    }
    return true;
}

void QwenModelSet::unload_text_encoder()
{
    text_encoder.reset();
}

void QwenModelSet::unload_vision_encoder()
{
    vision_encoder.reset();
}

void QwenModelSet::unload_vae_encoder()
{
    vae_encoder.reset();
}

void QwenModelSet::unload_vae_decoder()
{
    vae_decoder.reset();
}

void QwenModelSet::unload_transformer()
{
    // Release the large merged block graph before the small input/output
    // graphs, so both model weights and Vulkan pipelines are reclaimed.
    transformer_blocks.clear();
    transformer_input.reset();
    transformer_output.reset();
}

void QwenModelSet::unload_all()
{
    unload_text_encoder();
    unload_vision_encoder();
    unload_vae_encoder();
    unload_vae_decoder();
    unload_transformer();
}
}
