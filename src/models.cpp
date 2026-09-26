// qwen-image implemented with ncnn library

#include "models.h"

#include <filesystem>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

namespace qwenimage {
namespace {
bool exists(const std::string& path)
{
    std::ifstream stream(path, std::ios::binary);
    return static_cast<bool>(stream);
}
bool valid_file_pair(const ModelFiles& files)
{
    return exists(files.param) && exists(files.bin);
}

std::string join_tokens(const std::vector<std::string>& tokens)
{
    std::string line;
    for (size_t i = 0; i < tokens.size(); i++)
    {
        if (i) line += ' ';
        line += tokens[i];
    }
    return line;
}

bool make_transformer_kvcache_param(const std::string& path, std::string& param)
{
    std::ifstream file(path);
    if (!file)
        return false;

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        lines.push_back(line);
    }
    if (!file.eof() || lines.size() < 2)
        return false;

    int layer_count = 0;
    int blob_count = 0;
    std::istringstream header(lines[1]);
    if (!(header >> layer_count >> blob_count))
        return false;

    int sdpa_count = 0;
    int cached_sdpa_count = 0;
    std::vector<std::string> transformed(lines.size());
    std::vector<std::string> input_layers;
    for (size_t i = 2; i < lines.size(); i++)
    {
        std::istringstream line_stream(lines[i]);
        std::vector<std::string> tokens;
        std::string token;
        while (line_stream >> token)
            tokens.push_back(token);
        if (tokens.size() < 4 || tokens[0] != "SDPA")
        {
            transformed[i] = lines[i];
            continue;
        }

        sdpa_count++;
        int bottom_count = 0;
        int top_count = 0;
        std::istringstream bottom_stream(tokens[2]);
        std::istringstream top_stream(tokens[3]);
        if (!(bottom_stream >> bottom_count) || !(top_stream >> top_count))
            return false;
        if (bottom_count == 6 && top_count == 3)
        {
            bool has_kv_cache = false;
            for (size_t j = 4 + bottom_count + top_count; j < tokens.size(); j++)
                if (tokens[j] == "7=1") has_kv_cache = true;
            if (!has_kv_cache)
                return false;
            cached_sdpa_count++;
            transformed[i] = lines[i];
            continue;
        }
        if (bottom_count != 4 || top_count != 1 || tokens.size() < 9)
            return false;

        const std::string& name = tokens[1];
        if (name.size() < 5 || name[0] != 'b'
            || name[1] < '0' || name[1] > '9'
            || name[2] < '0' || name[2] > '9')
            return false;
        const int block = (name[1] - '0') * 10 + name[2] - '0';
        if (block >= 32 || name.compare(3, 5, "_sdpa") != 0)
            return false;

        char suffix[8];
        char layer_name[32];
        std::snprintf(suffix, sizeof(suffix), "%02d", block);
        std::snprintf(layer_name, sizeof(layer_name), "kvcache_input_%02d", block);
        const std::string key_in = std::string("b") + suffix + "_cache_k_in";
        const std::string value_in = std::string("b") + suffix + "_cache_v_in";
        const std::string key_out = std::string("b") + suffix + "_cache_k_out";
        const std::string value_out = std::string("b") + suffix + "_cache_v_out";

        std::vector<std::string> upgraded;
        upgraded.reserve(tokens.size() + 6);
        upgraded.insert(upgraded.end(), tokens.begin(), tokens.begin() + 4);
        upgraded[2] = "6";
        upgraded[3] = "3";
        upgraded.insert(upgraded.end(), tokens.begin() + 4, tokens.begin() + 8);
        upgraded.push_back(key_in);
        upgraded.push_back(value_in);
        upgraded.push_back(tokens[8]);
        upgraded.push_back(key_out);
        upgraded.push_back(value_out);
        bool has_kv_cache = false;
        for (size_t j = 9; j < tokens.size(); j++)
        {
            if (tokens[j].compare(0, 2, "7=") == 0)
            {
                upgraded.push_back("7=1");
                has_kv_cache = true;
            }
            else
            {
                upgraded.push_back(tokens[j]);
            }
        }
        if (!has_kv_cache)
            upgraded.push_back("7=1");
        transformed[i] = join_tokens(upgraded);

        std::snprintf(layer_name, sizeof(layer_name), "kvcache_k_%02d", block);
        input_layers.push_back(std::string("Input ") + layer_name + " 0 1 " + key_in);
        std::snprintf(layer_name, sizeof(layer_name), "kvcache_v_%02d", block);
        input_layers.push_back(std::string("Input ") + layer_name + " 0 1 " + value_in);
    }

    if (sdpa_count != 32)
        return false;
    if (cached_sdpa_count == sdpa_count)
    {
        param.clear();
        for (const std::string& source_line : lines)
            param += source_line + '\n';
        return true;
    }
    if (cached_sdpa_count != 0 || input_layers.size() != 64)
        return false;

    std::ostringstream counts;
    counts << layer_count + 64 << ' ' << blob_count + 128;
    param = lines[0] + '\n' + counts.str() + '\n';
    for (const std::string& input_layer : input_layers)
        param += input_layer + '\n';
    for (size_t i = 2; i < lines.size(); i++)
        param += transformed[i] + '\n';
    return true;
}

bool append_control_projection_graph(std::string& param, float control_scale)
{
    std::istringstream input(param);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line))
        lines.push_back(line);
    if (lines.size() < 2)
        return false;

    int layers = 0;
    int blobs = 0;
    std::istringstream counts(lines[1]);
    if (!(counts >> layers >> blobs))
        return false;
    char scale[64];
    std::snprintf(scale, sizeof(scale), "%.9g", control_scale);
    std::vector<std::string> extra;
    extra.push_back("Input control_base_input_layer 0 1 control_base_input");
    extra.push_back("Input control_before_input_layer 0 1 control_before_input");
    extra.push_back("BinaryOp control_add_before 2 1 control_before_input control_base_input control_add_before_output 0=0");
    for (int i = 0; i < 16; i++)
    {
        char hint[32];
        char scaled[40];
        char block[32];
        char scale_layer[40];
        char add[40];
        std::snprintf(hint, sizeof(hint), "control_after_%02d_input", i);
        std::snprintf(scaled, sizeof(scaled), "control_scaled_%02d", i);
        std::snprintf(block, sizeof(block), "b%02d_out", i * 2);
        std::snprintf(scale_layer, sizeof(scale_layer), "control_scale_%02d", i);
        std::snprintf(add, sizeof(add), "control_add_%02d", i);
        extra.push_back(std::string("Input ") + hint + "_layer 0 1 " + hint);
        extra.push_back(std::string("BinaryOp ") + scale_layer + " 1 1 " + hint + " " + scaled
                        + " 0=2 1=1 2=" + scale);
        extra.push_back(std::string("BinaryOp ") + add + " 2 1 " + block + " " + scaled
                        + " control_add_" + (i < 10 ? "0" : "") + std::to_string(i) + "_output 0=0");
    }
    std::ostringstream output;
    output << lines[0] << '\n' << layers + (int)extra.size() << ' '
           << blobs + (int)extra.size() << '\n';
    for (size_t i = 2; i < lines.size(); i++)
        output << lines[i] << '\n';
    for (const std::string& extra_line : extra)
        output << extra_line << '\n';
    param = output.str();
    return true;
}

bool load_transformer_blocks_net(ncnn::Net& net, const ModelFiles& files, const RuntimeConfig& config, TransformerLoRA* lora, bool controlnet, float control_scale)
{
    net.opt = make_ncnn_option(config, ModelStage::Transformer);
    net.opt.lightmode = true;

    std::string param;
    if (controlnet)
    {
        std::ifstream file(files.param);
        if (!file)
            return false;
        std::ostringstream contents;
        contents << file.rdbuf();
        param = contents.str();
        if (param.empty() || !append_control_projection_graph(param, control_scale))
        {
            fprintf(stderr, "failed to prepare in-memory ControlNet graph %s\n", files.param.c_str());
            return false;
        }
    }
    else if (!make_transformer_kvcache_param(files.param, param))
    {
        fprintf(stderr, "failed to prepare transformer KV-cache param %s\n", files.param.c_str());
        return false;
    }
    if (lora && register_transformer_lora(net, lora, TransformerPart::Blocks) != 0)
    {
        fprintf(stderr, "failed to register transformer block LoRA layers\n");
        return false;
    }
    if (net.load_param_mem(param.c_str()) != 0)
    {
        fprintf(stderr, "failed to load transformer KV-cache param %s\n", files.param.c_str());
        return false;
    }
    if (net.load_model(files.bin.c_str()) != 0)
    {
        fprintf(stderr, "failed to load ncnn model %s\n", files.bin.c_str());
        return false;
    }
    return true;
}
#if NCNN_VULKAN
void set_stage_vkallocators(ncnn::Net& net, std::unique_ptr<ncnn::VkBlobAllocator>& blob_vkallocator, std::unique_ptr<ncnn::VkStagingAllocator>& staging_vkallocator)
{
    if (!net.opt.use_vulkan_compute)
        return;

    if (!blob_vkallocator)
        blob_vkallocator.reset(new ncnn::VkBlobAllocator(net.vulkan_device()));
    if (!staging_vkallocator)
        staging_vkallocator.reset(new ncnn::VkStagingAllocator(net.vulkan_device()));
    net.opt.blob_vkallocator = blob_vkallocator.get();
    net.opt.workspace_vkallocator = blob_vkallocator.get();
    net.opt.staging_vkallocator = staging_vkallocator.get();
}
#endif
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
    paths.transformer_blocks = {
        (transformer / "blocks.ncnn.param").string(),
        (transformer / "blocks.ncnn.bin").string()};
    paths.transformer_output = {(transformer / "output.ncnn.param").string(),
                                (transformer / "output.ncnn.bin").string()};

    return paths;
}
bool validate_model_paths(const ModelPaths& paths, std::string* error)
{
    if (!valid_file_pair(paths.text_encoder)
        || !valid_file_pair(paths.vae_decoder)
        || !valid_file_pair(paths.transformer_input)
        || !valid_file_pair(paths.transformer_blocks)
        || !valid_file_pair(paths.transformer_output))
    {
        if (error) *error = "model graph files are incomplete";
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

bool get_transformer_weight_size(const ModelPaths& paths, uint64_t& bytes, const std::string& controlnet_param, const std::string& lora_path)
{
    bytes = 0;
    for (const ModelFiles* files : {&paths.transformer_input, &paths.transformer_blocks, &paths.transformer_output})
    {
        std::error_code error;
        const uintmax_t size = std::filesystem::file_size(files->bin, error);
        if (error || size > UINT64_MAX - bytes)
        {
            fprintf(stderr, "failed to get transformer model size %s\n", files->bin.c_str());
            return false;
        }
        bytes += size;
    }
    if (!controlnet_param.empty())
    {
        std::string controlnet_bin = controlnet_param;
        const std::string suffix = ".param";
        if (controlnet_bin.size() < suffix.size()
            || controlnet_bin.compare(controlnet_bin.size() - suffix.size(), suffix.size(), suffix) != 0)
        {
            fprintf(stderr, "ControlNet path must point to a .param file\n");
            return false;
        }
        controlnet_bin.replace(controlnet_bin.size() - suffix.size(), suffix.size(), ".bin");
        std::error_code error;
        const uintmax_t size = std::filesystem::file_size(controlnet_bin, error);
        if (error || size > UINT64_MAX - bytes)
        {
            fprintf(stderr, "failed to get executable ControlNet model size %s\n", controlnet_bin.c_str());
            return false;
        }
        bytes += size;
    }
    if (!lora_path.empty())
    {
        std::error_code error;
        const uintmax_t size = std::filesystem::file_size(lora_path, error);
        // safetensors LoRA tensors are expanded to float32 Mat weights
        if (error || size > (UINT64_MAX - bytes) / 2)
        {
            fprintf(stderr, "failed to estimate LoRA adapter weight size %s\n", lora_path.c_str());
            return false;
        }
        bytes += (uint64_t)size * 2;
    }
    return true;
}

QwenModelSet::~QwenModelSet()
{
    unload_all();
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
#if NCNN_VULKAN
    set_stage_vkallocators(*text_encoder, text_encoder_blob_vkallocator, text_encoder_staging_vkallocator);
#endif
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
#if NCNN_VULKAN
    set_stage_vkallocators(*vision_encoder, vision_encoder_blob_vkallocator, vision_encoder_staging_vkallocator);
#endif
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

bool QwenModelSet::load_transformer(const ModelPaths& paths, const RuntimeConfig& config, const std::string& lora_path, float lora_scale, const std::string& controlnet_path, float control_scale)
{
    unload_transformer();
    if (!lora_path.empty())
    {
        transformer_lora.reset(new TransformerLoRA(lora_path, lora_scale));
        if (!transformer_lora->valid())
        {
            fprintf(stderr, "failed to load transformer LoRA: %s\n", transformer_lora->error().c_str());
            unload_transformer();
            return false;
        }
    }

    RuntimeConfig transformer_config = config;
    if (!controlnet_path.empty())
    {
        transformer_controlnet.reset(new ncnn::Net());
        transformer_controlnet->opt = make_ncnn_option(transformer_config, ModelStage::Transformer);
        transformer_controlnet->opt.lightmode = true;
        std::string controlnet_bin = controlnet_path;
        const std::string suffix = ".param";
        if (controlnet_bin.size() < suffix.size()
            || controlnet_bin.compare(controlnet_bin.size() - suffix.size(), suffix.size(), suffix) != 0)
        {
            fprintf(stderr, "ControlNet path must point to a .param file\n");
            unload_transformer();
            return false;
        }
        controlnet_bin.replace(controlnet_bin.size() - suffix.size(), suffix.size(), ".bin");
        if (transformer_controlnet->load_param(controlnet_path.c_str()) != 0
            || transformer_controlnet->load_model(controlnet_bin.c_str()) != 0)
        {
            fprintf(stderr, "failed to load executable ControlNet graph %s\n", controlnet_path.c_str());
            unload_transformer();
            return false;
        }
        fprintf(stderr, "ControlNet weights = %s-backed\n",
                transformer_config.use_weights_in_host_memory ? "host" : "device");
    }

    transformer_input.reset(new ncnn::Net());
    if (!load_net(*transformer_input, paths.transformer_input, transformer_config,
                  ModelStage::Transformer, transformer_lora.get(), TransformerPart::Input))
    {
        unload_transformer();
        return false;
    }
    transformer_blocks.reset(new ncnn::Net());
    if (!load_transformer_blocks_net(*transformer_blocks, paths.transformer_blocks,
                                     transformer_config, transformer_lora.get(),
                                     !controlnet_path.empty(), control_scale))
    {
        unload_transformer();
        return false;
    }
    transformer_output.reset(new ncnn::Net());
    if (!load_net(*transformer_output, paths.transformer_output, transformer_config,
                  ModelStage::Transformer, transformer_lora.get(), TransformerPart::Output))
    {
        unload_transformer();
        return false;
    }
    if (transformer_lora && transformer_lora->matched_targets() != transformer_lora->expected_targets())
    {
        fprintf(stderr, "LoRA matched %zu of %zu projection targets\n",
                transformer_lora->matched_targets(), transformer_lora->expected_targets());
        unload_transformer();
        return false;
    }
#if NCNN_VULKAN
    set_stage_vkallocators(*transformer_input, transformer_blob_vkallocator, transformer_staging_vkallocator);
    set_stage_vkallocators(*transformer_blocks, transformer_blob_vkallocator, transformer_staging_vkallocator);
    set_stage_vkallocators(*transformer_output, transformer_blob_vkallocator, transformer_staging_vkallocator);
    if (transformer_controlnet)
        set_stage_vkallocators(*transformer_controlnet, transformer_blob_vkallocator, transformer_staging_vkallocator);
#endif
    return true;
}

void QwenModelSet::unload_text_encoder()
{
    text_encoder.reset();
#if NCNN_VULKAN
    text_encoder_blob_vkallocator.reset();
    text_encoder_staging_vkallocator.reset();
#endif
}

void QwenModelSet::unload_vision_encoder()
{
    vision_encoder.reset();
#if NCNN_VULKAN
    vision_encoder_blob_vkallocator.reset();
    vision_encoder_staging_vkallocator.reset();
#endif
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
    // release the large block graph before the small input/output graphs
    transformer_controlnet.reset();
    transformer_blocks.reset();
    transformer_input.reset();
    transformer_output.reset();
    transformer_lora.reset();
#if NCNN_VULKAN
    transformer_blob_vkallocator.reset();
    transformer_staging_vkallocator.reset();
#endif
}

void QwenModelSet::clear_transformer_workspace() const
{
#if NCNN_VULKAN
    // call only after all extractors, commands and shared-pool vkmat views are released
    if (transformer_blob_vkallocator)
        transformer_blob_vkallocator->clear();
    if (transformer_staging_vkallocator)
        transformer_staging_vkallocator->clear();
#endif
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
