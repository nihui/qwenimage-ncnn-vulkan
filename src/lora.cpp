#include "lora.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "layer.h"
#include "modelbin.h"
#include "paramdict.h"

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qwenimage
{

namespace
{

class RecordingModelBin final : public ncnn::ModelBin
{
public:
    explicit RecordingModelBin(const ncnn::ModelBin& source)
        : source_(source)
    {
    }

    ncnn::Mat load(int w, int type) const override
    {
        return record(source_.load(w, type));
    }

    ncnn::Mat load(int w, int h, int type) const override
    {
        return record(source_.load(w, h, type));
    }

    ncnn::Mat load(int w, int h, int c, int type) const override
    {
        return record(source_.load(w, h, c, type));
    }

    ncnn::Mat load(int w, int h, int d, int c, int type) const override
    {
        return record(source_.load(w, h, d, c, type));
    }

    const std::vector<ncnn::Mat>& loaded() const
    {
        return loaded_;
    }

private:
    ncnn::Mat record(const ncnn::Mat& mat) const
    {
        loaded_.push_back(mat);
        return mat;
    }

private:
    const ncnn::ModelBin& source_;
    mutable std::vector<ncnn::Mat> loaded_;
};

struct TensorInfo
{
    enum DType
    {
        Unknown,
        F32,
        BF16,
        F16,
    } dtype = Unknown;
    std::vector<size_t> shape;
    size_t begin = 0;
    size_t end = 0;
};

// the gemm order follows the exported SwiGLU graph, where gemm_4 is the silu gate branch
const char* const kTargetModules[7] = {
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "img_mlp.gate_layer",
    "img_mlp.proj",
    "img_mlp.out",
};

bool starts_with(const std::string& value, const char* prefix)
{
    const size_t n = std::strlen(prefix);
    return value.size() >= n && value.compare(0, n, prefix) == 0;
}

bool parse_block_gemm(const std::string& layer_name, int& block, int& slot)
{
    if (layer_name.size() != 10 || layer_name[0] != 'b'
        || !std::isdigit((unsigned char)layer_name[1])
        || !std::isdigit((unsigned char)layer_name[2])
        || layer_name.compare(3, 6, "_gemm_") != 0
        || layer_name[9] < '0' || layer_name[9] > '6')
        return false;

    block = (layer_name[1] - '0') * 10 + layer_name[2] - '0';
    slot = layer_name[9] - '0';
    return block < 32;
}

bool get_lora_stem(TransformerPart part, const std::string& layer_name,
                   std::string& stem)
{
    if (part == TransformerPart::Blocks)
    {
        int block = 0;
        int slot = 0;
        if (!parse_block_gemm(layer_name, block, slot))
            return false;
        stem = "transformer_blocks." + std::to_string(block) + "."
             + kTargetModules[slot];
        return true;
    }

    if (part == TransformerPart::Input)
    {
        static const char* const names[] = {
            "img_in",
            "txt_in.in_layer",
            "txt_in.out_layer",
            "time_text_embed.timestep_embedder.linear_1",
            "time_text_embed.timestep_embedder.linear_2",
            "modulation.1",
        };
        if (layer_name.size() != 6 || layer_name.compare(0, 5, "gemm_") != 0
            || layer_name[5] < '0' || layer_name[5] > '5')
            return false;
        stem = names[layer_name[5] - '0'];
        return true;
    }

    if (part == TransformerPart::Output && layer_name == "gemm_0")
    {
        stem = "norm_out.linear";
        return true;
    }
    return false;
}

class HeaderParser
{
public:
    HeaderParser(const char* begin, const char* end)
        : p_(begin), end_(end)
    {
    }

    bool parse(std::unordered_map<std::string, TensorInfo>& tensors)
    {
        skip_space();
        if (!consume('{'))
            return false;
        skip_space();
        if (consume('}'))
            return true;

        while (p_ < end_)
        {
            std::string key;
            if (!parse_string(key))
                return false;
            skip_space();
            if (!consume(':'))
                return false;
            skip_space();
            if (key == "__metadata__")
            {
                if (!skip_value())
                    return false;
            }
            else
            {
                TensorInfo info;
                if (!parse_tensor(info))
                    return false;
                tensors.emplace(std::move(key), std::move(info));
            }
            skip_space();
            if (consume('}'))
                return true;
            if (!consume(','))
                return false;
            skip_space();
        }
        return false;
    }

private:
    void skip_space()
    {
        while (p_ < end_ && std::isspace((unsigned char)*p_))
            p_++;
    }

    bool consume(char c)
    {
        if (p_ >= end_ || *p_ != c)
            return false;
        p_++;
        return true;
    }

    bool parse_string(std::string& result)
    {
        if (!consume('"'))
            return false;
        result.clear();
        while (p_ < end_)
        {
            const char c = *p_++;
            if (c == '"')
                return true;
            if (c != '\\')
            {
                result.push_back(c);
                continue;
            }
            if (p_ >= end_)
                return false;
            const char escaped = *p_++;
            switch (escaped)
            {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u':
                // Tensor keys in the HF files are ASCII.  Keep a valid
                // parser for escaped ASCII rather than silently accepting
                // malformed JSON.
                if (end_ - p_ < 4)
                    return false;
                {
                    unsigned int value = 0;
                    for (int i = 0; i < 4; i++)
                    {
                        const char h = p_[i];
                        value <<= 4;
                        if (h >= '0' && h <= '9') value += h - '0';
                        else if (h >= 'a' && h <= 'f') value += h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') value += h - 'A' + 10;
                        else return false;
                    }
                    p_ += 4;
                    if (value <= 0x7f)
                        result.push_back((char)value);
                    else
                        return false;
                }
                break;
            default:
                return false;
            }
        }
        return false;
    }

    bool parse_uint(size_t& result)
    {
        skip_space();
        if (p_ >= end_ || !std::isdigit((unsigned char)*p_))
            return false;
        size_t value = 0;
        while (p_ < end_ && std::isdigit((unsigned char)*p_))
        {
            const size_t digit = (size_t)(*p_ - '0');
            if (value > (std::numeric_limits<size_t>::max() - digit) / 10)
                return false;
            value = value * 10 + digit;
            p_++;
        }
        result = value;
        return true;
    }

    bool parse_array(std::vector<size_t>& values)
    {
        if (!consume('['))
            return false;
        values.clear();
        skip_space();
        if (consume(']'))
            return true;
        while (p_ < end_)
        {
            size_t value = 0;
            if (!parse_uint(value))
                return false;
            values.push_back(value);
            skip_space();
            if (consume(']'))
                return true;
            if (!consume(','))
                return false;
            skip_space();
        }
        return false;
    }

    bool parse_dtype(TensorInfo::DType& dtype)
    {
        std::string value;
        if (!parse_string(value))
            return false;
        if (value == "F32") dtype = TensorInfo::F32;
        else if (value == "BF16") dtype = TensorInfo::BF16;
        else if (value == "F16") dtype = TensorInfo::F16;
        else dtype = TensorInfo::Unknown;
        return true;
    }

    bool parse_tensor(TensorInfo& info)
    {
        if (!consume('{'))
            return false;
        skip_space();
        if (consume('}'))
            return false;
        bool got_dtype = false;
        bool got_shape = false;
        bool got_offsets = false;
        while (p_ < end_)
        {
            std::string key;
            if (!parse_string(key))
                return false;
            skip_space();
            if (!consume(':'))
                return false;
            skip_space();
            if (key == "dtype")
            {
                if (!parse_dtype(info.dtype)) return false;
                got_dtype = true;
            }
            else if (key == "shape")
            {
                if (!parse_array(info.shape)) return false;
                got_shape = true;
            }
            else if (key == "data_offsets")
            {
                std::vector<size_t> offsets;
                if (!parse_array(offsets) || offsets.size() != 2)
                    return false;
                info.begin = offsets[0];
                info.end = offsets[1];
                got_offsets = true;
            }
            else
            {
                if (!skip_value()) return false;
            }
            skip_space();
            if (consume('}'))
                break;
            if (!consume(','))
                return false;
            skip_space();
        }
        return got_dtype && got_shape && got_offsets && info.end >= info.begin;
    }

    bool skip_value()
    {
        skip_space();
        if (p_ >= end_)
            return false;
        if (*p_ == '"')
        {
            std::string ignored;
            return parse_string(ignored);
        }
        if (*p_ == '{')
        {
            p_++;
            skip_space();
            if (consume('}')) return true;
            while (p_ < end_)
            {
                std::string ignored;
                if (!parse_string(ignored)) return false;
                skip_space();
                if (!consume(':')) return false;
                if (!skip_value()) return false;
                skip_space();
                if (consume('}')) return true;
                if (!consume(',')) return false;
                skip_space();
            }
            return false;
        }
        if (*p_ == '[')
        {
            p_++;
            skip_space();
            if (consume(']')) return true;
            while (p_ < end_)
            {
                if (!skip_value()) return false;
                skip_space();
                if (consume(']')) return true;
                if (!consume(',')) return false;
                skip_space();
            }
            return false;
        }
        while (p_ < end_ && *p_ != ',' && *p_ != '}' && *p_ != ']')
            p_++;
        return true;
    }

    const char* p_;
    const char* end_;
};

uint16_t read_u16(const unsigned char* p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

uint32_t read_u32(const unsigned char* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

float bfloat16_to_float(uint16_t value)
{
    const uint32_t bits = (uint32_t)value << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float float16_to_float(uint16_t value)
{
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0)
    {
        if (mantissa != 0)
        {
            uint32_t m = mantissa;
            int e = -1;
            while ((m & 0x400u) == 0)
            {
                m <<= 1;
                e--;
            }
            m &= 0x3ffu;
            bits = sign | (uint32_t)(e + 127) << 23 | (m << 13);
        }
        else
        {
            bits = sign;
        }
    }
    else if (exponent == 0x1fu)
    {
        bits = sign | 0x7f800000u | (mantissa << 13);
    }
    else
    {
        bits = sign | (exponent + 112u) << 23 | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

size_t tensor_element_count(const TensorInfo& info)
{
    size_t count = 1;
    for (size_t dim : info.shape)
    {
        if (dim != 0 && count > std::numeric_limits<size_t>::max() / dim)
            return 0;
        count *= dim;
    }
    return count;
}

class ControlWeightLayer final : public ncnn::Layer
{
public:
    ControlWeightLayer()
    {
        one_blob_only = true;
        support_inplace = false;
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        num_output = pd.get(0, 0);
        bias_term = pd.get(1, 0);
        weight_data_size = pd.get(2, 0);
        if (num_output <= 0 || weight_data_size <= 0
            || (bias_term != 0 && bias_term != 1))
            return -1;
        return 0;
    }

    int load_model(const ncnn::ModelBin& mb) override
    {
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty())
            return -100;
        if (bias_term)
        {
            bias_data = mb.load(num_output, 1);
            if (bias_data.empty())
                return -100;
        }
        return 0;
    }

    int num_output = 0;
    int bias_term = 0;
    int weight_data_size = 0;
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
};

ncnn::Layer* control_weight_layer_creator(void*)
{
    return new ControlWeightLayer;
}

void control_weight_layer_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}

} // namespace

struct LayerRegistry
{
    TransformerLoRA* lora = nullptr;
    TransformerPart part = TransformerPart::Input;
    bool capture_gemm_weights = false;
};

namespace
{

ncnn::Layer* lora_layer_creator(void* userdata);
void adapter_layer_destroyer(ncnn::Layer* layer, void*);

} // namespace

struct TransformerLoRA::Impl
{
    std::unordered_map<std::string, TensorInfo> tensors;
    std::unordered_set<std::string> matched;
    const unsigned char* bytes = nullptr;
    size_t file_size = 0;
    size_t data_offset = 0;
    std::vector<unsigned char> owned;
    LayerRegistry registries[3];
#if defined(__unix__) || defined(__APPLE__)
    int fd = -1;
    void* mapping = nullptr;
#endif

    ~Impl()
    {
#if defined(__unix__) || defined(__APPLE__)
        if (mapping)
            munmap(mapping, file_size);
        if (fd >= 0)
            close(fd);
#endif
    }

    bool open_and_parse(const std::string& path, std::string& error)
    {
#if defined(__unix__) || defined(__APPLE__)
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            error = "open failed: " + std::string(std::strerror(errno));
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8)
        {
            error = "invalid safetensors file size";
            return false;
        }
        file_size = (size_t)st.st_size;
        mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED)
        {
            mapping = nullptr;
            error = "mmap failed: " + std::string(std::strerror(errno));
            return false;
        }
        bytes = static_cast<const unsigned char*>(mapping);
#else
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream)
        {
            error = "failed to open safetensors file";
            return false;
        }
        const std::streamoff size = stream.tellg();
        if (size < 8)
        {
            error = "invalid safetensors file size";
            return false;
        }
        file_size = (size_t)size;
        owned.resize(file_size);
        stream.seekg(0);
        stream.read(reinterpret_cast<char*>(owned.data()), file_size);
        if (!stream)
        {
            error = "failed to read safetensors file";
            return false;
        }
        bytes = owned.data();
#endif

        const uint64_t header_size = (uint64_t)read_u32(bytes)
                                    | (uint64_t)read_u32(bytes + 4) << 32;
        if (header_size > file_size - 8
            || header_size > (uint64_t)std::numeric_limits<size_t>::max())
        {
            error = "invalid safetensors header size";
            return false;
        }
        data_offset = 8 + (size_t)header_size;
        HeaderParser parser(reinterpret_cast<const char*>(bytes + 8),
                            reinterpret_cast<const char*>(bytes + data_offset));
        if (!parser.parse(tensors) || tensors.empty())
        {
            error = "failed to parse safetensors JSON header";
            return false;
        }
        for (const auto& item : tensors)
        {
            const TensorInfo& info = item.second;
            if (info.dtype == TensorInfo::Unknown)
            {
                error = "unsupported tensor dtype for " + item.first;
                return false;
            }
            const size_t bytes_per_element = info.dtype == TensorInfo::F32 ? 4u : 2u;
            const size_t count = tensor_element_count(info);
            if (count == 0 || info.end < info.begin
                || info.end - info.begin != count * bytes_per_element
                || info.end > file_size - data_offset)
            {
                error = "invalid tensor offsets for " + item.first;
                return false;
            }
        }
        return true;
    }

    const TensorInfo* find(const std::string& key) const
    {
        const auto it = tensors.find(key);
        return it == tensors.end() ? nullptr : &it->second;
    }

    const TensorInfo* find_lora_down(const std::string& stem) const
    {
        const char* suffixes[] = {
            ".lora_down", ".lora_down.weight",
            ".lora_A.default.weight", ".lora_A.weight"
        };
        for (const char* suffix : suffixes)
        {
            const TensorInfo* info = find(stem + suffix);
            if (info)
                return info;
        }
        return nullptr;
    }

    const TensorInfo* find_lora_up(const std::string& stem) const
    {
        const char* suffixes[] = {
            ".lora_up", ".lora_up.weight",
            ".lora_B.default.weight", ".lora_B.weight"
        };
        for (const char* suffix : suffixes)
        {
            const TensorInfo* info = find(stem + suffix);
            if (info)
                return info;
        }
        return nullptr;
    }

    const unsigned char* tensor_data(const TensorInfo& info) const
    {
        return bytes + data_offset + info.begin;
    }

    float tensor_value(const TensorInfo& info, size_t index) const
    {
        const size_t element_size = info.dtype == TensorInfo::F32 ? 4u : 2u;
        const unsigned char* p = tensor_data(info) + index * element_size;
        if (info.dtype == TensorInfo::F32)
        {
            const uint32_t bits = read_u32(p);
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        const uint16_t bits = read_u16(p);
        return info.dtype == TensorInfo::F16
             ? float16_to_float(bits) : bfloat16_to_float(bits);
    }
};

struct TransformerLoRA::ControlNetWeights
{
    ncnn::Net net;
    std::unordered_map<std::string, ncnn::Layer*> layers;
    LayerRegistry capture_registry;

    bool open(const std::string& param_path, std::string& error)
    {
        if (param_path.size() < 6
            || param_path.compare(param_path.size() - 6, 6, ".param") != 0)
        {
            error = "ControlNet path must point to controlnet.ncnn.param";
            return false;
        }

        std::string bin_path = param_path.substr(0, param_path.size() - 6) + ".bin";
        net.opt.use_vulkan_compute = false;
        capture_registry.capture_gemm_weights = true;
        if (net.register_custom_layer("Gemm", lora_layer_creator,
                                     adapter_layer_destroyer, &capture_registry) != 0)
        {
            error = "failed to register ControlNet Gemm weight capture layer";
            return false;
        }
        if (net.register_custom_layer("ControlWeight", control_weight_layer_creator,
                                     control_weight_layer_destroyer) != 0)
        {
            error = "failed to register ControlNet weight layer";
            return false;
        }
        if (net.load_param(param_path.c_str()) != 0)
        {
            error = "failed to load ControlNet ncnn param";
            return false;
        }
        if (net.load_model(bin_path.c_str()) != 0)
        {
            error = "failed to load ControlNet ncnn bin";
            return false;
        }

        const size_t weight_layer_count = 162;
        std::vector<ncnn::Layer*> ordered(weight_layer_count, nullptr);
        size_t found = 0;
        for (ncnn::Layer* layer : net.layers())
        {
            if (layer->type != "Gemm" && layer->type != "ControlWeight")
                continue;

            const std::string& name = layer->name;
            static const char prefix[] = "control_weight_";
            if (name.compare(0, sizeof(prefix) - 1, prefix) != 0)
            {
                error = "ControlNet ncnn model has an unnamed weight layer";
                return false;
            }
            const char* number = name.c_str() + sizeof(prefix) - 1;
            char* end = nullptr;
            const unsigned long index = std::strtoul(number, &end, 10);
            if (end == number || !end || *end || index >= ordered.size() || ordered[index])
            {
                error = "ControlNet ncnn model has an invalid weight layer name";
                return false;
            }
            ordered[index] = layer;
            found++;
        }
        if (found != ordered.size())
        {
            error = "ControlNet ncnn model has an unexpected weight layer count";
            return false;
        }

        size_t index = 0;
        const auto add = [&](const std::string& key) -> bool
        {
            if (index >= ordered.size())
                return false;
            layers.emplace(key, ordered[index++]);
            return true;
        };
        static const char* const modules[7] = {
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"
        };
        if (!add("control_img_in"))
        {
            error = "ControlNet ncnn model has no weight layers";
            return false;
        }
        for (int block = 0; block < 16; block++)
        {
            const std::string prefix = "control_blocks." + std::to_string(block) + ".";
            for (int i = 0; i < 7; i++)
                if (!add(prefix + modules[i]))
                {
                    error = "ControlNet ncnn model is missing a projection layer";
                    return false;
                }
            if (!add(prefix + "attn.norm_q") || !add(prefix + "attn.norm_k"))
            {
                error = "ControlNet ncnn model is missing an attention norm layer";
                return false;
            }
            if (block == 0 && !add(prefix + "before_proj"))
            {
                error = "ControlNet ncnn model is missing before_proj";
                return false;
            }
            if (!add(prefix + "after_proj"))
            {
                error = "ControlNet ncnn model is missing after_proj";
                return false;
            }
        }
        if (index != ordered.size())
        {
            error = "ControlNet ncnn model has an unexpected Gemm layer count";
            return false;
        }
        return true;
    }

    bool get(const std::string& key, int output_dim, int input_dim,
             ncnn::Mat& weights, ncnn::Mat& bias) const;
};

TransformerLoRA::TransformerLoRA(const std::string& path, float strength,
                                 const std::string& controlnet_path)
    : impl_(new Impl), path_(path), control_path_(controlnet_path), strength_(strength)
{
    if (!std::isfinite(strength_))
    {
        error_ = "invalid LoRA strength";
        return;
    }
    if (!path_.empty())
    {
        if (!impl_->open_and_parse(path_, error_))
            return;

        const TensorInfo* head = impl_->find("proj_out.weight");
        pdd_output_ = head && head->shape.size() == 3;
        if (pdd_output_ && (head->shape[0] != 4 || head->shape[1] != 64
                            || head->shape[2] != 4096))
        {
            error_ = "unsupported PDD output head shape";
            return;
        }
        expected_targets_ = 0;
        const auto has_pair = [&](TransformerPart part, const std::string& layer_name)
        {
            std::string stem;
            return get_lora_stem(part, layer_name, stem)
                && impl_->find_lora_down(stem) && impl_->find_lora_up(stem);
        };
        for (int block = 0; block < 32; block++)
        {
            char layer_name[16];
            for (int slot = 0; slot < 7; slot++)
            {
                snprintf(layer_name, sizeof(layer_name), "b%02d_gemm_%d", block, slot);
                if (has_pair(TransformerPart::Blocks, layer_name))
                    expected_targets_++;
            }
        }
        for (int slot = 0; slot < 6; slot++)
        {
            char layer_name[16];
            snprintf(layer_name, sizeof(layer_name), "gemm_%d", slot);
            if (has_pair(TransformerPart::Input, layer_name))
                expected_targets_++;
        }
        if (has_pair(TransformerPart::Output, "gemm_0"))
            expected_targets_++;
        if (expected_targets_ == 0)
        {
            error_ = "LoRA file has no supported Qwen-Image projection targets";
            return;
        }
        valid_ = true;
    }
    if (!control_path_.empty())
    {
        control_impl_.reset(new ControlNetWeights);
        if (!control_impl_->open(control_path_, error_))
            return;
        control_valid_ = true;
    }
    if (!valid_ && !control_valid_)
        error_ = "no adapter file was specified";
}

TransformerLoRA::~TransformerLoRA() = default;

float TransformerLoRA::sigma(int index) const
{
    static const float sigmas[5] = {
        1.f, 0.9169867038726807f, 0.7861579060554504f,
        0.5494909882545471f, 0.f
    };
    if (index < 0 || index > 4)
        return 0.f;
    return sigmas[index];
}

namespace
{

template<typename TensorFile>
bool safetensors_tensor_to_mat(const TensorFile* impl,
                         const TensorInfo* info, const std::vector<size_t>& shape,
                         ncnn::Mat& mat)
{
    if (!impl || !info || info->shape != shape || shape.empty())
        return false;
    for (size_t dim : shape)
        if (dim > (size_t)std::numeric_limits<int>::max())
            return false;

    if (shape.size() == 1)
    {
        mat.create((int)shape[0]);
        if (mat.empty())
            return false;
        float* dst = (float*)mat.data;
        for (size_t i = 0; i < shape[0]; i++)
            dst[i] = impl->tensor_value(*info, i);
    }
    else if (shape.size() == 2)
    {
        const int width = (int)shape[1];
        const int height = (int)shape[0];
        mat.create(width, height);
        if (mat.empty())
            return false;
        for (int y = 0; y < height; y++)
        {
            float* dst = mat.row(y);
            for (int x = 0; x < width; x++)
                dst[x] = impl->tensor_value(*info, (size_t)y * width + x);
        }
    }
    else
        return false;

    return true;
}

const char* control_target_modules[7] = {
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"
};

bool parse_control_projection_name(const std::string& layer_name,
                                   std::string& stem)
{
    if (layer_name == "control_img_in")
    {
        stem = "control_img_in";
        return true;
    }
    if (layer_name == "control_before")
    {
        stem = "control_blocks.0.before_proj";
        return true;
    }
    if (layer_name.size() == 15 && starts_with(layer_name, "control_after_")
        && std::isdigit((unsigned char)layer_name[14]))
    {
        const int index = layer_name[14] - '0';
        stem = "control_blocks." + std::to_string(index) + ".after_proj";
        return index < 16;
    }
    if (layer_name.size() == 16 && starts_with(layer_name, "control_after_")
        && std::isdigit((unsigned char)layer_name[14])
        && std::isdigit((unsigned char)layer_name[15]))
    {
        const int index = (layer_name[14] - '0') * 10 + layer_name[15] - '0';
        stem = "control_blocks." + std::to_string(index) + ".after_proj";
        return index < 16;
    }
    return false;
}

} // namespace

bool TransformerLoRA::is_control_norm_layer(const std::string& layer_name) const
{
    if (!control_valid_ || layer_name.size() != 11 || layer_name[0] != 'b'
        || !std::isdigit((unsigned char)layer_name[1])
        || !std::isdigit((unsigned char)layer_name[2])
        || layer_name.compare(3, 6, "_rmsn_") != 0
        || layer_name[9] != '1'
        || (layer_name[10] != '1' && layer_name[10] != '2'))
        return false;
    const int block = (layer_name[1] - '0') * 10 + layer_name[2] - '0';
    return block < 32 && (block % 2) == 0;
}

bool TransformerLoRA::is_control_target_layer(const std::string& layer_name) const
{
    int block = 0;
    int slot = 0;
    return control_valid_ && parse_block_gemm(layer_name, block, slot)
        && (block % 2 == 0);
}

bool TransformerLoRA::load_control_layer_weights(const std::string& layer_name,
                                                  int output_dim, int input_dim,
                                                  ncnn::Mat& weights,
                                                  ncnn::Mat& bias)
{
    int block = 0;
    int slot = 0;
    if (!control_valid_ || !parse_block_gemm(layer_name, block, slot)
        || block % 2 != 0)
        return false;
    const int control_block = block / 2;
    const std::string key = "control_blocks." + std::to_string(control_block)
                          + "." + control_target_modules[slot];
    if (!control_impl_->get(key, output_dim, input_dim, weights, bias))
    {
        error_ = "control Gemm shape mismatch for " + key;
        return false;
    }
    return true;
}

bool TransformerLoRA::load_control_norm_weight(const std::string& layer_name,
                                                int size, ncnn::Mat& weight)
{
    if (!control_valid_ || layer_name.size() != 11 || layer_name[0] != 'b'
        || !std::isdigit((unsigned char)layer_name[1])
        || !std::isdigit((unsigned char)layer_name[2])
        || layer_name.compare(3, 6, "_rmsn_") != 0
        || layer_name[9] != '1'
        || (layer_name[10] != '1' && layer_name[10] != '2'))
        return false;

    const int block = (layer_name[1] - '0') * 10 + layer_name[2] - '0';
    if (block >= 32 || block % 2 != 0)
        return false;
    const int control_block = block / 2;
    const char* name = layer_name[10] == '1' ? "norm_q" : "norm_k";
    const std::string key = "control_blocks." + std::to_string(control_block)
                          + ".attn." + name;
    ncnn::Mat gamma;
    ncnn::Mat unused_bias;
    if (!control_impl_->get(key, size, 1, gamma, unused_bias))
    {
        error_ = "control RMSNorm shape mismatch for " + key;
        return false;
    }
    weight = gamma.reshape(size);
    return !weight.empty();
}

bool TransformerLoRA::load_control_projection(const std::string& layer_name,
                                               int output_dim, int input_dim,
                                               ncnn::Mat& weights,
                                               ncnn::Mat& bias)
{
    std::string key;
    if (!control_valid_ || !parse_control_projection_name(layer_name, key))
        return false;

    if (!control_impl_->get(key, output_dim, input_dim, weights, bias)
        || bias.empty())
    {
        error_ = "control projection shape mismatch for " + key;
        return false;
    }
    return true;
}

bool TransformerLoRA::is_target_layer(TransformerPart part,
                                      const std::string& layer_name) const
{
    std::string stem;
    return valid_ && get_lora_stem(part, layer_name, stem)
        && impl_->find_lora_down(stem) && impl_->find_lora_up(stem);
}

bool TransformerLoRA::load_layer_lora(TransformerPart part,
                                      const std::string& layer_name,
                                      int output_dim, int input_dim,
                                      ncnn::Mat& down, ncnn::Mat& up,
                                      float& output_scale)
{
    std::string stem;
    if (!valid_ || !get_lora_stem(part, layer_name, stem))
        return false;

    const TensorInfo* down_info = impl_->find_lora_down(stem);
    const TensorInfo* up_info = impl_->find_lora_up(stem);
    if (!down_info || !up_info)
    {
        error_ = "missing LoRA tensors for " + stem;
        return false;
    }
    if (down_info->shape.size() != 2 || up_info->shape.size() != 2
        || down_info->shape[1] != (size_t)input_dim
        || up_info->shape[0] != (size_t)output_dim
        || down_info->shape[0] != up_info->shape[1]
        || down_info->shape[0] == 0
        || down_info->shape[0] > (size_t)std::numeric_limits<int>::max())
    {
        error_ = "LoRA shape mismatch for " + stem;
        return false;
    }

    const int rank = (int)down_info->shape[0];
    float alpha = (float)rank;
    const TensorInfo* alpha_info = impl_->find(stem + ".alpha");
    if (!alpha_info) alpha_info = impl_->find(stem + ".lora_alpha");
    if (alpha_info)
    {
        if (tensor_element_count(*alpha_info) != 1)
        {
            error_ = "invalid LoRA alpha for " + stem;
            return false;
        }
        alpha = impl_->tensor_value(*alpha_info, 0);
    }

    down.create(input_dim, rank, (size_t)4u, 1);
    up.create(rank, output_dim, (size_t)4u, 1);
    if (down.empty() || up.empty())
    {
        error_ = "failed to allocate LoRA matrices for " + stem;
        return false;
    }

    const size_t down_count = tensor_element_count(*down_info);
    const size_t up_count = tensor_element_count(*up_info);
    float* down_ptr = static_cast<float*>(down.data);
    float* up_ptr = static_cast<float*>(up.data);
    for (size_t i = 0; i < down_count; i++)
        down_ptr[i] = impl_->tensor_value(*down_info, i);
    for (size_t i = 0; i < up_count; i++)
        up_ptr[i] = impl_->tensor_value(*up_info, i);

    output_scale = strength_ * alpha / (float)rank;
    if (!std::isfinite(output_scale))
    {
        error_ = "invalid LoRA scale for " + stem;
        return false;
    }
    if (impl_->matched.insert(stem).second)
        matched_targets_++;
    return true;
}

bool TransformerLoRA::load_norm_weight(TransformerPart part,
                                      const std::string& layer_name,
                                      int size, ncnn::Mat& weight)
{
    std::string key;
    if (part == TransformerPart::Input && layer_name == "rmsn_7")
    {
        key = "txt_in.text_norm.weight";
    }
    else if (part == TransformerPart::Blocks && layer_name.size() == 11
             && layer_name[0] == 'b'
             && std::isdigit((unsigned char)layer_name[1])
             && std::isdigit((unsigned char)layer_name[2])
             && layer_name.compare(3, 6, "_rmsn_") == 0
             && layer_name[9] == '1'
             && (layer_name[10] == '1' || layer_name[10] == '2'))
    {
        const int block = (layer_name[1] - '0') * 10 + layer_name[2] - '0';
        if (block >= 32)
            return false;
        const char* kind = layer_name[10] == '1' ? "norm_q" : "norm_k";
        key = "transformer_blocks." + std::to_string(block) + ".attn."
            + kind + ".weight";
    }
    else
    {
        return false;
    }

    const TensorInfo* info = impl_->find(key);
    if (!info)
        return false;
    if (tensor_element_count(*info) != (size_t)size)
    {
        error_ = "RMSNorm shape mismatch for " + key;
        return false;
    }
    weight.create(size);
    if (weight.empty())
        return false;
    float* ptr = weight;
    for (int i = 0; i < size; i++)
        ptr[i] = impl_->tensor_value(*info, (size_t)i);
    return true;
}

bool TransformerLoRA::load_output_heads(int output_dim, int input_dim,
                                        std::vector<ncnn::Mat>& heads)
{
    if (!valid_ || !pdd_output_)
        return false;
    const TensorInfo* info = impl_->find("proj_out.weight");
    if (!info || info->shape.size() != 3 || info->shape[0] != 4
        || info->shape[1] != (size_t)output_dim
        || info->shape[2] != (size_t)input_dim)
    {
        error_ = "PDD output head shape mismatch";
        return false;
    }
    heads.resize(4);
    const size_t count = (size_t)output_dim * input_dim;
    for (int h = 0; h < 4; h++)
    {
        heads[h].create(input_dim, output_dim, (size_t)4u, 1);
        if (heads[h].empty())
            return false;
        float* ptr = heads[h];
        for (size_t i = 0; i < count; i++)
            ptr[i] = impl_->tensor_value(*info, (size_t)h * count + i);
    }
    return true;
}

namespace
{

ncnn::Layer* create_gemm()
{
#if NCNN_VULKAN
    return ncnn::create_layer_vulkan("Gemm");
#else
    return ncnn::create_layer_cpu("Gemm");
#endif
}

ncnn::Layer* create_binaryop()
{
#if NCNN_VULKAN
    return ncnn::create_layer_vulkan("BinaryOp");
#else
    return ncnn::create_layer_cpu("BinaryOp");
#endif
}

ncnn::Layer* create_rmsnorm()
{
#if NCNN_VULKAN
    return ncnn::create_layer_vulkan("RMSNorm");
#else
    return ncnn::create_layer_cpu("RMSNorm");
#endif
}

void copy_layer_flags(ncnn::Layer* destination, const ncnn::Layer* source)
{
    destination->one_blob_only = source->one_blob_only;
    destination->support_inplace = source->support_inplace;
    destination->support_vulkan = source->support_vulkan;
    destination->support_packing = source->support_packing;
    destination->support_bf16_storage = source->support_bf16_storage;
    destination->support_fp16_storage = source->support_fp16_storage;
    destination->support_int8_storage = source->support_int8_storage;
    destination->support_tensor_storage = source->support_tensor_storage;
    destination->support_vulkan_packing = source->support_vulkan_packing;
    destination->support_any_packing = source->support_any_packing;
    destination->support_vulkan_any_packing = source->support_vulkan_any_packing;
    destination->support_batch = source->support_batch;
    destination->support_reserved_2 = source->support_reserved_2;
    destination->support_reserved_3 = source->support_reserved_3;
    destination->support_reserved_4 = source->support_reserved_4;
    destination->support_reserved_5 = source->support_reserved_5;
    destination->support_reserved_6 = source->support_reserved_6;
    destination->support_reserved_7 = source->support_reserved_7;
    destination->support_reserved_8 = source->support_reserved_8;
    destination->support_reserved_9 = source->support_reserved_9;
}

bool configure_gemm(ncnn::Layer* gemm, int output_dim, int input_dim,
                    const ncnn::Mat& weights, float alpha)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, 1.f);
    pd.set(2, 0);
    // weight matrices use output-by-input storage
    pd.set(3, 1);
    pd.set(4, 0);
    pd.set(5, 1);
    pd.set(6, 0);
    pd.set(7, 0);
    pd.set(8, output_dim);
    pd.set(9, input_dim);
    pd.set(10, 0);
    if (gemm->load_param(pd) != 0)
        return false;

    ncnn::ModelBinFromMatArray mb(&weights);
    return gemm->load_model(mb) == 0;
}

class AdapterGemm final : public ncnn::Layer
{
public:
    explicit AdapterGemm(LayerRegistry* registry)
        : registry_(registry), base_(create_gemm())
    {
        copy_layer_flags(this, base_);
#if NCNN_VULKAN
        base_->vkdev = vkdev;
#endif
    }

    ~AdapterGemm() override
    {
        for (ncnn::Layer* head : pdd_heads_)
            delete head;
        delete add_;
        delete up_;
        delete down_;
        delete control_;
        delete base_;
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        const int ret = base_->load_param(pd);
        if (ret != 0)
            return ret;
        copy_layer_flags(this, base_);
#if NCNN_VULKAN
        base_->vkdev = vkdev;
#endif
        if (registry_ && registry_->lora)
        {
            target_ = registry_->lora->is_target_layer(registry_->part, name);
            control_target_ = registry_->part == TransformerPart::Blocks
                           && registry_->lora->is_control_target_layer(name);
            pdd_head_ = registry_->part == TransformerPart::Output
                     && name == "gemm_1"
                     && registry_->lora->has_pdd_output();
        }
        pd_ = pd;
        return 0;
    }

    int load_model(const ncnn::ModelBin& mb) override
    {
        int ret = 0;
        if (registry_ && registry_->capture_gemm_weights)
        {
            RecordingModelBin recording(mb);
            ret = base_->load_model(recording);
            if (ret != 0)
                return ret;

            const std::vector<ncnn::Mat>& loaded = recording.loaded();
            size_t index = pd_.get(4, 0) == 1 ? 1u : 0u;
            if (pd_.get(5, 0) == 1)
            {
                if (index >= loaded.size())
                    return -1;
                captured_B_data_ = loaded[index++];
            }
            if (pd_.get(6, 0) == 1 && pd_.get(10, 0) != -1)
            {
                if (index >= loaded.size())
                    return -1;
                captured_C_data_ = loaded[index];
            }
        }
        else
        {
            ret = base_->load_model(mb);
            if (ret != 0)
                return ret;
        }

        const int output_dim = pd_.get(8, 0);
        const int input_dim = pd_.get(9, 0);
        if (control_target_)
        {
            ncnn::Mat weights;
            ncnn::Mat bias;
            if (!registry_->lora->load_control_layer_weights(
                    name, output_dim, input_dim, weights, bias))
                return -1;
            control_ = create_gemm();
            ncnn::ParamDict control_pd = pd_;
            control_pd.set(6, 0);
            if (!control_ || control_->load_param(control_pd) != 0)
                return -1;
            ncnn::ModelBinFromMatArray control_mb(&weights);
            if (control_->load_model(control_mb) != 0)
                return -1;
            (void)bias;
            copy_layer_flags(control_, base_);
#if NCNN_VULKAN
            control_->vkdev = vkdev;
#endif
        }

        if (target_)
        {
            ncnn::Mat down_weights;
            ncnn::Mat up_weights;
            float output_scale = 0.f;
            if (!registry_->lora->load_layer_lora(
                    registry_->part, name, output_dim, input_dim,
                    down_weights, up_weights, output_scale))
                return -1;
            const int rank = down_weights.h;
            down_ = create_gemm();
            up_ = create_gemm();
            add_ = create_binaryop();
            if (!down_ || !up_ || !add_
                || !configure_gemm(down_, rank, input_dim,
                                   down_weights, 1.f)
                || !configure_gemm(up_, output_dim, rank,
                                   up_weights, output_scale))
                return -1;

            ncnn::ParamDict add_pd;
            add_pd.set(0, 0);
            add_pd.set(1, 0);
            add_pd.set(2, 0.f);
            if (add_->load_param(add_pd) != 0)
                return -1;
#if NCNN_VULKAN
            down_->vkdev = vkdev;
            up_->vkdev = vkdev;
            add_->vkdev = vkdev;
#endif
        }

        if (pdd_head_)
        {
            std::vector<ncnn::Mat> heads;
            if (!registry_->lora->load_output_heads(output_dim, input_dim, heads))
                return -1;
            for (const ncnn::Mat& weights : heads)
            {
                ncnn::Layer* head = create_gemm();
                if (!head || !configure_gemm(head, output_dim,
                                              input_dim, weights, 1.f))
                {
                    delete head;
                    return -1;
                }
#if NCNN_VULKAN
                head->vkdev = vkdev;
#endif
                pdd_heads_.push_back(head);
            }
        }
        return 0;
    }

    bool captured_weights(ncnn::Mat& weights, ncnn::Mat& bias) const
    {
        if (captured_B_data_.empty())
            return false;
        weights = captured_B_data_;
        bias = captured_C_data_;
        return true;
    }

    int create_pipeline(const ncnn::Option& opt) override
    {
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (base_->create_pipeline(opt) != 0)
                return -1;
            if (control_target_ && control_->create_pipeline(opt) != 0)
                return -1;
            if (target_ && (down_->create_pipeline(opt) != 0
                            || up_->create_pipeline(opt) != 0
                            || add_->create_pipeline(opt) != 0))
                return -1;
            for (ncnn::Layer* head : pdd_heads_)
                if (head->create_pipeline(opt) != 0)
                    return -1;
        }
#else
        (void)opt;
#endif
        return 0;
    }

    int destroy_pipeline(const ncnn::Option& opt) override
    {
        int ret = 0;
        if (control_ && control_->destroy_pipeline(opt) != 0) ret = -1;
        if (add_ && add_->destroy_pipeline(opt) != 0) ret = -1;
        if (up_ && up_->destroy_pipeline(opt) != 0) ret = -1;
        if (down_ && down_->destroy_pipeline(opt) != 0) ret = -1;
        for (ncnn::Layer* head : pdd_heads_)
            if (head->destroy_pipeline(opt) != 0) ret = -1;
        if (base_ && base_->destroy_pipeline(opt) != 0) ret = -1;
        return ret;
    }

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override
    {
        if (base_->upload_model(cmd, opt) != 0)
            return -1;
        if (control_target_ && control_->upload_model(cmd, opt) != 0)
            return -1;
        if (target_ && (down_->upload_model(cmd, opt) != 0
                        || up_->upload_model(cmd, opt) != 0))
            return -1;
        for (ncnn::Layer* head : pdd_heads_)
            if (head->upload_model(cmd, opt) != 0)
                return -1;
        return 0;
    }
#endif

    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override
    {
        if (control_target_ && registry_->lora->control_active())
            return control_->forward(bottom_blobs, top_blobs, opt);
        if (pdd_head_)
        {
            const int step = registry_->lora->step();
            if (step < 0 || step >= (int)pdd_heads_.size())
                return -100;
            return pdd_heads_[step]->forward(bottom_blobs, top_blobs, opt);
        }
        if (!target_)
            return base_->forward(bottom_blobs, top_blobs, opt);

        std::vector<ncnn::Mat> base_top(1);
        std::vector<ncnn::Mat> down_top(1);
        std::vector<ncnn::Mat> lora_top(1);
        if (base_->forward(bottom_blobs, base_top, opt) != 0
            || down_->forward(bottom_blobs, down_top, opt) != 0
            || up_->forward(down_top, lora_top, opt) != 0)
            return -100;
        std::vector<ncnn::Mat> add_bottoms = {base_top[0], lora_top[0]};
        return add_->forward(add_bottoms, top_blobs, opt);
    }

    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                const ncnn::Option& opt) const override
    {
        std::vector<ncnn::Mat> bottom_blobs(1, bottom_blob);
        std::vector<ncnn::Mat> top_blobs(1);
        const int ret = forward(bottom_blobs, top_blobs, opt);
        if (ret == 0)
            top_blob = top_blobs[0];
        return ret;
    }

#if NCNN_VULKAN
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override
    {
        if (control_target_ && registry_->lora->control_active())
            return static_cast<ncnn::Layer*>(control_)->forward(
                bottom_blobs, top_blobs, cmd, opt);
        if (pdd_head_)
        {
            const int step = registry_->lora->step();
            if (step < 0 || step >= (int)pdd_heads_.size())
                return -100;
            return static_cast<ncnn::Layer*>(pdd_heads_[step])->forward(
                bottom_blobs, top_blobs, cmd, opt);
        }
        if (!target_)
            return static_cast<ncnn::Layer*>(base_)->forward(
                bottom_blobs, top_blobs, cmd, opt);

        std::vector<ncnn::VkMat> base_top(1);
        std::vector<ncnn::VkMat> down_top(1);
        std::vector<ncnn::VkMat> lora_top(1);
        if (static_cast<ncnn::Layer*>(base_)->forward(
                bottom_blobs, base_top, cmd, opt) != 0
            || static_cast<ncnn::Layer*>(down_)->forward(
                bottom_blobs, down_top, cmd, opt) != 0
            || static_cast<ncnn::Layer*>(up_)->forward(
                down_top, lora_top, cmd, opt) != 0)
            return -100;
        std::vector<ncnn::VkMat> add_bottoms = {base_top[0], lora_top[0]};
        return static_cast<ncnn::Layer*>(add_)->forward(
            add_bottoms, top_blobs, cmd, opt);
    }

    int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        std::vector<ncnn::VkMat> bottom_blobs(1, bottom_blob);
        std::vector<ncnn::VkMat> top_blobs(1);
        const int ret = forward(bottom_blobs, top_blobs, cmd, opt);
        if (ret == 0)
            top_blob = top_blobs[0];
        return ret;
    }
#endif

private:
    LayerRegistry* registry_ = nullptr;
    ncnn::Layer* base_ = nullptr;
    ncnn::Layer* control_ = nullptr;
    ncnn::Layer* down_ = nullptr;
    ncnn::Layer* up_ = nullptr;
    ncnn::Layer* add_ = nullptr;
    ncnn::ParamDict pd_;
    std::vector<ncnn::Layer*> pdd_heads_;
    ncnn::Mat captured_B_data_;
    ncnn::Mat captured_C_data_;
    bool target_ = false;
    bool control_target_ = false;
    bool pdd_head_ = false;
};

class AdapterRMSNorm final : public ncnn::Layer
{
public:
    explicit AdapterRMSNorm(LayerRegistry* registry)
        : registry_(registry), base_(create_rmsnorm())
    {
        copy_layer_flags(this, base_);
#if NCNN_VULKAN
        base_->vkdev = vkdev;
#endif
    }

    ~AdapterRMSNorm() override
    {
        delete control_;
        delete base_;
    }

    int load_param(const ncnn::ParamDict& pd) override
    {
        const int ret = base_->load_param(pd);
        if (ret != 0)
            return ret;
        copy_layer_flags(this, base_);
#if NCNN_VULKAN
        base_->vkdev = vkdev;
#endif
        control_target_ = registry_ && registry_->lora
                       && registry_->part == TransformerPart::Blocks
                       && registry_->lora->is_control_norm_layer(name);
        if (control_target_)
        {
            control_ = create_rmsnorm();
            if (!control_ || control_->load_param(pd) != 0)
                return -1;
            copy_layer_flags(control_, base_);
#if NCNN_VULKAN
            control_->vkdev = vkdev;
#endif
        }
        pd_ = pd;
        return 0;
    }

    int load_model(const ncnn::ModelBin& mb) override
    {
        const int ret = base_->load_model(mb);
        if (ret != 0 || !registry_ || !registry_->lora)
            return ret;

        const int affine_size = pd_.get(0, 0);
        if (control_target_)
        {
            ncnn::Mat control_weight;
            if (!registry_->lora->load_control_norm_weight(
                    name, affine_size, control_weight))
                return -1;
            ncnn::ModelBinFromMatArray control_mb(&control_weight);
            if (control_->load_model(control_mb) != 0)
                return -1;
        }

        ncnn::Mat weight;
        if (registry_->lora->load_norm_weight(registry_->part, name,
                                               affine_size, weight))
        {
            ncnn::Layer* replacement = create_rmsnorm();
            if (!replacement || replacement->load_param(pd_) != 0)
            {
                delete replacement;
                return -1;
            }
#if NCNN_VULKAN
            replacement->vkdev = vkdev;
#endif
            ncnn::ModelBinFromMatArray norm_mb(&weight);
            if (replacement->load_model(norm_mb) != 0)
            {
                delete replacement;
                return -1;
            }
            delete base_;
            base_ = replacement;
        }
        return 0;
    }

    int create_pipeline(const ncnn::Option& opt) override
    {
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
        {
            if (base_->create_pipeline(opt) != 0)
                return -1;
            if (control_target_ && control_->create_pipeline(opt) != 0)
                return -1;
        }
#else
        (void)opt;
#endif
        return 0;
    }

    int destroy_pipeline(const ncnn::Option& opt) override
    {
#if NCNN_VULKAN
        int ret = 0;
        if (control_ && control_->destroy_pipeline(opt) != 0) ret = -1;
        if (base_ && base_->destroy_pipeline(opt) != 0) ret = -1;
        return ret;
#else
        (void)opt;
        return 0;
#endif
    }

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override
    {
        if (base_->upload_model(cmd, opt) != 0)
            return -1;
        if (control_target_ && control_->upload_model(cmd, opt) != 0)
            return -1;
        return 0;
    }
#endif

    int forward_inplace(ncnn::Mat& bottom_top_blob,
                        const ncnn::Option& opt) const override
    {
        if (control_target_ && registry_->lora->control_active())
            return control_->forward_inplace(bottom_top_blob, opt);
        return base_->forward_inplace(bottom_top_blob, opt);
    }

#if NCNN_VULKAN
    int forward_inplace(ncnn::VkMat& bottom_top_blob, ncnn::VkCompute& cmd,
                        const ncnn::Option& opt) const override
    {
        if (control_target_ && registry_->lora->control_active())
            return static_cast<ncnn::Layer*>(control_)->forward_inplace(
                bottom_top_blob, cmd, opt);
        return static_cast<ncnn::Layer*>(base_)->forward_inplace(
            bottom_top_blob, cmd, opt);
    }
#endif

private:
    LayerRegistry* registry_ = nullptr;
    ncnn::Layer* base_ = nullptr;
    ncnn::Layer* control_ = nullptr;
    ncnn::ParamDict pd_;
    bool control_target_ = false;
};

class ControlProjection final : public ncnn::Layer
{
public:
    explicit ControlProjection(LayerRegistry* registry)
        : registry_(registry), gemm_(create_gemm())
    {
        copy_layer_flags(this, gemm_);
#if NCNN_VULKAN
        gemm_->vkdev = vkdev;
#endif
    }

    ~ControlProjection() override { delete gemm_; }

    int load_param(const ncnn::ParamDict& pd) override
    {
        if (!registry_ || !registry_->lora)
            return -1;
        const int output_dim = pd.get(0, 0);
        const int input_dim = pd.get(1, 0);
        if (output_dim <= 0 || input_dim <= 0)
            return -1;
        ncnn::Mat weights;
        ncnn::Mat bias;
        if (!registry_->lora->load_control_projection(
                name, output_dim, input_dim, weights, bias))
            return -1;

        is_after_ = starts_with(name, "control_after_");
        ncnn::ParamDict gemm_pd;
        gemm_pd.set(0, is_after_ ? registry_->lora->control_scale() : 1.f);
        gemm_pd.set(1, 1.f);
        gemm_pd.set(2, 0);
        gemm_pd.set(3, 0);
        gemm_pd.set(4, 0);
        gemm_pd.set(5, 1);
        gemm_pd.set(6, bias.empty() ? 0 : 1);
        gemm_pd.set(7, 0);
        gemm_pd.set(8, output_dim);
        gemm_pd.set(9, input_dim);
        gemm_pd.set(10, -1);
        if (gemm_->load_param(gemm_pd) != 0)
            return -1;
        ncnn::ModelBinFromMatArray gemm_mb(&weights);
        if (gemm_->load_model(gemm_mb) != 0)
            return -1;
        (void)bias;
        copy_layer_flags(this, gemm_);
#if NCNN_VULKAN
        gemm_->vkdev = vkdev;
#endif
        return 0;
    }

    int load_model(const ncnn::ModelBin&) override { return 0; }

    int create_pipeline(const ncnn::Option& opt) override
    {
#if NCNN_VULKAN
        if (opt.use_vulkan_compute)
            return gemm_->create_pipeline(opt);
#else
        (void)opt;
#endif
        return 0;
    }

    int destroy_pipeline(const ncnn::Option& opt) override
    {
#if NCNN_VULKAN
        return gemm_->destroy_pipeline(opt);
#else
        (void)opt;
        return 0;
#endif
    }

#if NCNN_VULKAN
    int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override
    {
        return gemm_->upload_model(cmd, opt);
    }
#endif

    int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                std::vector<ncnn::Mat>& top_blobs,
                const ncnn::Option& opt) const override
    {
        return gemm_->forward(bottom_blobs, top_blobs, opt);
    }

    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                const ncnn::Option& opt) const override
    {
        std::vector<ncnn::Mat> bottom_blobs(1, bottom_blob);
        std::vector<ncnn::Mat> top_blobs(1);
        const int ret = forward(bottom_blobs, top_blobs, opt);
        if (ret == 0) top_blob = top_blobs[0];
        return ret;
    }

#if NCNN_VULKAN
    int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                std::vector<ncnn::VkMat>& top_blobs, ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override
    {
        return static_cast<ncnn::Layer*>(gemm_)->forward(
            bottom_blobs, top_blobs, cmd, opt);
    }

    int forward(const ncnn::VkMat& bottom_blob, ncnn::VkMat& top_blob,
                ncnn::VkCompute& cmd, const ncnn::Option& opt) const override
    {
        std::vector<ncnn::VkMat> bottom_blobs(1, bottom_blob);
        std::vector<ncnn::VkMat> top_blobs(1);
        const int ret = forward(bottom_blobs, top_blobs, cmd, opt);
        if (ret == 0) top_blob = top_blobs[0];
        return ret;
    }
#endif

private:
    LayerRegistry* registry_ = nullptr;
    ncnn::Layer* gemm_ = nullptr;
    bool is_after_ = false;
};

ncnn::Layer* lora_layer_creator(void* userdata)
{
    return new AdapterGemm(static_cast<LayerRegistry*>(userdata));
}

ncnn::Layer* rmsnorm_layer_creator(void* userdata)
{
    return new AdapterRMSNorm(static_cast<LayerRegistry*>(userdata));
}

ncnn::Layer* control_projection_layer_creator(void* userdata)
{
    return new ControlProjection(static_cast<LayerRegistry*>(userdata));
}

void adapter_layer_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}

} // namespace

bool TransformerLoRA::ControlNetWeights::get(const std::string& key,
                                               int output_dim, int input_dim,
                                               ncnn::Mat& weights,
                                               ncnn::Mat& bias) const
{
    const auto it = layers.find(key);
    if (it == layers.end())
        return false;

    if (it->second->type == "Gemm")
    {
        const AdapterGemm* gemm = static_cast<const AdapterGemm*>(it->second);
        if (!gemm->captured_weights(weights, bias)
            || weights.w != input_dim || weights.h != output_dim)
            return false;
        return true;
    }

    if (it->second->type == "ControlWeight")
    {
        const ControlWeightLayer* weight_layer =
            static_cast<const ControlWeightLayer*>(it->second);
        const ncnn::Mat& weight_data = weight_layer->weight_data;
        const size_t weight_count = (size_t)weight_data.w * weight_data.h
                                  * weight_data.d * weight_data.c;
        if (weight_data.empty()
            || weight_count != (size_t)output_dim * input_dim)
            return false;
        weights = weight_data.reshape(input_dim, output_dim);
        bias = weight_layer->bias_data;
        return !weights.empty();
    }
    return false;
}

int register_transformer_lora(ncnn::Net& net, TransformerLoRA* lora,
                              TransformerPart part)
{
#if NCNN_STRING
    if (!lora || !lora->impl_)
        return -1;
    const int index = static_cast<int>(part);
    if (index < 0 || index >= 3)
        return -1;
    LayerRegistry& registry = lora->impl_->registries[index];
    registry.lora = lora;
    registry.part = part;
    if (net.register_custom_layer("Gemm", lora_layer_creator,
                                  adapter_layer_destroyer, &registry) != 0)
        return -1;
    if (net.register_custom_layer("RMSNorm", rmsnorm_layer_creator,
                                  adapter_layer_destroyer, &registry) != 0)
        return -1;
    if (lora->has_controlnet()
        && net.register_custom_layer("ControlProjection",
                                     control_projection_layer_creator,
                                     adapter_layer_destroyer, &registry) != 0)
        return -1;
    return 0;
#else
    (void)net;
    (void)lora;
    (void)part;
    return -1;
#endif
}

} // namespace qwenimage
