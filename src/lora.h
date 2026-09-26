#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "mat.h"
#include "net.h"

namespace qwenimage
{

enum class TransformerPart
{
    Input,
    Blocks,
    Output
};

class TransformerLoRA
{
public:
    TransformerLoRA(const std::string& path, float strength = 1.f, const std::string& controlnet_path = std::string());
    ~TransformerLoRA();

    bool valid() const { return valid_ || control_valid_; }
    bool has_lora() const { return valid_; }
    bool has_controlnet() const { return control_valid_; }
    const std::string& error() const { return error_; }
    const std::string& path() const { return path_; }
    float strength() const { return strength_; }
    bool has_pdd_output() const { return pdd_output_; }
    int required_steps() const { return pdd_output_ ? 4 : 0; }
    float sigma(int index) const;
    void set_step(int index) { step_ = index; }
    int step() const { return step_; }
    void set_control_active(bool active) { control_active_ = active; }
    bool control_active() const { return control_active_; }
    void set_control_scale(float scale) { control_scale_ = scale; }
    float control_scale() const { return control_scale_; }

    bool is_control_target_layer(const std::string& layer_name) const;
    bool is_control_norm_layer(const std::string& layer_name) const;
    bool load_control_layer_weights(const std::string& layer_name,
                                    int output_dim, int input_dim,
                                    ncnn::Mat& weights, ncnn::Mat& bias);
    bool load_control_norm_weight(const std::string& layer_name, int size,
                                  ncnn::Mat& weight);
    bool load_control_projection(const std::string& layer_name,
                                 int output_dim, int input_dim,
                                 ncnn::Mat& weights, ncnn::Mat& bias);

    bool is_target_layer(TransformerPart part, const std::string& layer_name) const;
    bool load_layer_lora(TransformerPart part, const std::string& layer_name,
                         int output_dim, int input_dim, ncnn::Mat& down,
                         ncnn::Mat& up, float& output_scale);
    bool load_norm_weight(TransformerPart part, const std::string& layer_name,
                          int size, ncnn::Mat& weight);
    bool load_output_heads(int output_dim, int input_dim,
                           std::vector<ncnn::Mat>& heads);

    std::size_t expected_targets() const { return expected_targets_; }
    std::size_t matched_targets() const { return matched_targets_; }

private:
    struct Impl;
    struct ControlNetWeights;
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<ControlNetWeights> control_impl_;
    std::string path_;
    std::string control_path_;
    std::string error_;
    float strength_ = 1.f;
    bool valid_ = false;
    bool control_valid_ = false;
    bool control_active_ = false;
    float control_scale_ = 1.f;
    bool pdd_output_ = false;
    int step_ = 0;
    std::size_t expected_targets_ = 0;
    std::size_t matched_targets_ = 0;

    friend int register_transformer_lora(ncnn::Net&, TransformerLoRA*, TransformerPart);
};

int register_transformer_lora(ncnn::Net& net, TransformerLoRA* lora,
                              TransformerPart part);

} // namespace qwenimage
