// qwen-image implemented with ncnn library

#pragma once
#include <array>
#include <memory>
#include <vector>
#include "allocator.h"
#include "models.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

namespace qwenimage {
struct TransformerImageShape
{
    int height = 0;
    int width = 0;
};

class QwenTransformer
{
public:
    QwenTransformer(const QwenModelSet& models, const RuntimeConfig& config, int text_tokens, int image_tokens)
        : models_(models), config_(config), text_tokens_(text_tokens), image_tokens_(image_tokens), prefix_tokens_(0), prefix_ready_(false), edit_ready_(false) {}

    static bool make_rope(int text_tokens, int valid_text_tokens, int latent_height, int latent_width, ncnn::Mat& cos, ncnn::Mat& sin);
    static bool make_attention_mask(int text_tokens, int valid_text_tokens, int image_tokens, ncnn::Mat& mask);
    bool prepare_text_to_image(const ncnn::Mat& text, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, float first_timestep);
    bool run(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise);

    // prepare the fixed image layout once before the denoising loop
    bool prepare_edit(const ncnn::Mat& condition_latents, const ncnn::Mat& text, const std::vector<unsigned char>& text_image_slots, const std::vector<TransformerImageShape>& image_shapes, float first_timestep);
    bool run_edit(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise);

private:
    struct BlockBlobs
    {
        int residual = -1;
        int normalized = -1;
        int modulation[4] = {-1, -1, -1, -1};
        int cos_q = -1;
        int cos_k = -1;
        int sin_q = -1;
        int sin_k = -1;
        int mask = -1;
        int output = -1;
        int cache_k_in = -1;
        int cache_v_in = -1;
        int cache_k_out = -1;
        int cache_v_out = -1;
    };

#if NCNN_VULKAN
    struct CacheLayout
    {
        int w = 0;
        int h = 0;
        int c = 0;
        size_t cstep = 0;
    };

    static bool download_cache(const ncnn::VkMat& source, ncnn::Mat& data, CacheLayout& layout, ncnn::VkCompute& cmd, const ncnn::Option& opt);
    static bool upload_cache(const ncnn::Mat& data, const CacheLayout& layout, ncnn::VkMat& cache, ncnn::VkCompute& cmd, const ncnn::Option& opt);
#endif

    struct LayerCache
    {
        ncnn::Mat key;
        ncnn::Mat value;
#if NCNN_VULKAN
        ncnn::VkMat key_vk;
        ncnn::VkMat value_vk;
        CacheLayout key_layout;
        CacheLayout value_layout;
#endif
    };

    bool initialize_block_blobs();
    bool project_text(const ncnn::Mat& text, ncnn::Mat& projected) const;
    bool project_latents(const ncnn::Mat& latents, ncnn::Mat& projected) const;
#if NCNN_VULKAN
    bool project_latents_vulkan(const ncnn::Mat& latents, ncnn::VkMat& projected) const;
#endif
    bool run_time_condition(float timestep, ncnn::Mat& modulation, ncnn::Mat& temb) const;
    bool prepare_prefix(const ncnn::Mat& hidden, const ncnn::Mat& modulation, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask);
    bool run_blocks_cpu(ncnn::Mat hidden, const std::array<ncnn::Mat, 4>& modulation, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, bool prefill, ncnn::Mat& output);
#if NCNN_VULKAN
    bool run_blocks_vulkan(ncnn::VkMat hidden, const std::array<ncnn::Mat, 4>& modulation, const ncnn::Mat& cos, const ncnn::Mat& sin, const ncnn::Mat& mask, bool prefill, ncnn::VkMat& output);
#endif
    bool run_target(const ncnn::Mat& latents, float timestep, ncnn::Mat& noise);

    const QwenModelSet& models_;
    const RuntimeConfig& config_;
    int text_tokens_;
    int image_tokens_;
    int prefix_tokens_;
    bool prefix_ready_;
    bool edit_ready_;
    ncnn::Mat target_cos_;
    ncnn::Mat target_sin_;
    ncnn::Mat target_mask_;
    std::array<BlockBlobs, 32> block_blobs_;
    std::unique_ptr<ncnn::PoolAllocator> prefix_kvcache_allocator_;
    std::unique_ptr<ncnn::PoolAllocator> decode_kvcache_allocator_;
#if NCNN_VULKAN
    std::unique_ptr<ncnn::VkBlobAllocator> prefix_kvcache_vkallocator_;
    std::unique_ptr<ncnn::VkBlobAllocator> decode_kvcache_vkallocator_;
#endif
    std::vector<LayerCache> prefix_cache_;
    std::vector<int> joint_rows_;
};
}
