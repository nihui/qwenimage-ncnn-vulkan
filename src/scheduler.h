// qwen-image implemented with ncnn library

#pragma once

#include <vector>

namespace qwenimage {

class QwenScheduler
{
public:
    // lightning=true selects the scheduler configuration used by the
    // Qwen-Image-Edit Lightning LoRAs: dynamic shifting with mu=log(3) and
    // no terminal sigma remapping.
    static std::vector<float> make_sigmas(int steps, int image_sequence_length, bool lightning = false);
};

} // namespace qwenimage
