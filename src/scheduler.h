// qwen-image implemented with ncnn library

#pragma once

#include <vector>

namespace qwenimage {

class QwenScheduler
{
public:
    static constexpr int turbo_steps = 8;

    // Qwen-Image-2.1-Turbo's fixed schedule, including the terminal zero.
    // Do not apply dynamic shifting or terminal sigma remapping to it.
    static std::vector<float> make_turbo_sigmas();

    // lightning=true selects the scheduler configuration used by the
    // Qwen-Image-Edit Lightning LoRAs: dynamic shifting with mu=log(3) and
    // no terminal sigma remapping.
    static std::vector<float> make_sigmas(int steps, int image_sequence_length, bool lightning = false);
};

} // namespace qwenimage
