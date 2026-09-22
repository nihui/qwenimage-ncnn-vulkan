// qwen-image implemented with ncnn library

#include "scheduler.h"

#include <cmath>

namespace qwenimage {

std::vector<float> QwenScheduler::make_sigmas(int steps, int image_sequence_length, bool lightning)
{
    double mu;
    if (lightning)
    {
        // Qwen-Image-Edit-2511-Lightning's scheduler config sets both
        // base_shift and max_shift to log(3), and leaves shift_terminal
        // unset.  Therefore the image length does not change mu.
        (void)image_sequence_length;
        mu = std::log(3.0);
    }
    else
    {
        // Qwen-Image-2.1 scheduler parameters: base_shift=0.5,
        // max_shift=0.9, base_image_seq_len=256, max_image_seq_len=8192.
        const double m = (0.9 - 0.5) / (8192.0 - 256.0);
        const double b = 0.5 - m * 256.0;
        mu = image_sequence_length * m + b;
    }
    const double emu = std::exp(mu);
    std::vector<float> sigmas(steps + 1);
    for (int i = 0; i < steps; i++)
    {
        const double t = steps == 1 ? 1.0
            : 1.0 - (double)i * (1.0 - 1.0 / steps) / (steps - 1);
        sigmas[i] = (float)(emu / (emu + (1.0 / t - 1.0)));
    }
    if (!lightning && steps > 1)
    {
        const double scale = (1.0 - sigmas[steps - 1]) / (1.0 - 0.02);
        for (float& sigma : sigmas)
            sigma = (float)(1.0 - (1.0 - sigma) / scale);
    }
    sigmas[steps] = 0.f;
    return sigmas;
}

} // namespace qwenimage
