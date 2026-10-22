#include "training_pixel_culling_reference.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    namespace ref = pixel_culling_reference;
    try {
        // Adversarial SPEC error envelope, not merely a libm parity test.
        // Allow 3e-6 cutoff overestimate and 4e-5 exp/multiply overestimate
        // (also covers exp2 lowering); the 2^-10 guard must still exclude
        // alphaMin. Test the supremum at the rejection boundary.
        for (uint32_t i = 0; i <= 4096u; ++i) {
            const double opacity =
                double(ref::alphaMin) + (1.0 - double(ref::alphaMin)) * double(i) / 4096.0;
            const double highestCutoff =
                std::log(double(ref::alphaMin) / opacity) + 3e-6 - double(ref::guard);
            const double highestAlpha = opacity * std::exp(highestCutoff) * (1.0 + 4e-5);
            if (!(highestAlpha < double(ref::alphaMin)))
                throw std::runtime_error(
                    "FP32 worst-case error budget does not preserve exclusion");
        }
        uint32_t rejected = 0;
        const auto inputs = ref::cases();
        for (const auto& input : inputs) {
            const float cutoff = ref::threshold(input[0]);
            if (ref::reject(input[1], cutoff)) {
                ++rejected;
                if (ref::contributes(input[0], input[1]))
                    throw std::runtime_error("EarlyCull rejected an original FP32 contributor");
            }
            if (cutoff != ref::disabled) {
                const double boundary = std::log(double(ref::alphaMin) / double(input[0]));
                if (!(double(cutoff) < boundary - double(ref::guard) * 0.9))
                    throw std::runtime_error("Threshold is not conservatively expanded");
            }
        }
        if (rejected == 0u) throw std::runtime_error("No useful culling cases exercised");
        std::cout << "EarlyCull method 1: " << inputs.size() << " classifications, " << rejected
                  << " safely rejected; no original contributors lost\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
