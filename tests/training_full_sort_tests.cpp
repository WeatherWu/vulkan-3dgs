#include "training_full_sort_reference.hpp"
#include "vulkan/diagnostic_radix_sort.hpp"
#include <iostream>
#include <tuple>
int main() {
    try {
        for (uint32_t n :
             {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4095u, 4096u, 4097u, 65537u}) {
            auto [input, source] = full_sort::synthetic(n);
            sort_replay::verify(input, source);
            full_sort::verifyChainOracle(input);
        }
        for (auto [n, wide, tiles, inactive] : {std::tuple{33u, true, 2u, false},
                                                {33u, false, 2u, true},
                                                {2u, false, 65536u, false},
                                                {2u, false, 65537u, false}}) {
            auto [input, source] = full_sort::synthetic(n, wide, tiles, inactive);
            sort_replay::verify(input, source);
            full_sort::verifyChainOracle(input);
        }
        sort_replay::check(full_sort::tileBits(65536) == 16 && full_sort::tileBits(65537) == 32,
                           "Effective radix width boundary failed");
        bool rejected = false;
        try {
            vulkan3DGS::recordDiagnosticRadixKeyValue({}, {}, 0, {}, {}, {}, 8);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        sort_replay::check(rejected, "Unsupported radix width accepted");
        std::cout << "Full sort CPU chain contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
