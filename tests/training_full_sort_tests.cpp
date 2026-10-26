#include "training_full_sort_reference.hpp"
#include "training_sort_downstream_reference.hpp"
#include "vulkan/radix_key_value.hpp"
#include <iostream>
#include <tuple>
int main() {
    try {
        using vulkan3DGS::radixKeyValueCapacity;
        sort_replay::check(
            radixKeyValueCapacity(822, 16) == 824 && radixKeyValueCapacity(822, 32) == 824 &&
                radixKeyValueCapacity(822, 256) == 832 && radixKeyValueCapacity(822, 4) == 822,
            "822-item external radix descriptor allocation regression");
        sort_replay::check(radixKeyValueCapacity(0, 16) == 0 && radixKeyValueCapacity(1, 16) == 4 &&
                               radixKeyValueCapacity(824, 16) == 824 &&
                               radixKeyValueCapacity(UINT32_MAX - 3, 16) == UINT32_MAX - 3,
                           "Radix capacity boundary regression");
        bool overflowRejected = false;
        try {
            (void)radixKeyValueCapacity(UINT32_MAX, 16);
        } catch (const std::overflow_error&) {
            overflowRejected = true;
        }
        sort_replay::check(overflowRejected, "Radix uint32 alignment overflow accepted");
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
            vulkan3DGS::recordRadixKeyValue({}, {}, 0, {}, {}, {}, 8);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        sort_replay::check(rejected, "Unsupported radix width accepted");
        for (auto count : {0u, 1u, 2u, 33u, 257u}) {
            auto [input, source] = full_sort::downstreamSynthetic(count, true);
            sort_replay::verify(input, source);
            full_sort::verifyChainOracle(input);
            sort_replay::check(source.pixelGrads.size() == 31u * 15u,
                               "Partial-edge fixture pixel dimensions");
        }
        full_sort::DownstreamSnapshot reference;
        reference.color.resize(1);
        reference.blends.resize(1);
        reference.gradients.resize(1);
        auto candidate = reference;
        sort_replay::check(full_sort::compareDownstream(reference, candidate).passed(),
                           "Identical downstream state failed");
        candidate.color[0].x = .01f;
        sort_replay::check(!full_sort::compareDownstream(reference, candidate).passed(),
                           "Rendered color mismatch accepted");
        candidate = reference;
        candidate.blends[0].processedCount = 1;
        sort_replay::check(!full_sort::compareDownstream(reference, candidate).passed(),
                           "Blend integer mismatch accepted");
        candidate = reference;
        candidate.gradients[0].color.x = .01f;
        sort_replay::check(!full_sort::compareDownstream(reference, candidate).passed(),
                           "Projected gradient mismatch accepted");
        candidate = reference;
        candidate.gradients[0].color.w = 1;
        sort_replay::check(!full_sort::compareDownstream(reference, candidate).passed(),
                           "Gradient padding mismatch accepted");
        candidate = reference;
        candidate.color[0].x = std::numeric_limits<float>::quiet_NaN();
        rejected = false;
        try {
            (void)full_sort::compareDownstream(reference, candidate);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        sort_replay::check(rejected, "Non-finite downstream color accepted");
        std::cout << "Full sort CPU chain contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
