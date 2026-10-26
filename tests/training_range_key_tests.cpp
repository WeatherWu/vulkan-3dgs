#include "training_range_key_reference.hpp"
#include "training_optimizer_r1_metrics.hpp"

#include <chrono>
#include <iostream>

using namespace range_keys;
namespace {
class TemporaryInput {
public:
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("v3d-range-input-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
    TemporaryInput() { check(!std::filesystem::exists(path), "Temporary input collision"); }
    ~TemporaryInput() { std::error_code error; std::filesystem::remove(path, error); }
    void write(const std::string& bytes) {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file.write(bytes.data(), bytes.size());
        check(bool(file), "Cannot write temporary test input");
    }
};
} // namespace
int main() {
    try {
        for (uint32_t n : {0u, 1u, 2u, 31u, 32u, 33u, 255u, 256u, 257u, 4095u, 4096u, 4097u}) {
            for (bool ties : {false, true}) {
                auto [data, source] = synthetic(n, ties);
                verify(data, source);
                const auto expected = oracle(data);
                check(std::is_sorted(expected.high.begin(), expected.high.end()) && expected.items == source.items,
                      "Range oracle ordered-key/item contract failed");
                check(std::memcmp(expected.ranges.data(), source.ranges.data(), expected.ranges.size() * 16) == 0,
                      "Range oracle final ranges failed");
            }
        }
        auto [data, source] = synthetic(257);
        TemporaryInput temp;
        const auto payload = serialize(data);
        temp.write(payload);
        const auto roundtrip = readInputs(temp.path);
        verify(roundtrip, source);
        check(roundtrip.tileKeys == data.tileKeys && roundtrip.indices == data.indices, "Range input roundtrip failed");
        auto rejectBytes = [&](const std::string& bytes) {
            temp.write(bytes);
            bool rejected = false;
            try { (void)readInputs(temp.path); } catch (const std::runtime_error&) { rejected = true; }
            check(rejected, "Malformed range input accepted");
        };
        rejectBytes(payload.substr(0, 40));
        rejectBytes(payload.substr(0, payload.size() - 1));
        rejectBytes(payload + 'x');
        for (size_t offset : {size_t(8), size_t(12), size_t(16 + 12)}) {
            auto changed = payload;
            const uint32_t bad = UINT32_MAX;
            std::memcpy(changed.data() + offset, &bad, 4);
            rejectBytes(changed);
        }
        auto invalid = data;
        invalid.indices[0] = 2;
        bool rejected = false;
        try { (void)oracle(invalid); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "Nonidentity unsorted seed accepted");
        const std::array<double, 8> a{1, 1, 1, 1, 1, 1, 1, 1};
        const std::array<double, 8> b{.8, .8, .8, .8, .8, .8, .8, .8};
        check(optimizer_r1::evaluate(a, b, a).verdict == "SHORT_GAIN_REQUIRES_NATIVE_REVIEW", "A/B/A gain gate failed");
        auto outlier = a; outlier[0] = 1.3;
        check(optimizer_r1::evaluate(outlier, b, a).verdict == "INCONCLUSIVE_SAMPLE_OUTLIER", "Outlier gate failed");
        std::cout << "Range key reader/oracle contracts passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
    return 0;
}
