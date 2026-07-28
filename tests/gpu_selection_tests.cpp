#include "context/device_selection.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

using vulkan3DGS::GpuSelectionCandidate;
using vulkan3DGS::GpuSelectionResult;
using vulkan3DGS::selectGpuCandidate;

const std::vector<GpuSelectionCandidate> kCandidates = {
    {0, "Intel(R) UHD Graphics", "00112233-4455-6677-8899-aabbccddeeff"},
    {2, "NVIDIA GeForce RTX 4090", "10213243-5465-7687-98a9-bacbdcedfe0f"},
    {3, "NVIDIA GeForce RTX 4090", "ffeeddcc-bbaa-9988-7766-554433221100"},
};

bool expectsCandidate(const std::string& selector, size_t expected) {
    const GpuSelectionResult result = selectGpuCandidate(kCandidates, selector);
    if (!result || *result.candidateIndex != expected) {
        std::cerr << "selector '" << selector << "' did not select candidate " << expected
                  << ": " << result.error << '\n';
        return false;
    }
    return true;
}

bool expectsError(const std::string& selector) {
    const GpuSelectionResult result = selectGpuCandidate(kCandidates, selector);
    if (result || result.error.empty()) {
        std::cerr << "selector '" << selector << "' unexpectedly succeeded\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!expectsCandidate(" 2 ", 1) ||
        !expectsCandidate("intel(r) uhd graphics", 0) ||
        !expectsCandidate("uhd", 0) ||
        !expectsCandidate("102132435465768798A9BACBDCEDFE0F", 1) ||
        !expectsError("1") ||
        !expectsError("RTX 4090") ||
        !expectsError("missing") ||
        !expectsError("")) {
        return 1;
    }
    return 0;
}
