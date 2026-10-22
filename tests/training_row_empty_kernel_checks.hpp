#pragma once
#include "training_checkpoint_reference.hpp"
#include "training_row_empty_kernel.hpp"

inline std::vector<vulkan3DGS::ProjectedGaussianGrad>
rowScalarGradients(const vulkan3DGS::TrainingScanlineCapture& data) {
    using namespace vulkan3DGS;
    namespace ref = checkpoint_reference;
    ref::Scene<float> scene;
    for (const auto& g : data.projected)
        scene.gaussians.push_back({g.centerRadius.x,
                                   g.centerRadius.y,
                                   g.conicOpacity.x,
                                   g.conicOpacity.y,
                                   g.conicOpacity.z,
                                   g.conicOpacity.w,
                                   {g.color.x, g.color.y, g.color.z}});
    std::vector<ProjectedGaussianGrad> out(data.projected.size());
    constexpr std::array<uint32_t, 9> active = {0, 1, 4, 5, 6, 7, 8, 9, 10};
    const auto& h = data.header;
    const uint32_t tx = (h.width + 15u) / 16u;
    std::vector<std::array<float, 12>> accumulated(data.projected.size());
    for (uint32_t tile = 0; tile < h.tileCount; ++tile) {
        const auto range = data.ranges[tile];
        scene.items.assign(data.items.begin() + range.x, data.items.begin() + range.x + range.y);
        const uint32_t ox = tile % tx * 16u, oy = tile / tx * 16u;
        for (uint32_t y = 0; y < 16u && oy + y < h.height; ++y)
            for (uint32_t x = 0; x < 16u && ox + x < h.width; ++x) {
                const size_t pixel = size_t(oy + y) * h.width + ox + x;
                ref::Forward<float> state;
                state.processedCount = std::min(data.blends[pixel].processedCount, range.y);
                state.finalState.transmittance = data.blends[pixel].finalTransmittance;
                const auto gradient = data.pixelGrads[pixel].color;
                const auto values = ref::reverse(scene, state, float(ox + x), float(oy + y),
                                                 {gradient.x, gradient.y, gradient.z});
                for (size_t id = 0; id < values.size(); ++id)
                    for (size_t component = 0; component < 9; ++component)
                        accumulated[id][active[component]] += values[id][component];
            }
    }
    for (size_t id = 0; id < out.size(); ++id)
        std::memcpy(&out[id], accumulated[id].data(), sizeof(out[id]));
    return out;
}

inline void checkRowKernelFixture(vk::Device device, vk::PhysicalDevice physical, vk::Queue queue,
                                  uint32_t family,
                                  const vulkan3DGS::TrainingScanlineCapture& data) {
    const bool portable = !supportsRowEmptyTensor(physical);
    RowEmptyKernelSession kernel(device, physical, queue, family, data, portable);
    (void)kernel.dispatch(false);
    const auto baseline = kernel.gradients();
    (void)kernel.dispatch(true);
    const auto candidate = kernel.gradients();
    const auto comparison = compareRowGradients(baseline, candidate);
    const auto cpu = rowScalarGradients(data);
    const auto scalarComparison = compareRowGradients(cpu, candidate);
    // Reuse the same buffers/pipeline; every batch must reset its own mask/pairs.
    (void)kernel.dispatch(true);
    const auto repeat = compareRowGradients(candidate, kernel.gradients());
    if (!comparison.passed() || !scalarComparison.passed() || !repeat.passed() ||
        !kernel.inputsUnchanged()) {
        std::cerr << "RowEmpty fixture differs: GPU=" << comparison.differing
                  << " CPU=" << scalarComparison.differing << " max=" << scalarComparison.maximum
                  << " at " << scalarComparison.maximumGaussian << ':'
                  << scalarComparison.maximumComponent << '\n';
        throw std::runtime_error("RowEmpty sample/matrix/gradient/synchronization fixture failed");
    }
    std::cout << "S3-3 R1 fixture "
              << (portable ? "portable batch4 Shared (NOT native Tensor16)"
                           : "actual Tensor16 Subgroup/Global")
              << " passed; shared=" << kernel.candidateShared << " bytes\n";
}
