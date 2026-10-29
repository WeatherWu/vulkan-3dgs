#pragma once
#include "training/core/gaussian_training.hpp"
#include "training_sort_baseline_session.hpp"
#include "training_sort_downstream_reference.hpp"

inline void checkTrainingSortingLifecycle(vk::Device device, vk::PhysicalDevice physical,
                                          vk::Queue queue, uint32_t family) {
    using namespace vulkan3DGS;
    GaussianTraining training;
    training.initialize(device, physical, queue, family, family);
    training.setFasterSortEnabled(true);
    const auto checkRenderer = [&](uint32_t count, bool requested) {
        training.initializeTrainingRenderers(device, physical, queue, family, count, {31, 15});
        const auto& forward = dynamic_cast<const GaussianForwardRenderer&>(training.forward());
        sort_replay::check(forward.fasterSortRequested() == requested,
                           "Training sorting preference lost on renderer creation/rebuild");
        sort_replay::check(!training.lastSortState().executed,
                           "Renderer initialization reported an executed sorting step");
    };
    checkRenderer(1, true);
    training.setFasterSortEnabled(false);
    sort_replay::check(
        !dynamic_cast<const GaussianForwardRenderer&>(training.forward()).fasterSortRequested(),
        "Stopped-owner sorting change did not reach renderer");
    checkRenderer(257, false);
    training.setFasterSortEnabled(true);
    checkRenderer(33, true);
    training.cleanup();
    training.initialize(device, physical, queue, family, family);
    checkRenderer(2, true);
}

inline std::string runFasterSortIntegration(vk::Device device, vk::PhysicalDevice physical,
                                            vk::Queue queue, uint32_t family,
                                            const vulkan3DGS::TrainingScanlineCapture* capture,
                                            bool physical16 = false) {
    using namespace vulkan3DGS;
    using sort_replay::check;
    if (!capture) checkTrainingSortingLifecycle(device, physical, queue, family);
    std::vector<TrainingScanlineCapture> fixtures;
    if (capture)
        fixtures.push_back(*capture);
    else
        for (uint32_t count : {1u, 257u, 33u, 0u, 513u, 2u})
            fixtures.push_back(std::move(full_sort::downstreamSynthetic(count, true).second));
    const auto& h = fixtures.front().header;
    checkOptimizerCaptureBudget(physical, h.gaussianCount,
                                uint64_t(h.gaussianCount) * 2048 +
                                    uint64_t(h.width) * h.height * 256 + (64ull << 20));
    sort_replay::BaselineSession session(device, physical, queue, family, physical16);
    session.initialize(fixtures.front());
    std::ostringstream out;
    out << std::boolalpha << '[';
    for (size_t frame = 0; frame < fixtures.size(); ++frame) {
        const auto& source = fixtures[frame];
        if (frame) session.loadProjected(source, source.header.gaussianCount != 0);
        std::vector<uint8_t> visible(source.header.gaussianCount, 0);
        for (auto id : source.items) {
            check(id < visible.size(), "Integrated source Gaussian ID");
            visible[id] = 1;
        }
        const uint32_t expectedNv = uint32_t(std::count(visible.begin(), visible.end(), 1u));
        std::vector<glm::vec4> referenceColor;
        std::vector<PixelBlendState> referenceBlend;
        uint32_t actualNv = 0;
        for (size_t i = 0; i < 3; ++i) {
            session.setFasterSortEnabled(physical16 || i == 1);
            session.setPhysicalTileKey16(physical16 && i == 1);
            session.run(false, true);
            session.verifySorted(source);
            check(session.fasterSortActive() == (physical16 || i == 1),
                  "Integrated requested/actual sorting mismatch");
            if (physical16)
                check(session.physicalTileKey16Active() == (i == 1),
                      "Physical16 integrated path mismatch");
            if (i == 1) {
                actualNv = session.participatingCount();
                check(actualNv == expectedNv, "Integrated Nv mismatch");
            }
            auto colors = session.colors();
            auto blends = session.blends();
            for (const auto& color : colors)
                for (int channel = 0; channel < 4; ++channel)
                    check(std::isfinite(color[channel]), "Integrated non-finite composite");
            if (i == 0) {
                referenceColor = std::move(colors);
                referenceBlend = std::move(blends);
            } else {
                check(std::memcmp(referenceColor.data(), colors.data(), colors.size() * 16) == 0,
                      "Integrated composite differs from original renderer");
                check(std::memcmp(referenceBlend.data(), blends.data(), blends.size() * 16) == 0,
                      "Integrated blend differs from original renderer");
            }
        }
        if (capture)
            check(std::memcmp(referenceBlend.data(), source.blends.data(),
                              source.blends.size() * 16) == 0,
                  "Integrated original blend differs from capture");
        bool fixed8Fallback = false, adaptiveFallback = false;
        if (!source.header.gaussianCount) {
            session.setFasterSortEnabled(true);
            session.setPhysicalTileKey16(physical16);
            for (uint32_t tileSize : {8u, 32u}) {
                session.setTileSize(tileSize);
                session.run(false, true);
                check(!session.fasterSortActive(), "Non-Fixed16 unexpectedly selected Faster sort");
                const auto fallbackColor = session.colors();
                const auto fallbackBlend = session.blends();
                check(std::memcmp(referenceColor.data(), fallbackColor.data(),
                                  referenceColor.size() * 16) == 0 &&
                          std::memcmp(referenceBlend.data(), fallbackBlend.data(),
                                      referenceBlend.size() * 16) == 0,
                      "Empty non-Fixed16 original fallback output changed");
            }
            session.setTileSize(16);
            fixed8Fallback = adaptiveFallback = true;
        }
        if (frame) out << ',';
        out << "{\"passed\":true,\"gaussians\":" << source.header.gaussianCount
            << ",\"items\":" << source.header.itemCount
            << ",\"participating_gaussians\":" << actualNv << ",\"actual_active_A_B_A\":"
            << (physical16 ? "[true,true,true]" : "[false,true,false]")
            << ",\"physical16_active_A_B_A\":"
            << (physical16 ? "[false,true,false]" : "[false,false,false]")
            << ",\"capture_blend_checked\":" << (capture != nullptr)
            << ",\"projected_adoption_exercised\":" << (frame && source.header.gaussianCount != 0)
            << ",\"tile_capacity\":" << session.capacity()
            << ",\"fixed8_empty_fallback_verified\":" << fixed8Fallback
            << ",\"adaptive_empty_fallback_verified\":" << adaptiveFallback
            << ",\"lists_ranges_projected_exact\":true,\"color_blend_exact\":true"
            << ",\"adam_state_equivalence_checked\":false}";
    }
    out << ']';
    return out.str();
}
