#pragma once
#include "training_optimizer_gpu_session.hpp"
#include "training_optimizer_r1_metrics.hpp"

#include <atomic>
#include <iostream>
#include <sstream>

namespace optimizer_r1 {
inline std::vector<optimizer_gpu::Sample> phase(optimizer_gpu::Session& session,
                                                optimizer_gpu::Variant variant,
                                                const std::atomic<uint32_t>& errors) {
    std::vector<optimizer_gpu::Sample> samples;
    samples.reserve(8);
    for (uint32_t i = 0; i < 2; ++i)
        session.dispatch(true, variant);
    for (uint32_t i = 0; i < 8; ++i) {
        samples.push_back(session.dispatch(true, variant));
        optimizer_gpu::check(!errors.load(), "S4-1R timing Vulkan validation failure");
    }
    return samples;
}
inline void writeSamples(std::ostream& out, const std::vector<optimizer_gpu::Sample>& samples,
                         bool timestamps) {
    out << '[';
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) out << ',';
        out << "{\"kernel_ms\":";
        if (timestamps)
            out << samples[i].kernelMs;
        else
            out << "null";
        out << ",\"restore_ms\":";
        if (timestamps)
            out << samples[i].restoreMs;
        else
            out << "null";
        out << ",\"cpu_submit_wait_ms\":" << samples[i].cpuSubmitWaitMs
            << ",\"restore_cpu_submit_wait_ms\":" << samples[i].restoreCpuMs << '}';
    }
    out << ']';
}
inline std::string evaluateReplay(optimizer_gpu::Session& session, const OptimizerCaptureHeader& h,
                                  const OptimizerCaptureSource& reference, uint64_t candidateHash,
                                  const std::atomic<uint32_t>& errors, uint32_t profileSeconds,
                                  optimizer_gpu::Variant profileVariant) {
    using optimizer_gpu::Variant;
    // Caller has already verified original post-reference and original A/A.
    for (uint32_t i = 0; i < 2; ++i) {
        session.dispatch(true, Variant::StreamAdam);
        optimizer_gpu::requireMatch(session.compare(reference),
                                    "S4-1R original post-reference / candidate repeat mismatch");
        optimizer_gpu::check(!errors.load(), "S4-1R correctness Vulkan validation failure");
    }
    std::ostringstream out;
    out.precision(10);
    out << ",\"r1\":{\"candidate\":\"stream_adam_v1\",\"layout\":\"unchanged typed AoS, scoped "
           "float4 moments\""
        << ",\"candidate_shader_fnv1a64\":\"" << candidateHash << '"'
        << ",\"bitwise_post_reference\":true,\"bitwise_candidate_repeat\":true"
        << ",\"all_mutable_state_restored_each_dispatch\":true,\"integrated_into_training\":false"
        << ",\"restore_bytes_per_dispatch\":" << uint64_t(h.gaussianCount) * 736
        << ",\"candidate_native\":{\"registers\":null,\"shared_bytes\":null,\"spill_bytes\":null}";
    if (profileSeconds) {
        // Explicit bounded resource-profiling window, never a performance admission run.
        std::cerr << "S4-1R PROFILE WINDOW READY: " << profileSeconds << " seconds, "
                  << (profileVariant == Variant::Baseline ? "ORIGINAL baseline" : "stream_adam_v1")
                  << ". Trigger Nsight now; every dispatch restores the SAME seed.\n";
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(profileSeconds);
        uint64_t dispatches = 0;
        while (std::chrono::steady_clock::now() < end) {
            session.dispatch(true, profileVariant);
            optimizer_gpu::check(!errors.load(), "S4-1R native profiling validation failure");
            ++dispatches;
        }
        optimizer_gpu::requireMatch(session.compare(reference), "S4-1R profile-window state drift");
        out << ",\"profiling_only\":true,\"profile_seconds\":" << profileSeconds
            << ",\"profile_variant\":\""
            << (profileVariant == Variant::Baseline ? "baseline" : "stream-adam")
            << "\",\"restored_dispatches\":" << dispatches
            << ",\"short_gain_percent\":null,\"verdict\":\"PROFILE_ONLY_NOT_TIMING\"}";
    } else {
        // No downloads/comparison between phases; no candidate-to-baseline Adam chaining.
        const auto a1 = phase(session, Variant::Baseline, errors);
        const auto b = phase(session, Variant::StreamAdam, errors);
        const auto a2 = phase(session, Variant::Baseline, errors);
        optimizer_gpu::requireMatch(session.compare(reference), "S4-1R timed A2 contamination");
        session.dispatch(true, Variant::StreamAdam);
        optimizer_gpu::requireMatch(session.compare(reference),
                                    "S4-1R post-timing candidate mismatch");
        out << ",\"profiling_only\":false,\"warmup_per_phase\":2,\"measured_per_phase\":8"
            << ",\"sample_policy\":\"all samples; max/min >1.20 is inconclusive; no automatic "
               "retry\""
            << ",\"A1\":";
        writeSamples(out, a1, session.timestampsAvailable());
        out << ",\"B\":";
        writeSamples(out, b, session.timestampsAvailable());
        out << ",\"A2\":";
        writeSamples(out, a2, session.timestampsAvailable());
        if (session.timestampsAvailable()) {
            std::array<double, 8> x{}, y{}, z{};
            for (size_t i = 0; i < 8; ++i) {
                x[i] = a1[i].kernelMs;
                y[i] = b[i].kernelMs;
                z[i] = a2[i].kernelMs;
            }
            const auto decision = evaluate(x, y, z);
            out << ",\"A1_mean_ms\":" << decision.a1.mean << ",\"B_mean_ms\":" << decision.b.mean
                << ",\"A2_mean_ms\":" << decision.a2.mean
                << ",\"baseline_drift_percent\":" << decision.baselineDriftPercent
                << ",\"short_gain_percent\":" << decision.gainPercent << ",\"timing_verdict\":\""
                << decision.verdict << "\",\"verdict\":\""
                << (h.synthetic ? "SYNTHETIC_ONLY_NOT_TARGET_EVIDENCE" : decision.verdict) << "\"}";
        } else {
            out << ",\"short_gain_percent\":null,\"verdict\":\"INCONCLUSIVE_TIMESTAMPS_"
                   "UNAVAILABLE\"}";
        }
    }
    session.verifyReadonly();
    return out.str();
}
} // namespace optimizer_r1
