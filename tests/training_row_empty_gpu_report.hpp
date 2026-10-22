#pragma once
#include "training_pixel_culling_gpu_probe.hpp"
#include "training_scanline_gpu_diagnostic.hpp"

inline void verifyRowEmptyShader() {
    verifyPixelCullingShader(
        "train_row_empty_census"); // FP32, no RelaxedPrecision, original Exp/Log.
    const auto path =
        std::string(CHECKPOINT_RUNTIME_DIR) + "/shaders/train_row_empty_census.comp.spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(file.tellg());
    if (!file || bytes < 20 || bytes % 4 != 0)
        throw std::runtime_error("Missing/invalid R0 SPIR-V");
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4u);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), bytes);
    bool precise = false;
    for (size_t i = 5; i < words.size(); i += words[i] >> 16u) {
        const auto count = words[i] >> 16u;
        if (!count || count > words.size() - i) throw std::runtime_error("Invalid R0 SPIR-V");
        precise |= (words[i] & 65535u) == 71u && count >= 3u && words[i + 2u] == 42u;
    }
    if (!precise) throw std::runtime_error("R0 lost NoContraction; re-audit numerical contract");
}

inline void writeRowEmptyGateReport(const std::filesystem::path& capturePath,
                                    vk::PhysicalDevice physical,
                                    const vulkan3DGS::TrainingScanlineCapture& capture,
                                    const ScanlineGpuResult& result) {
    const auto& c = result.counts;
    const auto& r = result.rowEmptyCounts;
    const auto props = physical.getProperties();
    const auto& h = capture.header;
    const bool matched = h.vendorID != 0u && h.deviceID != 0u && h.vendorID == props.vendorID &&
                         h.deviceID == props.deviceID && h.driverVersion == props.driverVersion;
    const double coverage = c[1] ? double(c[14]) / double(c[1]) : 0.0;
    std::string verdict = "R0_COVERAGE_PASS_R1_NOT_IMPLEMENTED";
    if (c[24] || c[25] || c[26] || c[31])
        verdict = "STOP_INVALID_CLASSIFICATION_OR_INPUT";
    else if (!c[1])
        verdict = "STOP_NO_ELIGIBLE_WORK";
    else if (!matched)
        verdict = "INCONCLUSIVE_SOURCE_GPU_OR_DRIVER_MISMATCH";
    else if (coverage < 0.10)
        verdict = "STOP_LOW_COMPACT_MASK_COVERAGE";
    const auto requested =
        capturePath.parent_path() / (capturePath.stem().string() + ".row_empty_r0.json");
    auto path = requested;
    for (uint32_t suffix = 2; std::filesystem::exists(path); ++suffix)
        path = requested.parent_path() /
               (requested.stem().string() + "_" + std::to_string(suffix) + ".json");
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write R0 report");
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &subgroup;
    physical.getProperties2(&properties);
    file << std::setprecision(10) << "{\n  \"version\": 1,\n  \"stage\": \"S3-3 R0\",\n"
         << "  \"method\": \"fp32_tangent_geometric_row_empty_v1\",\n"
         << "  \"gpu\": " << std::quoted(std::string(props.deviceName.data())) << ",\n"
         << "  \"source_vendor_id\": " << h.vendorID << ", \"source_device_id\": " << h.deviceID
         << ", \"current_vendor_id\": " << props.vendorID
         << ", \"current_device_id\": " << props.deviceID << ",\n"
         << "  \"source_driver\": " << h.driverVersion
         << ", \"current_driver\": " << props.driverVersion << ",\n"
         << "  \"source_gpu_and_driver_match\": " << (matched ? "true" : "false") << ",\n"
         << "  \"frame\": " << h.frame << ", \"iteration\": " << h.iteration
         << ", \"width\": " << h.width << ", \"height\": " << h.height << ",\n"
         << "  \"gaussians\": " << h.gaussianCount << ", \"items\": " << h.itemCount << ",\n"
         << "  \"native_subgroup_size\": " << subgroup.subgroupSize << ",\n  \"counters\": {\n";
    for (size_t i = 0; i < c.size(); ++i)
        file << "    " << std::quoted(vulkan3DGS::kScanlineCounterNames[i]) << ": " << c[i]
             << ",\n";
    constexpr std::array names = {
        "geometric_empty_rows_all_valid",     "geometric_empty_eligible_pairs",
        "oracle_empty_eligible_rows_kept",    "geometric_empty_already_guarded_rows",
        "native_waves_with_eligible_pairs",   "native_waves_all_eligible_rejected",
        "geometric_empty_rows_with_eligible", "fallback_rows_all_valid"};
    for (size_t i = 0; i < r.size(); ++i)
        file << "    " << std::quoted(names[i]) << ": " << r[i]
             << (i + 1u < r.size() ? ",\n" : "\n");
    file << "  },\n  \"compact_mask_eligible_pair_coverage\": " << coverage
         << ",\n  \"compact_mask_old_exp_coverage\": "
         << (c[3] ? double(c[15]) / double(c[3]) : 0.0)
         << ",\n  \"actual_native_wave_all_eligible_rejected_fraction\": "
         << (r[4] ? double(r[5]) / double(r[4]) : 0.0)
         << ",\n  \"min_coverage_heuristic\": 0.10,\n  \"verdict\": " << std::quoted(verdict)
         << ",\n  \"gradient_kernel_implemented\": false,\n  \"integrated_gradients_validated\": "
            "false,\n"
         << "  \"integrated_time_ms\": null,\n  \"estimated_speedup\": null,\n"
         << "  \"limitations\": \"R0 census only. No training gradients are changed or validated. "
            "Compact mask is purely geometric, independent of processedCount. Native waves refer "
            "to census sampling mapping, "
            "not Tensor Gaussian-reduction waves or issued instruction savings. Census time is not "
            "candidate cost. "
            "Old span-named counters refer to full-or-empty row masks in this report only.\"\n}\n";
    file.flush();
    if (!file) throw std::runtime_error("R0 report write failed");
    std::cout << "S3-3 R0 " << verdict << ": compact coverage=" << coverage
              << " false-drops=" << c[24] << " report=" << path.string() << '\n';
}
