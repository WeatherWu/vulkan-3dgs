#pragma once
#include "training_row_empty_kernel.hpp"

inline void writeRowPrototypeReport(const std::filesystem::path& capturePath,
                                    vk::PhysicalDevice physical,
                                    const vulkan3DGS::TrainingScanlineCapture& data,
                                    const RowPrototypeResult& result, bool resourcesReviewed,
                                    bool validation, bool syncValidation) {
    const auto requested =
        capturePath.parent_path() / (capturePath.stem().string() + ".row_empty_r2.json");
    auto path = requested;
    for (uint32_t suffix = 2; std::filesystem::exists(path); ++suffix)
        path = requested.parent_path() /
               (requested.stem().string() + "_" + std::to_string(suffix) + ".json");
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write RowEmpty R2 report");
    file << std::setprecision(12);
    const auto number = [&](double value) {
        if (std::isfinite(value))
            file << value;
        else
            file << "null";
    };
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    const auto timing = [&](const char* name, const std::vector<double>& samples) {
        file << "  " << std::quoted(std::string(name) + "_mean_ms") << ": ";
        if (samples.empty())
            file << "null";
        else
            number(rowMean(samples));
        file << ",\n  " << std::quoted(std::string(name) + "_median_ms") << ": ";
        if (samples.empty())
            file << "null";
        else
            number(scanlineMedian(samples));
        file << ",\n  " << std::quoted(std::string(name) + "_samples_ms") << ": [";
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i) file << ',';
            number(samples[i]);
        }
        file << "],\n";
    };
    const auto gradient = [&](const char* name, const RowGradientComparison& c) {
        file << "  " << std::quoted(name) << ": ";
        if (!result.gradientsRun) {
            file << "null,\n";
            return;
        }
        file << "{\"compared_components\":" << c.compared
             << ",\"differing_components\":" << c.differing << ",\"non_finite\":" << c.nonFinite
             << ",\"padding_errors\":" << c.paddingErrors
             << ",\"missing_extra_gaussians\":" << c.missingExtra << ",\"relative_l2\":";
        number(c.relativeL2);
        file << ",\"cosine\":";
        number(c.cosine);
        file << ",\"max_absolute_difference\":";
        number(c.maximum);
        file << ",\"max_gaussian\":" << c.maximumGaussian
             << ",\"max_component\":" << c.maximumComponent << ",\"passed\":" << boolean(c.passed())
             << "},\n";
    };
    const auto props = physical.getProperties();
    const auto& h = data.header;
    file << "{\n  \"version\": 1,\n  \"method\": \"tensor_row_empty_tangent_v1\",\n"
         << "  \"input_capture\": " << std::quoted(capturePath.filename().string()) << ",\n"
         << "  \"gpu\": " << std::quoted(std::string(props.deviceName.data())) << ",\n"
         << "  \"source_vendor_id\": " << h.vendorID << ", \"source_device_id\": " << h.deviceID
         << ", \"current_vendor_id\": " << props.vendorID
         << ", \"current_device_id\": " << props.deviceID << ",\n"
         << "  \"source_driver\": " << h.driverVersion
         << ", \"current_driver\": " << props.driverVersion << ",\n"
         << "  \"source_gpu_driver_match\": " << boolean(result.sourceMatches) << ",\n"
         << "  \"frame\": " << h.frame << ", \"iteration\": " << h.iteration
         << ", \"width\": " << h.width << ", \"height\": " << h.height << ",\n"
         << "  \"gaussians\": " << h.gaussianCount << ", \"items\": " << h.itemCount
         << ", \"ssim_mode\": " << h.ssimMode << ",\n"
         << "  \"tensor_capability_available\": " << boolean(result.capable) << ",\n"
         << "  \"r0_rechecked_on_current_device\": " << boolean(result.r0Run) << ",\n"
         << "  \"r0_false_drops\": ";
    if (result.r0Run)
        file << result.r0.counts[24];
    else
        file << "null";
    file << ",\n  \"r0_compact_mask_pair_coverage\": ";
    if (result.r0Run && result.r0.counts[1])
        number(double(result.r0.counts[14]) / double(result.r0.counts[1]));
    else
        file << "null";
    file << ",\n  \"gradient_kernel_implemented\": true,\n"
         << "  \"baseline_compiler_backend\": \"Slang direct SPIR-V\",\n"
         << "  \"candidate_compiler_backend\": \"Slang via GLSL, fp-mode precise\",\n"
         << "  \"integrated_gradients_validated\": "
         << boolean(result.error.empty() && result.gradientsRun && result.gradients.passed() &&
                    result.repeat.passed() && result.inputsSame)
         << ",\n"
         << "  \"pixel_grad_and_blend_inputs_unchanged\": ";
    if (result.gradientsRun)
        file << boolean(result.inputsSame);
    else
        file << "null";
    file << ",\n  \"atol\": 0.00001, \"rtol\": 0.001,\n";
    gradient("candidate_gradients", result.gradients);
    gradient("baseline_repeat_gradients", result.repeat);
    file << "  \"baseline_declared_shared_bytes\": ";
    if (result.referenceShared)
        file << result.referenceShared;
    else
        file << "null";
    file << ",\n  \"candidate_declared_shared_bytes\": ";
    if (result.candidateShared)
        file << result.candidateShared;
    else
        file << "null";
    file << ",\n  \"native_registers\": null, \"native_shared_allocation\": null, "
            "\"actual_residency_measured\": false,\n"
         << "  \"native_resources_reviewed_by_operator\": " << boolean(resourcesReviewed) << ",\n"
         << "  \"validation_enabled\": " << boolean(validation)
         << ", \"synchronization_validation_enabled\": " << boolean(syncValidation) << ",\n"
         << "  \"short_warmup\": 2, \"short_measured\": 8,\n"
         << "  \"long_timing_executed\": " << boolean(result.longRun) << ",\n"
         << "  \"long_warmup\": 200, \"long_measured\": 1000,\n";
    timing("short_baseline_before", result.shortBefore);
    timing("short_candidate", result.shortCandidate);
    timing("short_baseline_after", result.shortAfter);
    timing("long_baseline_before", result.longBefore);
    timing("long_candidate", result.longCandidate);
    timing("long_baseline_after", result.longAfter);
    file << "  \"verdict\": " << std::quoted(result.verdict)
         << ",\n  \"execution_error\": " << std::quoted(result.error) << ",\n"
         << "  \"production_training_enabled\": false,\n"
         << "  \"limitations\": \"Offline original Tensor vs isolated RowEmpty dispatch. "
            "Classification/barriers/mask use are included; "
            "clear, census, allocation, downloads and correctness checks are outside both "
            "timestamp intervals. "
            "Forward is not rerun. Declared shared bytes are not native residency. Operator "
            "resource review is an attestation, not an automatic measurement. "
            "No portable batch4 or Direct fallback is admitted as production Tensor. No "
            "full-training speed or quality claim.\"\n}\n";
    file.flush();
    if (!file) throw std::runtime_error("RowEmpty R2 report write failed");
    std::cout << "S3-3 R1/R2 " << result.verdict << " report=" << path.string() << '\n';
}
