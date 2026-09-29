#include "training/core/training_profiling_service.hpp"
#include "training/core/training_validation_service.hpp"

#include <cmath>
#include <iostream>

using namespace vulkan3DGS;

namespace {

bool near(float lhs, float rhs, float tolerance = 1e-6f) {
    return std::abs(lhs - rhs) <= tolerance;
}

bool testValidationConversionAndHistory() {
    TrainingValidationService service;
    TrainingValidationGpuResult result{};
    result.validationIteration = 1;
    result.lossSum = 0.5f;
    result.maxLoss = 0.75f;
    result.alphaSum = 2.0f;
    result.luminanceSum = 1.0f;
    result.validRenderedPixelCount = 4;
    result.processedCandidateSum = 8.0f;
    result.contributorSum = 4.0f;
    result.maxProcessedCandidates = 7;
    result.processedCandidateBucketsLow = glm::uvec4(1u, 1u, 1u, 1u);
    service.consumeResult(result, 12u, 2u, TrainingExtent{2u, 2u}, 8u);

    const TrainingValidationStats& stats = service.stats();
    const TrainingCandidateProfileStats& history =
        service.candidateProfileStats();
    if (!stats.valid || stats.tileItemCount != 12u ||
        !near(stats.meanLoss, 0.5f) ||
        !near(stats.meanRenderedAlpha, 0.5f) ||
        !near(stats.meanRenderedLuminance, 0.25f) ||
        !near(stats.meanProcessedCandidatesPerPixel, 2.0f) ||
        !near(stats.meanContributorsPerPixel, 1.0f) ||
        history.sampleCount != 1u ||
        !near(history.meanPixelFractionByProcessedBucket[0], 0.25f)) {
        return false;
    }

    result.validationIteration = 2;
    result.nonFiniteGaussianCount = 1;
    service.consumeResult(result, 12u, 2u, TrainingExtent{2u, 2u}, 8u);
    if (service.stats().valid ||
        service.candidateProfileStats().sampleCount != 2u) {
        return false;
    }

    service.resetCandidateProfileStats();
    return service.candidateProfileStats().sampleCount == 0u &&
           !service.stats().valid;
}

bool testProfilingAccumulators() {
    TrainingProfilingService service;
    service.recordCpu(TrainingCpuProfileStage::FrameUpload, 2.0f);
    service.recordCpu(TrainingCpuProfileStage::FrameUpload, 4.0f);
    const TrainingTiming first =
        service.stats().cpu[static_cast<size_t>(
            TrainingCpuProfileStage::FrameUpload)];
    if (first.sampleCount != 2u || !near(first.lastMs, 4.0f) ||
        !near(first.averageMs, 3.0f)) {
        return false;
    }

    service.resetLastSamples();
    const TrainingTiming afterLastReset =
        service.stats().cpu[static_cast<size_t>(
            TrainingCpuProfileStage::FrameUpload)];
    if (!near(afterLastReset.lastMs, 0.0f) ||
        !near(afterLastReset.averageMs, 3.0f) ||
        afterLastReset.sampleCount != 2u) {
        return false;
    }

    service.resetStats();
    const TrainingTiming afterFullReset =
        service.stats().cpu[static_cast<size_t>(
            TrainingCpuProfileStage::FrameUpload)];
    return afterFullReset.sampleCount == 0u &&
           near(afterFullReset.averageMs, 0.0f);
}

} // namespace

int main() {
    if (!testValidationConversionAndHistory()) {
        std::cerr << "Training validation service test failed\n";
        return 1;
    }
    if (!testProfilingAccumulators()) {
        std::cerr << "Training profiling service test failed\n";
        return 1;
    }
    return 0;
}
