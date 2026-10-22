#ifndef TRAIN_PIXEL_CULLING_SLANG
#define TRAIN_PIXEL_CULLING_SLANG

// Method 1: FP32 only, opacity in [TRAIN_ALPHA_MIN, 1], power in [-80, 0].
// Disabled sentinel is outside that power interval. Do not broaden the domain
// or lower the guard without a new method version and boundary validation.
// Vulkan GLSL.std.450 bounds: division 2.5 ULP, log 3 ULP (absolute 2^-21
// around 1), exp (3+2*abs(p)) ULP. Over this domain their combined log-domain
// error is <2^-13, including a lowering to exp2(p*log2(e)). The exact 2^-10
// guard is eight times that bound. No RelaxedPrecision / FP16 is permitted.
// See docs/DEVELOPMENT.md, Pixel-level EarlyCull experiment.
static const float TRAIN_PIXEL_CULL_GUARD = 0.0009765625;
static const float TRAIN_PIXEL_CULL_DISABLED = -81.0;

float trainPixelCullThreshold(float opacity) {
    if (!isfinite(opacity) || opacity < TRAIN_ALPHA_MIN || opacity > 1.0)
        return TRAIN_PIXEL_CULL_DISABLED;
    precise float ratio = TRAIN_ALPHA_MIN / opacity;
    precise float threshold = log(ratio) - TRAIN_PIXEL_CULL_GUARD;
    return isfinite(threshold) ? threshold : TRAIN_PIXEL_CULL_DISABLED;
}

bool trainPixelCullReject(float power, float threshold) {
    return power >= -80.0 && power < threshold;
}

// Evaluate the ORIGINAL power and retained sample arithmetic. Never use an
// algebraically reordered ellipse/scanline expression as the final power.
GaussianSample trainEvaluatePixelCulledSample(ProjectedGaussian gaussian,
                                              float2 samplePosition, float threshold) {
    GaussianSample sample;
    sample.delta = gaussianPixelDelta(gaussian, samplePosition);
    sample.power = evaluateGaussianPower(gaussian, sample.delta);
    sample.coverage = 0.0;
    if (isInsideGaussianSupport(sample.power) &&
        !trainPixelCullReject(sample.power, threshold))
        sample.coverage = evaluateGaussianCoverage(sample.power);
    sample.alpha = evaluateGaussianAlpha(gaussian, sample.coverage);
    sample.color = gaussian.color.xyz;
    return sample;
}

#endif
