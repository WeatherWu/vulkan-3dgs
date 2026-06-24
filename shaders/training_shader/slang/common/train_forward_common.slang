#include "train_common.slang"

static const uint TRAIN_TILE_SIZE = 16;
static const float TRAIN_ALPHA_MAX = 0.99;
static const float TRAIN_ALPHA_MIN = 1e-4;
static const float TRAIN_TRANSMITTANCE_MIN = 1e-4;

struct TileRange {
    uint start;
    uint count;
};

struct GaussianSample {
    float2 delta;
    float power;
    float coverage;
    float alpha;
    float3 color;
};

struct PixelComposite {
    float3 color;
    float transmittance;
    float accumulatedAlpha;
    uint processedCount;
    uint lastContributor;
};

uint tileCountX() {
    return (params.width + TRAIN_TILE_SIZE - 1) / TRAIN_TILE_SIZE;
}

uint tileIndexForPixel(uint2 pixel) {
    uint2 tile = uint2(pixel.x / TRAIN_TILE_SIZE, pixel.y / TRAIN_TILE_SIZE);
    return tile.y * tileCountX() + tile.x;
}

float2 pixelCenter(uint2 pixel) {
    return float2(float(pixel.x) + 0.5, float(pixel.y) + 0.5);
}

TileRange unpackTileRange(uint4 packedRange) {
    TileRange range;
    range.start = packedRange.x;
    range.count = packedRange.y;
    return range;
}

float2 gaussianPixelDelta(ProjectedGaussian gaussian, float2 samplePosition) {
    return samplePosition - gaussian.centerRadius.xy;
}

float evaluateGaussianPower(ProjectedGaussian gaussian, float2 delta) {
    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;
    return -0.5 * (a * delta.x * delta.x + 2.0 * b * delta.x * delta.y + c * delta.y * delta.y);
}

bool isInsideGaussianSupport(float power) {
    return power <= 0.0;
}

float evaluateGaussianCoverage(float power) {
    return exp(power);
}

float evaluateGaussianAlpha(ProjectedGaussian gaussian, float coverage) {
    return clamp(gaussian.conicOpacity.w * coverage, 0.0, TRAIN_ALPHA_MAX);
}

GaussianSample evaluateGaussianSample(ProjectedGaussian gaussian, float2 samplePosition) {
    GaussianSample sample;
    sample.delta = gaussianPixelDelta(gaussian, samplePosition);
    sample.power = evaluateGaussianPower(gaussian, sample.delta);
    sample.coverage = isInsideGaussianSupport(sample.power) ? evaluateGaussianCoverage(sample.power) : 0.0;
    sample.alpha = evaluateGaussianAlpha(gaussian, sample.coverage);
    sample.color = gaussian.color.xyz;
    return sample;
}

bool contributesToPixel(GaussianSample sample) {
    return sample.alpha >= TRAIN_ALPHA_MIN;
}

PixelComposite makeEmptyComposite() {
    PixelComposite composite;
    composite.color = float3(0.0);
    composite.transmittance = 1.0;
    composite.accumulatedAlpha = 0.0;
    composite.processedCount = 0;
    composite.lastContributor = 0xFFFFFFFFu;
    return composite;
}

void compositeGaussianFrontToBack(inout PixelComposite composite, GaussianSample sample) {
    composite.color += composite.transmittance * sample.alpha * sample.color;
    composite.accumulatedAlpha += composite.transmittance * sample.alpha;
    composite.transmittance *= 1.0 - sample.alpha;
}

float4 finalizeComposite(PixelComposite composite) {
    return float4(composite.color, 1.0 - composite.transmittance);
}
