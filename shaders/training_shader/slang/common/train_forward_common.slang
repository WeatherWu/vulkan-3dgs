#include "train_common.slang"

#ifndef TRAIN_TILE_SIZE
#define TRAIN_TILE_SIZE 16
#endif

#ifndef TRAIN_ADAPTIVE_TILES
#define TRAIN_ADAPTIVE_TILES 0
#endif

#ifndef TRAIN_ADAPTIVE_TILE_WRITES
#define TRAIN_ADAPTIVE_TILE_WRITES 0
#endif

static const uint TRAIN_TILE_PIXEL_COUNT = TRAIN_TILE_SIZE * TRAIN_TILE_SIZE;
static const uint TRAIN_ADAPTIVE_CHILD_DISPATCH_WIDTH = 65535u;
static const float TRAIN_ALPHA_MAX = 0.99;
static const float TRAIN_ALPHA_MIN = 1.0 / 255.0;
static const float TRAIN_TRANSMITTANCE_MIN = 1e-4;

struct TileRange {
    uint start;
    uint count;
};

#if TRAIN_ADAPTIVE_TILES
#if TRAIN_ADAPTIVE_TILE_WRITES
[[vk::binding(34, 0)]]
RWStructuredBuffer<uint> adaptiveTileSplitFlags;
#else
[[vk::binding(34, 0)]]
StructuredBuffer<uint> adaptiveTileSplitFlags;
#endif
#endif

#if TRAIN_ADAPTIVE_TILES && TRAIN_TILE_SIZE == 8
[[vk::binding(40, 0)]]
StructuredBuffer<uint> adaptiveActiveParents;
[[vk::binding(41, 0)]]
StructuredBuffer<uint4> adaptiveChildDispatch;
#endif

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
    uint contributionCount;
};

uint tileCountX() {
    return (params.width + TRAIN_TILE_SIZE - 1) / TRAIN_TILE_SIZE;
}

uint tileCountY() {
    return (params.height + TRAIN_TILE_SIZE - 1) / TRAIN_TILE_SIZE;
}

uint tileRangeCapacity() {
#if TRAIN_ADAPTIVE_TILES
    return params.tileSize == 32u ? tileCountX() * tileCountY() * 4u
                                  : tileCountX() * tileCountY();
#else
    return tileCountX() * tileCountY();
#endif
}

uint tileIndexForPixel(uint2 pixel) {
#if TRAIN_ADAPTIVE_TILES
    if (params.tileSize == 32u) {
        uint2 parent = pixel / 16u;
        uint parentIndex = parent.y * tileCountX() + parent.x;
        if (adaptiveTileSplitFlags[parentIndex] != 0u) {
            uint2 localChild = (pixel % 16u) / 8u;
            uint childIndex = localChild.y * 2u + localChild.x;
            return parentIndex * 4u + childIndex;
        }
        return parentIndex * 4u;
    }
#endif
    uint2 tile = uint2(pixel.x / TRAIN_TILE_SIZE, pixel.y / TRAIN_TILE_SIZE);
    return tile.y * tileCountX() + tile.x;
}

struct TrainingTileWorkgroup {
    uint tileIndex;
    uint2 origin;
    bool active;
};

TrainingTileWorkgroup trainingTileWorkgroup(uint2 groupID) {
    TrainingTileWorkgroup workgroup;
    workgroup.tileIndex = 0u;
    workgroup.origin = uint2(0u);
    workgroup.active = true;
#if TRAIN_ADAPTIVE_TILES
    uint parentCountX = (params.width + 15u) / 16u;
#if TRAIN_TILE_SIZE == 16
    uint2 parent = groupID;
    uint parentIndex = parent.y * parentCountX + parent.x;
    workgroup.active = adaptiveTileSplitFlags[parentIndex] == 0u;
    workgroup.tileIndex = parentIndex * 4u;
    workgroup.origin = parent * 16u;
#elif TRAIN_TILE_SIZE == 8
    uint childWorkIndex = groupID.y * TRAIN_ADAPTIVE_CHILD_DISPATCH_WIDTH + groupID.x;
    if (childWorkIndex / 4u >= adaptiveChildDispatch[0].w) {
        workgroup.active = false;
        return workgroup;
    }
    uint parentIndex = adaptiveActiveParents[childWorkIndex / 4u];
    if (parentIndex >= parentCountX * ((params.height + 15u) / 16u)) {
        workgroup.active = false;
        return workgroup;
    }
    uint2 parent = uint2(parentIndex % parentCountX, parentIndex / parentCountX);
    uint childIndex = childWorkIndex % 4u;
    uint2 child = uint2(childIndex & 1u, childIndex >> 1u);
    workgroup.tileIndex = parentIndex * 4u + child.y * 2u + child.x;
    workgroup.origin = parent * 16u + child * 8u;
    workgroup.active = workgroup.origin.x < params.width &&
                       workgroup.origin.y < params.height;
#else
    workgroup.active = false;
    workgroup.tileIndex = 0u;
    workgroup.origin = uint2(0u);
#endif
#else
    workgroup.tileIndex = groupID.y * tileCountX() + groupID.x;
    workgroup.origin = groupID * TRAIN_TILE_SIZE;
#endif
    return workgroup;
}

float2 pixelCenter(uint2 pixel) {
    return float2(pixel);
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
    composite.contributionCount = 0u;
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
