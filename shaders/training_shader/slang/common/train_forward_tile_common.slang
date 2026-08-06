#include "train_forward_common.slang"

struct TileBounds {
    uint2 minTile;
    uint2 maxTileExclusive;
    bool valid;
};

uint trainTileCountY() {
    return (params.height + TRAIN_TILE_SIZE - 1) / TRAIN_TILE_SIZE;
}

uint trainTileCount() {
    return tileCountX() * trainTileCountY();
}

TileBounds trainProjectedTileBounds(ProjectedGaussian gaussian) {
    TileBounds bounds;
    bounds.valid = false;
    bounds.minTile = uint2(0);
    bounds.maxTileExclusive = uint2(0);

    if (gaussian.centerRadius.w <= 0.0 || gaussian.conicOpacity.w <= 0.0) {
        return bounds;
    }

    int radius = int(gaussian.centerRadius.w);
    int2 tileGrid = int2(int(tileCountX()), int(trainTileCountY()));
    int2 minTile = int2(
        int((gaussian.centerRadius.x - float(radius)) / float(TRAIN_TILE_SIZE)),
        int((gaussian.centerRadius.y - float(radius)) / float(TRAIN_TILE_SIZE))
    );
    int2 maxTileExclusive = int2(
        int((gaussian.centerRadius.x + float(radius) + float(TRAIN_TILE_SIZE - 1u)) / float(TRAIN_TILE_SIZE)),
        int((gaussian.centerRadius.y + float(radius) + float(TRAIN_TILE_SIZE - 1u)) / float(TRAIN_TILE_SIZE))
    );

    minTile = clamp(minTile, int2(0), tileGrid);
    maxTileExclusive = clamp(maxTileExclusive, int2(0), tileGrid);
    bounds.minTile = uint2(minTile);
    bounds.maxTileExclusive = uint2(maxTileExclusive);
    bounds.valid = all(maxTileExclusive > minTile);
    return bounds;
}

uint trainTileIndex(uint2 tile) {
    return tile.y * tileCountX() + tile.x;
}

bool trainGaussianDepthBefore(uint lhsGaussianIndex, uint rhsGaussianIndex, StructuredBuffer<ProjectedGaussian> projectedGaussians) {
    if (lhsGaussianIndex >= params.gaussianCount) {
        return false;
    }
    if (rhsGaussianIndex >= params.gaussianCount) {
        return true;
    }
    return projectedGaussians[lhsGaussianIndex].centerRadius.z < projectedGaussians[rhsGaussianIndex].centerRadius.z;
}

float trainConicQuadratic(ProjectedGaussian gaussian, float2 delta) {
    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;
    return a * delta.x * delta.x + 2.0 * b * delta.x * delta.y + c * delta.y * delta.y;
}

float trainMinimumConicQuadraticOnRectangle(ProjectedGaussian gaussian,
                                             float2 minimumDelta,
                                             float2 maximumDelta) {
    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;
    float minimumQuadratic = trainConicQuadratic(
        gaussian, clamp(float2(0.0), minimumDelta, maximumDelta));

    float yAtMinimumX = clamp(-b * minimumDelta.x / c, minimumDelta.y, maximumDelta.y);
    float yAtMaximumX = clamp(-b * maximumDelta.x / c, minimumDelta.y, maximumDelta.y);
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(gaussian, float2(minimumDelta.x, yAtMinimumX)));
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(gaussian, float2(maximumDelta.x, yAtMaximumX)));

    float xAtMinimumY = clamp(-b * minimumDelta.y / a, minimumDelta.x, maximumDelta.x);
    float xAtMaximumY = clamp(-b * maximumDelta.y / a, minimumDelta.x, maximumDelta.x);
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(gaussian, float2(xAtMinimumY, minimumDelta.y)));
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(gaussian, float2(xAtMaximumY, maximumDelta.y)));
    return minimumQuadratic;
}

bool trainProjectedGaussianMayContributeToTile(ProjectedGaussian gaussian, uint2 tile) {
    float opacity = gaussian.conicOpacity.w;
    if (!(opacity >= TRAIN_ALPHA_MIN)) {
        return false;
    }

    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;
    float determinant = a * c - b * b;
    if (!all(isfinite(gaussian.centerRadius.xy)) ||
        !all(isfinite(gaussian.conicOpacity)) ||
        !(a > 0.0) || !(c > 0.0) || !(determinant > 0.0)) {
        return true;
    }

    uint2 minimumPixel = tile * TRAIN_TILE_SIZE;
    uint2 maximumPixelExclusive = min(
        minimumPixel + uint2(TRAIN_TILE_SIZE),
        uint2(params.width, params.height));
    if (any(maximumPixelExclusive <= minimumPixel)) {
        return false;
    }

    float2 minimumDelta = float2(minimumPixel) - gaussian.centerRadius.xy;
    float2 maximumDelta = float2(maximumPixelExclusive - uint2(1u)) - gaussian.centerRadius.xy;
    float minimumQuadratic = trainMinimumConicQuadraticOnRectangle(
        gaussian, minimumDelta, maximumDelta);
    float supportQuadratic = 2.0 * log(opacity / TRAIN_ALPHA_MIN);
    float tolerance = 1e-5 * max(1.0, abs(supportQuadratic));
    return !isfinite(minimumQuadratic) || minimumQuadratic <= supportQuadratic + tolerance;
}

uint trainProjectedGaussianTileCount(ProjectedGaussian gaussian) {
    if (!(gaussian.conicOpacity.w >= TRAIN_ALPHA_MIN)) {
        return 0u;
    }
    TileBounds bounds = trainProjectedTileBounds(gaussian);
    if (!bounds.valid) {
        return 0u;
    }

    uint count = 0u;
    for (uint tileY = bounds.minTile.y; tileY < bounds.maxTileExclusive.y; ++tileY) {
        for (uint tileX = bounds.minTile.x; tileX < bounds.maxTileExclusive.x; ++tileX) {
            if (trainProjectedGaussianMayContributeToTile(gaussian, uint2(tileX, tileY))) {
                ++count;
            }
        }
    }
    return count;
}
