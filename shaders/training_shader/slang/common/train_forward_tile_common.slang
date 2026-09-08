#include "train_forward_common.slang"

struct TileBounds {
    uint2 minTile;
    uint2 maxTileExclusive;
    bool valid;
};

struct GaussianTileCullingData {
    TileBounds bounds;
    float2 center;
    float3 conic;
    float negativeBOverA;
    float negativeBOverC;
    float maximumQuadratic;
    bool active;
    bool useExactEllipse;
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

TileBounds trainProjectedTileBoundsForExtent(float2 center, float2 extent) {
    TileBounds bounds;
    bounds.valid = false;
    bounds.minTile = uint2(0);
    bounds.maxTileExclusive = uint2(0);

    float tileSize = float(TRAIN_TILE_SIZE);
    float lastPixelOffset = float(TRAIN_TILE_SIZE - 1u);
    int2 tileGrid = int2(int(tileCountX()), int(trainTileCountY()));
    int2 minTile = int2(ceil(
        (center - extent - float2(lastPixelOffset)) / float2(tileSize)));
    int2 maxTileExclusive = int2(floor(
        (center + extent) / float2(tileSize))) + int2(1);
    minTile = clamp(minTile, int2(0), tileGrid);
    maxTileExclusive = clamp(maxTileExclusive, int2(0), tileGrid);
    bounds.minTile = uint2(minTile);
    bounds.maxTileExclusive = uint2(maxTileExclusive);
    bounds.valid = all(maxTileExclusive > minTile);
    return bounds;
}

TileBounds trainIntersectTileBounds(TileBounds lhs, TileBounds rhs) {
    TileBounds bounds;
    bounds.minTile = max(lhs.minTile, rhs.minTile);
    bounds.maxTileExclusive = min(lhs.maxTileExclusive, rhs.maxTileExclusive);
    bounds.valid = lhs.valid && rhs.valid && all(bounds.maxTileExclusive > bounds.minTile);
    return bounds;
}

GaussianTileCullingData trainPrepareGaussianTileCulling(ProjectedGaussian gaussian) {
    GaussianTileCullingData culling;
    culling.bounds.valid = false;
    culling.bounds.minTile = uint2(0);
    culling.bounds.maxTileExclusive = uint2(0);
    culling.center = gaussian.centerRadius.xy;
    culling.conic = gaussian.conicOpacity.xyz;
    culling.negativeBOverA = 0.0;
    culling.negativeBOverC = 0.0;
    culling.maximumQuadratic = 0.0;
    culling.active = false;
    culling.useExactEllipse = false;

    if (!(gaussian.conicOpacity.w >= TRAIN_ALPHA_MIN) ||
        !(gaussian.centerRadius.w > 0.0) ||
        !all(isfinite(gaussian.centerRadius.xy)) ||
        !isfinite(gaussian.centerRadius.w)) {
        return culling;
    }

    culling.bounds = trainProjectedTileBounds(gaussian);
    if (!culling.bounds.valid) {
        return culling;
    }
    culling.active = true;

    float a = gaussian.conicOpacity.x;
    float b = gaussian.conicOpacity.y;
    float c = gaussian.conicOpacity.z;
    float determinant = a * c - b * b;
    if (!all(isfinite(gaussian.conicOpacity)) ||
        !(a > 0.0) || !(c > 0.0) || !(determinant > 0.0)) {
        return culling;
    }

    culling.useExactEllipse = true;
    culling.negativeBOverA = -b / a;
    culling.negativeBOverC = -b / c;
    float supportQuadratic = 2.0 * log(gaussian.conicOpacity.w / TRAIN_ALPHA_MIN);
    float tolerance = 1e-5 * max(1.0, abs(supportQuadratic));
    culling.maximumQuadratic = supportQuadratic + tolerance;

    float acceptedQuadratic = max(0.0, culling.maximumQuadratic);
    float2 extent = sqrt(float2(
        acceptedQuadratic * c / determinant,
        acceptedQuadratic * a / determinant));
    if (all(isfinite(extent))) {
        TileBounds tightBounds = trainProjectedTileBoundsForExtent(culling.center, extent);
        culling.bounds = trainIntersectTileBounds(culling.bounds, tightBounds);
    }
    return culling;
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

float trainConicQuadratic(GaussianTileCullingData culling, float2 delta) {
    float a = culling.conic.x;
    float b = culling.conic.y;
    float c = culling.conic.z;
    return a * delta.x * delta.x + 2.0 * b * delta.x * delta.y + c * delta.y * delta.y;
}

float trainMinimumConicQuadraticOnRectangle(GaussianTileCullingData culling,
                                             float2 minimumDelta,
                                             float2 maximumDelta) {
    float yAtMinimumX = clamp(
        culling.negativeBOverC * minimumDelta.x, minimumDelta.y, maximumDelta.y);
    float yAtMaximumX = clamp(
        culling.negativeBOverC * maximumDelta.x, minimumDelta.y, maximumDelta.y);
    float minimumQuadratic = trainConicQuadratic(
        culling, float2(minimumDelta.x, yAtMinimumX));
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(culling, float2(maximumDelta.x, yAtMaximumX)));

    float xAtMinimumY = clamp(
        culling.negativeBOverA * minimumDelta.y, minimumDelta.x, maximumDelta.x);
    float xAtMaximumY = clamp(
        culling.negativeBOverA * maximumDelta.y, minimumDelta.x, maximumDelta.x);
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(culling, float2(xAtMinimumY, minimumDelta.y)));
    minimumQuadratic = min(
        minimumQuadratic,
        trainConicQuadratic(culling, float2(xAtMaximumY, maximumDelta.y)));
    return minimumQuadratic;
}

bool trainProjectedGaussianMayContributeToTile(GaussianTileCullingData culling, uint2 tile) {
    if (!culling.useExactEllipse) {
        return true;
    }

    uint2 minimumPixel = tile * TRAIN_TILE_SIZE;
    uint2 maximumPixelExclusive = min(
        minimumPixel + uint2(TRAIN_TILE_SIZE),
        uint2(params.width, params.height));
    if (any(maximumPixelExclusive <= minimumPixel)) {
        return false;
    }

    float2 minimumDelta = float2(minimumPixel) - culling.center;
    float2 maximumDelta = float2(maximumPixelExclusive - uint2(1u)) - culling.center;
    if (all(minimumDelta <= float2(0.0)) && all(maximumDelta >= float2(0.0))) {
        return true;
    }

    float minimumQuadratic = trainMinimumConicQuadraticOnRectangle(
        culling, minimumDelta, maximumDelta);
    return !isfinite(minimumQuadratic) || minimumQuadratic <= culling.maximumQuadratic;
}

uint trainProjectedGaussianTileCount(GaussianTileCullingData culling) {
    if (!culling.active || !culling.bounds.valid) {
        return 0u;
    }

    uint count = 0u;
    for (uint tileY = culling.bounds.minTile.y;
         tileY < culling.bounds.maxTileExclusive.y; ++tileY) {
        for (uint tileX = culling.bounds.minTile.x;
             tileX < culling.bounds.maxTileExclusive.x; ++tileX) {
            if (trainProjectedGaussianMayContributeToTile(culling, uint2(tileX, tileY))) {
                ++count;
            }
        }
    }
    return count;
}
