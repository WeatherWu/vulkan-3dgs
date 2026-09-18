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
    float determinant;
    float2 ellipseExtent;
    float2 scanExtrema;
    float maximumQuadratic;
    bool active;
    bool useExactEllipse;
};

struct TileScanRange {
    uint minimum;
    uint maximumExclusive;
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
    culling.determinant = 0.0;
    culling.ellipseExtent = float2(0.0);
    culling.scanExtrema = float2(0.0);
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
        culling.determinant = determinant;
        culling.ellipseExtent = extent;
        culling.scanExtrema = float2(
            (-b / c) * sqrt(c * acceptedQuadratic / determinant),
            (-b / a) * sqrt(a * acceptedQuadratic / determinant));
        culling.useExactEllipse = all(isfinite(culling.scanExtrema));
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

float2 trainEllipseRangeBound(float3 conic,
                              float determinant,
                              float maximumQuadratic,
                              float stripExtremum,
                              float stripMinimum,
                              float stripMaximum) {
    float a = conic.x;
    float b = conic.y;
    float minimumAt = clamp(-stripExtremum, stripMinimum, stripMaximum);
    float maximumAt = clamp(stripExtremum, stripMinimum, stripMaximum);
    float minimumRoot = sqrt(max(0.0,
        a * maximumQuadratic - determinant * minimumAt * minimumAt));
    float maximumRoot = sqrt(max(0.0,
        a * maximumQuadratic - determinant * maximumAt * maximumAt));
    return float2(
        (-b * minimumAt - minimumRoot) / a,
        (-b * maximumAt + maximumRoot) / a);
}

TileScanRange trainTileRangeForStrip(GaussianTileCullingData culling,
                                     uint stripTile,
                                     bool scanRows) {
    TileScanRange range;
    uint boundMinimum = scanRows ? culling.bounds.minTile.x : culling.bounds.minTile.y;
    uint boundMaximum = scanRows
        ? culling.bounds.maxTileExclusive.x
        : culling.bounds.maxTileExclusive.y;
    range.minimum = boundMinimum;
    range.maximumExclusive = boundMaximum;
    if (!culling.useExactEllipse) {
        return range;
    }

    uint stripImageExtent = scanRows ? params.height : params.width;
    uint outputImageExtent = scanRows ? params.width : params.height;
    uint minimumPixel = stripTile * TRAIN_TILE_SIZE;
    uint maximumPixelExclusive = min(minimumPixel + TRAIN_TILE_SIZE, stripImageExtent);
    if (maximumPixelExclusive <= minimumPixel) {
        range.maximumExclusive = range.minimum;
        return range;
    }

    float stripCenter = scanRows ? culling.center.y : culling.center.x;
    float outputCenter = scanRows ? culling.center.x : culling.center.y;
    float stripMinimum = float(minimumPixel) - stripCenter;
    float stripMaximum = float(maximumPixelExclusive - 1u) - stripCenter;
    float stripExtent = scanRows ? culling.ellipseExtent.y : culling.ellipseExtent.x;
    if (stripMaximum < -stripExtent || stripMinimum > stripExtent) {
        range.maximumExclusive = range.minimum;
        return range;
    }
    float3 orientedConic = scanRows ? culling.conic : culling.conic.zyx;
    float stripExtremum = scanRows ? culling.scanExtrema.x : culling.scanExtrema.y;
    float2 bound = trainEllipseRangeBound(
        orientedConic,
        culling.determinant,
        culling.maximumQuadratic,
        stripExtremum,
        stripMinimum,
        stripMaximum);

    float tileSize = float(TRAIN_TILE_SIZE);
    float lastPixelOffset = float(TRAIN_TILE_SIZE - 1u);
    int minimumTile = int(ceil((bound.x + outputCenter - lastPixelOffset) / tileSize));
    int maximumTileExclusive = int(floor((bound.y + outputCenter) / tileSize)) + 1;
    minimumTile = clamp(minimumTile, int(boundMinimum), int(boundMaximum));
    if (minimumTile < int(boundMaximum)) {
        uint lastPixel = min(
            (uint(minimumTile) + 1u) * TRAIN_TILE_SIZE,
            outputImageExtent) - 1u;
        if (float(lastPixel) < bound.x + outputCenter) {
            ++minimumTile;
        }
    }
    maximumTileExclusive = clamp(
        maximumTileExclusive, minimumTile, int(boundMaximum));
    range.minimum = uint(minimumTile);
    range.maximumExclusive = uint(maximumTileExclusive);
    return range;
}

bool trainGaussianScansRows(GaussianTileCullingData culling) {
    uint width = culling.bounds.maxTileExclusive.x - culling.bounds.minTile.x;
    uint height = culling.bounds.maxTileExclusive.y - culling.bounds.minTile.y;
    return height <= width;
}

uint trainProjectedGaussianTileCount(GaussianTileCullingData culling) {
    if (!culling.active || !culling.bounds.valid) {
        return 0u;
    }

    uint count = 0u;
    bool scanRows = trainGaussianScansRows(culling);
    uint stripMinimum = scanRows ? culling.bounds.minTile.y : culling.bounds.minTile.x;
    uint stripMaximum = scanRows
        ? culling.bounds.maxTileExclusive.y
        : culling.bounds.maxTileExclusive.x;
    for (uint stripTile = stripMinimum; stripTile < stripMaximum; ++stripTile) {
        TileScanRange range = trainTileRangeForStrip(culling, stripTile, scanRows);
        count += range.maximumExclusive - range.minimum;
    }
    return count;
}

uint2 trainTileFromScan(uint stripTile, uint rangeTile, bool scanRows) {
    if (scanRows) {
        return uint2(rangeTile, stripTile);
    } else {
        return uint2(stripTile, rangeTile);
    }
}
