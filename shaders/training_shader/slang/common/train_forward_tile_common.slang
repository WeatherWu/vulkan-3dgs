#include "train_forward_common.slang"

struct TileBounds {
    uint2 minTile;
    uint2 maxTile;
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
    bounds.maxTile = uint2(0);

    if (gaussian.centerRadius.w <= 0.0 || gaussian.conicOpacity.w <= 0.0) {
        return bounds;
    }

    float2 minPixelF = floor(gaussian.centerRadius.xy - gaussian.centerRadius.ww);
    float2 maxPixelF = ceil(gaussian.centerRadius.xy + gaussian.centerRadius.ww);
    int2 minPixel = int2(int(minPixelF.x), int(minPixelF.y));
    int2 maxPixel = int2(int(maxPixelF.x), int(maxPixelF.y));

    if (maxPixel.x < 0 || maxPixel.y < 0 || minPixel.x >= int(params.width) || minPixel.y >= int(params.height)) {
        return bounds;
    }

    minPixel = clamp(minPixel, int2(0), int2(int(params.width) - 1, int(params.height) - 1));
    maxPixel = clamp(maxPixel, int2(0), int2(int(params.width) - 1, int(params.height) - 1));
    bounds.minTile = uint2(uint(minPixel.x) / TRAIN_TILE_SIZE, uint(minPixel.y) / TRAIN_TILE_SIZE);
    bounds.maxTile = uint2(uint(maxPixel.x) / TRAIN_TILE_SIZE, uint(maxPixel.y) / TRAIN_TILE_SIZE);
    bounds.valid = true;
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
