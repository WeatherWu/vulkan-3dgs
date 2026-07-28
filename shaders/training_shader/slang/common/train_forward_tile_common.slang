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
