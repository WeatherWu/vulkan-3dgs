#ifndef TRAIN_SCANLINE_DIAGNOSTIC_SLANG
#define TRAIN_SCANLINE_DIAGNOSTIC_SLANG
#include "train_pixel_culling.slang"

// DIAGNOSTIC CANDIDATE ONLY. Never used to skip a real gradient. The census
// checks every rejected pixel against the original GPU alpha predicate.
// Non-SPD, ill-conditioned, non-finite and near-tangent rows retain all pixels.
float4 trainScanlinePrepare(ProjectedGaussian g) {
    float a = g.conicOpacity.x, b = g.conicOpacity.y, c = g.conicOpacity.z;
    float ac = a * c, determinant = ac - b * b;
    float threshold = trainPixelCullThreshold(g.conicOpacity.w);
    if (!all(isfinite(g.centerRadius.xy)) || !all(isfinite(g.conicOpacity)) ||
        !(a > 0.0) || !(c > 0.0) || !isfinite(ac) ||
        !(determinant > 0.0001 * ac) || threshold == TRAIN_PIXEL_CULL_DISABLED)
        return float4(0.0, 0.0, 0.0, -1.0);
    float4 result = float4(1.0 / a, b / a, determinant / a, -2.0 * threshold);
    return all(isfinite(result)) ? result : float4(0.0, 0.0, 0.0, -1.0);
}

uint trainScanlinePack(uint lo, uint hi, bool fallback) {
    return lo | (hi << 5u) | (fallback ? 1024u : 0u);
}

uint trainScanlineSpan(ProjectedGaussian g, float4 prepared, uint2 origin, uint row) {
    uint width = min(16u, params.width - min(origin.x, params.width));
    if (origin.y + row >= params.height || width == 0u) return 0u;
    uint full = trainScanlinePack(0u, width, true);
    if (prepared.w < 0.0) return full;
    float dy = float(origin.y + row) - g.centerRadius.y;
    float dx0 = float(origin.x) - g.centerRadius.x;
    float dx1 = float(origin.x + width - 1u) - g.centerRadius.x;
    float dx = max(abs(dx0), abs(dx1));
    float magnitude = abs(g.conicOpacity.x) * dx * dx +
        2.0 * abs(g.conicOpacity.y) * dx * abs(dy) +
        abs(g.conicOpacity.z) * dy * dy + prepared.w + 1.0;
    float error = 0.00000762939453125 * magnitude; // 64 * float32 epsilon.
    float budget = prepared.w + error - prepared.z * dy * dy;
    if (!isfinite(budget) || !isfinite(error)) return full;
    if (budget < -error) return 0u;
    if (budget <= error) return full;
    float radius = sqrt(budget * prepared.x);
    float shift = prepared.y * dy;
    float center = g.centerRadius.x - shift - float(origin.x);
    float pad = max(1.0, 0.00000762939453125 *
        (abs(g.centerRadius.x) + abs(shift) + radius + float(origin.x) + 16.0));
    if (!isfinite(radius) || !isfinite(center) || !isfinite(pad)) return full;
    // Clamp before uint conversion; expand outward at integer sample positions.
    uint lo = uint(clamp(floor(center - radius - pad), 0.0, float(width)));
    uint hi = uint(clamp(ceil(center + radius + pad) + 1.0, 0.0, float(width)));
    return trainScanlinePack(lo, max(lo, hi), false);
}
#endif
