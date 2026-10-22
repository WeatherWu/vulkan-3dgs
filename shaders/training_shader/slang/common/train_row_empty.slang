#ifndef TRAIN_ROW_EMPTY_SLANG
#define TRAIN_ROW_EMPTY_SLANG
#include "train_pixel_culling.slang"

// Experimental classification/isolated candidate predicate: 0 = proven empty, 1 = keep, 2 = fallback/keep.
// Contract and forward-error ledger: docs/training-row-empty-contract.md.
// Unlike the diagnostic span, no determinant division or square root is used
// to prove exclusion. Division selects ANY tangent point, not an exact vertex.
float trainRowEmptyThreshold(ProjectedGaussian g) {
    const float low = 0.0000152587890625; // 2^-16
    const float high = 65536.0;
    if (!all(isfinite(g.centerRadius.xy)) ||
        any(abs(g.centerRadius.xy) > 1048576.0) || !all(isfinite(g.conicOpacity)))
        return TRAIN_PIXEL_CULL_DISABLED;
    float a = g.conicOpacity.x, b = g.conicOpacity.y, c = g.conicOpacity.z;
    if (a < low || a > high || c < low || c > high || abs(b) > high ||
        (b != 0.0 && abs(b) < low)) return TRAIN_PIXEL_CULL_DISABLED;
    precise float ac = a * c;
    precise float bb = b * b;
    // A scope restriction only. The tangent lower bound needs a>0, not SPD.
    if (!(ac - bb > ac * 0.0009765625)) return TRAIN_PIXEL_CULL_DISABLED;
    return trainPixelCullThreshold(g.conicOpacity.w);
}

uint trainRowEmptyClassify(ProjectedGaussian g, float threshold,
                          uint2 origin, uint row, out float lowerBound) {
    lowerBound = 0.0;
    if (params.width > 1048576u || params.height > 1048576u ||
        origin.x >= params.width || origin.y >= params.height ||
        row >= params.height - origin.y || threshold == TRAIN_PIXEL_CULL_DISABLED)
        return 2u;
    uint width = min(16u, params.width - origin.x);
    precise float lo = float(origin.x) - g.centerRadius.x;
    precise float hi = float(origin.x + width - 1u) - g.centerRadius.x;
    precise float dy = float(origin.y + row) - g.centerRadius.y;
    precise float dx = max(abs(lo), abs(hi));
    precise float a = g.conicOpacity.x, b = g.conicOpacity.y, c = g.conicOpacity.z;
    precise float m = a * dx * dx + 2.0 * abs(b) * dx * abs(dy) + c * dy * dy + 1.0;
    // Ensures ALL original powers are in [-80,0] if the row is rejected;
    // do not extend the existing exp/log contract into the underflow region.
    if (!isfinite(m) || m > 128.0) return 2u;
    precise float v = clamp((-b * dy) / a, lo, hi);
    precise float qv = a * v * v + 2.0 * b * v * dy + c * dy * dy;
    precise float slope = 2.0 * (a * v + b * dy);
    precise float widthDelta = hi - lo;
    precise float tangent = qv - abs(slope) * widthDelta;
    // 256*epsilon*M covers qv, slope, width, subtraction, original power,
    // either rounding direction and absolute FTZ errors (M>=1).
    precise float error = 0.000030517578125 * m;
    precise float bound = tangent - error;
    precise float limit = -2.0 * threshold;
    if (!isfinite(v) || !isfinite(bound)) return 2u;
    lowerBound = bound;
    return bound > limit ? 0u : 1u;
}
#endif
