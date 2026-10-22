// Scratch has three float4 coefficient planes; RGB is used and W is zero.
// Keep weights/normalization identical to the original 11x11 shaders. Normalize
// only after the vertical pass; zero padding never renormalizes edge windows.
static const int SEPARABLE_SSIM_RADIUS = 5;
static const float SEPARABLE_SSIM_INV_WEIGHT_SUM = 0.0707622378;
static const float SEPARABLE_SSIM_WEIGHTS[6] = {
    1.0, 0.8007374029, 0.4111122905,
    0.1353352832, 0.0285655008, 0.0038659201
};
