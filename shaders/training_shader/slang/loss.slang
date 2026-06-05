[Differentiable]
float SSIM(float3 a, float3 b) {
    float3 diff = a - b;
    float ssim = 1.0;
    for (int i = 0; i < 3; i++) {
        float diffValue = diff[i];
        float ssimValue = 1.0 - diffValue * diffValue;
        ssim *= ssimValue;
    }
    return ssim;
}

[Differentiable]
float L1(float3 a, float3 b) {
    float3 diff = abs(a - b);
    return (diff.x + diff.y + diff.z) / 3.0;
}