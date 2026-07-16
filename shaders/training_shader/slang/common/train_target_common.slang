float4 unpackTargetColor(uint packedColor) {
    const float inverse255 = 1.0 / 255.0;
    return float4(
        float(packedColor & 0xffu),
        float((packedColor >> 8u) & 0xffu),
        float((packedColor >> 16u) & 0xffu),
        float((packedColor >> 24u) & 0xffu)
    ) * inverse255;
}
