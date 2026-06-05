float l1Color(float3 a, float3 b) {
    float3 diff = abs(a - b);
    return (diff.x + diff.y + diff.z) / 3.0;
}

float simpleSsimColor(float3 a, float3 b) {
    float3 diff = a - b;
    float3 value = 1.0 - diff * diff;
    return value.x * value.y * value.z;
}
