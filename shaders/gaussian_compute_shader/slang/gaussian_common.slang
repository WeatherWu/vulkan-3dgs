struct UniformBufferObject {
    float4x4 view;
    float4x4 projection;
    float4x4 model;
    float4 cameraPosition_time;
    float4 focal;
};

struct GaussianInstance {
    float4 position;
    float4 scale;
    float4 rotation;
    uint2 sh0;
    uint2 sh1[3];
    uint2 sh2[5];
    uint2 sh3[7];
};

float unpackHalf(uint value) {
    uint halfValue = value & 0xffffu;
    uint sign = (halfValue & 0x8000u) << 16;
    uint exponent = (halfValue >> 10) & 0x1fu;
    uint mantissa = halfValue & 0x03ffu;

    uint floatBits = 0u;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            floatBits = sign;
        } else {
            int normalizedExponent = -14;
            while ((mantissa & 0x0400u) == 0u) {
                mantissa <<= 1;
                normalizedExponent -= 1;
            }
            mantissa &= 0x03ffu;
            floatBits = sign | (uint(normalizedExponent + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        floatBits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        floatBits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    return asfloat(floatBits);
}

float2 unpackHalf2(uint value) {
    return float2(unpackHalf(value), unpackHalf(value >> 16));
}

float3 unpackSH(uint2 packed) {
    float2 xy = unpackHalf2(packed.x);
    float2 z0 = unpackHalf2(packed.y);
    return float3(xy, z0.x);
}

float3 restore3DGSColor(float3 shColor) {
    return max(shColor + float3(0.5, 0.5, 0.5), float3(0.0, 0.0, 0.0));
}

float3x3 quatToMat3(float4 q) {
    return float3x3(
        1.0 - 2.0 * q.y * q.y - 2.0 * q.z * q.z,
        2.0 * q.x * q.y - 2.0 * q.z * q.w,
        2.0 * q.x * q.z + 2.0 * q.y * q.w,

        2.0 * q.x * q.y + 2.0 * q.z * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.z * q.z,
        2.0 * q.y * q.z - 2.0 * q.x * q.w,

        2.0 * q.x * q.z - 2.0 * q.y * q.w,
        2.0 * q.y * q.z + 2.0 * q.x * q.w,
        1.0 - 2.0 * q.x * q.x - 2.0 * q.y * q.y
    );
}

float3 evaluateSH(GaussianInstance instance, float3 viewDirection) {
    static const float SH_C0 = 0.28209479177387814;
    static const float SH_C1 = 0.4886025119029199;
    static const float SH_C2[5] = {
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    };
    static const float SH_C3[7] = {
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    };

    float3 result = unpackSH(instance.sh0) * SH_C0;
    if (viewDirection.x == 0.0 && viewDirection.y == 0.0 && viewDirection.z == 0.0) {
        return restore3DGSColor(result);
    }

    float x = viewDirection.x;
    float y = viewDirection.y;
    float z = viewDirection.z;

    result += SH_C1 * (-y * unpackSH(instance.sh1[0]) + z * unpackSH(instance.sh1[1]) - x * unpackSH(instance.sh1[2]));

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;
    result +=
        SH_C2[0] * xy * unpackSH(instance.sh2[0]) +
        SH_C2[1] * yz * unpackSH(instance.sh2[1]) +
        SH_C2[2] * (2.0 * zz - xx - yy) * unpackSH(instance.sh2[2]) +
        SH_C2[3] * xz * unpackSH(instance.sh2[3]) +
        SH_C2[4] * (xx - yy) * unpackSH(instance.sh2[4]);

    result +=
        SH_C3[0] * y * (3.0 * xx - yy) * unpackSH(instance.sh3[0]) +
        SH_C3[1] * xy * z * unpackSH(instance.sh3[1]) +
        SH_C3[2] * y * (4.0 * zz - xx - yy) * unpackSH(instance.sh3[2]) +
        SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * unpackSH(instance.sh3[3]) +
        SH_C3[4] * x * (4.0 * zz - xx - yy) * unpackSH(instance.sh3[4]) +
        SH_C3[5] * z * (xx - yy) * unpackSH(instance.sh3[5]) +
        SH_C3[6] * x * (xx - 3.0 * yy) * unpackSH(instance.sh3[6]);

    return restore3DGSColor(result);
}
