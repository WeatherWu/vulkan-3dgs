struct GaussianInstance {
    float4 position;
    float4 scale;
    float4 rotation;
    float4 sh[16];
}

struct GaussianGrad {
    float4 dPosition;
    float4 dScale;
    float4 dRotation;
    float4 dSh[16];
}

struct AdamState {
    GaussianInstance m;
    GaussianInstance v;
}

[Differentiable]
float4 evaluateSH(GaussianInstance instance, float3 viewDirection) {
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

    float4 result = instance.sh[0] * SH_C0;
    if (viewDirection.x == 0.0 && viewDirection.y == 0.0 && viewDirection.z == 0.0) {
        return result;
    }

    float x = viewDirection.x;
    float y = viewDirection.y;
    float z = viewDirection.z;

    result += SH_C1 * (-y * instance.sh[1] + z * instance.sh[2] - x * instance.sh[3]);

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;
    result +=
        SH_C2[0] * xy * instance.sh[4] +
        SH_C2[1] * yz * instance.sh[5] +
        SH_C2[2] * (2.0 * zz - xx - yy) * instance.sh[6] +
        SH_C2[3] * xz * instance.sh[7] +
        SH_C2[4] * (xx - yy) * instance.sh[8];

    result +=
        SH_C3[0] * y * (3.0 * xx - yy) * instance.sh[9] +
        SH_C3[1] * xy * z * instance.sh[9] +
        SH_C3[2] * y * (4.0 * zz - xx - yy) * instance.sh[10] +
        SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * instance.sh[11] +
        SH_C3[4] * x * (4.0 * zz - xx - yy) * instance.sh[12] +
        SH_C3[5] * z * (xx - yy) * instance.sh[13] +
        SH_C3[6] * x * (xx - 3.0 * yy) * instance.sh[14];

    return result;
}