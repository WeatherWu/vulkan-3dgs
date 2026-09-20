struct FragmentInput {
    float3 color : TEXCOORD0;
    float alpha : TEXCOORD1;
    float2 gaussianUV : TEXCOORD2;
    float compatibleProfile : TEXCOORD3;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    FragmentOutput output;
    float opacity = clamp(input.alpha, 0.0, 1.0);
    if (input.compatibleProfile > 0.5) {
        float A = dot(input.gaussianUV, input.gaussianUV);
        if (A > 1.0) {
            discard;
        }

        static const float EXP4 = 0.01831563888873418;
        static const float INV_ONE_MINUS_EXP4 = 1.018657360363774;
        float normalizedGaussian =
            (exp(-4.0 * A) - EXP4) * INV_ONE_MINUS_EXP4;
        float finalAlpha = opacity * max(normalizedGaussian, 0.0);
        if (finalAlpha < 1.0 / 255.0) {
            discard;
        }
        output.color = float4(input.color * finalAlpha, finalAlpha);
    } else {
        float gaussianAlpha = exp(-0.5 * dot(input.gaussianUV, input.gaussianUV));
        float finalAlpha = opacity * gaussianAlpha;
        output.color = float4(input.color, finalAlpha);
    }
    return output;
}
