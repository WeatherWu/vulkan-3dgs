struct FragmentInput {
    float3 color : TEXCOORD0;
    float alpha : TEXCOORD1;
    float2 gaussianUV : TEXCOORD2;
};

struct FragmentOutput {
    float4 color : SV_Target0;
};

[shader("fragment")]
FragmentOutput main(FragmentInput input) {
    FragmentOutput output;
    float gaussianAlpha = exp(-0.5 * dot(input.gaussianUV, input.gaussianUV));
    float finalAlpha = clamp(input.alpha, 0.0, 1.0) * gaussianAlpha;
    output.color = float4(input.color, finalAlpha);
    return output;
}
