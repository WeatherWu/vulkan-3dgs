#include "../vksplat/native.slang"
[[vk::binding(3,0)]] StructuredBuffer<ProjectedGaussian> projected;
[[vk::binding(21,0)]] RWStructuredBuffer<uint> tileKeys;
[[vk::binding(24,0)]] RWStructuredBuffer<uint> gaussianIDs;
[[vk::binding(32,0)]] StructuredBuffer<uint2> orderedRanges;
[[vk::binding(62,0)]] StructuredBuffer<uint> visibleIDs;
[shader("compute")]
[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
 uint rank=tid.x;
 if (rank>=params.gaussianCount) return;
 uint id=visibleIDs[rank];
 float2 xy=projected[id].centerRadius.xy;
 float4 inv_cov_opac=projected[id].conicOpacity;
 float extend=extend_from_opacity(inv_cov_opac.w);
 float3 inv_cov=inv_cov_opac.xyz/(extend*extend);
 uint4 bounds=nativeRectangles[id];
 rectangle rect={bounds.x,bounds.y,bounds.z,bounds.w};
 uint grid_width=(params.width+15u)/16u,grid_height=(params.height+15u)/16u;
 uint2 range=orderedRanges[rank];
 uint offset=range.x,offset_max=min(range.x+range.y,params.tileItemCount);


    if (rect.max_y-rect.min_y <= rect.max_x-rect.min_x) {
        for (uint32_t y = rect.min_y; y < rect.max_y; y++) {
            float y0 = y * TILE_HEIGHT - xy.y;
            float y1 = y0 + TILE_HEIGHT;

            float2 bound = ellipse_range_bound(inv_cov, y0, y1);
            uint32_t min_x = uint32_t(clamp(
                floor((bound.x + xy.x) / TILE_WIDTH),
                rect.min_x, rect.max_x
            ));
            uint32_t max_x = uint32_t(clamp(
                ceil((bound.y + xy.x) / TILE_WIDTH),
                rect.min_x, rect.max_x
            ));
            for (uint32_t x = min_x; x < max_x && offset < offset_max; x++) {
                uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
                tileKeys[offset] = key;
                gaussianIDs[offset] = id;
                offset++;
            }
        }
    } else {
        inv_cov = inv_cov.zyx;
        for (uint32_t x = rect.min_x; x < rect.max_x; x++) {
            float x0 = x * TILE_WIDTH - xy.x;
            float x1 = x0 + TILE_WIDTH;

            float2 bound = ellipse_range_bound(inv_cov, x0, x1);
            uint32_t min_y = uint32_t(clamp(
                floor((bound.x + xy.y) / TILE_HEIGHT),
                rect.min_y, rect.max_y
            ));
            uint32_t max_y = uint32_t(clamp(
                ceil((bound.y + xy.y) / TILE_HEIGHT),
                rect.min_y, rect.max_y
            ));
            for (uint32_t y = min_y; y < max_y && offset < offset_max; y++) {
                uint32_t key = uint32_t(y) * grid_width + uint32_t(x);
                tileKeys[offset] = key;
                gaussianIDs[offset] = id;
                offset++;
            }
        }
    }

    for (; offset < offset_max; offset++) {
        // tileKeys[offset] = sortingKey_t(-1);
        tileKeys[offset] = uint(offset % (grid_width * grid_height));
        gaussianIDs[offset] = id;
    }


}
