# VkSplat shader source

Reference: https://github.com/harry7557558/vksplat/tree/b3ad2b048918496616e0e9346e35c606d813ab55

License: Apache-2.0; see LICENSE and the source copyright notices.

The SSIM, geometry, SH and optimizer arithmetic comes from this fixed source.
Project adapters provide Vulkan bindings, explicit uniforms, loss/statistics
outputs and canonical state boundaries. Rectangle storage and temporary tile
coordinates use uint32, within the official image-size bounds, without requesting
shaderInt16. Quaternion adapters map project xyzw and upstream wxyz, including
moments. The padding buffer preserves unused canonical fields at state boundaries.

The composite variants use the official forward and 4x8 subtile organization;
32/64 subgroup configurations are explicit modules. The Tensor organization and
common projected-gradient interface come from the project reference shaders,
with upstream alpha-cap and saturation-gradient conventions. These components
are combined with the project ADC/optimizer schedule and exact-depth sorting,
not the complete upstream scheduler or quantized sorting.
