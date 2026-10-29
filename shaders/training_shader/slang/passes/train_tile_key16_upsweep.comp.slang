/*
MIT License

Copyright (c) 2024 jaesung-cs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/
#define KEY_VALUE 1
#include "../common/tile_key16_constants.slang"

StructuredBuffer<uint> elementCounts : register(t0, space0);
RWStructuredBuffer<uint> globalHistogram : register(u1, space0);
RWStructuredBuffer<uint> partitionHistogram : register(u2, space0);
StructuredBuffer<uint16_t> keys : register(t3, space0);

groupshared uint localHistogram[RADIX];

[shader("compute")]
[numthreads(WORKGROUP_SIZE)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID, uniform int pass) {
  uint elementCount = elementCounts[0];

  uint index = groupThreadID.x;
  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  // discard all workgroup invocations
  if (partitionStart >= elementCount) {
    return;
  }

  if (index < RADIX) {
    localHistogram[index] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + WORKGROUP_SIZE * i + index;
    uint key = keyIndex < elementCount ? keys[keyIndex] : 0xffffffff;
    uint radix = bitfieldExtract(key, 8 * pass, 8);
    __atomic_add(localHistogram[radix], 1, MemoryOrder.Relaxed);
  }
  GroupMemoryBarrierWithGroupSync();

  if (index < RADIX) {
    // set to partition histogram
    partitionHistogram[RADIX * partitionIndex + index] = localHistogram[index];

    // add to global histogram
    __atomic_add(globalHistogram[RADIX * pass + index], localHistogram[index], MemoryOrder.Relaxed);
  }
}
