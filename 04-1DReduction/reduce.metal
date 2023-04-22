#include <metal_stdlib>
using namespace metal;

kernel void reduce1D(device const float* X [[buffer(0)]],
                     device float* result [[buffer(1)]],
                     uint index [[thread_position_in_grid]])
{
    // TODO
}