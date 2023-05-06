#include <metal_stdlib>
using namespace metal;

// cf. https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu


kernel void reduceSum1D_2(device const float* X [[buffer(0)]],
                          device const unsigned long *xLength [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          threadgroup float* shared_data [[threadgroup(0)]],
                          uint global_id [[thread_position_in_grid]],
                          uint group_id [[threadgroup_position_in_grid]],
                          uint local_id [[thread_position_in_threadgroup]],
                          uint num_threads [[threads_per_threadgroup]])
{
    uint tid = local_id;
    uint i = global_id;
    uint n = xLength[0]; // Note: the [[grid_size]] attribute is not set for dispatchThreadgroups()

    shared_data[tid] = (i < n) ? X[i] : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // do reduction in shared memory
    for (uint s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // write result for this block to global memory
    if (tid == 0) {
        result[group_id] = shared_data[0];
    }
}
