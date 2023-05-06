#include <metal_stdlib>
using namespace metal;

// cf. https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/reduction/reduction_kernel.cu


kernel void reduceSum1D_0(device const float* X [[buffer(0)]],
                        device float* result [[buffer(1)]],
                        threadgroup float* shared_data [[threadgroup(0)]],
                        uint num_elements [[grid_size]],
                        uint global_id [[thread_position_in_grid]],
                        uint group_id [[threadgroup_position_in_grid]],
                        uint local_id [[thread_position_in_threadgroup]],
                        uint num_threads [[threads_per_threadgroup]])
{
    // if (global_id < num_elements) {
    //     shared_data[local_id] = X[global_id];
    // } else {
    //     shared_data[local_id] = 0.0;
    // }
    // threadgroup_barrier(mem_flags::mem_threadgroup);

    // for (uint stride = num_threads / 2; stride > 0; stride >>= 1) {
    //     if (local_id < stride) {
    //         reduceSum_local(shared_data, local_id, stride);
    //     }
    // }

    // if (local_id == 0) {
    //     result[group_id] = shared_data[0];
    // }
}
