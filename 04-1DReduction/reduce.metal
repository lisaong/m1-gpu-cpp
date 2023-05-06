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
    uint tid = local_id;
    uint i = global_id;

    shared_data[tid] = (i < num_elements) ? X[i] : 0.0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // do reduction in shared memory
    for (uint s = 1; s < num_threads; s *= 2) {
        // modulo arithmetic is slow!
        if ((tid % (s*2)) == 0) {
            shared_data[tid] += shared_data[tid + s];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // write result for this block to global memory
    if (tid == 0) {
        result[group_id] = shared_data[0];
    }
}
