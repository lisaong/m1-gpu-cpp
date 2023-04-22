#include <metal_stdlib>
using namespace metal;


kernel void add_arrays(device const float* X [[buffer(0)]],
                       device const float* Y [[buffer(1)]],
                       device float* result  [[buffer(2)]],
                       uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] + Y[index];
}


kernel void multiply_arrays(device const float* X [[buffer(0)]],
                            device const float* Y [[buffer(1)]],
                            device float* result  [[buffer(2)]],
                            uint index            [[thread_position_in_grid]])
{
    result[index] = X[index] * Y[index];
}

kernel void saxpy(device const float* a [[buffer(0)]],
                  device const float* X [[buffer(1)]],
                  device const float* Y [[buffer(2)]],
                  device float* result  [[buffer(3)]],
                  uint index            [[thread_position_in_grid]])
{
    result[index] = (*a) * X[index] + Y[index];
}

kernel void central_difference(
                  device const float* delta [[buffer(0)]],
                  device const float* X     [[buffer(1)]],
                  device float* result      [[buffer(2)]],
                  uint index                [[thread_position_in_grid]],
                  uint arrayLength          [[threads_per_grid]])
{
    if (index == 0)
    {
        result[index] = (X[index + 1] - X[index]) /  *delta;
    }
    else if (index == arrayLength - 1)
    {
        result[index] = (X[index] - X[index - 1]) /  *delta;
    }
    else
    {
        result[index] = (X[index + 1] - X[index - 1]) / (2 * *delta);
    }
}

kernel void inspector(
                  device uint* store                           [[buffer(0)]],
                  uint thread_position_in_grid                 [[thread_position_in_grid]],
                  uint threads_per_grid                        [[threads_per_grid]],
                  uint dispatch_quadgroups_per_threadgroup     [[dispatch_quadgroups_per_threadgroup]],
                  uint dispatch_simdgroups_per_threadgroup     [[dispatch_simdgroups_per_threadgroup]], 
                  uint dispatch_threads_per_threadgroup        [[dispatch_threads_per_threadgroup]], 
                  uint grid_origin                             [[grid_origin]], 
                  uint grid_size                               [[grid_size]], 
                  uint quadgroup_index_in_threadgroup          [[quadgroup_index_in_threadgroup]], 
                  uint quadgroups_per_threadgroup              [[quadgroups_per_threadgroup]], 
                  uint simdgroup_index_in_threadgroup          [[simdgroup_index_in_threadgroup]], 
                  uint simdgroups_per_threadgroup              [[simdgroups_per_threadgroup]], 
                  uint thread_execution_width                  [[thread_execution_width]],
                  uint thread_index_in_quadgroup               [[thread_index_in_quadgroup]], 
                  uint thread_index_in_simdgroup               [[thread_index_in_simdgroup]], 
                  uint thread_index_in_threadgroup             [[thread_index_in_threadgroup]], 
                  uint thread_position_in_threadgroup          [[thread_position_in_threadgroup]], 
                  uint threadgroup_position_in_grid            [[threadgroup_position_in_grid]],
                  uint threadgroups_per_grid                   [[threadgroups_per_grid]], 
                  uint threads_per_threadgroup                 [[threads_per_threadgroup]])
{
    if (thread_position_in_grid == 0){
        store[0] = threads_per_grid;
        store[1] = dispatch_quadgroups_per_threadgroup; // quadgroup is 4 simd groups
        store[2] = dispatch_simdgroups_per_threadgroup;
        store[3] = dispatch_threads_per_threadgroup;
        store[4] = grid_origin;
        store[5] = grid_size;
        store[6] = quadgroup_index_in_threadgroup;
        store[7] = quadgroups_per_threadgroup;      
        store[8] = simdgroup_index_in_threadgroup;
        store[9] = simdgroups_per_threadgroup;
        store[9] = simdgroups_per_threadgroup;
        store[10] = thread_execution_width;
        store[12] = thread_index_in_simdgroup;
        store[13] = thread_index_in_threadgroup;
        store[14] = thread_position_in_threadgroup;
        store[15] = threadgroup_position_in_grid;
        store[16] = threadgroups_per_grid;
        store[17] = threads_per_threadgroup;
    }

}
