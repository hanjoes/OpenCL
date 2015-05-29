// this kernel can run on both CPU and GPU
__kernel void vadd(__global uint4 *op1,
                   __global uint4 *op2,
                   __global uint4 *ret,
                   uint dev,
                   uint numItems)
{
    // For both CPU and GPU, 'count' gives the number of
    // instructions each core/warp/wavefront will need
    // to finish in the logical 'vector processor'
    uint count = (numItems / 4) / get_global_size(0);

    // For CPU, the work starts at the the beginning of each segment.
    // For example: [0 1 2 3 4 5 6 7] processed on two cores, the start index
    //              for the first work-item is 0, and for the second work-item it
    //              is 4.
    // For GPU, since each warp/wavefront executes in lock step, the start
    // index for each work-group will be the global id for the current
    // work-group. And the stride is the number of work-items in each group.
    // Different from CPU execution model, the kernel instance on the GPU is
    // executed in SIMD fashion.
    uint idx =  (dev == 0) ? (get_global_id(0) * count) : get_global_id(0);

    // For CPU, the stride is 1, which gives us better utilize space locality
    // For GPU, the stride is the number of total work-items, since the kernel
    //          instance is executed in SIMD lanes.
    uint stride = (dev == 0) ? 1 : get_global_size(0);

    // For all devices, the number of iterations is specified by 'count'.
    for (uint i = 0; i < count; ++i, idx += stride) {
        ret[idx] = op1[idx] + op2[idx];
    }
}

