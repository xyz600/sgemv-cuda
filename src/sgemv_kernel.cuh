#include <cuda_runtime_api.h>
#include <cassert>

constexpr int unroll_size = 2;
constexpr int base_block_size = 32;
constexpr int block_size_y = base_block_size * unroll_size;
constexpr int block_size_x = base_block_size;

__device__ int reduce_warp(float value)
{
    value += __shfl_down_sync(0xffffffff, value, 16);
    value += __shfl_down_sync(0xffffffff, value, 8);
    value += __shfl_down_sync(0xffffffff, value, 4);
    value += __shfl_down_sync(0xffffffff, value, 2);
    value += __shfl_down_sync(0xffffffff, value, 1);
    return value;
}
__global__ void sgemv_dev(const int m, const int n, const float *__restrict__ A, const float *__restrict__ x, float *y)
{
    assert(gridDim.z == 1);
    assert(blockDim.x == warpSize);
    assert(blockDim.y == warpSize);
    assert(m % block_size == 0);
    assert(n % block_size == 0);

    float x_cur = 0, x_next = 0;
    float a_upper = 0, a_lower = 0;
    float upper_result = 0, lower_result = 0;

    A += (blockIdx.y * block_size_y + threadIdx.y) * n + blockIdx.x * block_size_x + threadIdx.x;
    x += blockIdx.x * block_size_x + threadIdx.x;

    const int a_offset = base_block_size * n;
    const int width = gridDim.x * block_size_x;

    __shared__ float global_upper_result[base_block_size];
    __shared__ float global_lower_result[base_block_size];
    if (threadIdx.y == 0)
    {
        global_upper_result[threadIdx.x] = 0;
        global_lower_result[threadIdx.x] = 0;
    }

    // load upper
    a_upper = *A;
    x_next = *x;

    for (int j = blockIdx.x * block_size_x; j < n; j += width)
    {
        // load lower
        a_lower = A[a_offset];

        // compute upper
        x_cur = x_next;
        upper_result += a_upper * x_cur;

        // if needed load upper
        if (j + width < n)
        {
            a_upper = A[width];
            x_next = x[width];
        }

        // compute lower
        lower_result += a_lower * x_cur;

        x += width;
        A += width;
    }

    // 重いかも？
    const auto upper_reduce = reduce_warp(upper_result);
    const auto lower_reduce = reduce_warp(lower_result);
    if (threadIdx.x == 0)
    {
        atomicAdd(&global_upper_result[threadIdx.y], upper_reduce);
        atomicAdd(&global_lower_result[threadIdx.y], lower_reduce);
    }

    __syncthreads();

    if (threadIdx.y == 0)
    {
        atomicAdd(&y[blockIdx.y * block_size_y + threadIdx.x], global_upper_result[threadIdx.x]);
        atomicAdd(&y[blockIdx.y * block_size_y + base_block_size + threadIdx.x], global_lower_result[threadIdx.x]);
    }
}