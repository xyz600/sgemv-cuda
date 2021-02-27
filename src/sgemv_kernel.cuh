#include <cuda_runtime_api.h>
#include <cassert>

constexpr int block_size = 32;

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

    float x_left = 0, x_right = 0;
    float a_left = 0, a_right = 0;

    float result = 0;

    A += (blockIdx.y * block_size + threadIdx.y) * n + blockIdx.x * block_size + threadIdx.x;
    x += blockIdx.x * block_size + threadIdx.x;

    const int width = 2 * gridDim.x * block_size;

    __shared__ float global_result[block_size];
    if (threadIdx.y == 0)
    {
        global_result[threadIdx.x] = 0;
    }

    // load left
    a_left = *A;
    x_left = *x;

    for (int j = blockIdx.x * block_size; j < n; j += width)
    {
        // load right
        a_right = A[width / 2];
        x_right = x[width / 2];

        // compute left
        result += a_left * x_left;

        // if needed load left
        if (j + width < n)
        {
            a_left = A[width];
            x_left = x[width];
        }

        // compute right
        result += a_right * x_right;

        x += width;
        A += width;
    }

    // 重いかも？
    global_result[threadIdx.y] = reduce_warp(result);
    __syncthreads();

    if (threadIdx.y == 0)
    {
        atomicAdd(&y[blockIdx.y * block_size + threadIdx.x], global_result[threadIdx.x]);
    }
}