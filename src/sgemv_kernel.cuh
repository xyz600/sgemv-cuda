#include <cuda_runtime_api.h>
#include <cassert>

constexpr int unroll_size = 8;
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
    assert(m % base_block_size == 0);
    assert(n % base_block_size == 0);

    float x_cur = 0, x_next = 0;

    float a_cache[unroll_size] = {};
    float result[unroll_size] = {};

    A += (blockIdx.y * block_size_y + threadIdx.y) * n + blockIdx.x * block_size_x + threadIdx.x;
    x += blockIdx.x * block_size_x + threadIdx.x;

    const int width = gridDim.x * block_size_x;

    __shared__ float global_result[block_size_y];
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < block_size_y; i += blockDim.x * blockDim.y)
    {
        global_result[i] = 0;
    }

    // load upper
    x_next = *x;
    for (int k = 0; k < unroll_size / 2; k++)
    {
        a_cache[k] = A[k * base_block_size * n];
    }

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += width)
    {
        // load lower
        for (int k = unroll_size / 2; k < unroll_size; k++)
        {
            a_cache[k] = A[k * base_block_size * n];
        }

        // compute upper
        x_cur = x_next;
        for (int k = 0; k < unroll_size / 2; k++)
        {
            result[k] += a_cache[k] * x_cur;
        }

        // if needed load upper
        if (j + width < n)
        {
            x_next = x[width];
            for (int k = 0; k < unroll_size / 2; k++)
            {
                a_cache[k] = A[k * base_block_size * n + width];
            }
        }

        // compute lower
        for (int k = unroll_size / 2; k < unroll_size; k++)
        {
            result[k] += a_cache[k] * x_cur;
        }

        A += width;
        x += width;
    }

    // 重いかも？
    for (int k = 0; k < unroll_size; k++)
    {
        const auto sum = reduce_warp(result[k]);
        if (threadIdx.x == 0)
        {
            global_result[k * base_block_size + threadIdx.y] = sum;
        }
    }
    __syncthreads();

    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < block_size_y; i += blockDim.y * blockDim.x)
    {
        atomicAdd(&y[blockIdx.y * block_size_y + i], global_result[i]);
    }
}