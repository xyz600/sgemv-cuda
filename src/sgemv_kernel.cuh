#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>

constexpr int block_size = 32;
constexpr int div_size = 4;
constexpr int val_per_thread = block_size / div_size / 2;

__global__ void sgemv_dev(const int m, const int n, const float *__restrict__ A, const float *__restrict__ x, float *y)
{
    assert(gridDim.x == gridDim.z == 1);
    assert(blockDim.x == warpSize);
    assert(blockDim.y == div_size);
    assert(m % block_size == 0);
    assert(n % block_size == 0);

    // warp 単位で連続 32 個の x を読む
    float xelem, xelem_next;

    float upper[val_per_thread];
    float lower[val_per_thread];
    float upper_result[val_per_thread] = {};
    float lower_result[val_per_thread] = {};

    A += (blockIdx.y * block_size + threadIdx.y * val_per_thread) * n;
    y += blockIdx.y * block_size + threadIdx.y * val_per_thread;

    // load upper
    for (int k = 0; k < val_per_thread; k++)
    {
        upper[k] = A[k * n + threadIdx.x];
    }
    xelem_next = x[threadIdx.x];

    for (int j = 0; j < n; j += block_size)
    {
        // load lower
        for (int k = 0; k < val_per_thread; k++)
        {
            lower[k] = A[(k + block_size / 2) * n + threadIdx.x];
        }
        __syncwarp();

        // compute upper
        xelem = xelem_next;
        for (int k = 0; k < val_per_thread; k++)
        {
            const auto xk = __shfl_sync(0xffffffff, xelem, threadIdx.y * val_per_thread + k);
            upper_result[k] += upper[k] * xk;
        }

        // if needed load upper
        if (j + block_size < n)
        {
            for (int k = 0; k < val_per_thread; k++)
            {
                upper[k] = A[k * n + block_size + threadIdx.x];
            }
            xelem_next = x[block_size + threadIdx.x];
        }
        __syncwarp();

        // compute lower
        for (int k = 0; k < val_per_thread; k++)
        {
            const auto xk = __shfl_sync(0xffffffff, xelem, block_size / 2 + threadIdx.y * val_per_thread + k);
            lower_result[k] += upper[k] * xk;
        }

        x += block_size;
        A += block_size;
    }

    for (int k = 0; k < val_per_thread; k++)
    {
        atomicAdd(&y[k], upper_result[k]);
        atomicAdd(&y[k + block_size / 2], lower_result[k]);
    }
}