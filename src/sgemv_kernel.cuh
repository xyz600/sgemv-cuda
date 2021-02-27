#include <cuda_runtime_api.h>
#include <cassert>

constexpr int block_size = 32;
constexpr int div_size = 4;
constexpr int val_per_thread = block_size / div_size / 2;
__global__ void sgemv_dev(const int m, const int n, const float *__restrict__ A, const float *__restrict__ x, float *y)
{
    assert(gridDim.z == 1);
    assert(blockDim.x == warpSize);
    assert(blockDim.y == div_size);
    assert(m % block_size == 0);
    assert(n % block_size == 0);

    // warp 単位で連続 32 個の x を読む
    float xelem = 0, xelem_next = 0;

    float upper[val_per_thread] = {};
    float lower[val_per_thread] = {};
    float upper_result[val_per_thread] = {};
    float lower_result[val_per_thread] = {};

    A += (blockIdx.y * block_size + threadIdx.y * val_per_thread) * n + blockIdx.x * block_size + threadIdx.x;
    x += blockIdx.x * block_size + threadIdx.x;

    const int width = gridDim.x * block_size;

    __shared__ float global_result[block_size];
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < block_size; i += blockDim.x * blockDim.y)
    {
        global_result[i] = 0;
    }

    // load upper
    for (int k = 0; k < val_per_thread; k++)
    {
        upper[k] = A[k * n];
    }
    xelem_next = *x;

    for (int j = blockIdx.x; j < n; j += width)
    {
        // load lower
        for (int k = 0; k < val_per_thread; k++)
        {
            lower[k] = A[(k + block_size / 2) * n];
        }

        // compute upper
        xelem = xelem_next;
        for (int k = 0; k < val_per_thread; k++)
        {
            upper_result[k] += upper[k] * xelem;
        }

        // if needed load upper
        if (j + width < n)
        {
            for (int k = 0; k < val_per_thread; k++)
            {
                upper[k] = A[k * n + width];
            }
            xelem_next = x[width];
        }

        // compute lower
        for (int k = 0; k < val_per_thread; k++)
        {
            lower_result[k] += lower[k] * xelem;
        }

        x += width;
        A += width;
    }

    for (int k = 0; k < val_per_thread; k++)
    {
        atomicAdd(&global_result[threadIdx.y * val_per_thread + k], upper_result[k]);
        atomicAdd(&global_result[block_size / 2 + threadIdx.y * val_per_thread + k], lower_result[k]);
    }
    __syncthreads();

    if (threadIdx.y == 0)
    {
        atomicAdd(&y[blockIdx.y * block_size + threadIdx.x], global_result[threadIdx.x]);
    }
}