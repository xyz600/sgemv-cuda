#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>

/**
 * @details Producer / Consumer モデルのデータの受け渡しのタイミングを制御する
 * Producer: Consumer に必要なデータを load する役割
 * Consumer: Producer が load してくれたデータを使って計算処理を行う
 * 何に対して待つのかは、 id で指定する。双方向の伝達が必要なので、単一目的のデータに対して2つ id が必要になる
 *
 * 通知の方法
 * ダブルバッファリングで、メモリネックなアプリを高速化することを想定する。そのうち、単一のバッファに関して、以下のようなやり取りが必要になる
 * - Producer:
 *     - notify_produce(): Producer が、data load 命令を発行し終えたことを通知する
 *     - wait_consume(): バッファを上書きするために、Consumer の処理が完了したことを待つ必要がある
 * - Consumer:
 *     - notify_consume(): Producer が新しくデータを上書きするために、Consumer 側の処理が完了したことを通知する
 *     - wait_produce(): Consumer が計算を開始するために、Producer が開始してくれたメモリの load を待つ必要がある
 * bar に渡す id は CTA 単位で 0..15 の計 16 個までしかない点に注意する
 */
struct ResourceManager
{
    __device__ void notify_consumed()
    {
        asm volatile("barrier.arrive %0, %1;" ::"r"(consumer_resource_id_), "r"(num_thread_in_group_) : "memory");
    }

    __device__ void wait_produced()
    {
        asm volatile("barrier.sync %0, %1;" ::"r"(producer_resource_id_), "r"(num_thread_in_group_) : "memory");
    }

    __device__ void notify_produced()
    {
        asm volatile("barrier.arrive %0, %1;" ::"r"(producer_resource_id_), "r"(num_thread_in_group_) : "memory");
    }

    __device__ void wait_consumed()
    {
        asm volatile("barrier.sync %0, %1;" ::"r"(consumer_resource_id_), "r"(num_thread_in_group_) : "memory");
    }

    __device__ void setup(int data_id, int consumer_num_thread, int producer_num_thread)
    {
        consumer_resource_id_ = 2 * data_id;
        producer_resource_id_ = 2 * data_id + 1;
        num_thread_in_group_ = consumer_num_thread + producer_num_thread;
    }

    int consumer_resource_id_;
    int producer_resource_id_;
    int num_thread_in_group_;
};

__global__ void sgemv_dev(const float *__restrict__ A, const float *__restrict__ x, float *y, const int n, const int m)
{
    // CTA 単位で、 block_size * block_size の行列を更新する
    constexpr int block_size = 64;
    constexpr int column_width = block_size / 2;

    assert(blockDim.x == warpSize);
    // use producer / consumer model
    assert(blockDim.y == (block_size / warpSize) * 2 * 2);

    // for double buffering
    // remove bank conflict, increment inner sh_A size
    __shared__ float sh_A[2][block_size][block_size + 1];
    __shared__ float sh_x[2][block_size + 1];

    const int consumer_num_thread = blockDim.x * blockDim.y / 2;
    const int producer_num_thread = blockDim.x * blockDim.y / 2;

    // double buffer それぞれのメモリ領域の管理用
    ResourceManager manager[2];
    // 半分が Consumer, 半分が Producer
    manager[0].setup(0, consumer_num_thread, producer_num_thread);
    manager[1].setup(1, consumer_num_thread, producer_num_thread);

    int execute_block_num = 0;
    int resource_idx = 0;

    auto inverse = [](int resource_id) { return resource_id == 0 ? 1 : 0; };

    if (threadIdx.y < blockDim.y / 2)
    {
        const int producer_threadIdx = threadIdx.y * blockDim.x + threadIdx.x;
        const int row_index = producer_threadIdx % block_size;
        const int column_index = producer_threadIdx / block_size;

        // Producer
        for (int i = blockIdx.y * block_size; i < n; i += gridDim.y * block_size)
        {
            for (int j = blockIdx.x * block_size; j < m; j += gridDim.x * block_size)
            {
                // 各バッファの最初の1回(合計2回)は完了通知がないので待つ必要がない
                if (2 <= execute_block_num)
                {
                    manager[resource_idx].wait_consumed();
                }

                // 行列を転置して sh_A に保存
                for (int ii = 0; ii < column_width; ii++)
                {
                    sh_A[resource_idx][row_index][ii + column_index * column_width] =
                        A[(i + ii + column_index * column_width) * m + j + row_index];
                }
                // ベクトルを sh_x に保存
                for (int jj = producer_threadIdx; jj < block_size; jj += producer_num_thread)
                {
                    sh_x[resource_idx][jj] = x[j + jj];
                }

                manager[resource_idx].notify_produced();

                execute_block_num++;
                resource_idx = inverse(resource_idx);
            }
        }
    }
    else
    {
        const int consumer_threadIdx = (threadIdx.y - blockDim.y / 2) * blockDim.x + threadIdx.x;
        const int row_index = consumer_threadIdx % block_size;
        const int column_index = consumer_threadIdx / block_size;

        // Consumer
        for (int i = blockIdx.y * block_size; i < m; i += gridDim.y * block_size)
        {
            float yi = 0;

            for (int j = blockIdx.x * block_size; j < m; j += gridDim.x * block_size)
            {
                manager[resource_idx].wait_produced();

                // y[i] = \sum_j a^T_ji x_j
                for (int jj = 0; jj < column_width; jj++)
                {
                    yi += sh_A[resource_idx][jj + column_index * column_width][row_index] *
                          sh_x[resource_idx][jj + column_index * column_width];
                }

                // 各バッファから見て直前の処理が終わったことをアナウンスする
                manager[resource_idx].notify_consumed();

                execute_block_num++;
                resource_idx = inverse(resource_idx);
            }
            atomicAdd(&y[i + row_index], yi);
        }
    }
}
