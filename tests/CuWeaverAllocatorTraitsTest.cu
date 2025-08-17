#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuweaver_utils/AllocatorTraits.cuh>
#include <cuweaver/GlobalAllocator.cuh>

namespace {
    bool cudaAvailable() {
        int count = 0;
        auto st = cudaGetDeviceCount(&count);
        return (st == cudaSuccess) && (count > 0);
    }

    __global__ void testKernel(int* data, const size_t n) {
        if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n) {
            data[idx] = static_cast<int>(idx);
        }
    }
}

template <typename Alloc>
using traits = cuweaver::cudaAllocatorTraits<Alloc>;

template <typename T>
using GMEM = cuweaver::GlobalAllocator<T>;

#define BLOCK_SIZE 1024

TEST(CuWeaverAllocator, GlobalAllocator) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    int* hdata = new int[1024];
    const size_t n = 1024;
    GMEM<int> allocator;
    // Allocate device memory
    auto data = traits<GMEM<int>>::allocate(allocator, n);
    ASSERT_NE(data, nullptr);
    // Launch kernel to initialize data
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);
    testKernel<<<grid, block>>>(data, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    // Copy data back to host
    traits<GMEM<int>>::memcpy(allocator, hdata, data, n, cudaMemcpyDeviceToHost);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    // Verify data
    for (size_t i = 0; i < n; ++i) {
        EXPECT_EQ(hdata[i], static_cast<int>(i)) << "Mismatch at index " << i;
    }
    // Check max size
    size_t free;
    size_t total;
    ASSERT_EQ(cudaMemGetInfo(&free, &total), cudaSuccess);
    size_t maxSize = traits<GMEM<int>>::maxSize(allocator);
    EXPECT_LE(maxSize, free / sizeof(int)) << "Max size exceeds available memory.";
    // Deallocate device memory
    traits<GMEM<int>>::deallocate(allocator, data, n);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    // Clean up host memory
    delete[] hdata;
}