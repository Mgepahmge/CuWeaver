#include <gtest/gtest.h>
#include <cuweaver/StreamPool.cuh>
#include <cuweaver/Stream.cuh>

#include <cuda_runtime.h>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <set>

// Simple CUDA kernel for testing purposes
__global__ void dummyKernel(float* data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = val * 1.001f + 0.001f;
        }
        data[idx] = val;
    }
}

// Memory copy test kernel
__global__ void memcpyKernel(const float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx] * 2.0f;
    }
}

class CuWeaverStreamPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA device
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        ASSERT_GT(deviceCount, 0) << "No CUDA devices available";
        
        cudaSetDevice(0);
        
        // Allocate test data
        dataSize = 1024 * 1024;  // 1M elements
        cudaMalloc(&d_data1, dataSize * sizeof(float));
        cudaMalloc(&d_data2, dataSize * sizeof(float));
        cudaMalloc(&d_data3, dataSize * sizeof(float));
        
        h_data.resize(dataSize, 1.0f);
        cudaMemcpy(d_data1, h_data.data(), dataSize * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    void TearDown() override {
        cudaFree(d_data1);
        cudaFree(d_data2);
        cudaFree(d_data3);
        cudaDeviceReset();
    }
    
    int dataSize;
    float* d_data1;
    float* d_data2;
    float* d_data3;
    std::vector<float> h_data;
};

// Test basic constructor functionality
TEST_F(CuWeaverStreamPoolTest, BasicConstruction) {
    // Test default construction
    EXPECT_NO_THROW({
        cuweaver::StreamPool pool;
    });
    
    // Test construction with custom parameters
    EXPECT_NO_THROW({
        cuweaver::StreamPool pool(8, 2);
    });
    
    // Test invalid parameters - zero execution streams
    EXPECT_THROW({
        cuweaver::StreamPool pool(0, 4);
    }, std::invalid_argument);
    
    // Test invalid parameters - zero resource streams
    EXPECT_THROW({
        cuweaver::StreamPool pool(4, 0);
    }, std::invalid_argument);
    
    // Test invalid parameters - both zero
    EXPECT_THROW({
        cuweaver::StreamPool pool(0, 0);
    }, std::invalid_argument);
}

// Test stream retrieval and circular allocation behavior
TEST_F(CuWeaverStreamPoolTest, StreamRetrieval) {
    cuweaver::StreamPool pool(4, 2);
    
    // Test execution stream retrieval
    std::vector<cuweaver::cudaStream*> executionStreams;
    std::set<cudaStream_t> uniqueExecutionStreams;
    
    // Retrieve more streams than pool size to test circular behavior
    for (int i = 0; i < 8; ++i) {
        cuweaver::cudaStream& stream = pool.getExecutionStream();
        executionStreams.push_back(&stream);
        uniqueExecutionStreams.insert(stream.nativeHandle());
    }
    
    // Should have exactly 4 unique streams (circular reuse)
    EXPECT_EQ(uniqueExecutionStreams.size(), 4);
    
    // Test resource stream retrieval
    std::vector<cuweaver::cudaStream*> resourceStreams;
    std::set<cudaStream_t> uniqueResourceStreams;
    
    // Retrieve more streams than pool size to test circular behavior
    for (int i = 0; i < 6; ++i) {
        cuweaver::cudaStream& stream = pool.getResourceStream();
        resourceStreams.push_back(&stream);
        uniqueResourceStreams.insert(stream.nativeHandle());
    }
    
    // Should have exactly 2 unique streams (circular reuse)
    EXPECT_EQ(uniqueResourceStreams.size(), 2);
    
    // Ensure execution streams and resource streams are distinct
    for (auto execHandle : uniqueExecutionStreams) {
        EXPECT_EQ(uniqueResourceStreams.count(execHandle), 0);
    }
}

// Test basic CUDA operations with stream pool
TEST_F(CuWeaverStreamPoolTest, BasicCudaOperations) {
    cuweaver::StreamPool pool(4, 2);
    
    // Launch kernels on different streams
    cuweaver::cudaStream& stream1 = pool.getExecutionStream();
    cuweaver::cudaStream& stream2 = pool.getExecutionStream();
    
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    
    // Launch kernel on first stream
    dummyKernel<<<gridSize, blockSize, 0, stream1.nativeHandle()>>>(d_data1, dataSize, 100);
    
    // Launch kernel on second stream
    dummyKernel<<<gridSize, blockSize, 0, stream2.nativeHandle()>>>(d_data2, dataSize, 100);
    
    // Synchronize all execution streams
    EXPECT_TRUE(pool.syncExecutionStreams());
    
    // Verify results
    std::vector<float> result1(dataSize), result2(dataSize);
    cudaMemcpy(result1.data(), d_data1, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result2.data(), d_data2, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify computation results are different from initial values (kernel executed)
    EXPECT_NE(result1[0], 1.0f);
    EXPECT_NE(result2[0], 1.0f);
}

// Test realistic memory transfer scenarios
TEST_F(CuWeaverStreamPoolTest, MemoryTransferScenario) {
    cuweaver::StreamPool pool(2, 2);
    
    // Get resource stream for memory transfers and execution stream for computation
    cuweaver::cudaStream& resourceStream = pool.getResourceStream();
    cuweaver::cudaStream& execStream = pool.getExecutionStream();
    
    // Perform device-to-device memory copy on resource stream
    cudaMemcpyAsync(d_data2, d_data1, dataSize * sizeof(float), 
                    cudaMemcpyDeviceToDevice, resourceStream.nativeHandle());
    
    // Launch computation on execution stream
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    memcpyKernel<<<gridSize, blockSize, 0, execStream.nativeHandle()>>>(d_data2, d_data3, dataSize);
    
    // Synchronize all streams
    EXPECT_TRUE(pool.syncAllStreams());
    
    // Verify results
    std::vector<float> result(dataSize);
    cudaMemcpy(result.data(), d_data3, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Result should be original value multiplied by 2
    EXPECT_FLOAT_EQ(result[0], 2.0f);
    EXPECT_FLOAT_EQ(result[dataSize-1], 2.0f);
}

// Test thread safety with concurrent access from multiple threads
TEST_F(CuWeaverStreamPoolTest, MultithreadedAccess) {
    cuweaver::StreamPool pool(8, 4);
    const int numThreads = 16;
    const int opsPerThread = 10;
    
    std::atomic<int> successCount{0};
    std::atomic<int> failureCount{0};
    
    std::vector<std::thread> threads;
    
    // Launch multiple threads performing concurrent operations
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                for (int op = 0; op < opsPerThread; ++op) {
                    // Randomly select operation type based on thread and operation index
                    if ((t + op) % 3 == 0) {
                        // Get execution stream and launch kernel
                        cuweaver::cudaStream& stream = pool.getExecutionStream();
                        dim3 blockSize(128);
                        dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
                        dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(
                            d_data1, dataSize, 10);
                    } else if ((t + op) % 3 == 1) {
                        // Get resource stream and perform memory copy
                        cuweaver::cudaStream& stream = pool.getResourceStream();
                        cudaMemcpyAsync(d_data2, d_data1, dataSize * sizeof(float), 
                                      cudaMemcpyDeviceToDevice, stream.nativeHandle());
                    } else {
                        // Perform synchronization operations
                        if (op % 2 == 0) {
                            pool.syncExecutionStreams();
                        } else {
                            pool.syncResourceStreams();
                        }
                    }
                    successCount++;
                }
            } catch (...) {
                failureCount++;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Final synchronization
    EXPECT_TRUE(pool.syncAllStreams());
    
    // Verify most operations succeeded
    EXPECT_GT(successCount.load(), numThreads * opsPerThread * 0.8);
    EXPECT_LT(failureCount.load(), numThreads * opsPerThread * 0.2);
}

// Test stream synchronization functionality
TEST_F(CuWeaverStreamPoolTest, StreamSynchronization) {
    cuweaver::StreamPool pool(3, 2);
    
    // Launch long-running kernels on multiple streams
    std::vector<cuweaver::cudaStream*> streams;
    
    for (int i = 0; i < 5; ++i) {
        cuweaver::cudaStream& stream = pool.getExecutionStream();
        streams.push_back(&stream);
        
        dim3 blockSize(256);
        dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
        dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(
            d_data1, dataSize, 1000);  // Long-running computation
    }
    
    // Test execution stream synchronization timing
    auto start = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(pool.syncExecutionStreams());
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    EXPECT_GT(duration.count(), 0);  // Should take some time
    
    // Launch operation on resource stream
    cuweaver::cudaStream& resourceStream = pool.getResourceStream();
    cudaMemcpyAsync(d_data2, d_data1, dataSize * sizeof(float), 
                    cudaMemcpyDeviceToDevice, resourceStream.nativeHandle());
    
    // Test resource stream synchronization
    EXPECT_TRUE(pool.syncResourceStreams());
    
    // Test synchronization of all streams
    EXPECT_TRUE(pool.syncAllStreams());
}

// Test error handling scenarios
TEST_F(CuWeaverStreamPoolTest, ErrorHandling) {
    cuweaver::StreamPool pool(2, 2);
    
    // Normal synchronization should succeed
    EXPECT_TRUE(pool.syncExecutionStreams());
    EXPECT_TRUE(pool.syncResourceStreams());
    EXPECT_TRUE(pool.syncAllStreams());
    
    // Test synchronization with active operations
    cuweaver::cudaStream& stream = pool.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(d_data1, dataSize, 100);
    
    // Synchronization should still succeed even with active operations
    EXPECT_TRUE(pool.syncExecutionStreams());
}

// Test performance under high load scenarios
TEST_F(CuWeaverStreamPoolTest, PerformanceScenario) {
    cuweaver::StreamPool pool(8, 4);
    const int numIterations = 100;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < numIterations; ++i) {
        // Get streams and launch operations
        cuweaver::cudaStream& execStream = pool.getExecutionStream();
        cuweaver::cudaStream& resourceStream = pool.getResourceStream();
        
        dim3 blockSize(256);
        dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
        
        // Launch computation
        dummyKernel<<<gridSize, blockSize, 0, execStream.nativeHandle()>>>(
            d_data1, dataSize, 10);
        
        // Launch memory transfer on alternating iterations
        if (i % 2 == 0) {
            cudaMemcpyAsync(d_data2, d_data1, dataSize * sizeof(float), 
                          cudaMemcpyDeviceToDevice, resourceStream.nativeHandle());
        }
        
        // Synchronize every 10 iterations
        if (i % 10 == 9) {
            EXPECT_TRUE(pool.syncAllStreams());
        }
    }
    
    // Final synchronization
    EXPECT_TRUE(pool.syncAllStreams());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Performance test: should complete within reasonable time
    EXPECT_LT(duration.count(), 5000);  // Should complete within 5 seconds
    
    std::cout << "Performance test completed in " << duration.count() << "ms" << std::endl;
}

// Test resource cleanup and destructor behavior
TEST_F(CuWeaverStreamPoolTest, ResourceCleanup) {
    {
        cuweaver::StreamPool pool(4, 2);
        
        // Use streams for various operations
        for (int i = 0; i < 10; ++i) {
            cuweaver::cudaStream& stream = pool.getExecutionStream();
            dim3 blockSize(128);
            dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
            dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(
                d_data1, dataSize, 10);
        }
        
        EXPECT_TRUE(pool.syncAllStreams());
    } // StreamPool destructor called here
    
    // Ensure CUDA context is still valid after destruction
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess);
}

// Test edge cases and boundary conditions
TEST_F(CuWeaverStreamPoolTest, EdgeCases) {
    // Test with minimal pool sizes
    cuweaver::StreamPool smallPool(1, 1);
    
    // Multiple calls to getExecutionStream should return the same stream
    cuweaver::cudaStream& stream1 = smallPool.getExecutionStream();
    cuweaver::cudaStream& stream2 = smallPool.getExecutionStream();
    EXPECT_EQ(stream1.nativeHandle(), stream2.nativeHandle());
    
    // Multiple calls to getResourceStream should return the same stream
    cuweaver::cudaStream& rstream1 = smallPool.getResourceStream();
    cuweaver::cudaStream& rstream2 = smallPool.getResourceStream();
    EXPECT_EQ(rstream1.nativeHandle(), rstream2.nativeHandle());
    
    // Test repeated synchronization calls
    EXPECT_TRUE(smallPool.syncAllStreams());
    EXPECT_TRUE(smallPool.syncAllStreams());
    EXPECT_TRUE(smallPool.syncExecutionStreams());
    EXPECT_TRUE(smallPool.syncResourceStreams());
}

// Test stream pool with large numbers of streams
TEST_F(CuWeaverStreamPoolTest, LargePoolSize) {
    // Test with large pool sizes
    cuweaver::StreamPool largePool(32, 16);
    
    std::set<cudaStream_t> uniqueExecStreams;
    std::set<cudaStream_t> uniqueResourceStreams;
    
    // Collect all unique execution streams
    for (int i = 0; i < 64; ++i) {  // Request more than pool size
        cuweaver::cudaStream& stream = largePool.getExecutionStream();
        uniqueExecStreams.insert(stream.nativeHandle());
    }
    
    // Collect all unique resource streams
    for (int i = 0; i < 32; ++i) {  // Request more than pool size
        cuweaver::cudaStream& stream = largePool.getResourceStream();
        uniqueResourceStreams.insert(stream.nativeHandle());
    }
    
    // Should have exactly the pool size number of unique streams
    EXPECT_EQ(uniqueExecStreams.size(), 32);
    EXPECT_EQ(uniqueResourceStreams.size(), 16);
    
    // Launch operations on multiple streams and synchronize
    for (int i = 0; i < 10; ++i) {
        cuweaver::cudaStream& stream = largePool.getExecutionStream();
        dim3 blockSize(256);
        dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
        dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(d_data1, dataSize, 50);
    }
    
    EXPECT_TRUE(largePool.syncAllStreams());
}