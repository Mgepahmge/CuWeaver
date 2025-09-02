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

// Test move constructor functionality
TEST_F(CuWeaverStreamPoolTest, MoveConstructor) {
    // Create original pool
    cuweaver::StreamPool originalPool(4, 2);

    // Collect all unique streams from original pool
    std::set<cudaStream_t> originalExecStreams;
    std::set<cudaStream_t> originalResourceStreams;

    // Get all execution streams (should cycle through 4 unique streams)
    for (int i = 0; i < 8; ++i) {
        originalExecStreams.insert(originalPool.getExecutionStream().nativeHandle());
    }
    // Get all resource streams (should cycle through 2 unique streams)
    for (int i = 0; i < 4; ++i) {
        originalResourceStreams.insert(originalPool.getResourceStream().nativeHandle());
    }

    EXPECT_EQ(originalExecStreams.size(), 4);
    EXPECT_EQ(originalResourceStreams.size(), 2);

    // Move construct new pool
    cuweaver::StreamPool movedPool = std::move(originalPool);

    // Collect all streams from moved pool
    std::set<cudaStream_t> movedExecStreams;
    std::set<cudaStream_t> movedResourceStreams;

    for (int i = 0; i < 8; ++i) {
        movedExecStreams.insert(movedPool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 4; ++i) {
        movedResourceStreams.insert(movedPool.getResourceStream().nativeHandle());
    }

    // Moved pool should have the same streams as original
    EXPECT_EQ(movedExecStreams.size(), 4);
    EXPECT_EQ(movedResourceStreams.size(), 2);
    EXPECT_EQ(movedExecStreams, originalExecStreams);
    EXPECT_EQ(movedResourceStreams, originalResourceStreams);

    // Moved pool should be fully functional
    cuweaver::cudaStream& stream = movedPool.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(d_data1, dataSize, 10);

    EXPECT_TRUE(movedPool.syncAllStreams());

    // Verify computation worked
    std::vector<float> result(dataSize);
    cudaMemcpy(result.data(), d_data1, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NE(result[0], 1.0f);
}

// Test move assignment operator functionality
TEST_F(CuWeaverStreamPoolTest, MoveAssignment) {
    // Create source pool with specific configuration
    cuweaver::StreamPool sourcePool(6, 3);

    // Create target pool with different configuration
    cuweaver::StreamPool targetPool(2, 1);

    // Collect all streams from source pool before move
    std::set<cudaStream_t> sourceExecStreams;
    std::set<cudaStream_t> sourceResourceStreams;

    for (int i = 0; i < 12; ++i) {  // Get enough to cycle through all streams
        sourceExecStreams.insert(sourcePool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 6; ++i) {
        sourceResourceStreams.insert(sourcePool.getResourceStream().nativeHandle());
    }

    EXPECT_EQ(sourceExecStreams.size(), 6);
    EXPECT_EQ(sourceResourceStreams.size(), 3);

    // Get target pool's original streams for verification they're different
    std::set<cudaStream_t> originalTargetExecStreams;
    std::set<cudaStream_t> originalTargetResourceStreams;

    for (int i = 0; i < 4; ++i) {
        originalTargetExecStreams.insert(targetPool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 2; ++i) {
        originalTargetResourceStreams.insert(targetPool.getResourceStream().nativeHandle());
    }

    EXPECT_EQ(originalTargetExecStreams.size(), 2);
    EXPECT_EQ(originalTargetResourceStreams.size(), 1);

    // Perform move assignment
    targetPool = std::move(sourcePool);

    // Collect streams from target pool after move assignment
    std::set<cudaStream_t> targetExecStreams;
    std::set<cudaStream_t> targetResourceStreams;

    for (int i = 0; i < 12; ++i) {
        targetExecStreams.insert(targetPool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 6; ++i) {
        targetResourceStreams.insert(targetPool.getResourceStream().nativeHandle());
    }

    // Target should now have same streams as original source
    EXPECT_EQ(targetExecStreams.size(), 6);
    EXPECT_EQ(targetResourceStreams.size(), 3);
    EXPECT_EQ(targetExecStreams, sourceExecStreams);
    EXPECT_EQ(targetResourceStreams, sourceResourceStreams);

    // Target pool should be fully functional
    cuweaver::cudaStream& stream = targetPool.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(d_data1, dataSize, 20);

    EXPECT_TRUE(targetPool.syncAllStreams());

    // Verify results
    std::vector<float> result(dataSize);
    cudaMemcpy(result.data(), d_data1, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NE(result[0], 1.0f);
}

// Test self-assignment protection
TEST_F(CuWeaverStreamPoolTest, MoveAssignmentSelfAssignment) {
    cuweaver::StreamPool pool(4, 2);

    // Collect original streams
    std::set<cudaStream_t> originalExecStreams;
    std::set<cudaStream_t> originalResourceStreams;

    for (int i = 0; i < 8; ++i) {
        originalExecStreams.insert(pool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 4; ++i) {
        originalResourceStreams.insert(pool.getResourceStream().nativeHandle());
    }

    // Launch operation to establish working state
    cuweaver::cudaStream& stream = pool.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(d_data1, dataSize, 10);

    // Self-assignment (should be handled gracefully)
    pool = std::move(pool);

    // Pool should still be functional after self-assignment
    std::set<cudaStream_t> postAssignmentExecStreams;
    std::set<cudaStream_t> postAssignmentResourceStreams;

    for (int i = 0; i < 8; ++i) {
        postAssignmentExecStreams.insert(pool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 4; ++i) {
        postAssignmentResourceStreams.insert(pool.getResourceStream().nativeHandle());
    }

    // Configuration and streams should remain unchanged
    EXPECT_EQ(postAssignmentExecStreams.size(), 4);
    EXPECT_EQ(postAssignmentResourceStreams.size(), 2);
    EXPECT_EQ(postAssignmentExecStreams, originalExecStreams);
    EXPECT_EQ(postAssignmentResourceStreams, originalResourceStreams);

    // Pool should still be functional
    cuweaver::cudaStream& newStream = pool.getExecutionStream();
    dummyKernel<<<gridSize, blockSize, 0, newStream.nativeHandle()>>>(d_data2, dataSize, 10);
    EXPECT_TRUE(pool.syncAllStreams());
}

// Test move semantics with active operations
TEST_F(CuWeaverStreamPoolTest, MoveWithActiveOperations) {
    cuweaver::StreamPool sourcePool(3, 2);

    // Collect original stream set before launching operations
    std::set<cudaStream_t> originalExecStreams;
    std::set<cudaStream_t> originalResourceStreams;

    for (int i = 0; i < 6; ++i) {
        originalExecStreams.insert(sourcePool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 4; ++i) {
        originalResourceStreams.insert(sourcePool.getResourceStream().nativeHandle());
    }

    // Launch long-running operations
    for (int i = 0; i < 5; ++i) {
        cuweaver::cudaStream& stream = sourcePool.getExecutionStream();

        dim3 blockSize(256);
        dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
        dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(
            d_data1, dataSize, 500);  // Long-running operation
    }

    // Move construct while operations are potentially still running
    cuweaver::StreamPool movedPool = std::move(sourcePool);

    // Verify moved pool has same streams
    std::set<cudaStream_t> movedExecStreams;
    std::set<cudaStream_t> movedResourceStreams;

    for (int i = 0; i < 6; ++i) {
        movedExecStreams.insert(movedPool.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 4; ++i) {
        movedResourceStreams.insert(movedPool.getResourceStream().nativeHandle());
    }

    EXPECT_EQ(movedExecStreams, originalExecStreams);
    EXPECT_EQ(movedResourceStreams, originalResourceStreams);

    // Synchronize moved pool (should wait for active operations)
    EXPECT_TRUE(movedPool.syncAllStreams());

    // Launch new operations on moved pool
    cuweaver::cudaStream& newStream = movedPool.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, newStream.nativeHandle()>>>(
        d_data2, dataSize, 10);

    EXPECT_TRUE(movedPool.syncAllStreams());

    // Verify results
    std::vector<float> result1(dataSize), result2(dataSize);
    cudaMemcpy(result1.data(), d_data1, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result2.data(), d_data2, dataSize * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_NE(result1[0], 1.0f);
    EXPECT_NE(result2[0], 1.0f);
}

// Test chained move operations
TEST_F(CuWeaverStreamPoolTest, ChainedMoveOperations) {
    cuweaver::StreamPool pool1(5, 3);

    // Collect original stream configuration
    std::set<cudaStream_t> originalExecStreams;
    std::set<cudaStream_t> originalResourceStreams;

    for (int i = 0; i < 10; ++i) {
        originalExecStreams.insert(pool1.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 6; ++i) {
        originalResourceStreams.insert(pool1.getResourceStream().nativeHandle());
    }

    EXPECT_EQ(originalExecStreams.size(), 5);
    EXPECT_EQ(originalResourceStreams.size(), 3);

    // Launch operation to establish state
    cuweaver::cudaStream& stream1 = pool1.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, stream1.nativeHandle()>>>(d_data1, dataSize, 10);

    // Chain of moves: pool1 -> pool2 -> pool3
    cuweaver::StreamPool pool2 = std::move(pool1);
    cuweaver::StreamPool pool3 = std::move(pool2);

    // Final pool should have original stream configuration
    std::set<cudaStream_t> finalExecStreams;
    std::set<cudaStream_t> finalResourceStreams;

    for (int i = 0; i < 10; ++i) {
        finalExecStreams.insert(pool3.getExecutionStream().nativeHandle());
    }
    for (int i = 0; i < 6; ++i) {
        finalResourceStreams.insert(pool3.getResourceStream().nativeHandle());
    }

    EXPECT_EQ(finalExecStreams.size(), 5);
    EXPECT_EQ(finalResourceStreams.size(), 3);
    EXPECT_EQ(finalExecStreams, originalExecStreams);
    EXPECT_EQ(finalResourceStreams, originalResourceStreams);

    // Should be fully functional
    cuweaver::cudaStream& finalStream = pool3.getExecutionStream();
    dummyKernel<<<gridSize, blockSize, 0, finalStream.nativeHandle()>>>(
        d_data2, dataSize, 10);
    EXPECT_TRUE(pool3.syncAllStreams());
}

// Test moved-from object state
TEST_F(CuWeaverStreamPoolTest, MovedFromObjectState) {
    cuweaver::StreamPool originalPool(3, 2);

    // Use original pool
    cuweaver::cudaStream& stream = originalPool.getExecutionStream();
    dim3 blockSize(256);
    dim3 gridSize((dataSize + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<gridSize, blockSize, 0, stream.nativeHandle()>>>(d_data1, dataSize, 10);
    EXPECT_TRUE(originalPool.syncAllStreams());

    // Move construct
    cuweaver::StreamPool movedPool = std::move(originalPool);

    // Moved-from object should be in a valid but unspecified state
    // We can't guarantee what operations will succeed, but they shouldn't crash
    // This tests the basic requirement that moved-from objects remain destructible

    // The destructor should handle the moved-from state gracefully
    // (This will be tested when originalPool goes out of scope)

    // Moved-to object should work perfectly
    cuweaver::cudaStream& movedStream = movedPool.getExecutionStream();
    dummyKernel<<<gridSize, blockSize, 0, movedStream.nativeHandle()>>>(d_data2, dataSize, 10);
    EXPECT_TRUE(movedPool.syncAllStreams());

    std::vector<float> result(dataSize);
    cudaMemcpy(result.data(), d_data2, dataSize * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_NE(result[0], 1.0f);
}