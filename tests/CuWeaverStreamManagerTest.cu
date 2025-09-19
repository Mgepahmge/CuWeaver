#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuweaver/StreamManager.cuh>
#include <memory>
#include <vector>

__global__ void addKernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void multiplyKernel(const float* a, float* b, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        b[idx] = a[idx] * factor;
    }
}

__global__ void initKernel(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

class StreamManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        ASSERT_GT(deviceCount, 0) << "No CUDA devices available";

        cudaSetDevice(0);

        manager = &cuweaver::StreamManager::getInstance();

        bool success = manager->initialize(10, 4, 2); // 10个事件，4个执行流，2个内存流
        ASSERT_TRUE(success) << "Failed to initialize StreamManager";

        const size_t dataSize = sizeof(float) * arraySize;
        cudaMalloc(&d_a, dataSize);
        cudaMalloc(&d_b, dataSize);
        cudaMalloc(&d_c, dataSize);
        cudaMalloc(&d_result, dataSize);

        ASSERT_NE(d_a, nullptr);
        ASSERT_NE(d_b, nullptr);
        ASSERT_NE(d_c, nullptr);
        ASSERT_NE(d_result, nullptr);

        h_a.resize(arraySize, 1.0f);
        h_b.resize(arraySize, 2.0f);
        h_expected.resize(arraySize, 3.0f); // a + b = 1 + 2 = 3
        h_result.resize(arraySize, 0.0f);

        cudaMemcpy(d_a, h_a.data(), dataSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), dataSize, cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
        if (d_result) cudaFree(d_result);

        manager->clear();
        cudaDeviceReset();
    }

    cuweaver::StreamManager* manager;
    static constexpr int arraySize = 1024;

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* d_result = nullptr;

    std::vector<float> h_a;
    std::vector<float> h_b;
    std::vector<float> h_expected;
    std::vector<float> h_result;
};

TEST_F(StreamManagerTest, SingletonTest) {
    auto& instance1 = cuweaver::StreamManager::getInstance();
    auto& instance2 = cuweaver::StreamManager::getInstance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(StreamManagerTest, InitializationTest) {
    auto& newManager = cuweaver::StreamManager::getInstance();

    EXPECT_TRUE(newManager.clear());

    EXPECT_TRUE(newManager.initialize(5, 2, 1));
    EXPECT_TRUE(newManager.clear());

    EXPECT_TRUE(newManager.initialize(20, 8, 4));
    EXPECT_TRUE(newManager.clear());

    EXPECT_TRUE(newManager.initialize(10, 4, 2));
}

TEST_F(StreamManagerTest, GetStreamTest) {
    auto& execStream1 = manager->getStream(0, false);
    auto& execStream2 = manager->getStream(0, false);

    EXPECT_NE(&execStream1, &execStream2);

    auto& resourceStream1 = manager->getStream(0, true);
    auto& resourceStream2 = manager->getStream(0, true);

    EXPECT_NE(&resourceStream1, &resourceStream2);
}

TEST_F(StreamManagerTest, GetEventTest) {
    auto& event1 = manager->getEvent(0);
    auto& event2 = manager->getEvent(0);

    EXPECT_NE(&event1, &event2);
}

TEST_F(StreamManagerTest, BasicKernelLaunchTest) {
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    auto readA = cuweaver::makeRead(d_a, 0);
    auto readB = cuweaver::makeRead(d_b, 0);
    auto writeC = cuweaver::makeWrite(d_c, 0);

    manager->launchKernel(addKernel, grid, block, 0, 0, readA, readB, writeC, arraySize);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_c, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], h_expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(StreamManagerTest, MemoryDependencyTest) {
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    auto writeA = cuweaver::makeWrite(d_a, 0);
    manager->launchKernel(initKernel, grid, block, 0, 0, writeA, arraySize, 5.0f);

    auto readA = cuweaver::makeRead(d_a, 0);
    auto writeB = cuweaver::makeWrite(d_b, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA, writeB, arraySize, 2.0f);

    auto readA2 = cuweaver::makeRead(d_a, 0);
    auto readB2 = cuweaver::makeRead(d_b, 0);
    auto writeC = cuweaver::makeWrite(d_c, 0);
    manager->launchKernel(addKernel, grid, block, 0, 0, readA2, readB2, writeC, arraySize);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_c, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 15.0f) << "Mismatch at index " << i;
    }
}

TEST_F(StreamManagerTest, ConcurrentKernelsTest) {
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    float *d_temp1, *d_temp2;
    cudaMalloc(&d_temp1, sizeof(float) * arraySize);
    cudaMalloc(&d_temp2, sizeof(float) * arraySize);

    auto readA = cuweaver::makeRead(d_a, 0);
    auto writeTemp1 = cuweaver::makeWrite(d_temp1, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA, writeTemp1, arraySize, 3.0f);

    auto readB = cuweaver::makeRead(d_b, 0);
    auto writeTemp2 = cuweaver::makeWrite(d_temp2, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readB, writeTemp2, arraySize, 4.0f);

    auto readTemp1 = cuweaver::makeRead(d_temp1, 0);
    auto readTemp2 = cuweaver::makeRead(d_temp2, 0);
    auto writeC = cuweaver::makeWrite(d_c, 0);
    manager->launchKernel(addKernel, grid, block, 0, 0, readTemp1, readTemp2, writeC, arraySize);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_c, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 11.0f) << "Mismatch at index " << i;
    }

    cudaFree(d_temp1);
    cudaFree(d_temp2);
}

TEST_F(StreamManagerTest, ReadWriteConflictTest) {
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    auto writeA = cuweaver::makeWrite(d_a, 0);
    manager->launchKernel(initKernel, grid, block, 0, 0, writeA, arraySize, 10.0f);

    auto readA1 = cuweaver::makeRead(d_a, 0);
    auto writeA1 = cuweaver::makeWrite(d_a, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA1, writeA1, arraySize, 0.5f);

    auto readA2 = cuweaver::makeRead(d_a, 0);
    auto writeA2 = cuweaver::makeWrite(d_a, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA2, writeA2, arraySize, 2.0f);

    auto readA3 = cuweaver::makeRead(d_a, 0);
    auto writeC = cuweaver::makeWrite(d_c, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA3, writeC, arraySize, 1.0f);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_c, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(h_result[i], 10.0f) << "Mismatch at index " << i;
    }
}

TEST_F(StreamManagerTest, StressTest) {
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    const int numIterations = 50;

    auto writeA = cuweaver::makeWrite(d_a, 0);
    manager->launchKernel(initKernel, grid, block, 0, 0, writeA, arraySize, 1.0f);

    for (int i = 0; i < numIterations; ++i) {
        auto readA = cuweaver::makeRead(d_a, 0);
        auto writeA_new = cuweaver::makeWrite(d_a, 0);
        manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA, writeA_new, arraySize, 1.01f);
    }

    auto readA_final = cuweaver::makeRead(d_a, 0);
    auto writeC = cuweaver::makeWrite(d_c, 0);
    manager->launchKernel(multiplyKernel, grid, block, 0, 0, readA_final, writeC, arraySize, 1.0f);

    cudaDeviceSynchronize();

    cudaMemcpy(h_result.data(), d_c, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    float expected = std::pow(1.01f, numIterations);
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_NEAR(h_result[i], expected, 0.01f) << "Mismatch at index " << i;
    }
}

TEST_F(StreamManagerTest, ErrorHandlingTest) {

    dim3 grid(1);
    dim3 block(1);

    auto readA = cuweaver::makeRead(d_a, 999); // 无效设备ID
    auto writeC = cuweaver::makeWrite(d_c, 999);

    EXPECT_NO_THROW(
        manager->launchKernel(addKernel, grid, block, 0, 999, readA, readA, writeC, arraySize)
    );
}

// Test basic functionality of malloc method
TEST_F(StreamManagerTest, MallocBasicTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    // Create a pointer to store the allocated memory address
    void* allocated_ptr = nullptr;

    // Use StreamManager's malloc method to allocate memory
    EXPECT_NO_THROW(manager->malloc(&allocated_ptr, allocSize, 0));

    // Verify memory was actually allocated
    EXPECT_NE(allocated_ptr, nullptr);

    // Synchronize to wait for async malloc to complete
    cudaDeviceSynchronize();

    // Test if allocated memory is usable - try writing and reading
    std::vector<float> test_data(arraySize, 42.0f);
    EXPECT_EQ(cudaMemcpy(allocated_ptr, test_data.data(), allocSize, cudaMemcpyHostToDevice), cudaSuccess);

    std::vector<float> read_data(arraySize, 0.0f);
    EXPECT_EQ(cudaMemcpy(read_data.data(), allocated_ptr, allocSize, cudaMemcpyDeviceToHost), cudaSuccess);

    // Verify data correctness
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(read_data[i], 42.0f) << "Data mismatch at index " << i;
    }

    // Clean up allocated memory
    if (allocated_ptr) {
        cudaFree(allocated_ptr);
    }
}

// Test integration of malloc method with kernel launch
TEST_F(StreamManagerTest, MallocWithKernelLaunchTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_malloc_a = nullptr;
    void* d_malloc_b = nullptr;
    void* d_malloc_result = nullptr;

    // Use StreamManager to allocate memory
    manager->malloc(&d_malloc_a, allocSize, 0);
    manager->malloc(&d_malloc_b, allocSize, 0);
    manager->malloc(&d_malloc_result, allocSize, 0);

    ASSERT_NE(d_malloc_a, nullptr);
    ASSERT_NE(d_malloc_b, nullptr);
    ASSERT_NE(d_malloc_result, nullptr);

    // Cast void* to float* and create memory wrappers
    float* f_malloc_a = static_cast<float*>(d_malloc_a);
    float* f_malloc_b = static_cast<float*>(d_malloc_b);
    float* f_malloc_result = static_cast<float*>(d_malloc_result);

    // Initialize first array (dependency management should handle malloc completion)
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_malloc_a, 0), arraySize, 7.0f);

    // Initialize second array
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_malloc_b, 0), arraySize, 3.0f);


    // Perform addition operation
    manager->launchKernel(addKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_malloc_a, 0),
                         cuweaver::makeRead(f_malloc_b, 0),
                         cuweaver::makeWrite(f_malloc_result, 0),
                         arraySize);

    // Synchronize and verify results
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_malloc_result, allocSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 10.0f) << "Expected 7.0 + 3.0 = 10.0 at index " << i;
    }

    // Clean up
    cudaFree(d_malloc_a);
    cudaFree(d_malloc_b);
    cudaFree(d_malloc_result);
}

// Test concurrent malloc operations
TEST_F(StreamManagerTest, ConcurrentMallocTest) {
    const size_t allocSize = sizeof(float) * arraySize;
    const int numAllocations = 10;

    std::vector<void*> allocated_ptrs(numAllocations, nullptr);

    // Concurrently allocate multiple memory blocks
    for (int i = 0; i < numAllocations; ++i) {
        EXPECT_NO_THROW(manager->malloc(&allocated_ptrs[i], allocSize, 0));
        EXPECT_NE(allocated_ptrs[i], nullptr) << "Allocation " << i << " failed";
    }

    // Verify all allocated memory has different addresses
    for (int i = 0; i < numAllocations; ++i) {
        for (int j = i + 1; j < numAllocations; ++j) {
            EXPECT_NE(allocated_ptrs[i], allocated_ptrs[j])
                << "Memory collision between allocation " << i << " and " << j;
        }
    }

    // Test each allocated memory block can be used independently
    // The dependency management should ensure malloc completion before kernel execution
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    for (int i = 0; i < numAllocations; ++i) {
        float initValue = static_cast<float>(i + 1);
        float* f_ptr = static_cast<float*>(allocated_ptrs[i]);
        manager->launchKernel(initKernel, grid, block, 0, 0,
                             cuweaver::makeWrite(f_ptr, 0), arraySize, initValue);
    }

    // Synchronize and verify data in each memory block
    cudaDeviceSynchronize();

    for (int i = 0; i < numAllocations; ++i) {
        std::vector<float> data(arraySize);
        cudaMemcpy(data.data(), allocated_ptrs[i], allocSize, cudaMemcpyDeviceToHost);

        float expectedValue = static_cast<float>(i + 1);
        for (int j = 0; j < arraySize; ++j) {
            EXPECT_FLOAT_EQ(data[j], expectedValue)
                << "Data mismatch in allocation " << i << " at index " << j;
        }
    }

    // Clean up all allocated memory
    for (void* ptr : allocated_ptrs) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
}

// Test memory event recording functionality of malloc method
TEST_F(StreamManagerTest, MallocEventRecordingTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_malloc_mem = nullptr;

    // Allocate memory (this should record a write event)
    manager->malloc(&d_malloc_mem, allocSize, 0);
    ASSERT_NE(d_malloc_mem, nullptr);

    float* f_malloc_mem = static_cast<float*>(d_malloc_mem);

    // The dependency system should handle waiting for malloc completion
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // This kernel should wait for malloc operation to complete before executing
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_malloc_mem, 0), arraySize, 99.0f);

    // Read again to verify data
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_malloc_mem, 0),
                         cuweaver::makeWrite(d_result, 0),
                         arraySize, 1.0f);

    // Synchronize and verify results
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_result, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 99.0f)
            << "Expected initialization value 99.0 at index " << i;
    }

    cudaFree(d_malloc_mem);
}

// Test memory allocation of different sizes
TEST_F(StreamManagerTest, MallocDifferentSizesTest) {
    std::vector<size_t> sizes = {
        sizeof(float) * 10,        // Small allocation
        sizeof(float) * 1024,      // Medium allocation
        sizeof(float) * 100000,    // Large allocation
        sizeof(double) * 50000,    // Different type
        1                          // Minimum allocation
    };

    std::vector<void*> allocated_ptrs;
    allocated_ptrs.reserve(sizes.size());

    // Allocate memory of different sizes
    for (size_t size : sizes) {
        void* ptr = nullptr;
        EXPECT_NO_THROW(manager->malloc(&ptr, size, 0));
        EXPECT_NE(ptr, nullptr) << "Failed to allocate " << size << " bytes";
        allocated_ptrs.push_back(ptr);
    }

    // Wait for all async malloc operations to complete
    cudaDeviceSynchronize();

    // Test each allocated memory can be written to
    for (size_t i = 0; i < sizes.size(); ++i) {
        // Simple write test - write one byte
        char testByte = static_cast<char>(i + 1);
        EXPECT_EQ(cudaMemcpy(allocated_ptrs[i], &testByte, 1, cudaMemcpyHostToDevice),
                  cudaSuccess) << "Failed to write to allocation " << i;

        // Read back to verify
        char readByte = 0;
        EXPECT_EQ(cudaMemcpy(&readByte, allocated_ptrs[i], 1, cudaMemcpyDeviceToHost),
                  cudaSuccess) << "Failed to read from allocation " << i;
        EXPECT_EQ(readByte, testByte) << "Data mismatch in allocation " << i;
    }

    // Clean up
    for (void* ptr : allocated_ptrs) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
}

// Test malloc integration with complex dependency chains
TEST_F(StreamManagerTest, MallocComplexDependencyTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_step1 = nullptr;
    void* d_step2 = nullptr;
    void* d_step3 = nullptr;

    // Step 1: Allocate and initialize first array
    manager->malloc(&d_step1, allocSize, 0);
    ASSERT_NE(d_step1, nullptr);

    float* f_step1 = static_cast<float*>(d_step1);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);
    // Dependency system should wait for malloc completion
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_step1, 0), arraySize, 5.0f);

    // Step 2: Allocate second array and compute based on first array
    manager->malloc(&d_step2, allocSize, 0);
    ASSERT_NE(d_step2, nullptr);

    float* f_step2 = static_cast<float*>(d_step2);

    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_step1, 0),
                         cuweaver::makeWrite(f_step2, 0),
                         arraySize, 3.0f);

    // Step 3: Allocate third array and combine previous two results
    manager->malloc(&d_step3, allocSize, 0);
    ASSERT_NE(d_step3, nullptr);

    float* f_step3 = static_cast<float*>(d_step3);

    manager->launchKernel(addKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_step1, 0),
                         cuweaver::makeRead(f_step2, 0),
                         cuweaver::makeWrite(f_step3, 0),
                         arraySize);

    // Synchronize and verify final results
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_step3, allocSize, cudaMemcpyDeviceToHost);

    // Expected result: 5.0 + (5.0 * 3.0) = 5.0 + 15.0 = 20.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 20.0f)
            << "Expected 5.0 + 15.0 = 20.0 at index " << i;
    }

    // Clean up
    cudaFree(d_step1);
    cudaFree(d_step2);
    cudaFree(d_step3);
}

// Test malloc with read-write dependency chains
TEST_F(StreamManagerTest, MallocReadWriteDependencyTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_chain_mem = nullptr;

    // Allocate memory for dependency chain testing
    manager->malloc(&d_chain_mem, allocSize, 0);
    ASSERT_NE(d_chain_mem, nullptr);

    float* f_chain_mem = static_cast<float*>(d_chain_mem);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // Initialize memory (write after malloc) - dependency system handles malloc completion
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_chain_mem, 0), arraySize, 2.0f);

    // Read and modify in place multiple times to test dependency chain
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_chain_mem, 0),
                         cuweaver::makeWrite(f_chain_mem, 0),
                         arraySize, 2.0f);

    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_chain_mem, 0),
                         cuweaver::makeWrite(f_chain_mem, 0),
                         arraySize, 2.0f);

    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_chain_mem, 0),
                         cuweaver::makeWrite(f_chain_mem, 0),
                         arraySize, 2.0f);

    // Final read to copy result
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_chain_mem, 0),
                         cuweaver::makeWrite(d_result, 0),
                         arraySize, 1.0f);

    // Synchronize and verify results
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_result, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    // Expected result: 2.0 * 2.0 * 2.0 * 2.0 = 16.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 16.0f)
            << "Expected 2.0^4 = 16.0 at index " << i;
    }

    cudaFree(d_chain_mem);
}

// Test malloc memory allocation stress test
TEST_F(StreamManagerTest, MallocStressTest) {
    const size_t smallAllocSize = sizeof(float) * 100;
    const int numAllocations = 100;

    std::vector<void*> stress_ptrs(numAllocations, nullptr);

    // Rapidly allocate many small memory blocks
    for (int i = 0; i < numAllocations; ++i) {
        manager->malloc(&stress_ptrs[i], smallAllocSize, 0);
        EXPECT_NE(stress_ptrs[i], nullptr) << "Stress allocation " << i << " failed";
    }

    std::cout << "Allocated " << numAllocations << " blocks of "
              << smallAllocSize << " bytes each." << std::endl;

    // Use each memory block with kernel operations
    // Dependency system should handle malloc completion automatically
    dim3 grid((100 + 255) / 256);
    dim3 block(256);

    for (int i = 0; i < numAllocations; ++i) {
        float* f_ptr = static_cast<float*>(stress_ptrs[i]);
        float value = static_cast<float>(i % 10 + 1);
        manager->launchKernel(initKernel, grid, block, 0, 0,
                             cuweaver::makeWrite(f_ptr, 0), 100, value);
    }

    // Verify some of the allocations
    cudaDeviceSynchronize();

    for (int i = 0; i < 10; ++i) { // Only check first 10 to avoid too much verification
        std::vector<float> data(100);
        cudaMemcpy(data.data(), stress_ptrs[i], smallAllocSize, cudaMemcpyDeviceToHost);

        float expectedValue = static_cast<float>(i % 10 + 1);
        for (int j = 0; j < 100; ++j) {
            EXPECT_FLOAT_EQ(data[j], expectedValue)
                << "Stress test data mismatch in allocation " << i << " at index " << j;
        }
    }

    // Clean up all allocations
    for (void* ptr : stress_ptrs) {
        if (ptr) {
            cudaFree(ptr);
        }
    }
}

// Test synchronous operations after malloc
TEST_F(StreamManagerTest, MallocSynchronousOperationsTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_malloc_mem = nullptr;

    // Allocate memory asynchronously
    manager->malloc(&d_malloc_mem, allocSize, 0);
    ASSERT_NE(d_malloc_mem, nullptr);

    // Must synchronize before using synchronous CUDA operations
    cudaDeviceSynchronize();

    // Now safe to use synchronous operations
    std::vector<float> host_data(arraySize, 123.45f);
    EXPECT_EQ(cudaMemcpy(d_malloc_mem, host_data.data(), allocSize, cudaMemcpyHostToDevice),
              cudaSuccess);

    // Verify data was written correctly
    std::vector<float> read_data(arraySize, 0.0f);
    EXPECT_EQ(cudaMemcpy(read_data.data(), d_malloc_mem, allocSize, cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(read_data[i], 123.45f) << "Data verification failed at index " << i;
    }

    // Now use the memory with kernels (no extra sync needed due to dependency system)
    float* f_malloc_mem = static_cast<float*>(d_malloc_mem);
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_malloc_mem, 0),
                         cuweaver::makeWrite(d_result, 0),
                         arraySize, 2.0f);

    cudaDeviceSynchronize();

    std::vector<float> final_result(arraySize);
    cudaMemcpy(final_result.data(), d_result, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(final_result[i], 246.9f) << "Final result mismatch at index " << i;
    }

    cudaFree(d_malloc_mem);
}

// Test basic functionality of memcpy method
TEST_F(StreamManagerTest, MemcpyBasicTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_src = nullptr;
    void* d_dst = nullptr;

    // Allocate source and destination memory
    manager->malloc(&d_src, allocSize, 0);
    manager->malloc(&d_dst, allocSize, 0);

    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    float* f_src = static_cast<float*>(d_src);
    float* f_dst = static_cast<float*>(d_dst);

    // Initialize source data
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_src, 0), arraySize, 42.0f);

    // Perform memcpy operation (device to device)
    EXPECT_NO_THROW(manager->memcpy(f_dst, 0, f_src, 0, allocSize, cuweaver::memcpyFlags::DeviceToDevice));

    // Synchronize and verify results
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_dst, allocSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 42.0f) << "Data mismatch at index " << i;
    }

    // Clean up
    cudaFree(d_src);
    cudaFree(d_dst);
}

// Test memcpy with different memory copy directions
TEST_F(StreamManagerTest, MemcpyDifferentDirectionsTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_mem = nullptr;
    manager->malloc(&d_mem, allocSize, 0);
    ASSERT_NE(d_mem, nullptr);

    float* f_mem = static_cast<float*>(d_mem);

    // Test Host to Device
    std::vector<float> host_data(arraySize, 15.0f);
    EXPECT_NO_THROW(manager->memcpy(f_mem, 0, host_data.data(), cuweaver::deviceFlags::Host,
                                   allocSize, cuweaver::memcpyFlags::HostToDevice));

    cudaDeviceSynchronize();

    // Test Device to Host
    std::vector<float> host_result(arraySize, 0.0f);
    EXPECT_NO_THROW(manager->memcpy(host_result.data(), cuweaver::deviceFlags::Host, f_mem, 0,
                                   allocSize, cuweaver::memcpyFlags::DeviceToHost));

    // Synchronize and verify
    cudaDeviceSynchronize();

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(host_result[i], 15.0f) << "Host-Device-Host copy failed at index " << i;
    }

    cudaFree(d_mem);
}

// Test memcpy integration with kernel operations
TEST_F(StreamManagerTest, MemcpyWithKernelIntegrationTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_src = nullptr;
    void* d_dst = nullptr;

    manager->malloc(&d_src, allocSize, 0);
    manager->malloc(&d_dst, allocSize, 0);

    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    float* f_src = static_cast<float*>(d_src);
    float* f_dst = static_cast<float*>(d_dst);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // Initialize source with kernel
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_src, 0), arraySize, 8.0f);

    // Copy data (dependency system should handle kernel completion)
    manager->memcpy(f_dst, 0, f_src, 0, allocSize, cuweaver::memcpyFlags::DeviceToDevice);

    // Modify copied data with kernel
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_dst, 0),
                         cuweaver::makeWrite(d_result, 0),
                         arraySize, 2.0f);

    // Synchronize and verify
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_result, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    // Expected: 8.0 * 2.0 = 16.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 16.0f) << "Expected 8.0 * 2.0 = 16.0 at index " << i;
    }

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Test memcpy with invalid device combinations
TEST_F(StreamManagerTest, MemcpyInvalidDeviceTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_mem = nullptr;
    manager->malloc(&d_mem, allocSize, 0);
    ASSERT_NE(d_mem, nullptr);

    std::vector<float> host_data(arraySize, 1.0f);

    // Test with invalid source device (should be handled gracefully)
    EXPECT_NO_THROW(manager->memcpy(static_cast<float*>(d_mem), 0, host_data.data(), 999,
                                   allocSize, cuweaver::memcpyFlags::HostToDevice));

    // Test with invalid destination device (should be handled gracefully)
    EXPECT_NO_THROW(manager->memcpy(host_data.data(), 999, static_cast<float*>(d_mem), 0,
                                   allocSize, cuweaver::memcpyFlags::DeviceToHost));

    cudaFree(d_mem);
}

// Test memcpyPeer functionality (only in multi-GPU environment)
TEST_F(StreamManagerTest, MemcpyPeerBasicTest) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        GTEST_SKIP() << "Skipping memcpyPeer test: requires multiple GPUs";
        return;
    }

    // Check if peer access is available between device 0 and 1
    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    if (!canAccessPeer) {
        GTEST_SKIP() << "Skipping memcpyPeer test: peer access not available";
        return;
    }

    const size_t allocSize = sizeof(float) * arraySize;

    // Allocate memory on device 0
    cudaSetDevice(0);
    void* d_src_dev0 = nullptr;
    manager->malloc(&d_src_dev0, allocSize, 0);
    ASSERT_NE(d_src_dev0, nullptr);

    // Allocate memory on device 1
    cudaSetDevice(1);
    void* d_dst_dev1 = nullptr;
    manager->malloc(&d_dst_dev1, allocSize, 1);
    ASSERT_NE(d_dst_dev1, nullptr);

    // Initialize source data on device 0
    float* f_src = static_cast<float*>(d_src_dev0);
    float* f_dst = static_cast<float*>(d_dst_dev1);
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    cudaSetDevice(0);
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_src, 0), arraySize, 77.0f);

    // Perform peer-to-peer copy
    EXPECT_NO_THROW(manager->memcpyPeer(f_dst, 1, f_src, 0, allocSize));

    // Synchronize both devices
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    // Verify data on device 1
    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_dst_dev1, allocSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 77.0f) << "Peer copy failed at index " << i;
    }

    // Clean up
    cudaSetDevice(0);
    cudaFree(d_src_dev0);
    cudaSetDevice(1);
    cudaFree(d_dst_dev1);
}

// Test memcpyPeer with complex dependency chains across devices
TEST_F(StreamManagerTest, MemcpyPeerComplexDependencyTest) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount < 2) {
        GTEST_SKIP() << "Skipping multi-GPU memcpyPeer test: requires multiple GPUs";
        return;
    }

    int canAccessPeer = 0;
    cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    if (!canAccessPeer) {
        GTEST_SKIP() << "Skipping memcpyPeer test: peer access not available";
        return;
    }

    const size_t allocSize = sizeof(float) * arraySize;

    // Allocate memory on both devices
    cudaSetDevice(0);
    void* d_mem0 = nullptr;
    manager->malloc(&d_mem0, allocSize, 0);

    cudaSetDevice(1);
    void* d_mem1 = nullptr;
    void* d_result1 = nullptr;
    manager->malloc(&d_mem1, allocSize, 1);
    manager->malloc(&d_result1, allocSize, 1);

    ASSERT_NE(d_mem0, nullptr);
    ASSERT_NE(d_mem1, nullptr);
    ASSERT_NE(d_result1, nullptr);

    float* f_mem0 = static_cast<float*>(d_mem0);
    float* f_mem1 = static_cast<float*>(d_mem1);
    float* f_result1 = static_cast<float*>(d_result1);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // Step 1: Initialize data on device 0
    cudaSetDevice(0);
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_mem0, 0), arraySize, 12.0f);

    // Step 2: Copy from device 0 to device 1
    manager->memcpyPeer(f_mem1, 1, f_mem0, 0, allocSize);

    // Step 3: Process data on device 1
    cudaSetDevice(1);
    manager->launchKernel(multiplyKernel, grid, block, 0, 1,
                         cuweaver::makeRead(f_mem1, 1),
                         cuweaver::makeWrite(f_result1, 1),
                         arraySize, 3.0f);

    // Synchronize all devices
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    // Verify final result
    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_result1, allocSize, cudaMemcpyDeviceToHost);

    // Expected: 12.0 * 3.0 = 36.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 36.0f) << "Expected 12.0 * 3.0 = 36.0 at index " << i;
    }

    // Clean up
    cudaSetDevice(0);
    cudaFree(d_mem0);
    cudaSetDevice(1);
    cudaFree(d_mem1);
    cudaFree(d_result1);
}

// Test memcpyPeer with invalid devices
TEST_F(StreamManagerTest, MemcpyPeerInvalidDeviceTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_mem = nullptr;
    manager->malloc(&d_mem, allocSize, 0);
    ASSERT_NE(d_mem, nullptr);

    float* f_mem = static_cast<float*>(d_mem);

    // Test with invalid source device (should be handled gracefully)
    EXPECT_NO_THROW(manager->memcpyPeer(f_mem, 0, f_mem, 999, allocSize));

    // Test with invalid destination device (should be handled gracefully)
    EXPECT_NO_THROW(manager->memcpyPeer(f_mem, 999, f_mem, 0, allocSize));

    cudaFree(d_mem);
}

// Test free functionality
TEST_F(StreamManagerTest, FreeBasicTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_mem = nullptr;
    manager->malloc(&d_mem, allocSize, 0);
    ASSERT_NE(d_mem, nullptr);

    float* f_mem = static_cast<float*>(d_mem);

    // Use the memory first
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_mem, 0), arraySize, 25.0f);

    // Test free operation (should wait for kernel completion)
    EXPECT_NO_THROW(manager->free(&f_mem, 0));

    // Synchronize to ensure free completes
    cudaDeviceSynchronize();

    // Note: After free, we cannot verify memory content as it's deallocated
    // The test passes if no exceptions are thrown
}

// Test free with dependency management
TEST_F(StreamManagerTest, FreeDependencyTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_temp1 = nullptr;
    void* d_temp2 = nullptr;

    manager->malloc(&d_temp1, allocSize, 0);
    manager->malloc(&d_temp2, allocSize, 0);

    ASSERT_NE(d_temp1, nullptr);
    ASSERT_NE(d_temp2, nullptr);

    float* f_temp1 = static_cast<float*>(d_temp1);
    float* f_temp2 = static_cast<float*>(d_temp2);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // Create a computation chain
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_temp1, 0), arraySize, 6.0f);

    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_temp1, 0),
                         cuweaver::makeWrite(f_temp2, 0),
                         arraySize, 4.0f);

    // Copy result to final destination before freeing temps
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_temp2, 0),
                         cuweaver::makeWrite(d_result, 0),
                         arraySize, 1.0f);

    // Free temporary memories (should wait for all operations to complete)
    EXPECT_NO_THROW(manager->free(&f_temp1, 0));
    EXPECT_NO_THROW(manager->free(&f_temp2, 0));

    // Synchronize and verify final result is still valid
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_result, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    // Expected: 6.0 * 4.0 = 24.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 24.0f) << "Expected 6.0 * 4.0 = 24.0 at index " << i;
    }
}

// Test multiple free operations
TEST_F(StreamManagerTest, MultipleFreeTest) {
    const size_t allocSize = sizeof(float) * arraySize;
    const int numAllocations = 5;

    std::vector<void*> allocated_ptrs(numAllocations);

    // Allocate multiple memory blocks
    for (int i = 0; i < numAllocations; ++i) {
        manager->malloc(&allocated_ptrs[i], allocSize, 0);
        ASSERT_NE(allocated_ptrs[i], nullptr) << "Allocation " << i << " failed";
    }

    // Use each memory block
    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    for (int i = 0; i < numAllocations; ++i) {
        float initValue = static_cast<float>(i + 10);
        float* f_ptr = static_cast<float*>(allocated_ptrs[i]);
        manager->launchKernel(initKernel, grid, block, 0, 0,
                             cuweaver::makeWrite(f_ptr, 0), arraySize, initValue);
    }

    // Free all memory blocks (dependency system should handle completion)
    for (int i = 0; i < numAllocations; ++i) {
        float* f_ptr = static_cast<float*>(allocated_ptrs[i]);
        EXPECT_NO_THROW(manager->free(&f_ptr, 0))
            << "Free operation " << i << " failed";
    }

    // Synchronize to ensure all operations complete
    cudaDeviceSynchronize();

    // Test passes if no exceptions are thrown during free operations
}

// Test free with invalid device ID (edge case)
TEST_F(StreamManagerTest, FreeInvalidDeviceTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_mem = nullptr;
    manager->malloc(&d_mem, allocSize, 0);
    ASSERT_NE(d_mem, nullptr);

    float* f_mem = static_cast<float*>(d_mem);

    // Test free with invalid device ID (should be handled gracefully)
    EXPECT_NO_THROW(manager->free(&f_mem, 999));

    // Clean up properly
    cudaFree(d_mem);
}

// Test concurrent memory operations
TEST_F(StreamManagerTest, ConcurrentMemoryOperationsTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_mem1 = nullptr;
    void* d_mem2 = nullptr;
    void* d_mem3 = nullptr;

    // Allocate multiple memories
    manager->malloc(&d_mem1, allocSize, 0);
    manager->malloc(&d_mem2, allocSize, 0);
    manager->malloc(&d_mem3, allocSize, 0);

    ASSERT_NE(d_mem1, nullptr);
    ASSERT_NE(d_mem2, nullptr);
    ASSERT_NE(d_mem3, nullptr);

    float* f_mem1 = static_cast<float*>(d_mem1);
    float* f_mem2 = static_cast<float*>(d_mem2);
    float* f_mem3 = static_cast<float*>(d_mem3);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // Initialize first memory
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_mem1, 0), arraySize, 10.0f);

    // Copy from mem1 to mem2
    manager->memcpy(f_mem2, 0, f_mem1, 0, allocSize, cuweaver::memcpyFlags::DeviceToDevice);

    // Process mem2 and store in mem3
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_mem2, 0),
                         cuweaver::makeWrite(f_mem3, 0),
                         arraySize, 2.0f);

    // Copy final result
    manager->memcpy(d_result, 0, f_mem3, 0, allocSize, cuweaver::memcpyFlags::DeviceToDevice);

    // Free intermediate memories
    manager->free(&f_mem1, 0);
    manager->free(&f_mem2, 0);

    // Synchronize and verify
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_result, sizeof(float) * arraySize, cudaMemcpyDeviceToHost);

    // Expected: 10.0 * 2.0 = 20.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 20.0f) << "Expected 10.0 * 2.0 = 20.0 at index " << i;
    }

    // Final cleanup
    manager->free(&f_mem3, 0);
}

// Comprehensive integration test combining malloc, memcpy, and free
TEST_F(StreamManagerTest, MallocMemcpyFreeIntegrationTest) {
    const size_t allocSize = sizeof(float) * arraySize;

    void* d_source = nullptr;
    void* d_temp = nullptr;
    void* d_final = nullptr;

    // Step 1: Allocate memories
    manager->malloc(&d_source, allocSize, 0);
    manager->malloc(&d_temp, allocSize, 0);
    manager->malloc(&d_final, allocSize, 0);

    ASSERT_NE(d_source, nullptr);
    ASSERT_NE(d_temp, nullptr);
    ASSERT_NE(d_final, nullptr);

    float* f_source = static_cast<float*>(d_source);
    float* f_temp = static_cast<float*>(d_temp);
    float* f_final = static_cast<float*>(d_final);

    dim3 grid((arraySize + 255) / 256);
    dim3 block(256);

    // Step 2: Initialize source data
    manager->launchKernel(initKernel, grid, block, 0, 0,
                         cuweaver::makeWrite(f_source, 0), arraySize, 9.0f);

    // Step 3: Copy source to temp
    manager->memcpy(f_temp, 0, f_source, 0, allocSize, cuweaver::memcpyFlags::DeviceToDevice);

    // Step 4: Process temp data
    manager->launchKernel(multiplyKernel, grid, block, 0, 0,
                         cuweaver::makeRead(f_temp, 0),
                         cuweaver::makeWrite(f_final, 0),
                         arraySize, 5.0f);

    // Step 5: Free source and temp (final result should still be valid)
    manager->free(&f_source, 0);
    manager->free(&f_temp, 0);

    // Step 6: Copy final result to host and verify
    cudaDeviceSynchronize();

    std::vector<float> result(arraySize);
    cudaMemcpy(result.data(), d_final, allocSize, cudaMemcpyDeviceToHost);

    // Expected: 9.0 * 5.0 = 45.0
    for (int i = 0; i < arraySize; ++i) {
        EXPECT_FLOAT_EQ(result[i], 45.0f) << "Expected 9.0 * 5.0 = 45.0 at index " << i;
    }

    // Final cleanup
    manager->free(&f_final, 0);
}

