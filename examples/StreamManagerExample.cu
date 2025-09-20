#include <iostream>
#include <cuweaver/StreamManager.cuh>

// CUDA kernel to fill an array with a specific value
__global__ void fillKernel(int* data, const int value, const size_t size) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size) {
        data[idx] = value;
    }
}

// CUDA kernel to perform element-wise addition of two arrays
__global__ void addKernel(int* c, const int* a, const int* b, const size_t size) {
    if (const size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Import cuweaver utilities for memory access specification
    using cuweaver::makeRead;
    using cuweaver::makeWrite;
    using namespace cuweaver::deviceFlags;

    // Get singleton instance of StreamManager - the core component for automatic stream management
    auto manager = &cuweaver::StreamManager::getInstance();

    // Initialize StreamManager with:
    // - 50 events for synchronization
    // - 8 execution streams for kernel launches
    // - 4 resource streams for memory operations
    manager->initialize(50, 8, 4);

    // Device memory pointers
    int *a, *b, *c;
    constexpr size_t size = 1 << 20;  // 1M elements

    // Allocate GPU memory using StreamManager (device 0)
    // StreamManager tracks memory allocations and their associated devices
    manager->malloc(&a, size * sizeof(int), 0);
    manager->malloc(&b, size * sizeof(int), 0);
    manager->malloc(&c, size * sizeof(int), 0);

    // Configure CUDA launch parameters
    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);

    // CONCURRENT OPERATION 1: Fill array 'a' with value 1
    // - Uses automatic device selection (Auto flag)
    // - Declares write access to array 'a' on device 0
    // - This operation runs on its own CUDA stream
    manager->launchKernel(fillKernel, grid, block, 0, Auto, makeWrite(a, 0), 1, size);

    // CONCURRENT OPERATION 2: Fill array 'b' with value 2
    // - Also uses automatic device selection
    // - Runs concurrently with the first fillKernel on a different stream
    // - StreamManager automatically handles stream assignment and synchronization
    manager->launchKernel(fillKernel, grid, block, 0, Auto, makeWrite(b, 0), 2, size);

    // CONCURRENT OPERATION 3: Add arrays 'a' and 'b', store result in 'c'
    // - Declares read access to 'a' and 'b', write access to 'c'
    // - StreamManager automatically inserts synchronization to ensure 'a' and 'b' are ready
    // - This kernel will wait for the previous two fillKernel operations to complete
    manager->launchKernel(addKernel, grid, block , 0, Auto, makeWrite(c, 0), makeRead(a, 0), makeRead(b, 0), size);

    // Copy result from device to host
    // StreamManager handles the dependency - this memcpy waits for addKernel to complete
    auto h_data = new int[size];
    manager->memcpy(h_data, Host, c, 0, size * sizeof(int), cuweaver::memcpyFlags::DeviceToHost);

    // Wait for all operations to complete
    cudaDeviceSynchronize();

    // Print first 10 results (should all be 3 = 1 + 2)
    for (size_t i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }

    // Cleanup
    delete[] h_data;
    manager->free(a, 0);
    manager->free(b, 0);
    manager->free(c, 0);
}

/*
 * COMPARISON: Manual CUDA Stream Management vs StreamManager
 *
 * To achieve the same automatic synchronization with raw CUDA APIs, you would need:
 *
 * cudaStream_t stream1, stream2, stream3;
 * cudaEvent_t event1, event2;
 *
 * // Create streams and events
 * cudaStreamCreate(&stream1);
 * cudaStreamCreate(&stream2);
 * cudaStreamCreate(&stream3);
 * cudaEventCreate(&event1);
 * cudaEventCreate(&event2);
 *
 * // Launch first kernel on stream1
 * fillKernel<<<grid, block, 0, stream1>>>(a, 1, size);
 * cudaEventRecord(event1, stream1);
 *
 * // Launch second kernel on stream2
 * fillKernel<<<grid, block, 0, stream2>>>(b, 2, size);
 * cudaEventRecord(event2, stream2);
 *
 * // Wait for both kernels before launching addKernel
 * cudaStreamWaitEvent(stream3, event1, 0);
 * cudaStreamWaitEvent(stream3, event2, 0);
 * addKernel<<<grid, block, 0, stream3>>>(c, a, b, size);
 *
 * // Wait for addKernel before memcpy
 * cudaStreamSynchronize(stream3);
 * cudaMemcpy(h_data, c, size * sizeof(int), cudaMemcpyDeviceToHost);
 *
 * // Cleanup streams and events
 * cudaStreamDestroy(stream1);
 * cudaStreamDestroy(stream2);
 * cudaStreamDestroy(stream3);
 * cudaEventDestroy(event1);
 * cudaEventDestroy(event2);
 *
 * StreamManager's advantages:
 * 1. Automatic dependency tracking through memory access declarations
 * 2. No manual stream/event creation and management
 * 3. Automatic stream assignment and reuse
 * 4. Simplified API - just declare read/write access patterns
 * 5. Reduced boilerplate code and potential for synchronization bugs
 */