#include <cuweaver/GlobalAllocator.cuh>      // cuweaver global memory allocator header
#include <cuweaver_utils/AllocatorTraits.cuh>  // cuweaver allocator traits utilities header
#include <iostream>

/**
 * CUDA kernel function: assigns each array element its index value
 * @param data pointer to integer array in GPU memory
 * @param size size of the array (number of elements)
 */
__global__ void kernel(int* data, const size_t size) {
    // Calculate the global index of the current thread
    // blockIdx.x: index of the current block
    // blockDim.x: number of threads per block
    // threadIdx.x: index of the thread within the block
    if (const int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size) {
        // Assign array element value: data[i] = i
        data[idx] = idx;
    }
}

int main() {
    // Define type aliases to simplify code
    using allocator_t = cuweaver::GlobalAllocator<int>;           // cuweaver global allocator type
    using traits_t = cuweaver::cudaAllocatorTraits<allocator_t>;  // allocator traits type

    // Define array size: 2^20 = 1,048,576 integer elements
    constexpr size_t size = 1 << 20; // 1M elements

    // Create cuweaver allocator instance
    allocator_t allocator;

    // Allocate memory on GPU using cuweaver allocator
    int* d_data = traits_t::allocate(allocator, size);

    // Check if memory allocation succeeded
    if (!d_data) {
        std::cerr << "Allocation failed!" << std::endl;
        return -1;
    }

    // Configure CUDA kernel launch parameters
    const int threadsPerBlock = 256;                              // number of threads per block
    const int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;  // calculate required number of blocks (ceiling division)

    // Launch CUDA kernel using <<<blocks, threadsPerBlock>>> syntax
    // Each thread will assign a value to the corresponding array element
    kernel<<<blocks, threadsPerBlock>>>(d_data, size);

    // Wait for all GPU operations to complete
    cudaDeviceSynchronize();

    // Allocate host memory to store data copied back from GPU
    int* h_data = new int[size];

    // Copy data from GPU memory to host memory
    cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Output first 10 elements to verify results
    for (size_t i = 0; i < 10; ++i) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;  // newline

    // Free host memory
    delete[] h_data;

    // Free GPU memory using cuweaver allocator
    traits_t::deallocate(allocator, d_data, size);

    return 0;
}