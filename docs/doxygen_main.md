/** @mainpage 
# CuWeaver: CUDA Concurrency Library

CuWeaver is a CUDA concurrency library designed to simplify parallel programming by automating concurrency flow management. It provides C++-style wrappers for selected CUDA Runtime APIs, helping to reduce the complexity of managing concurrency in multi-GPU environments.

## Key Features
- **Concurrency Automation**: Automatically manages memory streams, execution streams, and event pools for each GPU, ensuring isolation of memory and computation operations.
- **Multi-GPU Simplification**: Optimizes memory management and kernel invocation in multi-GPU environments, reducing the complexity of cross-GPU development.
- **Event-Driven Dependency Management**: Ensures correct data access order by maintaining dependencies between operations, preventing data races.
- **Modern C++ Wrappers**: Provides C++-style wrappers for CUDA's native C API, leveraging RAII and move semantics to simplify resource management.

## Dependencies
CuWeaver requires the following to build and run:
- **CMake**: Version 3.18 or higher
- **C++ Compiler**: Must support C++17 or later
- **CUDA Driver**: Required
- **CUDA Version**: 10.1 or higher
- **CUDA Runtime**

## Building and Installation
CuWeaver uses **CMake** for building and installation. Follow these steps to compile and install the library:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/CuWeaver.git
   cd CuWeaver
   ```

2. Create a build directory:

   ```bash
   mkdir build
   cd build
   ```

3. Generate build files:

   ```bash
   cmake ..
   ```

4. Build the project:

   ```bash
   cmake --build .
   ```

5. Install the library:

   ```bash
   sudo cmake --install .
   ```

## Usage

Once installed, CuWeaver simplifies concurrency management in CUDA applications. The library provides C++ interfaces to wrap CUDA Runtime APIs and manage memory and event flows.

```cpp
// CuWeaver - Automatic CUDA Stream Management Demo
#include <cuweaver/StreamManager.cuh>

__global__ void fillKernel(int* data, int value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = value;
}

__global__ void addKernel(int* c, const int* a, const int* b, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) c[idx] = a[idx] + b[idx];
}

int main() {
    using namespace cuweaver;
    
    auto manager = &StreamManager::getInstance();
    manager->initialize(50, 8, 4); // Event pool, execution streams, resource streams
    
    int *a, *b, *c;
    constexpr size_t size = 1 << 20;
    
    // GPU memory allocation
    manager->malloc(&a, size * sizeof(int), 0);
    manager->malloc(&b, size * sizeof(int), 0);
    manager->malloc(&c, size * sizeof(int), 0);
    
    dim3 grid((size + 511) / 512), block(512);
    
    // Concurrent operations - automatic stream management and dependency tracking
    // Note: All StreamManager operations are non-blocking "submissions" to CUDA,
    // they return immediately without blocking the host thread
    manager->launchKernel(fillKernel, grid, block, 0, deviceFlags::Auto, 
                         makeWrite(a, 0), 1, size);
    manager->launchKernel(fillKernel, grid, block, 0, deviceFlags::Auto, 
                         makeWrite(b, 0), 2, size);
    
    // Automatically waits for a and b to be ready before execution
    // This submission also returns immediately - dependency is handled by CUDA streams
    manager->launchKernel(addKernel, grid, block, 0, deviceFlags::Auto, 
                         makeWrite(c, 0), makeRead(a, 0), makeRead(b, 0), size);
    
    // Automatically waits for computation to complete before copying
    // This memcpy submission is also non-blocking to the host thread
    auto h_result = new int[size];
    manager->memcpy(h_result, Host, c, 0, size * sizeof(int), memcpyFlags::DeviceToHost);
    
    cudaDeviceSynchronize(); // Explicit synchronization needed to wait for all GPU operations to complete
    
    // Verify results: should all be 3 (1+2)
    std::cout << "Result: ";
    for (int i = 0; i < 5; ++i) std::cout << h_result[i] << " ";
    
    // Clean up resources
    delete[] h_result;
    manager->free(a, 0);
    manager->free(b, 0);
    manager->free(c, 0);
    
    return 0;
}
```

## Design Overview

CuWeaver works by automatically separating memory operations (e.g., allocation, copying) from computation (e.g., kernel calls) by dispatching them to distinct memory and execution streams. Multi-GPU optimization and event-driven dependency maintenance ensure smooth operation even in complex, multi-GPU systems.

## Project Plan and Progress

### Current Progress:

1. **C++-style CUDA Runtime API Wrappers** (Completed): Wrapping core CUDA Runtime functions (like `cudaMalloc`) with C++-style abstractions using RAII and move semantics.
2. **Automatic Concurrency Management** (Completed): Automating the flow control for memory and computation streams, including event-driven dependency management between different operations.
3. **Simplified Multi-GPU Management** (In Progress): Streamlining memory management and kernel invocation for multi-GPU systems, with automatic memory transfers and optimizations for cross-GPU communication.

### Upcoming Features:

4. **Automated, Portable CUDA Memory Management** (Planned): Developing a more streamlined, portable approach to memory management across different CUDA devices, including automatic memory allocation, transfers, and synchronization.

## Contributing

We welcome contributions! Please refer to the [Contributing Guide](CONTRIBUTING.md) for detailed guidelines on how to contribute.

## Testing

For instructions on building and running the test suite, refer to the [Testing Guide](docs/TESTING.md).

## License

CuWeaver is licensed under the [MIT License](LICENSE).
