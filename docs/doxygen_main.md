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

Once installed, CuWeaver simplifies concurrency management in CUDA applications. The library provides C++ interfaces to wrap CUDA Runtime APIs and manage memory and event flows. Example usage is forthcoming.

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
