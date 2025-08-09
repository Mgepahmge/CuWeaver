# CuWeaver: CUDA Concurrency Library

CuWeaver is a CUDA concurrency library designed to simplify parallel programming by automating concurrency flow management. It provides C++-style wrappers for selected CUDA Runtime APIs and helps reduce the complexity of managing concurrency in multi-GPU environments.

---

## Key Features

* **Concurrency Automation**: Automatically manages memory streams, execution streams, and event pools for each GPU, ensuring isolation of memory and computation operations.
* **Multi-GPU Simplification**: Optimizes memory management and kernel invocation in multi-GPU environments, reducing the complexity of cross-GPU development.
* **Event-Driven Dependency Management**: Ensures correct data access order by maintaining dependencies between operations, preventing data races.
* **Modern C++ Wrappers**: Provides C++-style wrappers for CUDA's native C API, leveraging RAII and move semantics to simplify resource management.

---

## Dependencies

* **CMake**: Version 3.18 or higher
* **C++ Compiler**: Must support C++17 or later
* **CUDA Driver**: Required
* **CUDA Version**: 10.1 or higher
* **CUDA Runtime**

---

## Building and Installation

This project uses **CMake** for building and installation. To compile and install the library, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/CuWeaver.git
   cd CuWeaver
   ```

2. **Create a build directory**:

   ```bash
   mkdir build
   cd build
   ```

3. **Generate build files**:

   ```bash
   cmake ..
   ```

4. **Build the project**:

   ```bash
   cmake --build .
   ```

5. **Install the library**:

   ```bash
   sudo cmake --install .
   ```

The library will be installed as a static library.

---

## Usage

Once the library is installed, you can use it to manage concurrency in your CUDA applications. The library provides simple and intuitive C++ interfaces to wrap CUDA Runtime APIs and manage memory and event flows.

```cpp
// Not yet completed
```

---

## Design Overview

* **Automatic Flow Management**: CuWeaver automatically separates memory operations (e.g., allocation, copying) from computation (e.g., kernel calls) by dispatching them to distinct memory and execution streams.
* **Multi-GPU Optimization**: CuWeaver simplifies GPU-to-GPU communication by automatically handling memory transfers across GPUs when required.
* **Event-Driven Dependency Maintenance**: Dependencies between operations are managed using events to ensure that operations are executed in the correct order without data races.

---

## Project Plan and Progress

### Current Progress:

1. **C++-style CUDA Runtime API Wrappers** (In Progress): Wrapping core CUDA Runtime functions (like `cudaMalloc`) with C++-style abstractions using RAII and move semantics.

### Upcoming Features:

2. **Automatic Concurrency Management** (Planned): Automating the flow control for memory and computation streams, including event-driven dependency management between different operations.
3. **Simplified Multi-GPU Management** (Planned): Streamlining memory management and kernel invocation for multi-GPU systems, with automatic memory transfers and optimizations for cross-GPU communication.

---

## License

CuWeaver is licensed under the [MIT License](LICENSE).
