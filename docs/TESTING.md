# Testing Guide for CuWeaver

This document explains how to build and run the test suite for
CuWeaver. The tests ensure that changes do not break existing
functionality and that new features behave as expected.

## Prerequisites

Before building the tests, ensure you have the following dependencies
installed:

- **CMake** version 3.18 or higher.
- A C++ compiler with C++17 support.
- **CUDA** driver and runtime (version 10.1 or higher).
- **Google Test** (GTest).  
  If GTest is not installed system‑wide, you can add it as a
  dependency in your CMake configuration or fetch it via `FetchContent`. The
  project’s CMake scripts assume that `find_package(GTest)` will
  succeed when testing is enabled.

## Building the Test Suite

CuWeaver uses CMake’s [`CTest`](https://cmake.org/cmake/help/latest/module/CTest.html) to manage its
unit tests. To build the tests, follow these steps from the root of
your clone:

```bash
# Create and enter the build directory
mkdir build
cd build

# Configure the project with testing enabled
cmake .. -DBUILD_TESTING=ON

# Build the library and tests
cmake --build .
````

The `BUILD_TESTING` option is OFF by default. Enabling it adds the  
`tests/` directory to the build and links the necessary test  
frameworks.

## Running Tests

After building, you can execute the entire test suite using `ctest`:

```bash
ctest
```

Some useful `ctest` options:

- `ctest -V` – run tests in verbose mode to see detailed output.

- `ctest -R <regex>` – run only tests whose names match the provided  
  regular expression. For example:

    ```bash
    ctest -R CudaErrorHandlingTest
    ```

- `ctest --output-on-failure` – show output only for tests that fail.


Tests that require a CUDA device will skip automatically when no GPU is  
available or when the code is not compiled with CUDA support. Skipped  
tests are reported but do not cause a failure.

## Writing New Tests

Contributors adding new features or fixing bugs must add or update  
unit tests accordingly. Tests are written using **Google Test** and  
located in the `tests/` directory. Follow these guidelines when  
creating tests:

1. **Use descriptive test names.** Test cases follow the format  
   `TEST(SuiteName, TestName)`. The suite name should describe the  
   component under test (e.g., `CuWeaverCudaStream`) and the test name  
   should describe the specific behaviour.

2. **Check for GPU availability.** Many tests depend on CUDA  
   functionality. At the beginning of each test, check whether CUDA is  
   available and skip the test if not:

    ```cpp
    #ifndef __CUDACC__
        GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
    #endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    ```

3. **Use assertions appropriately.** Prefer `EXPECT_*` for non‑fatal  
   checks and `ASSERT_*` when subsequent code depends on the result.  
   See the Google Test documentation for guidance.

4. **Test error handling.** When adding error handling logic, write  
   tests that verify exceptions are thrown with the correct error code,  
   context and message. The existing `ErrorHandlingTest.cu` offers  
   examples.

5. **Cover edge cases.** Include tests for boundary conditions,  
   invalid parameters and multi‑GPU scenarios as applicable.

6. **Register new test files.** To build a new test source file,  
   update the `tests/CMakeLists.txt` file and link against the  
   necessary libraries. The project uses `add_executable()` and  
   `target_link_libraries()` to build each test executable, and then  
   uses `add_test()` to register it with CTest.

7. **Keep tests fast and deterministic.** Tests should complete  
   quickly and produce the same results on repeated runs.


## Continuous Integration (CI)

The maintainers may configure CI to automatically run tests on pull  
requests. Passing tests locally before opening a pull request ensures  
that CI runs smoothly and helps reviewers focus on the content of your  
changes rather than fixing build issues.

## Troubleshooting

- **Tests fail to build.** Ensure that `BUILD_TESTING` is set to `ON` and  
  that Google Test is installed or available via CMake. Look at the  
  `tests/CMakeLists.txt` for guidance on adding libraries.

- **Tests are skipped unexpectedly.** Confirm that CUDA is installed  
  correctly and that your device is visible (`nvidia-smi` should list  
  GPUs). You can also run a simple CUDA sample to check.

- **CUDA errors or crashes.** Use CUDA’s debugging tools (e.g., `cuda-gdb`)  
  or run with `cuda-memcheck` to diagnose memory access errors and  
  synchronization issues.


By following these testing practices, you help maintain the quality  
and stability of CuWeaver while enabling a robust development workflow.