/**
 * @file cudaDeviceProperties.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Declares the cudaDeviceProperties class for type-safe access to CUDA device properties.
 *
 * @details This file defines the cudaDeviceProperties class, which wraps the raw `cudaDeviceProp`
 *          structure from the CUDA Runtime API. It provides validated, human-readable accessors for
 *          all device capabilities (e.g., memory capacity, compute mode, thread limits) and includes
 *          index validation for multi-dimensional properties (e.g., `maxThreadsDim`, `maxGridSize`)
 *          to avoid out-of-bounds errors. This class simplifies safe interaction with CUDA device
 *          metadata in C++ applications.
 */
#ifndef CUWEAVER_CUDADEVICEPROPERTIES_CUH
#define CUWEAVER_CUDADEVICEPROPERTIES_CUH

#ifdef __CUDACC__
#include <string>

namespace cuweaver::detail {
    /**
     * @class cudaDeviceProperties
     * @brief Encapsulates CUDA device properties with type-safe, validated accessors.
     *
     * @details Wraps the raw `cudaDeviceProp` structure from the CUDA Runtime API, providing
     *          structured access to device capabilities (e.g., memory size, thread limits,
     *          compute capability). Includes validation for indexed properties (e.g.,
     *          `maxThreadsDim`, `maxGridSize`) to prevent out-of-bounds access.
     */
    class cudaDeviceProperties {
    public:
        /**
         * @brief Constructs a `cudaDeviceProperties` object using a `cudaDeviceProp` object.
         *
         * @details The constructor stores the `cudaDeviceProp` object, which contains the properties
         * of the CUDA device, allowing easy access to the device's various capabilities and configurations.
         *
         * @param[in] prop The `cudaDeviceProp` object that holds the properties of the device.
         */
        explicit cudaDeviceProperties(const cudaDeviceProp& prop);

        /**
         * @brief Returns the name of the device.
         *
         * @details This function returns the name of the CUDA device as a string.
         *
         * @return The name of the device.
         */
        [[nodiscard]] std::string getName() const noexcept;

        /**
         * @brief Returns the UUID of the device.
         *
         * @details This function returns the UUID of the CUDA device as a string.
         *          The UUID is a unique identifier for the device.
         *
         * @return The UUID of the device.
         */
        [[nodiscard]] std::string getUuid() const noexcept;

        /**
         * @brief Returns the total global memory available on the device.
         *
         * @details This function returns the amount of total global memory (in bytes)
         *          available on the CUDA device.
         *
         * @return Total global memory in bytes.
         */
        [[nodiscard]] size_t getTotalGlobalMem() const noexcept;

        /**
         * @brief Returns the amount of shared memory available per block.
         *
         * @details This function returns the amount of shared memory available per block
         *          (in bytes) on the device.
         *
         * @return Shared memory per block in bytes.
         */
        [[nodiscard]] size_t getSharedMemPerBlock() const noexcept;

        /**
         * @brief Returns the number of registers per block.
         *
         * @details This function returns the number of registers available per block
         *          for use by kernels running on the device.
         *
         * @return Number of registers per block.
         */
        [[nodiscard]] int getRegsPerBlock() const noexcept;

        /**
         * @brief Returns the warp size of the device.
         *
         * @details This function returns the warp size (in threads) of the CUDA device, which
         *          is the number of threads that can be processed simultaneously in one warp.
         *
         * @return The warp size (in threads).
         */
        [[nodiscard]] int getWarpSize() const noexcept;

        /**
         * @brief Returns the memory pitch of the device.
         *
         * @details This function returns the memory pitch (in bytes) for device memory
         *          allocation. This is useful for calculating the spacing between rows of
         *          memory in 2D memory layouts.
         *
         * @return The memory pitch in bytes.
         */
        [[nodiscard]] size_t getMemPitch() const noexcept;

        /**
         * @brief Returns the maximum number of threads per block.
         *
         * @details This function returns the maximum number of threads allowed in a block
         *          for kernels executed on the CUDA device.
         *
         * @return The maximum number of threads per block.
         */
        [[nodiscard]] int getMaxThreadsPerBlock() const noexcept;

        /**
         * @brief Returns the size of each dimension of the maximum number of threads per block.
         *
         * @details This function returns the maximum size (in threads) of each dimension
         *          of the block. The index specifies which dimension (0 = x, 1 = y, 2 = z).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum number of threads in the specified dimension.
         */
        [[nodiscard]] int getMaxThreadsDim(size_t index) const noexcept;

        /**
         * @brief Returns the size of each dimension of the maximum grid size.
         *
         * @details This function returns the maximum size (in blocks) of the grid in each dimension.
         *          The index specifies which dimension (0 = x, 1 = y, 2 = z).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum number of blocks in the specified dimension.
         */
        [[nodiscard]] int getMaxGridSize(size_t index) const noexcept;

        /**
         * @brief Returns the clock rate of the device.
         *
         * @details This function returns the clock rate (in kilohertz) of the device, which indicates
         *          the speed at which the GPU cores operate.
         *
         * @return The clock rate in kilohertz.
         */
        [[nodiscard]] int getClockRate() const noexcept;

        /**
         * @brief Returns the total constant memory on the device.
         *
         * @details This function returns the total amount of constant memory (in bytes) available
         *          on the device.
         *
         * @return Total constant memory in bytes.
         */
        [[nodiscard]] size_t getTotalConstMem() const noexcept;

        /**
         * @brief Returns the major compute capability version of the device.
         *
         * @details This function returns the major version of the device's compute capability.
         *          The compute capability indicates the feature set supported by the device.
         *
         * @return The major version of the compute capability.
         */
        [[nodiscard]] int getMajor() const noexcept;

        /**
         * @brief Returns the minor compute capability version of the device.
         *
         * @details This function returns the minor version of the device's compute capability.
         *          The compute capability indicates the feature set supported by the device.
         *
         * @return The minor version of the compute capability.
         */
        [[nodiscard]] int getMinor() const noexcept;

        /**
         * @brief Returns the texture alignment in bytes.
         *
         * @details This function returns the alignment (in bytes) required for texture data in
         *          device memory.
         *
         * @return The texture alignment in bytes.
         */
        [[nodiscard]] size_t getTextureAlignment() const noexcept;

        /**
         * @brief Returns the texture pitch alignment in bytes.
         *
         * @details This function returns the alignment (in bytes) required for texture pitch
         *          in device memory.
         *
         * @return The texture pitch alignment in bytes.
         */
        [[nodiscard]] size_t getTexturePitchAlignment() const noexcept;

        /**
         * @brief Returns whether device overlap is supported.
         *
         * @details This function returns whether the device supports overlapping kernel execution
         *          and memory copy operations.
         *
         * @return Non-zero if overlap is supported, zero otherwise.
         */
        [[nodiscard]] int getDeviceOverlap() const noexcept;

        /**
         * @brief Returns the number of multiprocessors on the device.
         *
         * @details This function returns the total number of streaming multiprocessors (SMs)
         *          on the device, which is an important factor for determining the compute capacity.
         *
         * @return The number of multiprocessors.
         */
        [[nodiscard]] int getMultiProcessorCount() const noexcept;

        /**
         * @brief Returns whether kernel execution timeout is enabled.
         *
         * @details This function returns whether kernel execution timeout is enabled for the device.
         *
         * @return Non-zero if timeout is enabled, zero otherwise.
         */
        [[nodiscard]] int getKernelExecTimeoutEnabled() const noexcept;

        /**
         * @brief Returns whether the device is integrated.
         *
         * @details This function returns whether the device is an integrated GPU or a discrete GPU.
         *
         * @return Non-zero if integrated, zero if discrete.
         */
        [[nodiscard]] int getIntegrated() const noexcept;

        /**
         * @brief Returns whether the device can map host memory.
         *
         * @details This function returns whether the device can map host memory for use with CUDA
         *          operations.
         *
         * @return Non-zero if host memory can be mapped, zero otherwise.
         */
        [[nodiscard]] int getCanMapHostMemory() const noexcept;

        /**
         * @brief Returns the compute mode of the device.
         *
         * @details This function returns the compute mode of the device, which determines the
         *          level of access other applications have to the GPU.
         *
         * @return The compute mode.
         */
        [[nodiscard]] int getComputeMode() const noexcept;
                /**
         * @brief Returns the maximum texture size in 1D.
         *
         * @details This function returns the maximum size of a 1D texture that can be created
         *          on the device.
         *
         * @return The maximum size of a 1D texture.
         */
        [[nodiscard]] int getMaxTexture1D() const noexcept;

        /**
         * @brief Returns the maximum texture size in 1D with mipmaps.
         *
         * @details This function returns the maximum size of a 1D texture with mipmaps
         *          that can be created on the device.
         *
         * @return The maximum size of a 1D texture with mipmaps.
         */
        [[nodiscard]] int getMaxTexture1DMipmap() const noexcept;

        /**
         * @brief Returns the maximum linear size of a 1D texture.
         *
         * @details This function returns the maximum linear size of a 1D texture that can
         *          be created on the device.
         *
         * @return The maximum linear size of a 1D texture.
         */
        [[nodiscard]] int getMaxTexture1DLinear() const noexcept;

        /**
         * @brief Returns the maximum size of a 2D texture in a given dimension.
         *
         * @details This function returns the maximum size of a 2D texture that can be created
         *          on the device in the specified dimension. The index specifies which dimension
         *          (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of the 2D texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture2D(size_t index) const noexcept;

        /**
         * @brief Returns the maximum mipmap size of a 2D texture in a given dimension.
         *
         * @details This function returns the maximum mipmap size of a 2D texture in the
         *          specified dimension. The index specifies which dimension (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum mipmap size of the 2D texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture2DMipmap(size_t index) const noexcept;

        /**
         * @brief Returns the maximum linear size of a 2D texture.
         *
         * @details This function returns the maximum linear size of a 2D texture that can
         *          be created on the device. The index specifies the dimension (0 = width,
         *          1 = height, 2 = depth).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum linear size of a 2D texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture2DLinear(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 2D texture with gather operations in a given dimension.
         *
         * @details This function returns the maximum size of a 2D texture with gather operations
         *          in the specified dimension. The index specifies which dimension (0 = width,
         *          1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of the 2D texture with gather operations in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture2DGather(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 3D texture in a given dimension.
         *
         * @details This function returns the maximum size of a 3D texture in the specified dimension.
         *          The index specifies which dimension (0 = width, 1 = height, 2 = depth).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum size of the 3D texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture3D(size_t index) const noexcept;

        /**
         * @brief Returns the maximum alternative size of a 3D texture in a given dimension.
         *
         * @details This function returns the maximum alternative size of a 3D texture in the specified
         *          dimension. The index specifies which dimension (0 = width, 1 = height, 2 = depth).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum alternative size of the 3D texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture3DAlt(size_t index) const noexcept;
        /**
         * @brief Returns the maximum size of a cubemap texture.
         *
         * @details This function returns the maximum size of a cubemap texture that can be created
         *          on the device. A cubemap texture is used for 3D rendering, often for skyboxes or
         *          environment mapping.
         *
         * @return The maximum size of a cubemap texture.
         */
        [[nodiscard]] int getMaxTextureCubemap() const noexcept;

        /**
         * @brief Returns the maximum size of a 1D layered texture in a given dimension.
         *
         * @details This function returns the maximum size of a 1D layered texture in the specified dimension.
         *          The index specifies which dimension (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of the 1D layered texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture1DLayered(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 2D layered texture in a given dimension.
         *
         * @details This function returns the maximum size of a 2D layered texture in the specified dimension.
         *          The index specifies which dimension (0 = width, 1 = height, 2 = depth).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum size of the 2D layered texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTexture2DLayered(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a cubemap layered texture in a given dimension.
         *
         * @details This function returns the maximum size of a cubemap layered texture in the specified dimension.
         *          The index specifies which dimension (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of the cubemap layered texture in the specified dimension.
         */
        [[nodiscard]] int getMaxTextureCubemapLayered(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 1D surface texture.
         *
         * @details This function returns the maximum size of a 1D surface texture that can be created on the device.
         *          Surface textures are often used in CUDA for advanced memory access patterns.
         *
         * @return The maximum size of a 1D surface texture.
         */
        [[nodiscard]] int getMaxSurface1D() const noexcept;

        /**
         * @brief Returns the maximum size of a 2D surface texture in a given dimension.
         *
         * @details This function returns the maximum size of a 2D surface texture in the specified dimension.
         *          The index specifies which dimension (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of a 2D surface texture in the specified dimension.
         */
        [[nodiscard]] int getMaxSurface2D(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 3D surface texture in a given dimension.
         *
         * @details This function returns the maximum size of a 3D surface texture in the specified dimension.
         *          The index specifies which dimension (0 = width, 1 = height, 2 = depth).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum size of a 3D surface texture in the specified dimension.
         */
        [[nodiscard]] int getMaxSurface3D(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 1D surface layered texture in a given dimension.
         *
         * @details This function returns the maximum size of a 1D surface texture that can be created in layers,
         *          in the specified dimension. The index specifies which dimension (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of a 1D surface layered texture in the specified dimension.
         */
        [[nodiscard]] int getMaxSurface1DLayered(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a 2D surface layered texture in a given dimension.
         *
         * @details This function returns the maximum size of a 2D surface texture in layers,
         *          in the specified dimension. The index specifies which dimension (0 = width, 1 = height, 2 = depth).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 2).
         * @return The maximum size of a 2D surface layered texture in the specified dimension.
         */
        [[nodiscard]] int getMaxSurface2DLayered(size_t index) const noexcept;

        /**
         * @brief Returns the maximum size of a cubemap surface texture.
         *
         * @details This function returns the maximum size of a cubemap surface texture, used for 3D rendering.
         *          Cubemaps are often used in shaders for environment mapping.
         *
         * @return The maximum size of a cubemap surface texture.
         */
        [[nodiscard]] int getMaxSurfaceCubemap() const noexcept;

        /**
         * @brief Returns the maximum size of a cubemap surface layered texture in a given dimension.
         *
         * @details This function returns the maximum size of a cubemap surface texture in layers,
         *          in the specified dimension. The index specifies which dimension (0 = width, 1 = height).
         *
         * @param[in] index The index for the dimension (valid range: 0 to 1).
         * @return The maximum size of a cubemap surface layered texture in the specified dimension.
         */
        [[nodiscard]] int getMaxSurfaceCubemapLayered(size_t index) const noexcept;

        /**
         * @brief Returns the alignment for surfaces on the device.
         *
         * @details This function returns the alignment (in bytes) required for surfaces in device memory.
         *          It ensures that surfaces are correctly aligned for optimal performance.
         *
         * @return The surface alignment in bytes.
         */
        [[nodiscard]] size_t getSurfaceAlignment() const noexcept;

        /**
         * @brief Returns the number of concurrent kernels supported by the device.
         *
         * @details This function returns the maximum number of kernels that can be executed concurrently
         *          on the device.
         *
         * @return The number of concurrent kernels supported by the device.
         */
        [[nodiscard]] int getConcurrentKernels() const noexcept;

        /**
         * @brief Returns whether ECC (Error-Correcting Code) memory is enabled on the device.
         *
         * @details This function returns whether ECC memory is enabled on the CUDA device.
         *          ECC helps detect and correct memory errors.
         *
         * @return Non-zero if ECC is enabled, zero if it is disabled.
         */
        [[nodiscard]] int getECCEnabled() const noexcept;

        /**
         * @brief Returns the PCI bus ID of the device.
         *
         * @details This function returns the PCI bus ID for the device. The PCI bus ID is used
         *          to identify the physical location of the device in a multi-GPU setup.
         *
         * @return The PCI bus ID of the device.
         */
        [[nodiscard]] int getPciBusID() const noexcept;

        /**
         * @brief Returns the PCI device ID of the device.
         *
         * @details This function returns the PCI device ID for the device. The PCI device ID uniquely
         *          identifies the device on the PCI bus.
         *
         * @return The PCI device ID of the device.
         */
        [[nodiscard]] int getPciDeviceID() const noexcept;

        /**
         * @brief Returns the PCI domain ID of the device.
         *
         * @details This function returns the PCI domain ID for the device. The domain ID is used
         *          in systems with multiple PCI domains to distinguish different physical areas of
         *          the system's PCI bus.
         *
         * @return The PCI domain ID of the device.
         */
        [[nodiscard]] int getPciDomainID() const noexcept;

        /**
         * @brief Returns the TCC (Tesla Compute Cluster) driver status.
         *
         * @details This function returns whether the device is in TCC mode. TCC mode is used for
         *          compute workloads and provides maximum performance for CUDA applications.
         *
         * @return Non-zero if TCC mode is enabled, zero otherwise.
         */
        [[nodiscard]] int getTccDriver() const noexcept;

        /**
         * @brief Returns the number of asynchronous engines on the device.
         *
         * @details This function returns the number of asynchronous engines available on the device.
         *          Asynchronous engines enable overlapping of compute and memory operations.
         *
         * @return The number of asynchronous engines.
         */
        [[nodiscard]] int getAsyncEngineCount() const noexcept;

        /**
         * @brief Returns whether unified addressing is supported.
         *
         * @details This function returns whether unified addressing (CUDA's ability to access both
         *          host and device memory using a single pointer) is supported by the device.
         *
         * @return Non-zero if unified addressing is supported, zero otherwise.
         */
        [[nodiscard]] int getUnifiedAddressing() const noexcept;

        /**
         * @brief Returns the memory clock rate of the device.
         *
         * @details This function returns the memory clock rate of the device in megahertz (MHz).
         *          The memory clock rate affects memory access speed.
         *
         * @return The memory clock rate in MHz.
         */
        [[nodiscard]] int getMemoryClockRate() const noexcept;

        /**
         * @brief Returns the memory bus width of the device.
         *
         * @details This function returns the memory bus width of the device in bits, which indicates
         *          the bandwidth for memory access.
         *
         * @return The memory bus width in bits.
         */
        [[nodiscard]] int getMemoryBusWidth() const noexcept;

        /**
         * @brief Returns the L2 cache size of the device.
         *
         * @details This function returns the size of the L2 cache (in bytes) available on the device,
         *          which helps improve memory access speed by storing frequently used data.
         *
         * @return The L2 cache size in bytes.
         */
        [[nodiscard]] int getL2CacheSize() const noexcept;
        /**
         * @brief Returns the maximum size of the persisting L2 cache on the device.
         *
         * @details This function returns the maximum size of the persisting L2 cache (in bytes).
         *          The persisting L2 cache is used to store data that is frequently accessed by
         *          compute workloads, improving memory access speeds.
         *
         * @return The maximum size of the persisting L2 cache in bytes.
         */
        [[nodiscard]] int getPersistingL2CacheMaxSize() const noexcept;

        /**
         * @brief Returns the maximum number of threads per multiprocessor.
         *
         * @details This function returns the maximum number of threads that can be assigned
         *          to each streaming multiprocessor (SM) on the device.
         *
         * @return The maximum number of threads per multiprocessor.
         */
        [[nodiscard]] int getMaxThreadsPerMultiProcessor() const noexcept;

        /**
         * @brief Returns whether stream priorities are supported.
         *
         * @details This function returns whether the device supports prioritizing CUDA streams.
         *          Stream priorities allow for better control over the execution order of operations.
         *
         * @return Non-zero if stream priorities are supported, zero otherwise.
         */
        [[nodiscard]] int getStreamPrioritiesSupported() const noexcept;

        /**
         * @brief Returns whether global L1 cache is supported.
         *
         * @details This function returns whether the device supports global L1 cache, which allows
         *          for improved memory access speed for large workloads.
         *
         * @return Non-zero if global L1 cache is supported, zero otherwise.
         */
        [[nodiscard]] int getGlobalL1CacheSupported() const noexcept;

        /**
         * @brief Returns whether local L1 cache is supported.
         *
         * @details This function returns whether the device supports local L1 cache, which allows
         *          for faster memory access within each streaming multiprocessor.
         *
         * @return Non-zero if local L1 cache is supported, zero otherwise.
         */
        [[nodiscard]] int getLocalL1CacheSupported() const noexcept;

        /**
         * @brief Returns the shared memory per multiprocessor.
         *
         * @details This function returns the amount of shared memory (in bytes) available per
         *          streaming multiprocessor (SM) on the device.
         *
         * @return Shared memory per multiprocessor in bytes.
         */
        [[nodiscard]] size_t getSharedMemPerMultiprocessor() const noexcept;

        /**
         * @brief Returns the number of registers per multiprocessor.
         *
         * @details This function returns the number of registers available per multiprocessor for
         *          kernel execution on the device.
         *
         * @return The number of registers per multiprocessor.
         */
        [[nodiscard]] int getRegsPerMultiprocessor() const noexcept;

        /**
         * @brief Returns whether managed memory is supported.
         *
         * @details This function returns whether the device supports managed memory, allowing
         *          both the CPU and GPU to access the same memory space.
         *
         * @return Non-zero if managed memory is supported, zero otherwise.
         */
        [[nodiscard]] int getManagedMemory() const noexcept;

        /**
         * @brief Returns whether the device is a multi-GPU board.
         *
         * @details This function returns whether the device is part of a multi-GPU setup.
         *          This is useful for identifying devices in a multi-GPU configuration.
         *
         * @return Non-zero if the device is part of a multi-GPU board, zero otherwise.
         */
        [[nodiscard]] int getIsMultiGpuBoard() const noexcept;

        /**
         * @brief Returns the group ID for a multi-GPU board.
         *
         * @details This function returns the ID of the group to which the multi-GPU board belongs.
         *          This is useful for identifying devices in a multi-GPU configuration.
         *
         * @return The multi-GPU board group ID.
         */
        [[nodiscard]] int getMultiGpuBoardGroupID() const noexcept;

        /**
         * @brief Returns the single-to-double precision performance ratio.
         *
         * @details This function returns the performance ratio of single-precision (float) to
         *          double-precision (double) operations on the device. This can help in understanding
         *          the device's performance characteristics for scientific computations.
         *
         * @return The single-to-double precision performance ratio.
         */
        [[nodiscard]] int getSingleToDoublePrecisionPerfRatio() const noexcept;

        /**
         * @brief Returns whether pageable memory access is supported.
         *
         * @details This function returns whether the device supports pageable memory access.
         *          Pageable memory allows data to be transferred between host memory and device memory
         *          with lower latency compared to pinned memory.
         *
         * @return Non-zero if pageable memory access is supported, zero otherwise.
         */
        [[nodiscard]] int getPageableMemoryAccess() const noexcept;

        /**
         * @brief Returns whether concurrent managed memory access is supported.
         *
         * @details This function returns whether the device supports concurrent access to managed
         *          memory from both the CPU and GPU. This enables concurrent memory operations
         *          across multiple devices.
         *
         * @return Non-zero if concurrent managed access is supported, zero otherwise.
         */
        [[nodiscard]] int getConcurrentManagedAccess() const noexcept;

        /**
         * @brief Returns whether compute preemption is supported.
         *
         * @details This function returns whether the device supports compute preemption, which
         *          allows the GPU to stop a running kernel and schedule a new one.
         *
         * @return Non-zero if compute preemption is supported, zero otherwise.
         */
        [[nodiscard]] int getComputePreemptionSupported() const noexcept;

        /**
         * @brief Returns whether the device can use host pointer for registered memory.
         *
         * @details This function returns whether the device can directly access host memory through
         *          registered pointers, improving memory transfer efficiency.
         *
         * @return Non-zero if the device can use host pointers for registered memory, zero otherwise.
         */
        [[nodiscard]] int getCanUseHostPointerForRegisteredMem() const noexcept;

        /**
         * @brief Returns whether cooperative launch is supported.
         *
         * @details This function returns whether the device supports cooperative kernel launches.
         *          Cooperative kernel launches enable multiple kernels to run concurrently on the
         *          same device, improving overall utilization.
         *
         * @return Non-zero if cooperative launch is supported, zero otherwise.
         */
        [[nodiscard]] int getCooperativeLaunch() const noexcept;

        /**
         * @brief Returns whether cooperative multi-device launch is supported.
         *
         * @details This function returns whether the device supports launching cooperative kernels
         *          across multiple devices, enabling improved performance for multi-GPU configurations.
         *
         * @return Non-zero if cooperative multi-device launch is supported, zero otherwise.
         */
        [[nodiscard]] int getCooperativeMultiDeviceLaunch() const noexcept;

        /**
         * @brief Returns whether pageable memory access uses host page tables.
         *
         * @details This function returns whether pageable memory access uses host page tables
         *          instead of device-specific tables. This can improve memory management efficiency
         *          in certain use cases.
         *
         * @return Non-zero if pageable memory access uses host page tables, zero otherwise.
         */
        [[nodiscard]] int getPageableMemoryAccessUsesHostPageTables() const noexcept;

        /**
         * @brief Returns whether the device allows direct managed memory access from the host.
         *
         * @details This function returns whether the device allows the host to access managed
         *          memory directly without requiring intermediate device operations.
         *
         * @return Non-zero if direct managed memory access from host is supported, zero otherwise.
         */
        [[nodiscard]] int getDirectManagedMemAccessFromHost() const noexcept;

        /**
         * @brief Returns the maximum window size for access policy.
         *
         * @details This function returns the maximum window size (in bytes) for the access policy
         *          used by the device. This is important for managing memory regions accessed by
         *          CUDA kernels.
         *
         * @return The maximum window size for access policy in bytes.
         */
        [[nodiscard]] int getAccessPolicyMaxWindowSize() const noexcept;

        /**
         * @brief Gets the raw CUDA device properties structure.
         *
         * @return The underlying `cudaDeviceProp` structure from the CUDA Runtime API.
         */
        [[nodiscard]] cudaDeviceProp nativeHandle() const noexcept;

    private:
        cudaDeviceProp properties; //!< Raw CUDA device properties structure from the Runtime API

        /**
         * @brief Validates whether an index is within the valid range.
         *
         * @details This helper function ensures that an index is within the valid range of the
         *          corresponding array or list, based on the maximum index size.
         *
         * @param[in] index The index to check.
         * @param[in] maxIndex The maximum valid index (exclusive).
         * @return `true` if the index is valid, `false` otherwise.
         */
        static bool isValidIndex(size_t index, size_t maxIndex);
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_CUDADEVICEPROPERTIES_CUH
