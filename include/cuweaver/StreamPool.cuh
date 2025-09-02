/**
* @file StreamPool.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Declares the cuweaver::StreamPool class for managing CUDA streams for execution and resource tasks.
 *
 * @details This file defines the StreamPool class, which maintains two groups of CUDA streams:
 *          - **Execution streams**: Optimized for computation tasks (e.g., kernel launches).
 *          - **Resource streams**: Optimized for resource management (e.g., memory transfers).
 *
 *          Streams are stored in circular lists to enable round-robin acquisition, ensuring balanced workload distribution.
 *          All streams are created with `cudaStreamFlags::NonBlocking` for asynchronous operation. The class provides
 *          methods to sync individual stream groups or all streams, with error handling for synchronization failures.
 */
#ifndef CUWEAVER_STREAMPOOL_CUH
#define CUWEAVER_STREAMPOOL_CUH

#ifdef __CUDACC__
#include <cuweaver_utils/CircularList.cuh>
#include <cuweaver/Stream.cuh>

namespace cuweaver {
    /**
     * @class StreamPool StreamPool.cuh
     * @brief A pool of CUDA streams for balanced execution and resource management.
     *
     * @details The StreamPool manages two types of streams to separate computation from resource operations:
     *          - **Execution streams**: Used for CPU/GPU computation tasks.
     *          - **Resource streams**: Used for memory operations (e.g., `cudaMemcpy`) or event synchronization.
     *
     *          Streams are allocated in circular lists to support round-robin acquisition, preventing overloading
     *          of individual streams. The pool initializes streams with `cudaStreamFlags::NonBlocking` to maximize
     *          parallelism. Synchronization methods allow waiting for all operations on a stream group or all streams.
     */
    class StreamPool {
    public:
        /**
         * @brief Constructs a StreamPool with specified stream counts for execution and resource tasks.
         *
         * @details Initializes the pool with `numExecutionStreams` execution streams and `numResourceStreams`
         *          resource streams. All streams are created with `cudaStreamFlags::NonBlocking`. Throws an
         *          exception if either stream count is zero (invalid configuration).
         *
         * @param[in] numExecutionStreams Number of execution streams to pre-allocate (default: 16).
         * @param[in] numResourceStreams Number of resource streams to pre-allocate (default: 4).
         *
         * @throws std::invalid_argument If `numExecutionStreams` or `numResourceStreams` is 0.
         */
        explicit StreamPool(size_t numExecutionStreams = 16, size_t numResourceStreams = 4);

        StreamPool(const StreamPool&) = delete; //!< Disable copy constructor.

        StreamPool& operator=(const StreamPool&) = delete; //!< Disable copy assignment.

        /**
         * @brief Default move constructor for StreamPool.
         *
         * @details Constructs a StreamPool by transferring ownership of resources from another instance.
         *          The moved-from instance (`other`) is left in a valid but unspecified state after the operation.
         *          This constructor uses compiler-generated move semantics.
         *
         * @param[in] other The StreamPool instance to move resources from.
         */
        StreamPool(StreamPool&& other) noexcept = default;

        /**
         * @brief Default move assignment operator for StreamPool.
         *
         * @details Transfers ownership of resources from another StreamPool to this instance.
         *          Releases any resources previously held by this pool before acquiring resources from `other`.
         *          The moved-from instance (`other`) is left in a valid but unspecified state after the operation.
         *          This operator uses compiler-generated move semantics.
         *
         * @param[in] other The StreamPool instance to move resources from.
         *
         * @return Reference to this StreamPool instance after the move assignment.
         */
        StreamPool& operator=(StreamPool&& other) noexcept = default;

        /**
         * @brief Acquires the next execution stream in round-robin order.
         *
         * @details Cycles through the execution stream list to distribute workloads evenly. The next stream
         *          in the sequence is returned for computation tasks (e.g., kernel launches).
         *
         * @return Reference to the next available execution stream.
         */
        cudaStream& getExecutionStream();

        /**
         * @brief Acquires the next resource stream in round-robin order.
         *
         * @details Cycles through the resource stream list to distribute workloads evenly. The next stream
         *          in the sequence is returned for resource management tasks (e.g., memory transfers).
         *
         * @return Reference to the next available resource stream.
         */
        cudaStream& getResourceStream();

        /**
         * @brief Synchronizes all execution streams in the pool.
         *
         * @details Waits for all operations on every execution stream to complete. Catches exceptions from
         *          stream synchronization (e.g., CUDA errors) and returns `false` if any failure occurs.
         *
         * @return `true` if all execution streams synchronized successfully; `false` otherwise.
         */
        bool syncExecutionStreams();

        /**
         * @brief Synchronizes all resource streams in the pool.
         *
         * @details Waits for all operations on every resource stream to complete. Catches exceptions from
         *          stream synchronization (e.g., CUDA errors) and returns `false` if any failure occurs.
         *
         * @return `true` if all resource streams synchronized successfully; `false` otherwise.
         */
        bool syncResourceStreams();

        /**
         * @brief Synchronizes all streams (execution and resource) in the pool.
         *
         * @details Combines results from `syncExecutionStreams()` and `syncResourceStreams()`. Returns `true`
         *          only if both groups of streams synchronized without errors.
         *
         * @return `true` if all streams synchronized successfully; `false` otherwise.
         */
        bool syncAllStreams();

        /**
         * @brief Gets the number of execution streams pre-allocated in the pool.
         *
         * @return Total count of execution streams (for computation tasks) in the pool.
         */
        [[nodiscard]] size_t getNumExecutionStreams() const noexcept;

        /**
         * @brief Gets the number of resource streams pre-allocated in the pool.
         *
         * @return Total count of resource streams (for memory/resource tasks) in the pool.
         */
        [[nodiscard]] size_t getNumResourceStreams() const noexcept;

    private:
        detail::CircularList<cudaStream> executionStreams; //!< Circular list of execution streams (for computation).
        detail::CircularList<cudaStream> resourceStreams;  //!< Circular list of resource streams (for memory/resource tasks).
        size_t numExecutionStreams;                        //!< Number of pre-allocated execution streams.
        size_t numResourceStreams;                         //!< Number of pre-allocated resource streams.
    };
}

#endif

#ifndef __CUDACC__
#pragma message("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_STREAMPOOL_CUH
