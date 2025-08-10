/**
* @file EventStreamOps.cuh
 * @author Mgepahmge (https://github.com/Mgepahmge)
 * @brief Declarations for CUDA stream and event operation wrappers.
 *
 * @details This file provides type-safe encapsulations of CUDA Runtime API operations for streams and events.
 *          It depends on `Event.cuh` (for the `cudaEvent` type) and `Stream.cuh` (for the `cudaStream` type).
 *          Compilation is conditional on CUDA availability (`__CUDACC__`): if CUDA is not enabled, a warning is emitted
 *          and the file contents are skipped.
 */
#ifndef CUWEAVER_EVENTSTREAMOPS_CUH
#define CUWEAVER_EVENTSTREAMOPS_CUH

#ifdef __CUDACC__

#include "Event.cuh"
#include "Stream.cuh"

namespace cuweaver {
    /**
     * @brief Calculates the elapsed time between two CUDA events in milliseconds.
     *
     * @details Computes the time difference (in milliseconds) between the completion of the `start` event
     *          and the completion of the `end` event using the CUDA runtime API `cudaEventElapsedTime`.
     *          The result is the wall-clock time from when `start` finished executing to when `end` finished.
     *
     * @warning The function **requires both `start` and `end` events to have completed execution**
     *          (i.e., their underlying CUDA event handles must have been recorded and reached a
     *          completed state via CUDA operations like `cudaEventRecord` and synchronization).
     *          If either event is incomplete, invalid, `cudaEventElapsedTime` will fail,
     *          and a `cuweaver::cudaError` will be thrown.
     *
     * @param[in] start CUDA event representing the **start** of the time interval (must be completed).
     * @param[in] end   CUDA event representing the **end** of the time interval (must be completed).
     *
     * @return Elapsed time between the completion of `start` and `end` in milliseconds.
     *
     * @throws cuweaver::cudaError Thrown if `cudaEventElapsedTime` fails (e.g., events are incomplete,
     *                             invalid, or an underlying CUDA runtime error occurs).
     */
    float eventElapsedTime(const cudaEvent& start, const cudaEvent& end);

    /**
     * @brief Queries whether a CUDA event has completed execution.
     *
     * @details Uses `cudaEventQuery` to check the status of the provided CUDA event. Returns `true` if the event
     *          has completed (i.e., `cudaEventQuery` returns `cudaSuccess`). Returns `false` if the event is
     *          still pending (i.e., `cudaEventQuery` returns `cudaErrorNotReady`). Throws a `cuweaver::cudaError`
     *          for any other CUDA runtime error (e.g., invalid event handle or driver API mismatch).
     *
     * @param[in] event CUDA event to query for completion status.
     *
     * @return `true` if the event has completed execution; `false` if the event is still pending.
     *
     * @throws cuweaver::cudaError Thrown if `cudaEventQuery` returns any error code other than `cudaSuccess`
     *                             or `cudaErrorNotReady`.
     */
    bool eventQuery(const cudaEvent& event);

    /**
     * @brief Records a CUDA event into a CUDA stream.
     *
     * @details Uses the CUDA runtime API `cudaEventRecord` to schedule the specified `event` to be recorded
     *          in the given `stream`. The event will transition to the completed state once all operations
     *          in the stream that were enqueued before this call have finished executing. This function
     *          is used to mark synchronization or timing points in a stream for subsequent operations
     *          (e.g., timing with `EventElapsedTime` or polling with `EventQuery`).
     *
     * @param[in] event CUDA event to record into the stream.
     * @param[in] stream CUDA stream to which the event will be recorded.
     *
     * @throws cuweaver::cudaError Thrown if the underlying `cudaEventRecord` call fails (e.g., invalid
     *                             event handle, invalid stream handle, or an error in the CUDA runtime).
     *
     * @par Returns
     *      Nothing.
     */
    void eventRecord(const cudaEvent& event, const cudaStream& stream = cudaStream::defaultStream());

    /**
     * @brief Records a CUDA event into a CUDA stream with specified flags.
     *
     * @details Uses the CUDA runtime API `cudaEventRecordWithFlags` to schedule the given `event` into the `stream`
     *          with the provided `flags`. Unlike the flag-less `eventRecord` function, this variant allows
     *          customization of event recording behavior (e.g., enabling external synchronization) via the `flags`
     *          parameter. The event will transition to the completed state once all operations in the stream
     *          enqueued before this call finish executing, and can be used for subsequent synchronization or timing.
     *
     * @param[in] event CUDA event to record into the stream.
     * @param[in] stream CUDA stream to which the event will be recorded.
     * @param[in] flags Flags that configure event recording behavior (from `cudaEventRecordFlags` enumeration).
     *
     * @throws cuweaver::cudaError Thrown if the underlying `cudaEventRecordWithFlags` call fails (e.g., invalid
     *                             event handle, invalid stream handle, invalid flag value, or a CUDA runtime error).
     *
     * @par Returns
     *      Nothing.
     */
    void eventRecordWithFlags(const cudaEvent& event, const cudaStream& stream = cudaStream::defaultStream(),
                              cudaEventRecordFlags flags = cudaEventRecordFlags::Default);

    /**
     * @brief Blocks the host thread until the specified CUDA event completes.
     *
     * @details Waits for the provided `event` to reach the completed state by invoking `cudaEventSynchronize`
     *          on the event's native handle. The host thread will block until all operations in the event's
     *          associated stream (enqueued before the event was recorded) finish executing. This ensures
     *          host-side synchronization with the event's completion.
     *
     * @param[in] event CUDA event whose completion the host thread will wait for.
     *
     * @throws cuweaver::cudaError Thrown if the underlying `cudaEventSynchronize` call fails (e.g., invalid
     *                             event handle, unrecorded event, or a CUDA runtime error).
     *
     * @par Returns
     *      Nothing.
     */
    void eventSynchronize(const cudaEvent& event);

    /**
     * @brief Queries if the underlying CUDA stream of a `cudaStream` wrapper has completed all operations.
     *
     * @details Attempts to check the completion status of the managed CUDA stream using `cudaStreamQuery`. If the stream
     *          has finished all queued work, returns `true`. If the stream is still executing operations (`cudaErrorNotReady`),
     *          returns `false`. Any other error from `cudaStreamQuery` (e.g., invalid handle) is rethrown as a `cudaError`.
     *
     * @param[in] stream Reference to the `cudaStream` wrapper whose underlying CUDA stream to query.
     *
     * @retval true The stream has completed all operations.
     * @retval false The stream is still busy (returned `cudaErrorNotReady`).
     *
     * @throws cudaError Thrown if `cudaStreamQuery` returns an error other than `cudaErrorNotReady`.
     */
    bool streamQuery(const cudaStream& stream);
}

#endif

#ifndef __CUDACC__
#pragma warning("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_EVENTSTREAMOPS_CUH
