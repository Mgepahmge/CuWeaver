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

}

#endif

#ifndef __CUDACC__
#pragma warning("CUDA is not available. " __FILE__ " will not be compiled.")
#endif

#endif //CUWEAVER_EVENTSTREAMOPS_CUH