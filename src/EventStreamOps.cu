#include <cuweaver/EventStreamOps.cuh>

#include "cuweaver_utils/ErrorCheck.cuh"

namespace cuweaver {
    float eventElapsedTime(const cudaEvent& start, const cudaEvent& end) {
        float ms;
        CUW_THROW_IF_ERROR(cudaEventElapsedTime(&ms, start.nativeHandle(), end.nativeHandle()));
        return ms;
    }

    bool eventQuery(const cudaEvent& event) {
        try {
            CUW_THROW_IF_ERROR(cudaEventQuery(event.nativeHandle()));
            return true;
        }
        catch (const cudaError& e) {
            if (e.codeNative() == cudaErrorNotReady) {
                return false;
            }
            throw;
        }
    }

    void eventRecord(const cudaEvent& event, const cudaStream& stream) {
        CUW_THROW_IF_ERROR(cudaEventRecord(event.nativeHandle(), stream.nativeHandle()));
    }

    void eventRecordWithFlags(const cudaEvent& event, const cudaStream& stream, cudaEventRecordFlags flags) {
        CUW_THROW_IF_ERROR(
            cudaEventRecordWithFlags(event.nativeHandle(), stream.nativeHandle(), static_cast<unsigned int>(flags)));
    }

    void eventSynchronize(const cudaEvent& event) {
        CUW_THROW_IF_ERROR(cudaEventSynchronize(event.nativeHandle()));
    }

    bool streamQuery(const cudaStream& stream) {
        try {
            CUW_THROW_IF_ERROR(cudaStreamQuery(stream.nativeHandle()));
            return true;
        }
        catch (const cudaError& e) {
            if (e.codeNative() == cudaErrorNotReady) {
                return false;
            }
            throw;
        }
    }

    void streamSynchronize(const cudaStream& stream) {
        CUW_THROW_IF_ERROR(cudaStreamSynchronize(stream.nativeHandle()));
    }

    void streamAddCallback(const cudaStream& stream, cudaStreamCallback_t callback, void* userData,
                           const unsigned int flags) {
        if (flags) {
            throw std::invalid_argument("Use streamAddCallback with flags set to 0.");
        }
        CUW_THROW_IF_ERROR(cudaStreamAddCallback(stream.nativeHandle(), callback, userData, flags));
    }

    void streamWaitEvent(const cudaStream& stream, const cudaEvent& event, cudaEventWait flags) {
        CUW_THROW_IF_ERROR(
            cudaStreamWaitEvent(stream.nativeHandle(), event.nativeHandle(), static_cast<unsigned int>(flags)));
    }
}
