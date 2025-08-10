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
        } catch (const cudaError& e) {
            if (e.codeNative() == cudaErrorNotReady) {
                return false;
            }
            throw;
        }
    }
}
