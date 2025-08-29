#include <cuweaver/StreamPool.cuh>

#include "cuweaver/EventStreamOps.cuh"

namespace cuweaver {
    StreamPool::StreamPool(const size_t numExecutionStreams, const size_t numResourceStreams) :
        numExecutionStreams(numExecutionStreams), numResourceStreams(numResourceStreams) {
        if (numExecutionStreams == 0 || numResourceStreams == 0) {
            throw std::invalid_argument("StreamPool must be initialized with at least one stream of each type.");
        }
        for (auto i = 0; i < numExecutionStreams; ++i) {
            executionStreams.add(cudaStreamFlags::NonBlocking);
        }
        for (auto i = 0; i < numResourceStreams; ++i) {
            resourceStreams.add(cudaStreamFlags::NonBlocking);
        }
    }

    cudaStream& StreamPool::getExecutionStream() {
        return executionStreams.getNext();
    }

    cudaStream& StreamPool::getResourceStream() {
        return resourceStreams.getNext();
    }

    bool StreamPool::syncExecutionStreams() {
        try {
            executionStreams.forEach([](const cudaStream& stream) {
                streamSynchronize(stream);
            });
        } catch (...) {
            return false;
        }
        return true;
    }

    bool StreamPool::syncResourceStreams() {
        try {
            resourceStreams.forEach([](const cudaStream& stream) {
                streamSynchronize(stream);
            });
        } catch (...) {
            return false;
        }
        return true;
    }

    bool StreamPool::syncAllStreams() {
        return syncExecutionStreams() && syncResourceStreams();
    }

    size_t StreamPool::getNumExecutionStreams() const noexcept {
        return numExecutionStreams;
    }

    size_t StreamPool::getNumResourceStreams() const noexcept {
        return numResourceStreams;
    }
}
