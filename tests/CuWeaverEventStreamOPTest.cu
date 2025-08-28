#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuweaver/EventStreamOps.cuh>

namespace {

inline bool cudaAvailable() {
    int count = 0;
    auto st = cudaGetDeviceCount(&count);
    return (st == cudaSuccess) && (count > 0);
}

__global__ void BusyKernel(int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int acc = 0;
    for (int i = 0; i < iterations; ++i) {
        acc += i + idx;
    }
    if (acc == 0) {
        asm volatile("");
    }
}

    __global__ void VoidKernel(void** args) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int acc = 0;
    const size_t iterations = 1 << 24;
    for (int i = 0; i < iterations; ++i) {
        acc += i + idx;
    }
    if (acc == 0) {
        asm volatile("");
    }
}

} // namespace

TEST(CuWeaverCudaEvent, EventElapsedTimeNonNegative) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cuweaver::cudaEvent start;
    cuweaver::cudaEvent end;

    ASSERT_EQ(cudaEventRecord(start.nativeHandle(), 0), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(end.nativeHandle(), 0), cudaSuccess);

    ASSERT_EQ(cudaEventSynchronize(end.nativeHandle()), cudaSuccess);

    float ms = 0.0f;
    ASSERT_NO_THROW({ ms = eventElapsedTime(start, end); });
    EXPECT_GE(ms, 0.0f);
}

TEST(CuWeaverCudaEvent, EventQueryReportsPendingAndCompleted) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    cuweaver::cudaEvent ev;

    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1>>>(kIterations);
    ASSERT_EQ(cudaEventRecord(ev.nativeHandle(), 0), cudaSuccess);

    bool done = true;
    ASSERT_NO_THROW({ done = eventQuery(ev); });
    EXPECT_FALSE(done);

    ASSERT_EQ(cudaEventSynchronize(ev.nativeHandle()), cudaSuccess);
    ASSERT_NO_THROW({ done = eventQuery(ev); });
    EXPECT_TRUE(done);
}

TEST(CuWeaverCudaEvent, EventRecordOnStream) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cuweaver::cudaEvent start;
    cuweaver::cudaEvent end;

    ASSERT_NO_THROW(cuweaver::eventRecord(start));
    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1>>>(kIterations);
    ASSERT_NO_THROW(cuweaver::eventRecord(end));

    ASSERT_EQ(cudaEventSynchronize(end.nativeHandle()), cudaSuccess);

    float ms = 0.0f;
    ASSERT_NO_THROW({ ms = eventElapsedTime(start, end); });
    EXPECT_NE(ms, 0.0f);
}

TEST(CuWeaverCudaEvent, EventRecordWithFlagsOnStream) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cuweaver::cudaEvent start;
    cuweaver::cudaEvent end;
    cuweaver::cudaStream stream;

    ASSERT_NO_THROW(cuweaver::eventRecordWithFlags(start, stream, cuweaver::cudaEventRecordFlags::Default));
    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1, 0, stream.nativeHandle()>>>(kIterations);
    ASSERT_NO_THROW(cuweaver::eventRecordWithFlags(end, stream, cuweaver::cudaEventRecordFlags::Default));

    ASSERT_EQ(cudaEventSynchronize(end.nativeHandle()), cudaSuccess);

    float ms = 0.0f;
    ASSERT_NO_THROW({ ms = eventElapsedTime(start, end); });
    EXPECT_NE(ms, 0.0f);
}

TEST(CuWeaverCudaEvent, EventSynchronize) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    cuweaver::cudaEvent ev;

    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1>>>(kIterations);
    eventRecord(ev);

    bool done = true;
    ASSERT_NO_THROW({ done = eventQuery(ev); });
    EXPECT_FALSE(done);

    ASSERT_NO_THROW(cuweaver::eventSynchronize(ev));
    ASSERT_NO_THROW({ done = eventQuery(ev); });
    EXPECT_TRUE(done);
}

TEST(CuWeaverCudaStream, StreamQueryReportsPendingAndCompleted) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    cuweaver::cudaStream stream;

    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1, 0, stream.nativeHandle()>>>(kIterations);
    bool done = true;
    ASSERT_NO_THROW({ done = cuweaver::streamQuery(stream); });
    EXPECT_FALSE(done);
    ASSERT_EQ(cudaStreamSynchronize(stream.nativeHandle()), cudaSuccess);
    ASSERT_NO_THROW({ done = cuweaver::streamQuery(stream); });
    EXPECT_TRUE(done);
}

TEST(CuWeaverCudaStream, StreamSynchronize) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    cuweaver::cudaStream stream;

    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1, 0, stream.nativeHandle()>>>(kIterations);
    ASSERT_FALSE(cuweaver::streamQuery(stream));
    ASSERT_NO_THROW(cuweaver::streamSynchronize(stream));
    ASSERT_TRUE(cuweaver::streamQuery(stream));
}

TEST(CuWeaverCudaStream, StreamAddCallback) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    cuweaver::cudaStream stream;

    ASSERT_THROW(cuweaver::streamAddCallback(stream, nullptr, nullptr, 1), std::invalid_argument);

    int value = 0;
    auto callback = [](cudaStream_t, cudaError_t, void* userData) {
        int* value = static_cast<int*>(userData);
        *value = 42;
    };
    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1, 0, stream.nativeHandle()>>>(kIterations);
    ASSERT_NO_THROW(cuweaver::streamAddCallback(stream, callback, &value, 0));
    ASSERT_NO_THROW(cuweaver::streamSynchronize(stream));
    EXPECT_EQ(value, 42);
}

TEST(CuWeaverCudaStream, StreamWaitEvent) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cuweaver::cudaStream stream1;
    cuweaver::cudaStream stream2;
    cuweaver::cudaEvent event1;
    cuweaver::cudaEvent event2;

    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1, 0, stream1.nativeHandle()>>>(kIterations);
    ASSERT_NO_THROW(cuweaver::eventRecord(event1, stream1));
    ASSERT_NO_THROW(cuweaver::streamWaitEvent(stream2, event1, cuweaver::cudaEventWait::Default));
    ASSERT_NO_THROW(cuweaver::eventRecord(event2, stream2));
    ASSERT_NO_THROW(cuweaver::streamSynchronize(stream2));
    ASSERT_TRUE(cuweaver::eventQuery(event2));
    ASSERT_TRUE(cuweaver::eventQuery(event1));
}

TEST(CuWeaverCudaStream, LaunchHostFunction) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";
    cuweaver::cudaStream stream;

    ASSERT_THROW(cuweaver::streamAddCallback(stream, nullptr, nullptr, 1), std::invalid_argument);

    int value = 0;
    auto callback = [](void* userData) {
        int* value = static_cast<int*>(userData);
        *value = 42;
    };
    constexpr int kIterations = 1 << 24;
    BusyKernel<<<1, 1, 0, stream.nativeHandle()>>>(kIterations);
    ASSERT_NO_THROW(cuweaver::launchHostFunc(stream, callback, &value));
    ASSERT_NO_THROW(cuweaver::streamSynchronize(stream));
    EXPECT_EQ(value, 42);
}
