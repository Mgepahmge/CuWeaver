#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuweaver/Stream.cuh"

using cuweaver::cudaStream;

namespace {

inline bool cudaAvailable() {
    int count = 0;
    auto st = cudaGetDeviceCount(&count);
    return (st == cudaSuccess) && (count > 0);
}

__global__ void NopKernel() {}

inline void launchAndSync(cudaStream_t s) {
    ASSERT_NE(s, nullptr);
    NopKernel<<<1, 1, 0, s>>>();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
}

} // namespace

TEST(CuWeaverCudaStream, PriorityRangeStaticsAreConsistent) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    const int least = cudaStream::getLeastPriority();
    const int greatest = cudaStream::getGreatestPriority();

    EXPECT_LE(greatest, least);

    EXPECT_TRUE(cudaStream::isPriorityValid(greatest));
    EXPECT_TRUE(cudaStream::isPriorityValid(least));
    EXPECT_FALSE(cudaStream::isPriorityValid(least + 1));
    EXPECT_FALSE(cudaStream::isPriorityValid(greatest - 1));
}

TEST(CuWeaverCudaStream, DefaultConstructorCreatesValidStream) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaStream s;
    EXPECT_TRUE(s.isValid());
    EXPECT_NE(s.nativeHandle(), nullptr);
    EXPECT_EQ(s.getFlags(), static_cast<cudaStream::cudaStreamFlags_t>(cuweaver::cudaStreamFlags::Default));
    EXPECT_EQ(s.getPriority(), cudaStream::DefaultPriority);

    launchAndSync(s.nativeHandle());
}

TEST(CuWeaverCudaStream, EnumFlagsAndPriorityConstructor) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    const int hi = cudaStream::getGreatestPriority();

    cudaStream s{cuweaver::cudaStreamFlags::NonBlocking, hi};
    EXPECT_TRUE(s.isValid());
    EXPECT_NE(s.nativeHandle(), nullptr);
    EXPECT_EQ(s.getFlags(), static_cast<unsigned int>(cuweaver::cudaStreamFlags::NonBlocking));
    EXPECT_EQ(s.getPriority(), hi);

    launchAndSync(s.nativeHandle());
}

TEST(CuWeaverCudaStream, RawFlagsAndPriorityConstructorWithInvalidPriorityFallsBackToDefault) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    const int invalidPriority = cudaStream::getGreatestPriority() - 10;
    unsigned int rawFlags = static_cast<unsigned int>(cuweaver::cudaStreamFlags::NonBlocking);

    cudaStream s{rawFlags, invalidPriority};
    EXPECT_TRUE(s.isValid());
    EXPECT_NE(s.nativeHandle(), nullptr);
    EXPECT_EQ(s.getFlags(), rawFlags);
    EXPECT_EQ(s.getPriority(), cudaStream::DefaultPriority);

    launchAndSync(s.nativeHandle());
}

TEST(CuWeaverCudaStream, AdoptExistingNativeHandleConstructor) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaStream_t raw = nullptr;
    ASSERT_EQ(cudaStreamCreateWithPriority(&raw,
              static_cast<unsigned int>(cuweaver::cudaStreamFlags::NonBlocking),
              cudaStream::getGreatestPriority()), cudaSuccess);
    ASSERT_NE(raw, nullptr);

    {
        cudaStream s{raw};
        EXPECT_TRUE(s.isValid());
        EXPECT_EQ(s.nativeHandle(), raw);

        EXPECT_EQ(s.getFlags(), static_cast<unsigned int>(cuweaver::cudaStreamFlags::NonBlocking));
        EXPECT_EQ(s.getPriority(), cudaStream::getGreatestPriority());
        cudaStream::cudaStreamId_t id;
        EXPECT_EQ(cudaStreamGetId(raw, &id), cudaSuccess);
        EXPECT_EQ(s.getId(), id);

        launchAndSync(s.nativeHandle());
    }

}

TEST(CuWeaverCudaStream, MoveConstructorTransfersOwnership) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaStream s1{cuweaver::cudaStreamFlags::NonBlocking, cudaStream::DefaultPriority};
    ASSERT_TRUE(s1.isValid());
    auto h = s1.nativeHandle();

    cudaStream s2{std::move(s1)};
    EXPECT_FALSE(s1.isValid());
    EXPECT_TRUE(s2.isValid());
    EXPECT_EQ(s2.nativeHandle(), h);

    launchAndSync(s2.nativeHandle());
}

TEST(CuWeaverCudaStream, MoveAssignmentTransfersOwnership) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaStream src;
    ASSERT_TRUE(src.isValid());
    auto hsrc = src.nativeHandle();

    cudaStream dst{cuweaver::cudaStreamFlags::NonBlocking, cudaStream::DefaultPriority};
    ASSERT_TRUE(dst.isValid());

    dst = std::move(src);
    EXPECT_FALSE(src.isValid());
    EXPECT_TRUE(dst.isValid());
    EXPECT_EQ(dst.nativeHandle(), hsrc);

    launchAndSync(dst.nativeHandle());
}

TEST(CuWeaverCudaStream, ResetToNewHandleAndNullptr) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaStream s;
    ASSERT_TRUE(s.isValid());
    auto old = s.nativeHandle();

    cudaStream_t nraw = nullptr;
    ASSERT_EQ(cudaStreamCreateWithPriority(&nraw,
              static_cast<unsigned int>(cuweaver::cudaStreamFlags::Default),
              cudaStream::DefaultPriority), cudaSuccess);
    ASSERT_NE(nraw, nullptr);

    s.reset(nraw);
    EXPECT_TRUE(s.isValid());
    EXPECT_EQ(s.nativeHandle(), nraw);
    launchAndSync(s.nativeHandle());

    s.reset(nullptr);
    EXPECT_FALSE(s.isValid());
    EXPECT_EQ(s.nativeHandle(), static_cast<cudaStream_t>(nullptr));

    (void)old;
}

TEST(CuWeaverCudaStream, DefaultStreamReturnsValidDefaultStream) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaStream s = cudaStream::defaultStream();

    EXPECT_FALSE(s.isValid());
    EXPECT_EQ(s.nativeHandle(), static_cast<cudaStream_t>(nullptr));
    EXPECT_EQ(s.getFlags(), static_cast<cudaStream::cudaStreamFlags_t>(cuweaver::cudaStreamFlags::Default));
    EXPECT_EQ(s.getPriority(), cudaStream::DefaultPriority);

    NopKernel<<<1, 1, 0, s.nativeHandle()>>>();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

