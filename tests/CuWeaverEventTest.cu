#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "cuweaver/Event.cuh"

using cuweaver::cudaEvent;

namespace {

inline bool cudaAvailable() {
    int count = 0;
    auto st = cudaGetDeviceCount(&count);
    return (st == cudaSuccess) && (count > 0);
}

inline void recordAndSync(cudaEvent_t ev) {
    ASSERT_NE(ev, nullptr);
    ASSERT_EQ(cudaEventRecord(ev, /*stream=*/0), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(ev), cudaSuccess);
}

} // namespace

TEST(CuWeaverCudaEvent, DefaultConstructorCreatesValidEvent) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaEvent e; // 默认构造
    EXPECT_TRUE(e.isValid());
    EXPECT_NE(e.nativeHandle(), nullptr);
    EXPECT_EQ(e.getFlags(), static_cast<cudaEvent::cudaEventFlags_t>(cuweaver::cudaEventFlags::Default));

    recordAndSync(e.nativeHandle());
}

TEST(CuWeaverCudaEvent, EnumFlagsConstructorDisableTiming) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaEvent e{cuweaver::cudaEventFlags::DisableTiming};
    EXPECT_TRUE(e.isValid());
    EXPECT_NE(e.nativeHandle(), nullptr);
    EXPECT_EQ(e.getFlags(), static_cast<unsigned int>(cuweaver::cudaEventFlags::DisableTiming));

    recordAndSync(e.nativeHandle());
}

TEST(CuWeaverCudaEvent, RawFlagsConstructor) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    unsigned int rawFlags = static_cast<unsigned int>(cuweaver::cudaEventFlags::BlockingSync) |
                            static_cast<unsigned int>(cuweaver::cudaEventFlags::DisableTiming);
    cudaEvent e{rawFlags};
    EXPECT_TRUE(e.isValid());
    EXPECT_NE(e.nativeHandle(), nullptr);
    EXPECT_EQ(e.getFlags(), rawFlags);

    recordAndSync(e.nativeHandle());
}

TEST(CuWeaverCudaEvent, AdoptExistingNativeHandleConstructor) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaEvent_t raw = nullptr;
    unsigned int rawFlags = static_cast<unsigned int>(cuweaver::cudaEventFlags::DisableTiming);
    ASSERT_EQ(cudaEventCreateWithFlags(&raw, rawFlags), cudaSuccess);
    ASSERT_NE(raw, nullptr);

    {
        cudaEvent e{raw};
        EXPECT_TRUE(e.isValid());
        EXPECT_EQ(e.nativeHandle(), raw);

        EXPECT_EQ(e.getFlags(), static_cast<unsigned int>(cuweaver::cudaEventFlags::Default));

        recordAndSync(e.nativeHandle());
    }

}


TEST(CuWeaverCudaEvent, MoveConstructorTransfersOwnership) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaEvent e1;
    ASSERT_TRUE(e1.isValid());
    auto h = e1.nativeHandle();

    cudaEvent e2{std::move(e1)};
    EXPECT_FALSE(e1.isValid());
    EXPECT_TRUE(e2.isValid());
    EXPECT_EQ(e2.nativeHandle(), h);

    recordAndSync(e2.nativeHandle());
}

TEST(CuWeaverCudaEvent, MoveAssignmentTransfersOwnership) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaEvent src;
    ASSERT_TRUE(src.isValid());
    auto hsrc = src.nativeHandle();

    cudaEvent dst{cuweaver::cudaEventFlags::DisableTiming};
    ASSERT_TRUE(dst.isValid());

    dst = std::move(src);
    EXPECT_FALSE(src.isValid());
    EXPECT_TRUE(dst.isValid());
    EXPECT_EQ(dst.nativeHandle(), hsrc);

    recordAndSync(dst.nativeHandle());
}

// -------------------- reset / isValid --------------------

TEST(CuWeaverCudaEvent, ResetToNewHandleAndNullptr) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    cudaEvent e;
    EXPECT_TRUE(e.isValid());
    auto old = e.nativeHandle();

    cudaEvent_t nraw = nullptr;
    ASSERT_EQ(cudaEventCreate(&nraw), cudaSuccess);
    ASSERT_NE(nraw, nullptr);

    e.reset(nraw);
    EXPECT_TRUE(e.isValid());
    EXPECT_EQ(e.nativeHandle(), nraw);
    recordAndSync(e.nativeHandle());

    e.reset(nullptr);
    EXPECT_FALSE(e.isValid());
    EXPECT_EQ(e.nativeHandle(), static_cast<cudaEvent_t>(nullptr));

    (void)old;
}

TEST(CuWeaverCudaEvent, NativeHandleConstCorrectness) {
#ifndef __CUDACC__
    GTEST_SKIP() << "Not compiled with CUDA (__CUDACC__ not defined).";
#endif
    if (!cudaAvailable()) GTEST_SKIP() << "No CUDA device available.";

    const cudaEvent e;
    EXPECT_TRUE(e.isValid());
    EXPECT_NE(e.nativeHandle(), nullptr);
}

