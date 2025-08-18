#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cuweaver_utils/MemEventMap.cuh>

using cuweaver::detail::MemEventMap;

class MemEventMapCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_EQ(cudaFree(0), cudaSuccess);

        ASSERT_EQ(cudaEventCreateWithFlags(&ev1, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&ev2, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&ev3, cudaEventDisableTiming), cudaSuccess);

        memA = &dummyA;
        memB = &dummyB;
    }

    void TearDown() override {
        EXPECT_EQ(cudaEventDestroy(ev1), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(ev2), cudaSuccess);
        EXPECT_EQ(cudaEventDestroy(ev3), cudaSuccess);
    }

    MemEventMap map;

    cudaEvent_t ev1{}, ev2{}, ev3{};
    int dummyA{0}, dummyB{0};
    void* memA{nullptr};
    void* memB{nullptr};
};

TEST_F(MemEventMapCudaTest, HasMem_FalseBeforeRecord_TrueAfterRecord) {
    EXPECT_FALSE(map.hasMem(memA));
    EXPECT_FALSE(map.hasMem(memB));

    (void)map.recordEvent(memA, ev1);
    EXPECT_TRUE(map.hasMem(memA));
    EXPECT_FALSE(map.hasMem(memB));

    (void)map.recordEvent(memB, ev2);
    EXPECT_TRUE(map.hasMem(memB));
}

TEST_F(MemEventMapCudaTest, HasEvent_OnlyForRecordedOnes) {
    EXPECT_FALSE(map.hasEvent(memA, ev1));
    EXPECT_FALSE(map.hasEvent(memA, ev2));

    (void)map.recordEvent(memA, ev1);
    EXPECT_TRUE(map.hasEvent(memA, ev1));
    EXPECT_FALSE(map.hasEvent(memA, ev2));

    (void)map.recordEvent(memA, ev2);
    EXPECT_TRUE(map.hasEvent(memA, ev2));
    EXPECT_FALSE(map.hasEvent(memA, ev3));

    EXPECT_FALSE(map.hasEvent(memB, ev1));
    (void)map.recordEvent(memB, ev3);
    EXPECT_TRUE(map.hasEvent(memB, ev3));
}

TEST_F(MemEventMapCudaTest, DuplicateRecord_IsIdempotent) {
    (void)map.recordEvent(memA, ev1);
    EXPECT_TRUE(map.hasEvent(memA, ev1));

    (void)map.recordEvent(memA, ev1);
    EXPECT_TRUE(map.hasEvent(memA, ev1));

    (void)map.recordEvent(memA, ev2);
    EXPECT_TRUE(map.hasEvent(memA, ev2));
    EXPECT_TRUE(map.hasEvent(memA, ev1));
}

TEST_F(MemEventMapCudaTest, MultipleKeys_AreIndependent) {
    (void)map.recordEvent(memA, ev1);
    (void)map.recordEvent(memB, ev2);

    EXPECT_TRUE(map.hasMem(memA));
    EXPECT_TRUE(map.hasMem(memB));

    EXPECT_TRUE(map.hasEvent(memA, ev1));
    EXPECT_FALSE(map.hasEvent(memA, ev2));

    EXPECT_TRUE(map.hasEvent(memB, ev2));
    EXPECT_FALSE(map.hasEvent(memB, ev1));
}
