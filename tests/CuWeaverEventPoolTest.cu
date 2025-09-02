#ifdef __CUDACC__

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cuweaver/EventPool.cuh>

namespace cuweaver {

class EventPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context for testing
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
        
        // Create test pool with moderate size
        pool = std::make_unique<EventPool>(5);
    }

    void TearDown() override {
        pool.reset();
        cudaDeviceReset();
    }

    std::unique_ptr<EventPool> pool;
};

// Basic functionality tests
TEST_F(EventPoolTest, ConstructorInitializesCorrectly) {
    // Test that constructor creates a pool with specified size
    auto testPool = std::make_unique<EventPool>(3);
    ASSERT_NE(testPool, nullptr);
}

TEST_F(EventPoolTest, AcquireSingleEvent) {
    // Test acquiring a single event from the pool
    cudaEvent& event = pool->acquire();
    
    // Verify the event is valid by checking its native handle
    cudaEvent_t handle = event.nativeHandle();
    ASSERT_NE(handle, nullptr);
    
    // Test that the event can be used in CUDA operations
    cudaError_t err = cudaEventRecord(handle, 0);
    EXPECT_EQ(err, cudaSuccess);
}

TEST_F(EventPoolTest, AcquireMultipleEvents) {
    // Test acquiring multiple events from the pool
    std::vector<std::reference_wrapper<cudaEvent>> events;
    
    // Acquire initial pool size events
    for (size_t i = 0; i < 5; ++i) {
        events.emplace_back(pool->acquire());
    }
    
    // Verify all events are unique
    std::set<cudaEvent_t> handles;
    for (auto& eventRef : events) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_NE(handle, nullptr);
        EXPECT_TRUE(handles.insert(handle).second) << "Duplicate event handle found";
    }
}

TEST_F(EventPoolTest, ReleaseValidEvent) {
    // Test releasing a valid event back to the pool
    cudaEvent& event = pool->acquire();
    cudaEvent_t originalHandle = event.nativeHandle();
    
    // Use the event for some CUDA operation
    ASSERT_EQ(cudaEventRecord(originalHandle, 0), cudaSuccess);
    
    // Release the event
    bool result = pool->release(event);
    EXPECT_TRUE(result) << "Failed to release valid event";
}

TEST_F(EventPoolTest, ReleaseInvalidEvent) {
    // Test releasing an event that wasn't acquired from this pool
    cudaEvent externalEvent(cudaEventFlags::DisableTiming);
    
    // Try to release an external event
    bool result = pool->release(externalEvent);
    EXPECT_FALSE(result) << "Should fail to release external event";
}

TEST_F(EventPoolTest, AcquireReleaseAcquireCycle) {
    // Test the complete acquire-release-acquire cycle
    cudaEvent& event1 = pool->acquire();
    cudaEvent_t handle1 = event1.nativeHandle();
    
    // Use the event
    ASSERT_EQ(cudaEventRecord(handle1, 0), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(handle1), cudaSuccess);
    
    // Release the event
    ASSERT_TRUE(pool->release(event1));
    
    // Acquire another event (should potentially reuse the released one)
    cudaEvent& event2 = pool->acquire();
    cudaEvent_t handle2 = event2.nativeHandle();
    ASSERT_NE(handle2, nullptr);
    
    // The handle might be the same (reused) or different (new allocation)
    // Both scenarios are valid depending on pool implementation
}

TEST_F(EventPoolTest, PoolExpansionBeyondInitialSize) {
    // Test that pool can expand beyond initial size
    std::vector<std::reference_wrapper<cudaEvent>> events;
    
    // Acquire more events than initial pool size
    for (size_t i = 0; i < 10; ++i) {
        events.emplace_back(pool->acquire());
    }
    
    // Verify all events are valid and unique
    std::set<cudaEvent_t> handles;
    for (auto& eventRef : events) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_NE(handle, nullptr);
        EXPECT_TRUE(handles.insert(handle).second);
    }
    
    EXPECT_EQ(handles.size(), 10);
}

TEST_F(EventPoolTest, MultipleReleasesSameEvent) {
    // Test releasing the same event multiple times
    cudaEvent& event = pool->acquire();
    
    // First release should succeed
    EXPECT_TRUE(pool->release(event));
    
    // Second release of same event should fail
    EXPECT_FALSE(pool->release(event));
}

// Performance and stress tests
TEST_F(EventPoolTest, HighFrequencyAcquireRelease) {
    // Test high-frequency acquire and release operations
    const int iterations = 1000;
    
    for (int i = 0; i < iterations; ++i) {
        cudaEvent& event = pool->acquire();
        cudaEvent_t handle = event.nativeHandle();
        
        // Simulate some work
        ASSERT_EQ(cudaEventRecord(handle, 0), cudaSuccess);
        
        // Release immediately
        ASSERT_TRUE(pool->release(event));
    }
}

TEST_F(EventPoolTest, BatchAcquireAndRelease) {
    // Test acquiring a batch of events and then releasing them
    const size_t batchSize = 50;
    std::vector<std::reference_wrapper<cudaEvent>> events;

    // Acquire batch
    for (size_t i = 0; i < batchSize; ++i) {
        events.emplace_back(pool->acquire());
    }
    
    // Use all events
    for (auto& eventRef : events) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_EQ(cudaEventRecord(handle, 0), cudaSuccess);
    }
    
    // Wait for all events
    for (auto& eventRef : events) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_EQ(cudaEventSynchronize(handle), cudaSuccess);
    }
    
    // Release all events
    for (auto& eventRef : events) {
        EXPECT_TRUE(pool->release(eventRef.get()));
    }
}

// Real-world simulation tests
TEST_F(EventPoolTest, StreamSynchronizationSimulation) {
    // Simulate real-world stream synchronization scenario
    cudaStream_t stream1, stream2;
    ASSERT_EQ(cudaStreamCreate(&stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream2), cudaSuccess);
    
    // Acquire events for synchronization
    cudaEvent& event1 = pool->acquire();
    cudaEvent& event2 = pool->acquire();
    
    cudaEvent_t handle1 = event1.nativeHandle();
    cudaEvent_t handle2 = event2.nativeHandle();
    
    // Record events on different streams
    ASSERT_EQ(cudaEventRecord(handle1, stream1), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(handle2, stream2), cudaSuccess);
    
    // Synchronize streams using events
    ASSERT_EQ(cudaStreamWaitEvent(stream2, handle1, 0), cudaSuccess);
    ASSERT_EQ(cudaStreamWaitEvent(stream1, handle2, 0), cudaSuccess);
    
    // Synchronize and cleanup
    ASSERT_EQ(cudaStreamSynchronize(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream2), cudaSuccess);
    
    // Release events
    EXPECT_TRUE(pool->release(event1));
    EXPECT_TRUE(pool->release(event2));
    
    // Cleanup streams
    ASSERT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream2), cudaSuccess);
}

TEST_F(EventPoolTest, ResourceExhaustionAndRecovery) {
    // Test behavior when exhausting and recovering pool resources
    std::vector<std::reference_wrapper<cudaEvent>> events;
    
    // Acquire many events to potentially exhaust initial pool
    for (size_t i = 0; i < 100; ++i) {
        events.emplace_back(pool->acquire());
    }
    
    // Release half of them
    for (size_t i = 0; i < 50; ++i) {
        EXPECT_TRUE(pool->release(events[i].get()));
    }
    
    // Acquire new events (should reuse released ones)
    std::vector<std::reference_wrapper<cudaEvent>> newEvents;
    for (size_t i = 0; i < 25; ++i) {
        newEvents.emplace_back(pool->acquire());
    }
    
    // Verify new events are valid
    for (auto& eventRef : newEvents) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_NE(handle, nullptr);
        ASSERT_EQ(cudaEventRecord(handle, 0), cudaSuccess);
    }
    
    // Release remaining events
    for (size_t i = 50; i < 100; ++i) {
        EXPECT_TRUE(pool->release(events[i].get()));
    }
    for (auto& eventRef : newEvents) {
        EXPECT_TRUE(pool->release(eventRef.get()));
    }
}

// Edge cases and error handling
TEST_F(EventPoolTest, ZeroSizePool) {
    // Test creating a pool with zero initial size
    auto zeroPool = std::make_unique<EventPool>(0);
    
    // Should still be able to acquire events (pool should expand)
    cudaEvent& event = zeroPool->acquire();
    cudaEvent_t handle = event.nativeHandle();
    ASSERT_NE(handle, nullptr);
    
    EXPECT_TRUE(zeroPool->release(event));
}

TEST_F(EventPoolTest, LargeBatchOperations) {
    // Test with larger batch sizes to stress test the implementation
    const size_t largeBatch = 500;
    std::vector<std::reference_wrapper<cudaEvent>> events;
    
    // Acquire large batch
    for (size_t i = 0; i < largeBatch; ++i) {
        events.emplace_back(pool->acquire());
    }
    
    // Verify all are unique and valid
    std::set<cudaEvent_t> handles;
    for (auto& eventRef : events) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_NE(handle, nullptr);
        EXPECT_TRUE(handles.insert(handle).second);
    }
    
    // Release in reverse order
    for (auto it = events.rbegin(); it != events.rend(); ++it) {
        EXPECT_TRUE(pool->release(it->get()));
    }
}

    // Move semantics tests
TEST_F(EventPoolTest, MoveConstructor) {
    // Test move constructor
    auto originalPool = std::make_unique<EventPool>(3);

    // Acquire some events from original pool
    cudaEvent& event1 = originalPool->acquire();
    cudaEvent& event2 = originalPool->acquire();

    // Move construct new pool
    EventPool movedPool = std::move(*originalPool);

    // Original pool should be in valid but unspecified state
    // Moved pool should have the resources

    // Test that moved pool can still acquire events
    cudaEvent& event3 = movedPool.acquire();
    cudaEvent_t handle3 = event3.nativeHandle();
    ASSERT_NE(handle3, nullptr);
    ASSERT_EQ(cudaEventRecord(handle3, 0), cudaSuccess);

    // Release events through moved pool
    EXPECT_TRUE(movedPool.release(event1));
    EXPECT_TRUE(movedPool.release(event2));
    EXPECT_TRUE(movedPool.release(event3));
}

TEST_F(EventPoolTest, MoveAssignment) {
    // Test move assignment operator
    auto sourcePool = std::make_unique<EventPool>(4);
    auto targetPool = std::make_unique<EventPool>(2);

    // Acquire events from both pools
    cudaEvent& sourceEvent = sourcePool->acquire();
    cudaEvent& targetEvent = targetPool->acquire();

    // Store handles for later verification
    cudaEvent_t sourceHandle = sourceEvent.nativeHandle();
    cudaEvent_t targetHandle = targetEvent.nativeHandle();

    // Move assign source to target
    *targetPool = std::move(*sourcePool);

    // Target pool should now have source pool's resources
    // Test that target pool can acquire new events
    cudaEvent& newEvent = targetPool->acquire();
    cudaEvent_t newHandle = newEvent.nativeHandle();
    ASSERT_NE(newHandle, nullptr);

    // Should be able to release the original source event through target pool
    EXPECT_TRUE(targetPool->release(sourceEvent));
    EXPECT_TRUE(targetPool->release(newEvent));
}

TEST_F(EventPoolTest, SelfMoveAssignment) {
    // Test self move assignment (should be safe)
    cudaEvent& event = pool->acquire();
    cudaEvent_t originalHandle = event.nativeHandle();

    // Self move assignment
    *pool = std::move(*pool);

    // Pool should still be functional
    cudaEvent& newEvent = pool->acquire();
    ASSERT_NE(newEvent.nativeHandle(), nullptr);

    // Original event should still be manageable
    EXPECT_TRUE(pool->release(event));
    EXPECT_TRUE(pool->release(newEvent));
}

TEST_F(EventPoolTest, MoveConstructorWithActiveEvents) {
    // Test move constructor when pool has active (unreleased) events
    auto originalPool = std::make_unique<EventPool>(5);

    // Acquire multiple events and use them
    std::vector<std::reference_wrapper<cudaEvent>> events;
    for (int i = 0; i < 3; ++i) {
        events.emplace_back(originalPool->acquire());
        cudaEvent_t handle = events.back().get().nativeHandle();
        ASSERT_EQ(cudaEventRecord(handle, 0), cudaSuccess);
    }

    // Move construct while events are still active
    EventPool movedPool = std::move(*originalPool);

    // Synchronize all events
    for (auto& eventRef : events) {
        cudaEvent_t handle = eventRef.get().nativeHandle();
        ASSERT_EQ(cudaEventSynchronize(handle), cudaSuccess);
    }

    // Should be able to release events through moved pool
    for (auto& eventRef : events) {
        EXPECT_TRUE(movedPool.release(eventRef.get()));
    }

    // Moved pool should continue to work normally
    cudaEvent& newEvent = movedPool.acquire();
    ASSERT_NE(newEvent.nativeHandle(), nullptr);
    EXPECT_TRUE(movedPool.release(newEvent));
}

TEST_F(EventPoolTest, ChainedMoveOperations) {
    // Test chained move operations
    auto pool1 = std::make_unique<EventPool>(2);
    auto pool2 = std::make_unique<EventPool>(3);
    auto pool3 = std::make_unique<EventPool>(4);

    // Acquire events from each pool
    cudaEvent& event1 = pool1->acquire();
    cudaEvent& event2 = pool2->acquire();
    cudaEvent& event3 = pool3->acquire();

    // Chain move assignments: pool3 = pool2 = pool1
    *pool2 = std::move(*pool1);
    *pool3 = std::move(*pool2);

    // pool3 should now have pool1's original resources
    EXPECT_TRUE(pool3->release(event1));

    // pool3 should still be functional
    cudaEvent& newEvent = pool3->acquire();
    ASSERT_NE(newEvent.nativeHandle(), nullptr);
    EXPECT_TRUE(pool3->release(newEvent));
}

TEST_F(EventPoolTest, MoveAfterPoolExpansion) {
    // Test move operations after pool has expanded beyond initial size
    auto originalPool = std::make_unique<EventPool>(2);

    // Expand pool by acquiring more events than initial size
    std::vector<std::reference_wrapper<cudaEvent>> events;
    for (int i = 0; i < 10; ++i) {
        events.emplace_back(originalPool->acquire());
    }

    // Release some events to create a mixed state
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(originalPool->release(events[i].get()));
    }

    // Move construct from expanded pool
    EventPool movedPool = std::move(*originalPool);

    // Should be able to manage remaining active events
    for (int i = 5; i < 10; ++i) {
        EXPECT_TRUE(movedPool.release(events[i].get()));
    }

    // Moved pool should continue to work with its expanded capacity
    cudaEvent& newEvent = movedPool.acquire();
    ASSERT_NE(newEvent.nativeHandle(), nullptr);
    EXPECT_TRUE(movedPool.release(newEvent));
}

}

#endif // __CUDACC__