#include <gtest/gtest.h>
#include <stdexcept>
#include <cuweaver_utils/CircularList.cuh>

namespace cuweaver::detail {

    // 自定义的类，用于测试
    class TestData {
    public:
        explicit TestData(int val) : value(val) {}

        bool operator==(const TestData& other) const {
            return value == other.value;
        }

        int value;
    };

    TEST(CuWeaverCircularListTest, AddElementTest) {
        CircularList<TestData> list;
        list.add(1);
        list.add(2);
        list.add(3);

        EXPECT_EQ(list.getSize(), 3);
    }

    TEST(CuWeaverCircularListTest, GetElementTest) {
        CircularList<TestData> list;
        list.add(10);
        list.add(20);

        EXPECT_EQ(list.get().value, 10);
        list.next();
        EXPECT_EQ(list.get().value, 20);
    }

    TEST(CuWeaverCircularListTest, NextElementTest) {
        CircularList<TestData> list;
        list.add(1);
        list.add(2);
        list.add(3);

        list.next();
        EXPECT_EQ(list.get().value, 3);
        list.next();
        EXPECT_EQ(list.get().value, 2);
        list.next();
        EXPECT_EQ(list.get().value, 1);
    }

    TEST(CuWeaverCircularListTest, IsEmptyTest) {
        CircularList<TestData> list;
        EXPECT_TRUE(list.isEmpty());

        list.add(5);
        EXPECT_FALSE(list.isEmpty());
    }

    TEST(CuWeaverCircularListTest, ContainsTest) {
        CircularList<TestData> list;
        list.add(1);
        list.add(2);
        list.add(3);

        EXPECT_TRUE(list.contains(TestData(2)));
        EXPECT_FALSE(list.contains(TestData(4)));
    }

    TEST(CuWeaverCircularListTest, ClearTest) {
        CircularList<TestData> list;
        list.add(1);
        list.add(2);
        list.add(3);

        EXPECT_EQ(list.getSize(), 3);

        list.clear();
        EXPECT_EQ(list.getSize(), 0);
        EXPECT_TRUE(list.isEmpty());
    }

    TEST(CuWeaverCircularListTest, GetNextThrowsWhenEmpty) {
        CircularList<TestData> list;
        EXPECT_THROW(list.getNext(), std::runtime_error);
    }

    TEST(CuWeaverCircularListTest, ForEachTest) {
        CircularList<TestData> list;
        list.add(10);
        list.add(20);
        list.add(30);

        int sum = 0;
        list.forEach([&sum](const TestData& data) {
            sum += data.value;
        });

        EXPECT_EQ(sum, 60);
    }
}
