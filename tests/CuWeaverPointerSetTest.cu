#include <gtest/gtest.h>
#include "cuweaver_utils/PointerSet.cuh"

#include <vector>
#include <algorithm>

namespace {
  using cuweaver::detail::PointerSet;

  struct Foo {
    int v;
    explicit Foo(int x) : v(x) {}
  };
} // namespace

TEST(PointerSet_Basic, InsertAndSize) {
  PointerSet<Foo> s;
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0u);

  auto* a = new Foo(1);
  auto* b = new Foo(2);

  EXPECT_TRUE(s.insert(a));
  EXPECT_TRUE(s.insert(b));
  EXPECT_FALSE(s.empty());
  EXPECT_EQ(s.size(), 2u);

  EXPECT_FALSE(s.insert(a));
  EXPECT_EQ(s.size(), 2u);

  delete a;
  delete b;
}

TEST(PointerSet_Basic, ContainsAndErase) {
  PointerSet<Foo> s;
  auto* a = new Foo(10);
  auto* b = new Foo(20);
  auto* c = new Foo(30);

  ASSERT_TRUE(s.insert(a));
  ASSERT_TRUE(s.insert(b));
  ASSERT_TRUE(s.insert(c));

  EXPECT_TRUE(s.contains(a));
  EXPECT_TRUE(s.contains(b));
  EXPECT_TRUE(s.contains(c));

  Foo* fake = reinterpret_cast<Foo*>(0x1);
  EXPECT_FALSE(s.erase(fake));
  EXPECT_EQ(s.size(), 3u);

  EXPECT_TRUE(s.erase(b));
  EXPECT_EQ(s.size(), 2u);
  EXPECT_FALSE(s.contains(b));
  EXPECT_TRUE(s.contains(a));
  EXPECT_TRUE(s.contains(c));

  EXPECT_TRUE(s.erase(a));
  EXPECT_TRUE(s.erase(c));
  EXPECT_TRUE(s.empty());

  delete a; delete b; delete c;
}

TEST(PointerSet_Basic, Clear) {
  PointerSet<Foo> s;
  auto* a = new Foo(1);
  auto* b = new Foo(2);

  ASSERT_TRUE(s.insert(a));
  ASSERT_TRUE(s.insert(b));
  EXPECT_EQ(s.size(), 2u);

  s.clear();
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0u);

  EXPECT_TRUE(s.insert(a));
  EXPECT_TRUE(s.insert(b));
  EXPECT_EQ(s.size(), 2u);

  delete a; delete b;
}

TEST(PointerSet_Basic, ReserveDoesNotChangeSemantics) {
  PointerSet<Foo> s;
  s.reserve(128);
  EXPECT_TRUE(s.empty());
  EXPECT_EQ(s.size(), 0u);

  auto* a = new Foo(7);
  EXPECT_TRUE(s.insert(a));
  EXPECT_TRUE(s.contains(a));
  EXPECT_EQ(s.size(), 1u);

  delete a;
}

TEST(PointerSet_Iteration, RangeForCoversAllElements) {
  PointerSet<Foo> s;
  std::vector<Foo*> ptrs;
  for (int i = 0; i < 10; ++i) {
    auto* p = new Foo(i);
    ptrs.push_back(p);
    ASSERT_TRUE(s.insert(p));
  }
  EXPECT_EQ(s.size(), ptrs.size());

  size_t seen = 0;
  for (Foo* p : s) {
    ASSERT_NE(p, nullptr);
    EXPECT_NE(std::find(ptrs.begin(), ptrs.end(), p), ptrs.end());
    ++seen;
  }
  EXPECT_EQ(seen, ptrs.size());

  ASSERT_TRUE(s.erase(ptrs[3]));
  ASSERT_TRUE(s.erase(ptrs[7]));

  std::vector<Foo*> alive;
  for (size_t i = 0; i < ptrs.size(); ++i) {
    if (i == 3 || i == 7) continue;
    alive.push_back(ptrs[i]);
  }
  EXPECT_EQ(s.size(), alive.size());

  for (Foo* p : s) {
    EXPECT_NE(std::find(alive.begin(), alive.end(), p), alive.end());
  }

  for (auto* p : ptrs) delete p;
}

TEST(PointerSet_Const, ConstContainsAndIteration) {
  PointerSet<Foo> s;
  auto* a = new Foo(1);
  auto* b = new Foo(2);
  ASSERT_TRUE(s.insert(a));
  ASSERT_TRUE(s.insert(b));

  const PointerSet<Foo>& cs = s;
  EXPECT_TRUE(cs.contains(a));
  EXPECT_TRUE(cs.contains(b));
  EXPECT_EQ(cs.size(), 2u);
  EXPECT_FALSE(cs.empty());

  size_t cnt = 0;
  for (Foo* p : cs) {
    (void)p;
    ++cnt;
  }
  EXPECT_EQ(cnt, 2u);

  delete a; delete b;
}
