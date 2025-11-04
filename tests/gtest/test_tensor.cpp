// Google Test main header
#include <gtest/gtest.h>

// What we're testing
#include "tensor.h"

// TEST(TestSuiteName, TestName) - Google Test's macro
// Convention: TestSuiteName describes the class/feature, TestName describes the scenario

TEST(TensorTest, CreateWithShapeOnly) {
    Tensor t({2, 3});
    
    // EXPECT_EQ: check equality, continues if fails
    // ASSERT_EQ: check equality, stops test if fails
    EXPECT_EQ(t.get_shape().size(), 2);
    EXPECT_EQ(t.get_shape()[0], 2);
    EXPECT_EQ(t.get_shape()[1], 3);
    EXPECT_EQ(t.get_total_size(), 6);
    
    // All elements should be zero
    EXPECT_FLOAT_EQ(t.at(0, 0), 0.0f);  // FLOAT_EQ handles floating point comparison
    EXPECT_FLOAT_EQ(t.at(1, 2), 0.0f);
}

TEST(TensorTest, CreateWithShapeAndData) {
    Tensor t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    EXPECT_EQ(t.get_total_size(), 6);
    
    // Check specific elements
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(t.at(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(t.at(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(t.at(1, 2), 6.0f);
}

TEST(TensorTest, ElementAccessAndModification) {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Read element
    EXPECT_FLOAT_EQ(t.at(0, 1), 2.0f);
    
    // Modify element
    t.set(0, 1, 99.5f);
    EXPECT_FLOAT_EQ(t.at(0, 1), 99.5f);
    
    // Other elements unchanged
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 3.0f);
}

TEST(TensorTest, BoundsChecking) {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // EXPECT_THROW: verifies that an exception is thrown
    EXPECT_THROW(t.at(2, 0), std::runtime_error);   // Row out of bounds
    EXPECT_THROW(t.at(0, 2), std::runtime_error);   // Column out of bounds
    EXPECT_THROW(t.at(-1, 0), std::runtime_error);  // Negative index
}

TEST(TensorTest, DataSizeValidation) {
    // Should throw if data size doesn't match shape
    EXPECT_THROW(Tensor({2, 3}, {1.0, 2.0}), std::runtime_error);  // Too few
    EXPECT_THROW(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0}), std::runtime_error);  // Too many
}

TEST(TensorTest, DifferentShapes) {
    // 1x1 tensor
    Tensor t1({1, 1}, {42.0});
    EXPECT_FLOAT_EQ(t1.at(0, 0), 42.0f);
    
    // 4x2 tensor
    Tensor t2({4, 2});
    EXPECT_EQ(t2.get_total_size(), 8);
    EXPECT_FLOAT_EQ(t2.at(3, 1), 0.0f);
}

// TEST_F: Fixture tests (for setup/teardown shared across tests)
// We'll use this pattern later when tests need common initialization
class TensorFixture : public ::testing::Test {
protected:
    // SetUp() runs before each test
    void SetUp() override {
        // We'll use this later for complex setups
    }
    
    // TearDown() runs after each test
    void TearDown() override {
        // Cleanup if needed
    }
};

// Example of using a fixture (we'll use this pattern more later)
TEST_F(TensorFixture, ExampleFixtureTest) {
    // This test has access to anything set up in SetUp()
    SUCCEED();  // Placeholder - always passes
}