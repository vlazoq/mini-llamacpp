// Google Test main header
#include <gtest/gtest.h>

// What we're testing
#include "tensor.h"

// ===== BASIC TENSOR CREATION TESTS =====

// TEST(TestSuiteName, TestName) - Google Test's macro for defining tests
// TestSuiteName: groups related tests (like "TensorTest")
// TestName: describes what this specific test checks
TEST(TensorTest, CreateWithShapeOnly) {
    // Create a 2x3 tensor filled with zeros
    Tensor t({2, 3});
    
    // EXPECT_EQ(actual, expected): checks equality, continues if fails
    // ASSERT_EQ would stop the test immediately if it fails
    EXPECT_EQ(t.get_shape().size(), 2);     // 2D tensor
    EXPECT_EQ(t.get_shape()[0], 2);         // 2 rows
    EXPECT_EQ(t.get_shape()[1], 3);         // 3 columns
    EXPECT_EQ(t.get_total_size(), 6);       // 2*3 = 6 elements
    
    // EXPECT_FLOAT_EQ: proper floating-point comparison
    // Don't use == for floats due to precision issues
    EXPECT_FLOAT_EQ(t.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t.at(1, 2), 0.0f);
}

TEST(TensorTest, CreateWithShapeAndData) {
    // Create tensor with explicit data
    Tensor t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    EXPECT_EQ(t.get_total_size(), 6);
    
    // Check all elements in the matrix:
    // [1.0, 2.0, 3.0]
    // [4.0, 5.0, 6.0]
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(t.at(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(t.at(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(t.at(1, 2), 6.0f);
}

// ===== ELEMENT ACCESS TESTS =====

TEST(TensorTest, ElementAccessAndModification) {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Test read access
    EXPECT_FLOAT_EQ(t.at(0, 1), 2.0f);
    
    // Test write access
    t.set(0, 1, 99.5f);
    EXPECT_FLOAT_EQ(t.at(0, 1), 99.5f);
    
    // Verify other elements unchanged
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(t.at(1, 0), 3.0f);
}

// ===== ERROR HANDLING TESTS =====

TEST(TensorTest, BoundsChecking) {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // EXPECT_THROW(statement, exception_type)
    // Verifies that the statement throws the specified exception
    EXPECT_THROW(t.at(2, 0), std::runtime_error);    // Row out of bounds
    EXPECT_THROW(t.at(0, 2), std::runtime_error);    // Column out of bounds
    EXPECT_THROW(t.at(-1, 0), std::runtime_error);   // Negative index
}

TEST(TensorTest, DataSizeValidation) {
    // These should throw because data size doesn't match shape requirements
    EXPECT_THROW(Tensor({2, 3}, {1.0, 2.0}), std::runtime_error);                    // Too few
    EXPECT_THROW(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0}), std::runtime_error);    // Too many
}

// ===== EDGE CASES =====

TEST(TensorTest, DifferentShapes) {
    // Smallest possible tensor: 1x1
    Tensor t1({1, 1}, {42.0});
    EXPECT_FLOAT_EQ(t1.at(0, 0), 42.0f);
    
    // Larger tensor: 4x2
    Tensor t2({4, 2});
    EXPECT_EQ(t2.get_total_size(), 8);
    EXPECT_FLOAT_EQ(t2.at(3, 1), 0.0f);
}

// ===== ELEMENT-WISE OPERATIONS TESTS =====
// Group related operations in a new test suite

TEST(TensorOperations, ElementWiseAddition) {
    // Setup: create two tensors to add
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({2, 2}, {5.0, 6.0, 7.0, 8.0});
    
    // Action: perform element-wise addition
    Tensor c = Tensor::add(a, b);
    
    // Verification: check each result element
    // Expected result: [6, 8]
    //                  [10, 12]
    EXPECT_FLOAT_EQ(c.at(0, 0), 6.0f);
    EXPECT_FLOAT_EQ(c.at(0, 1), 8.0f);
    EXPECT_FLOAT_EQ(c.at(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 12.0f);
}

TEST(TensorOperations, ElementWiseMultiplication) {
    // Setup
    Tensor a({2, 2}, {2.0, 3.0, 4.0, 5.0});
    Tensor b({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Action: Hadamard product (element-wise multiplication)
    // Note: This is different from matrix multiplication!
    Tensor c = Tensor::multiply(a, b);
    
    // Verification
    // Expected: [2, 6]
    //           [12, 20]
    EXPECT_FLOAT_EQ(c.at(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(c.at(0, 1), 6.0f);
    EXPECT_FLOAT_EQ(c.at(1, 0), 12.0f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 20.0f);
}

TEST(TensorOperations, ScalarMultiplication) {
    // Setup
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Action: scale all elements by 2.5
    Tensor b = Tensor::scale(a, 2.5f);
    
    // Verification: each element should be multiplied by 2.5
    // Expected: [2.5, 5.0]
    //           [7.5, 10.0]
    EXPECT_FLOAT_EQ(b.at(0, 0), 2.5f);
    EXPECT_FLOAT_EQ(b.at(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(b.at(1, 0), 7.5f);
    EXPECT_FLOAT_EQ(b.at(1, 1), 10.0f);
}

TEST(TensorOperations, MismatchedShapesThrow) {
    // Setup: tensors with incompatible shapes
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});           // 2x2
    Tensor b({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}); // 2x3
    
    // Verification: operations should fail with shape mismatch
    EXPECT_THROW(Tensor::add(a, b), std::runtime_error);
    EXPECT_THROW(Tensor::multiply(a, b), std::runtime_error);
}

// ===== TEST FIXTURES =====
// For when multiple tests need the same setup

// TEST_F: Fixture-based tests
// Useful when multiple tests need identical setup/teardown
class TensorFixture : public ::testing::Test {
protected:
    // SetUp() runs before EACH test using this fixture
    void SetUp() override {
        // Example: initialize common test data here
        // We'll use this pattern more as tests get complex
    }
    
    // TearDown() runs after EACH test
    void TearDown() override {
        // Example: cleanup resources
    }
    
    // Member variables available to all tests using this fixture
    // Tensor test_tensor;
};

// Example fixture test (placeholder for now)
TEST_F(TensorFixture, ExampleFixtureTest) {
    // This test has access to anything in TensorFixture
    // We'll use this pattern more in later phases
    SUCCEED();  // Always passes - placeholder
}