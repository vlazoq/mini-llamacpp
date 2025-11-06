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

// ===== MATRIX MULTIPLICATION TESTS =====
// Matrix multiplication is the foundation of neural network computation

TEST(MatrixMultiplication, Basic2x2) {
    // Setup: Two 2×2 matrices
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({2, 2}, {5.0, 6.0, 7.0, 8.0});
    
    // Action: Multiply them
    Tensor c = Tensor::matmul(a, b);
    
    // Verification: Check shape and values
    // Expected: [19, 22]
    //           [43, 50]
    EXPECT_EQ(c.get_shape()[0], 2);
    EXPECT_EQ(c.get_shape()[1], 2);
    EXPECT_FLOAT_EQ(c.at(0, 0), 19.0f);   // 1*5 + 2*7
    EXPECT_FLOAT_EQ(c.at(0, 1), 22.0f);   // 1*6 + 2*8
    EXPECT_FLOAT_EQ(c.at(1, 0), 43.0f);   // 3*5 + 4*7
    EXPECT_FLOAT_EQ(c.at(1, 1), 50.0f);   // 3*6 + 4*8
}

TEST(MatrixMultiplication, DifferentDimensions) {
    // Setup: (2×3) × (3×2) should give (2×2)
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b({3, 2}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    
    // Action
    Tensor c = Tensor::matmul(a, b);
    
    // Verification
    EXPECT_EQ(c.get_shape()[0], 2);
    EXPECT_EQ(c.get_shape()[1], 2);
    EXPECT_FLOAT_EQ(c.at(0, 0), 58.0f);    // 1*7 + 2*9 + 3*11
    EXPECT_FLOAT_EQ(c.at(0, 1), 64.0f);    // 1*8 + 2*10 + 3*12
    EXPECT_FLOAT_EQ(c.at(1, 0), 139.0f);   // 4*7 + 5*9 + 6*11
    EXPECT_FLOAT_EQ(c.at(1, 1), 154.0f);   // 4*8 + 5*10 + 6*12
}

TEST(MatrixMultiplication, IdentityMatrix) {
    // Setup: Test with identity matrix
    // Property: A × I = A for any matrix A
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor identity({2, 2}, {1.0, 0.0, 0.0, 1.0});
    
    // Action
    Tensor result = Tensor::matmul(a, identity);
    
    // Verification: Should equal original matrix
    EXPECT_FLOAT_EQ(result.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result.at(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result.at(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(result.at(1, 1), 4.0f);
}

TEST(MatrixMultiplication, ZeroMatrix) {
    // Setup: Test with zero matrix
    // Property: A × 0 = 0 for any matrix A
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor zeros({2, 2}, {0.0, 0.0, 0.0, 0.0});
    
    // Action
    Tensor result = Tensor::matmul(a, zeros);
    
    // Verification: All zeros
    EXPECT_FLOAT_EQ(result.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(result.at(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(result.at(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(result.at(1, 1), 0.0f);
}

TEST(MatrixMultiplication, DimensionMismatch) {
    // Setup: Incompatible dimensions
    // (2×3) cannot multiply (2×2) - inner dimensions don't match
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Verification: Should throw error
    EXPECT_THROW(Tensor::matmul(a, b), std::runtime_error);
}

TEST(MatrixMultiplication, Requires2DTensors) {
    // Setup: Try with non-2D tensor
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({4}, {1.0, 2.0, 3.0, 4.0});  // 1D vector
    
    // Verification: Should require 2D tensors
    EXPECT_THROW(Tensor::matmul(a, b), std::runtime_error);
}

// ===== TRANSPOSE TESTS =====
// Transpose swaps rows and columns - critical for attention mechanisms

TEST(TensorUtilities, TransposeSquareMatrix) {
    // Setup: 2×2 matrix
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Action: Transpose
    Tensor b = Tensor::transpose(a);
    
    // Verification: Dimensions stay same for square matrix
    EXPECT_EQ(b.get_shape()[0], 2);
    EXPECT_EQ(b.get_shape()[1], 2);
    
    // Elements should be swapped across diagonal
    EXPECT_FLOAT_EQ(b.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0, 1), 3.0f);  // Was (1,0)
    EXPECT_FLOAT_EQ(b.at(1, 0), 2.0f);  // Was (0,1)
    EXPECT_FLOAT_EQ(b.at(1, 1), 4.0f);
}

TEST(TensorUtilities, TransposeRectangularMatrix) {
    // Setup: 2×3 matrix
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    // Action: Transpose to 3×2
    Tensor b = Tensor::transpose(a);
    
    // Verification: Dimensions should be swapped
    EXPECT_EQ(b.get_shape()[0], 3);
    EXPECT_EQ(b.get_shape()[1], 2);
    
    // Check transposed values
    EXPECT_FLOAT_EQ(b.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(b.at(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(b.at(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(b.at(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(b.at(2, 1), 6.0f);
}

TEST(TensorUtilities, DoubleTransposeReturnsOriginal) {
    // Setup: Test mathematical property (A^T)^T = A
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    // Action: Double transpose
    Tensor b = Tensor::transpose(a);
    Tensor c = Tensor::transpose(b);
    
    // Verification: Should match original
    EXPECT_EQ(c.get_shape()[0], 2);
    EXPECT_EQ(c.get_shape()[1], 3);
    EXPECT_FLOAT_EQ(c.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c.at(1, 2), 6.0f);
}

// ===== RESHAPE TESTS =====
// Reshape reinterprets data with different dimensions (same total size)

TEST(TensorUtilities, ReshapeSameSizeDifferentDimensions) {
    // Setup: 1×6 matrix
    Tensor a({1, 6}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    // Action: Reshape to 2×3
    Tensor b = Tensor::reshape(a, {2, 3});
    
    // Verification: New shape, same data order
    EXPECT_EQ(b.get_shape()[0], 2);
    EXPECT_EQ(b.get_shape()[1], 3);
    EXPECT_EQ(b.get_total_size(), 6);
    
    // Data order preserved
    EXPECT_FLOAT_EQ(b.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(b.at(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(b.at(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(b.at(1, 2), 6.0f);
}

TEST(TensorUtilities, ReshapeMultipleWays) {
    // Setup: 6 elements can be arranged multiple ways
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    // Action: Try different valid reshapes
    Tensor b = Tensor::reshape(a, {3, 2});
    Tensor c = Tensor::reshape(a, {6, 1});
    Tensor d = Tensor::reshape(a, {1, 6});
    
    // Verification: All should have same total size
    EXPECT_EQ(b.get_total_size(), 6);
    EXPECT_EQ(c.get_total_size(), 6);
    EXPECT_EQ(d.get_total_size(), 6);
    
    // Spot check values (data order preserved)
    EXPECT_FLOAT_EQ(b.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(c.at(5, 0), 6.0f);
    EXPECT_FLOAT_EQ(d.at(0, 5), 6.0f);
}

TEST(TensorUtilities, ReshapeWrongSizeThrows) {
    // Setup: 6 elements
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    // Verification: Invalid reshapes should throw
    EXPECT_THROW(Tensor::reshape(a, {2, 2}), std::runtime_error);  // 4 != 6
    EXPECT_THROW(Tensor::reshape(a, {3, 3}), std::runtime_error);  // 9 != 6
    EXPECT_THROW(Tensor::reshape(a, {5, 1}), std::runtime_error);  // 5 != 6
}

// ===== ACTIVATION FUNCTIONS TESTS =====
#include "activations.h"

// ReLU Tests
TEST(ActivationFunctions, ReLUBasic) {
    // Setup: Mix of negative, zero, and positive values
    Tensor input({2, 3}, {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
    
    // Action: Apply ReLU
    Tensor output = activations::relu(input);
    
    // Verification: Negatives become 0, positives unchanged
    EXPECT_EQ(output.get_shape()[0], 2);
    EXPECT_EQ(output.get_shape()[1], 3);
    EXPECT_FLOAT_EQ(output.at(0, 0), 0.0f);  // -2 -> 0
    EXPECT_FLOAT_EQ(output.at(0, 1), 0.0f);  // -1 -> 0
    EXPECT_FLOAT_EQ(output.at(0, 2), 0.0f);  //  0 -> 0
    EXPECT_FLOAT_EQ(output.at(1, 0), 1.0f);  //  1 -> 1
    EXPECT_FLOAT_EQ(output.at(1, 1), 2.0f);  //  2 -> 2
    EXPECT_FLOAT_EQ(output.at(1, 2), 3.0f);  //  3 -> 3
}

TEST(ActivationFunctions, ReLUAllNegative) {
    // Edge case: all negative values
    Tensor input({2, 2}, {-5.0, -3.0, -1.0, -0.5});
    Tensor output = activations::relu(input);
    
    // All should become zero
    EXPECT_FLOAT_EQ(output.at(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(output.at(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(output.at(1, 1), 0.0f);
}

TEST(ActivationFunctions, ReLUAllPositive) {
    // Edge case: all positive values
    Tensor input({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor output = activations::relu(input);
    
    // All should remain unchanged
    EXPECT_FLOAT_EQ(output.at(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(output.at(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(output.at(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(output.at(1, 1), 4.0f);
}

// GELU Tests
TEST(ActivationFunctions, GELUBasic) {
    // Setup: Range of values to show GELU behavior
    Tensor input({1, 5}, {-2.0, -1.0, 0.0, 1.0, 2.0});
    
    // Action: Apply GELU
    Tensor output = activations::gelu(input);
    
    // Verification: Check expected approximate values
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 5);
    
    // GELU(-2) ≈ -0.05 (small negative, almost zero)
    EXPECT_LT(output.at(0, 0), 0.0f);
    EXPECT_GT(output.at(0, 0), -0.1f);
    
    // GELU(0) = 0
    EXPECT_NEAR(output.at(0, 2), 0.0f, 0.01f);
    
    // GELU(1) ≈ 0.84
    EXPECT_NEAR(output.at(0, 3), 0.84f, 0.01f);
    
    // GELU(2) ≈ 1.95
    EXPECT_NEAR(output.at(0, 4), 1.95f, 0.01f);
}

TEST(ActivationFunctions, GELUSmoothness) {
    // Test that GELU is smooth around zero (no sharp corner)
    Tensor input({1, 3}, {-0.1, 0.0, 0.1});
    Tensor output = activations::gelu(input);
    
    // Should transition smoothly (all small values, no discontinuity)
    EXPECT_LT(output.at(0, 0), 0.0f);  // Negative
    EXPECT_NEAR(output.at(0, 1), 0.0f, 0.001f);  // Zero
    EXPECT_GT(output.at(0, 2), 0.0f);  // Positive
}

// Softmax Tests
TEST(ActivationFunctions, SoftmaxProbabilityDistribution) {
    // Setup: Simple increasing sequence
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    // Action: Apply softmax
    Tensor output = activations::softmax(input);
    
    // Verification: Creates valid probability distribution
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    // All values positive
    EXPECT_GT(output.at(0, 0), 0.0f);
    EXPECT_GT(output.at(0, 1), 0.0f);
    EXPECT_GT(output.at(0, 2), 0.0f);
    EXPECT_GT(output.at(0, 3), 0.0f);
    
    // Sum to 1.0
    float sum = output.at(0, 0) + output.at(0, 1) + output.at(0, 2) + output.at(0, 3);
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Monotonic: larger inputs get larger probabilities
    EXPECT_GT(output.at(0, 3), output.at(0, 2));
    EXPECT_GT(output.at(0, 2), output.at(0, 1));
    EXPECT_GT(output.at(0, 1), output.at(0, 0));
}

TEST(ActivationFunctions, SoftmaxMultipleRows) {
    // Setup: 2 rows, each should get independent distribution
    Tensor input({2, 3}, {1.0, 2.0, 3.0, 
                          4.0, 5.0, 6.0});
    
    // Action
    Tensor output = activations::softmax(input);
    
    // Verification: Each row sums to 1 independently
    float sum_row0 = output.at(0, 0) + output.at(0, 1) + output.at(0, 2);
    float sum_row1 = output.at(1, 0) + output.at(1, 1) + output.at(1, 2);
    
    EXPECT_NEAR(sum_row0, 1.0f, 0.001f);
    EXPECT_NEAR(sum_row1, 1.0f, 0.001f);
}

TEST(ActivationFunctions, SoftmaxUniformInput) {
    // Setup: All equal values
    Tensor input({1, 4}, {2.0, 2.0, 2.0, 2.0});
    
    // Action
    Tensor output = activations::softmax(input);
    
    // Verification: Should give uniform distribution (all 0.25)
    EXPECT_NEAR(output.at(0, 0), 0.25f, 0.001f);
    EXPECT_NEAR(output.at(0, 1), 0.25f, 0.001f);
    EXPECT_NEAR(output.at(0, 2), 0.25f, 0.001f);
    EXPECT_NEAR(output.at(0, 3), 0.25f, 0.001f);
}

TEST(ActivationFunctions, SoftmaxNumericalStability) {
    // Setup: Large values that would overflow without stability trick
    Tensor input({1, 3}, {1000.0, 1001.0, 1002.0});
    
    // Action: Should not overflow
    Tensor output = activations::softmax(input);
    
    // Verification: Still produces valid distribution
    float sum = output.at(0, 0) + output.at(0, 1) + output.at(0, 2);
    EXPECT_NEAR(sum, 1.0f, 0.001f);
    
    // Largest input should still dominate
    EXPECT_GT(output.at(0, 2), output.at(0, 1));
    EXPECT_GT(output.at(0, 1), output.at(0, 0));
}

// ===== LAYER NORMALIZATION TESTS =====
#include "layer_norm.h"

TEST(LayerNormalization, SimpleNormalization) {
    // Setup: Input with known statistics
    // [1, 2, 3, 4] has mean=2.5, variance=1.25
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    // Action: Normalize
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // Verification: Mean should be ≈0, variance should be ≈1
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    // Calculate mean
    float mean = 0.0f;
    for (int j = 0; j < 4; j++) {
        mean += output.at(0, j);
    }
    mean /= 4.0f;
    EXPECT_NEAR(mean, 0.0f, 0.01f);
    
    // Calculate variance
    float variance = 0.0f;
    for (int j = 0; j < 4; j++) {
        float diff = output.at(0, j) - mean;
        variance += diff * diff;
    }
    variance /= 4.0f;
    EXPECT_NEAR(variance, 1.0f, 0.01f);
}

TEST(LayerNormalization, MultipleRowsIndependent) {
    // Setup: Two rows with very different scales
    Tensor input({2, 3}, {1.0, 2.0, 3.0,      // Small values
                          10.0, 20.0, 30.0});  // Large values
    
    // Action: Each row normalized independently
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // Verification: Both rows should have mean≈0
    float mean_row0 = (output.at(0, 0) + output.at(0, 1) + output.at(0, 2)) / 3.0f;
    float mean_row1 = (output.at(1, 0) + output.at(1, 1) + output.at(1, 2)) / 3.0f;
    
    EXPECT_NEAR(mean_row0, 0.0f, 0.01f);
    EXPECT_NEAR(mean_row1, 0.0f, 0.01f);
}

TEST(LayerNormalization, ConstantValues) {
    // Edge case: All same value (variance = 0)
    Tensor input({1, 4}, {5.0, 5.0, 5.0, 5.0});
    
    // Action: Normalize (with epsilon preventing div-by-zero)
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // Verification: Should be all zeros (or near zero)
    EXPECT_NEAR(output.at(0, 0), 0.0f, 0.01f);
    EXPECT_NEAR(output.at(0, 1), 0.0f, 0.01f);
    EXPECT_NEAR(output.at(0, 2), 0.0f, 0.01f);
    EXPECT_NEAR(output.at(0, 3), 0.0f, 0.01f);
}

TEST(LayerNormalization, WithGammaAndBeta) {
    // Setup
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    Tensor gamma({1, 4}, {2.0, 2.0, 2.0, 2.0});  // Scale by 2
    Tensor beta({1, 4}, {1.0, 1.0, 1.0, 1.0});   // Shift by 1
    
    // Action: Apply full layer norm
    Tensor output = layer_norm::layer_norm(input, gamma, beta);
    
    // Verification: Should be normalized, then scaled, then shifted
    // First get the simple normalized version
    Tensor normalized = layer_norm::layer_norm_simple(input);
    
    // Check that gamma and beta were applied correctly
    for (int j = 0; j < 4; j++) {
        float expected = 2.0f * normalized.at(0, j) + 1.0f;
        EXPECT_NEAR(output.at(0, j), expected, 0.01f);
    }
}

TEST(LayerNormalization, GammaScalesBetaShifts) {
    // Setup: Different gamma/beta for each feature
    Tensor input({1, 3}, {-1.0, 0.0, 1.0});
    Tensor gamma({1, 3}, {1.0, 2.0, 3.0});  // Different scales
    Tensor beta({1, 3}, {0.5, 1.0, 1.5});   // Different shifts
    
    // Action
    Tensor output = layer_norm::layer_norm(input, gamma, beta);
    
    // Verification: Each feature gets its own scale and shift
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 3);
    
    // Values will be different due to different gamma/beta per feature
    // Just verify no crashes and reasonable output range
    for (int j = 0; j < 3; j++) {
        EXPECT_GT(output.at(0, j), -10.0f);
        EXPECT_LT(output.at(0, j), 10.0f);
    }
}

TEST(LayerNormalization, IdentityTransformation) {
    // Setup: Gamma=1, Beta=0 should equal simple layer norm
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    Tensor gamma({1, 4}, {1.0, 1.0, 1.0, 1.0});
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
    
    // Action
    Tensor output = layer_norm::layer_norm(input, gamma, beta);
    Tensor expected = layer_norm::layer_norm_simple(input);
    
    // Verification: Should be identical
    for (int j = 0; j < 4; j++) {
        EXPECT_NEAR(output.at(0, j), expected.at(0, j), 0.001f);
    }
}

TEST(LayerNormalization, DimensionMismatchThrows) {
    // Setup: Incompatible dimensions
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    Tensor gamma_wrong({1, 3}, {1.0, 1.0, 1.0});  // Wrong size
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
    
    // Verification: Should throw error
    EXPECT_THROW(layer_norm::layer_norm(input, gamma_wrong, beta), 
                 std::runtime_error);
}

TEST(LayerNormalization, LargeValues) {
    // Setup: Very large values to test numerical stability
    Tensor input({1, 3}, {1000.0, 2000.0, 3000.0});
    
    // Action: Should normalize without overflow
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // Verification: Mean ≈ 0, values in reasonable range
    float mean = (output.at(0, 0) + output.at(0, 1) + output.at(0, 2)) / 3.0f;
    EXPECT_NEAR(mean, 0.0f, 0.01f);
    
    // All values should be normalized (roughly in [-2, 2] range for normal data)
    EXPECT_GT(output.at(0, 0), -5.0f);
    EXPECT_LT(output.at(0, 0), 5.0f);
}