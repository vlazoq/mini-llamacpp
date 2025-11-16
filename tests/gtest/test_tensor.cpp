// Google Test main header
#include <gtest/gtest.h>

// What we're testing
#include "tensor.h"

#include "gguf_parser.h"
#include "quantization.h"
#include <fstream>

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

// ===== EMBEDDINGS TESTS =====
#include "embeddings.h"

TEST(Embeddings, TokenEmbeddingBasicLookup) {
    // Setup: Small vocabulary with 3D embeddings
    Tensor embedding_table({5, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    });
    
    // Action: Look up tokens
    std::vector<int> token_ids = {1, 3};
    Tensor result = embeddings::token_embedding(embedding_table, token_ids);
    
    // Verification
    EXPECT_EQ(result.get_shape()[0], 2);
    EXPECT_EQ(result.get_shape()[1], 3);
    
    EXPECT_FLOAT_EQ(result.at(0, 0), 4.0f);
    EXPECT_FLOAT_EQ(result.at(0, 1), 5.0f);
    EXPECT_FLOAT_EQ(result.at(0, 2), 6.0f);
    
    EXPECT_FLOAT_EQ(result.at(1, 0), 10.0f);
    EXPECT_FLOAT_EQ(result.at(1, 1), 11.0f);
    EXPECT_FLOAT_EQ(result.at(1, 2), 12.0f);
}

TEST(Embeddings, TokenEmbeddingSingleToken) {
    Tensor embedding_table({3, 2}, {1, 2, 3, 4, 5, 6});
    
    std::vector<int> token_ids = {2};
    Tensor result = embeddings::token_embedding(embedding_table, token_ids);
    
    EXPECT_EQ(result.get_shape()[0], 1);
    EXPECT_EQ(result.get_shape()[1], 2);
    EXPECT_FLOAT_EQ(result.at(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(result.at(0, 1), 6.0f);
}

TEST(Embeddings, TokenEmbeddingRepeatedTokens) {
    Tensor embedding_table({3, 2}, {1, 2, 3, 4, 5, 6});
    
    std::vector<int> token_ids = {1, 1, 1};
    Tensor result = embeddings::token_embedding(embedding_table, token_ids);
    
    EXPECT_EQ(result.get_shape()[0], 3);
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(result.at(i, 0), 3.0f);
        EXPECT_FLOAT_EQ(result.at(i, 1), 4.0f);
    }
}

TEST(Embeddings, TokenEmbeddingOutOfRangeThrows) {
    Tensor embedding_table({3, 2}, {1, 2, 3, 4, 5, 6});
    std::vector<int> token_ids = {5};
    
    EXPECT_THROW(embeddings::token_embedding(embedding_table, token_ids), 
                 std::runtime_error);
}

TEST(Embeddings, PositionalEmbeddingBasic) {
    Tensor position_table({4, 2}, {
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6,
        0.7, 0.8
    });
    
    int sequence_length = 3;
    Tensor result = embeddings::positional_embedding(position_table, sequence_length);
    
    EXPECT_EQ(result.get_shape()[0], 3);
    EXPECT_EQ(result.get_shape()[1], 2);
    
    EXPECT_FLOAT_EQ(result.at(0, 0), 0.1f);
    EXPECT_FLOAT_EQ(result.at(1, 0), 0.3f);
    EXPECT_FLOAT_EQ(result.at(2, 0), 0.5f);
}

TEST(Embeddings, PositionalEmbeddingTooLongThrows) {
    Tensor position_table({3, 2}, {1, 2, 3, 4, 5, 6});
    
    EXPECT_THROW(embeddings::positional_embedding(position_table, 5), 
                 std::runtime_error);
}

TEST(Embeddings, CombinedEmbedding) {
    // Setup
    Tensor token_table({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });
    
    Tensor position_table({3, 2}, {
        0.1, 0.2,
        0.3, 0.4,
        0.5, 0.6
    });
    
    std::vector<int> token_ids = {0, 2};
    
    // Action
    Tensor result = embeddings::combined_embedding(token_table, position_table, token_ids);
    
    // Verification: token + position at each position
    EXPECT_EQ(result.get_shape()[0], 2);
    EXPECT_EQ(result.get_shape()[1], 2);
    
    EXPECT_NEAR(result.at(0, 0), 1.1f, 0.001f);
    EXPECT_NEAR(result.at(0, 1), 2.2f, 0.001f);
    EXPECT_NEAR(result.at(1, 0), 5.3f, 0.001f);
    EXPECT_NEAR(result.at(1, 1), 6.4f, 0.001f);
}

// ===== ATTENTION MECHANISM TESTS =====
#include "attention.h"

TEST(Attention, ScaledDotProductBasic) {
    // Setup: Simple Q, K, V
    Tensor Q({2, 3}, {1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0});
    
    Tensor K({2, 3}, {1.0, 0.0, 0.0,
                      0.0, 1.0, 0.0});
    
    Tensor V({2, 3}, {1.0, 2.0, 3.0,
                      4.0, 5.0, 6.0});
    
    // Action
    Tensor output = attention::scaled_dot_product_attention(Q, K, V);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 2);
    EXPECT_EQ(output.get_shape()[1], 3);
    
    // Check output is in reasonable range
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_GE(output.at(i, j), 0.0f);
            EXPECT_LE(output.at(i, j), 10.0f);
        }
    }
}

TEST(Attention, SelfAttentionPattern) {
    // Setup: Q = K means each query is most similar to itself
    Tensor Q({2, 2}, {1.0, 0.0,
                      0.0, 1.0});
    
    Tensor K({2, 2}, {1.0, 0.0,
                      0.0, 1.0});
    
    Tensor V({2, 2}, {10.0, 20.0,
                      30.0, 40.0});
    
    // Action
    Tensor output = attention::scaled_dot_product_attention(Q, K, V);
    
    // Verification: Output should be close to V (self-attention)
    EXPECT_GE(output.at(0, 0), 5.0f);
    EXPECT_GE(output.at(0, 1), 10.0f);
    EXPECT_GE(output.at(1, 0), 15.0f);
    EXPECT_GE(output.at(1, 1), 20.0f);
}

TEST(Attention, DimensionMismatchThrows) {
    // Setup: Incompatible dimensions
    Tensor Q({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor K({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    Tensor V({2, 3}, {1, 2, 3, 4, 5, 6});
    
    // Verification
    EXPECT_THROW(attention::scaled_dot_product_attention(Q, K, V), 
                 std::runtime_error);
}

TEST(Attention, SingleTokenSequence) {
    // Edge case: Only one token
    Tensor Q({1, 2}, {1.0, 2.0});
    Tensor K({1, 2}, {3.0, 4.0});
    Tensor V({1, 2}, {5.0, 6.0});
    
    // Action
    Tensor output = attention::scaled_dot_product_attention(Q, K, V);
    
    // Verification: With one token, output = V
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 2);
    EXPECT_NEAR(output.at(0, 0), 5.0f, 0.01f);
    EXPECT_NEAR(output.at(0, 1), 6.0f, 0.01f);
}

TEST(Attention, MultiHeadBasic) {
    // Setup
    Tensor input({2, 4}, {1.0, 2.0, 3.0, 4.0,
                          5.0, 6.0, 7.0, 8.0});
    
    // Identity weight matrices
    Tensor W({4, 4}, {1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1});
    
    int num_heads = 2;
    
    // Action
    Tensor output = attention::multi_head_attention(input, W, W, W, W, num_heads);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 2);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    // Check all values are finite and reasonable
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_TRUE(std::isfinite(output.at(i, j)));
            EXPECT_GE(output.at(i, j), -100.0f);
            EXPECT_LE(output.at(i, j), 100.0f);
        }
    }
}

TEST(Attention, MultiHeadDivisionValidation) {
    // Setup: 5 dimensions, 2 heads (doesn't divide evenly)
    Tensor input({2, 5}, {1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10});
    
    Tensor W({5, 5}, {1, 0, 0, 0, 0,
                      0, 1, 0, 0, 0,
                      0, 0, 1, 0, 0,
                      0, 0, 0, 1, 0,
                      0, 0, 0, 0, 1});
    
    int num_heads = 2;
    
    // Verification: Should throw
    EXPECT_THROW(attention::multi_head_attention(input, W, W, W, W, num_heads),
                 std::runtime_error);
}

TEST(Attention, MultiHeadSingleHead) {
    // Setup: 1 head should work
    Tensor input({2, 4}, {1.0, 2.0, 3.0, 4.0,
                          5.0, 6.0, 7.0, 8.0});
    
    Tensor W({4, 4}, {1, 0, 0, 0,
                      0, 1, 0, 0,
                      0, 0, 1, 0,
                      0, 0, 0, 1});
    
    int num_heads = 1;
    
    // Action
    Tensor output = attention::multi_head_attention(input, W, W, W, W, num_heads);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 2);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_TRUE(std::isfinite(output.at(i, j)));
        }
    }
}

TEST(Attention, MultiHeadMultipleHeads) {
    // Setup: 4 tokens, 8 dims, 4 heads
    Tensor input({4, 8}, {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32
    });
    
    // 8×8 identity matrix
    std::vector<float> w_data(64, 0.0f);
    for (int i = 0; i < 8; i++) {
        w_data[i * 8 + i] = 1.0f;
    }
    
    Tensor W({8, 8}, w_data);
    
    int num_heads = 4;
    
    // Action
    Tensor output = attention::multi_head_attention(input, W, W, W, W, num_heads);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 4);
    EXPECT_EQ(output.get_shape()[1], 8);
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            EXPECT_TRUE(std::isfinite(output.at(i, j)));
        }
    }
}

// ===== FEEDFORWARD NETWORK TESTS =====
#include "feedforward.h"

TEST(FeedforwardNetwork, BasicFunctionality) {
    // Setup: 2 tokens, 4→8→4 dimensions
    Tensor input({2, 4}, {1.0, 2.0, 3.0, 4.0,
                          5.0, 6.0, 7.0, 8.0});
    
    std::vector<float> w1_data(32, 0.1f);
    Tensor W1({4, 8}, w1_data);
    Tensor b1({1, 8}, {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});
    
    std::vector<float> w2_data(32, 0.1f);
    Tensor W2({8, 4}, w2_data);
    Tensor b2({1, 4}, {0.1, 0.1, 0.1, 0.1});
    
    // Action
    Tensor output = feedforward::feedforward(input, W1, b1, W2, b2);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 2);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_TRUE(std::isfinite(output.at(i, j)));
            EXPECT_GE(output.at(i, j), -100.0f);
            EXPECT_LE(output.at(i, j), 100.0f);
        }
    }
}

TEST(FeedforwardNetwork, IdentityLikeWeights) {
    // Setup: Weights that approximately preserve input
    Tensor input({2, 3}, {1.0, 2.0, 3.0,
                          4.0, 5.0, 6.0});
    
    // W1: (3 × 6) - 18 elements total
    Tensor W1({3, 6}, {
        0.1, 0, 0, 0, 0, 0,      // Row 0
        0, 0.1, 0, 0, 0, 0,      // Row 1
        0, 0, 0.1, 0, 0, 0       // Row 2
    });
    
    Tensor b1({1, 6}, {0, 0, 0, 0, 0, 0});
    
    // W2: (6 × 3) - 18 elements total
    Tensor W2({6, 3}, {
        1, 0, 0,   // Row 0
        0, 1, 0,   // Row 1
        0, 0, 1,   // Row 2
        0, 0, 0,   // Row 3
        0, 0, 0,   // Row 4
        0, 0, 0    // Row 5
    });
    
    Tensor b2({1, 3}, {0, 0, 0});
    
    // Action
    Tensor output = feedforward::feedforward(input, W1, b1, W2, b2);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 2);
    EXPECT_EQ(output.get_shape()[1], 3);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_LT(std::abs(output.at(i, j)), 10.0f);
        }
    }
}

TEST(FeedforwardNetwork, SingleToken) {
    // Edge case: single token
    Tensor input({1, 2}, {1.0, 2.0});
    
    Tensor W1({2, 4}, {0.1, 0.2, 0.3, 0.4,
                       0.5, 0.6, 0.7, 0.8});
    Tensor b1({1, 4}, {0.1, 0.1, 0.1, 0.1});
    
    Tensor W2({4, 2}, {0.1, 0.2,
                       0.3, 0.4,
                       0.5, 0.6,
                       0.7, 0.8});
    Tensor b2({1, 2}, {0.1, 0.1});
    
    // Action
    Tensor output = feedforward::feedforward(input, W1, b1, W2, b2);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 2);
    EXPECT_TRUE(std::isfinite(output.at(0, 0)));
    EXPECT_TRUE(std::isfinite(output.at(0, 1)));
}

TEST(FeedforwardNetwork, BiasEffect) {
    // Test that biases are actually applied
    Tensor input({1, 2}, {0.0, 0.0});
    
    Tensor W1({2, 2}, {0, 0, 0, 0});
    Tensor b1({1, 2}, {5.0, 10.0});
    
    Tensor W2({2, 2}, {1, 0, 0, 1});
    Tensor b2({1, 2}, {1.0, 2.0});
    
    // Action
    Tensor output = feedforward::feedforward(input, W1, b1, W2, b2);
    
    // Verification: Output should be non-zero (bias effect)
    EXPECT_NE(output.at(0, 0), 0.0f);
    EXPECT_NE(output.at(0, 1), 0.0f);
}

TEST(FeedforwardNetwork, DimensionValidation) {
    // Setup: Wrong dimensions
    Tensor input({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    
    Tensor W1_wrong({3, 8}, {1, 2, 3, 4, 5, 6, 7, 8,
                             9, 10, 11, 12, 13, 14, 15, 16,
                             17, 18, 19, 20, 21, 22, 23, 24});
    
    Tensor b1({1, 8}, {0, 0, 0, 0, 0, 0, 0, 0});
    Tensor W2({8, 4}, std::vector<float>(32, 0.1f));
    Tensor b2({1, 4}, {0, 0, 0, 0});
    
    // Verification: Should throw
    EXPECT_THROW(feedforward::feedforward(input, W1_wrong, b1, W2, b2),
                 std::runtime_error);
}

TEST(FeedforwardNetwork, IndependentTokenProcessing) {
    // Setup: Duplicate token to verify independence
    Tensor input({2, 3}, {1.0, 2.0, 3.0,
                          1.0, 2.0, 3.0});
    
    Tensor W1({3, 6}, std::vector<float>(18, 0.1f));
    Tensor b1({1, 6}, {0, 0, 0, 0, 0, 0});
    Tensor W2({6, 3}, std::vector<float>(18, 0.1f));
    Tensor b2({1, 3}, {0, 0, 0});
    
    // Action
    Tensor output = feedforward::feedforward(input, W1, b1, W2, b2);
    
    // Verification: Same input → same output
    EXPECT_NEAR(output.at(0, 0), output.at(1, 0), 0.001f);
    EXPECT_NEAR(output.at(0, 1), output.at(1, 1), 0.001f);
    EXPECT_NEAR(output.at(0, 2), output.at(1, 2), 0.001f);
}

TEST(FeedforwardNetwork, LargeIntermediateDimension) {
    // Test typical transformer expansion ratio (4×)
    Tensor input({3, 4}, {1, 2, 3, 4,
                          5, 6, 7, 8,
                          9, 10, 11, 12});
    
    int intermediate_dim = 16;  // 4× expansion
    
    Tensor W1({4, 16}, std::vector<float>(64, 0.05f));
    Tensor b1({1, 16}, std::vector<float>(16, 0.0f));
    Tensor W2({16, 4}, std::vector<float>(64, 0.05f));
    Tensor b2({1, 4}, std::vector<float>(4, 0.0f));
    
    // Action
    Tensor output = feedforward::feedforward(input, W1, b1, W2, b2);
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 3);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            EXPECT_TRUE(std::isfinite(output.at(i, j)));
        }
    }
}

// ===== TRANSFORMER BLOCK TESTS =====
#include "transformer_block.h"

TEST(TransformerBlock, BasicFunctionality) {
    // Setup
    int seq_len = 2;
    int embedding_dim = 4;
    int num_heads = 2;
    
    Tensor input({seq_len, embedding_dim}, {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    });
    
    // Attention weights
    std::vector<float> attn_w_data(16, 0.0f);
    for (int i = 0; i < 4; i++) attn_w_data[i * 4 + i] = 1.0f;
    Tensor Wq({4, 4}, attn_w_data);
    Tensor Wk({4, 4}, attn_w_data);
    Tensor Wv({4, 4}, attn_w_data);
    Tensor Wo({4, 4}, attn_w_data);
    
    // Feedforward weights
    Tensor W1({4, 8}, std::vector<float>(32, 0.1f));
    Tensor b1({1, 8}, std::vector<float>(8, 0.0f));
    Tensor W2({8, 4}, std::vector<float>(32, 0.1f));
    Tensor b2({1, 4}, std::vector<float>(4, 0.0f));
    
    // Layer norm
    Tensor gamma({1, 4}, {1.0, 1.0, 1.0, 1.0});
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
    
    // Action
    Tensor output = transformer::transformer_block(
        input, Wq, Wk, Wv, Wo, num_heads,
        W1, b1, W2, b2,
        gamma, beta, gamma, beta
    );
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], seq_len);
    EXPECT_EQ(output.get_shape()[1], embedding_dim);
    
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            EXPECT_TRUE(std::isfinite(output.at(i, j)));
            EXPECT_GE(output.at(i, j), -100.0f);
            EXPECT_LE(output.at(i, j), 100.0f);
        }
    }
}

TEST(TransformerBlock, PreservesDimensions) {
    // Test various sequence lengths
    for (int seq_len : {1, 3, 5}) {
        int embedding_dim = 4;
        
        std::vector<float> input_data(seq_len * embedding_dim);
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = static_cast<float>(i + 1);
        }
        
        Tensor input({seq_len, embedding_dim}, input_data);
        
        std::vector<float> w_data(16, 0.0f);
        for (int i = 0; i < 4; i++) w_data[i * 4 + i] = 1.0f;
        Tensor W({4, 4}, w_data);
        
        Tensor W1({4, 8}, std::vector<float>(32, 0.05f));
        Tensor b1({1, 8}, std::vector<float>(8, 0.0f));
        Tensor W2({8, 4}, std::vector<float>(32, 0.05f));
        Tensor b2({1, 4}, std::vector<float>(4, 0.0f));
        
        Tensor gamma({1, 4}, {1.0, 1.0, 1.0, 1.0});
        Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
        
        // Action
        Tensor output = transformer::transformer_block(
            input, W, W, W, W, 2,
            W1, b1, W2, b2,
            gamma, beta, gamma, beta
        );
        
        // Verification
        EXPECT_EQ(output.get_shape()[0], seq_len);
        EXPECT_EQ(output.get_shape()[1], embedding_dim);
    }
}

TEST(TransformerBlock, SingleToken) {
    // Edge case: single token
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    std::vector<float> w_data(16, 0.0f);
    for (int i = 0; i < 4; i++) w_data[i * 4 + i] = 1.0f;
    Tensor W({4, 4}, w_data);
    
    Tensor W1({4, 8}, std::vector<float>(32, 0.1f));
    Tensor b1({1, 8}, std::vector<float>(8, 0.0f));
    Tensor W2({8, 4}, std::vector<float>(32, 0.1f));
    Tensor b2({1, 4}, std::vector<float>(4, 0.0f));
    
    Tensor gamma({1, 4}, {1.0, 1.0, 1.0, 1.0});
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
    
    // Action
    Tensor output = transformer::transformer_block(
        input, W, W, W, W, 1,
        W1, b1, W2, b2,
        gamma, beta, gamma, beta
    );
    
    // Verification
    EXPECT_EQ(output.get_shape()[0], 1);
    EXPECT_EQ(output.get_shape()[1], 4);
    
    for (int j = 0; j < 4; j++) {
        EXPECT_TRUE(std::isfinite(output.at(0, j)));
    }
}

TEST(TransformerBlock, ResidualConnectionsEffect) {
    // Residual connections should preserve input information
    Tensor input({2, 4}, {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0
    });
    
    // Zero weights
    Tensor W({4, 4}, std::vector<float>(16, 0.0f));
    Tensor W1({4, 8}, std::vector<float>(32, 0.0f));
    Tensor b1({1, 8}, std::vector<float>(8, 0.0f));
    Tensor W2({8, 4}, std::vector<float>(32, 0.0f));
    Tensor b2({1, 4}, std::vector<float>(4, 0.0f));
    
    Tensor gamma({1, 4}, {1.0, 1.0, 1.0, 1.0});
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
    
    // Action
    Tensor output = transformer::transformer_block(
        input, W, W, W, W, 2,
        W1, b1, W2, b2,
        gamma, beta, gamma, beta
    );
    
    // Verification: Output should be non-zero (residuals preserve input)
    bool has_nonzero = false;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            if (std::abs(output.at(i, j)) > 0.01f) {
                has_nonzero = true;
                break;
            }
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(TransformerBlock, MultipleHeads) {
    // Test with various numbers of heads
    Tensor input({2, 8}, {
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    });
    
    std::vector<float> w_data(64, 0.0f);
    for (int i = 0; i < 8; i++) w_data[i * 8 + i] = 0.5f;
    Tensor W({8, 8}, w_data);
    
    Tensor W1({8, 16}, std::vector<float>(128, 0.05f));
    Tensor b1({1, 16}, std::vector<float>(16, 0.0f));
    Tensor W2({16, 8}, std::vector<float>(128, 0.05f));
    Tensor b2({1, 8}, std::vector<float>(8, 0.0f));
    
    Tensor gamma({1, 8}, std::vector<float>(8, 1.0f));
    Tensor beta({1, 8}, std::vector<float>(8, 0.0f));
    
    // Test different head counts
    for (int num_heads : {2, 4, 8}) {
        Tensor output = transformer::transformer_block(
            input, W, W, W, W, num_heads,
            W1, b1, W2, b2,
            gamma, beta, gamma, beta
        );
        
        EXPECT_EQ(output.get_shape()[0], 2);
        EXPECT_EQ(output.get_shape()[1], 8);
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 8; j++) {
                EXPECT_TRUE(std::isfinite(output.at(i, j)));
            }
        }
    }
}

// ===== COMPLETE MODEL TESTS =====
#include "model.h"

TEST(CompleteModel, ConfigurationBasic) {
    model::ModelConfig config(100, 128, 64, 4, 4, 256);
    
    EXPECT_EQ(config.vocab_size, 100);
    EXPECT_EQ(config.max_seq_len, 128);
    EXPECT_EQ(config.embedding_dim, 64);
    EXPECT_EQ(config.num_layers, 4);
    EXPECT_EQ(config.num_heads, 4);
    EXPECT_EQ(config.intermediate_dim, 256);
}

TEST(CompleteModel, WeightsInitialization) {
    model::ModelConfig config(10, 16, 8, 2, 2, 16);
    model::ModelWeights weights(config);
    
    EXPECT_EQ(weights.token_embeddings.get_shape()[0], 10);
    EXPECT_EQ(weights.token_embeddings.get_shape()[1], 8);
    
    EXPECT_EQ(weights.position_embeddings.get_shape()[0], 16);
    EXPECT_EQ(weights.position_embeddings.get_shape()[1], 8);
    
    EXPECT_EQ(weights.blocks.size(), 2);
    
    EXPECT_EQ(weights.output_weight.get_shape()[0], 8);
    EXPECT_EQ(weights.output_weight.get_shape()[1], 10);
}

TEST(CompleteModel, BlockWeightsInitialization) {
    model::BlockWeights block(8, 16);
    
    EXPECT_EQ(block.Wq.get_shape()[0], 8);
    EXPECT_EQ(block.Wq.get_shape()[1], 8);
    
    EXPECT_EQ(block.W1.get_shape()[0], 8);
    EXPECT_EQ(block.W1.get_shape()[1], 16);
    
    EXPECT_EQ(block.b1.get_shape()[1], 16);
    EXPECT_EQ(block.b2.get_shape()[1], 8);
}

TEST(CompleteModel, ForwardPass) {
    model::ModelConfig config(20, 10, 8, 2, 2, 16);
    model::ModelWeights weights(config);
    model::TransformerModel model(config, weights);
    
    std::vector<int> token_ids = {5, 10, 15};
    
    Tensor logits = model.forward(token_ids);
    
    EXPECT_EQ(logits.get_shape()[0], 3);
    EXPECT_EQ(logits.get_shape()[1], 20);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 20; j++) {
            EXPECT_TRUE(std::isfinite(logits.at(i, j)));
        }
    }
}

TEST(CompleteModel, SingleTokenForward) {
    model::ModelConfig config(15, 10, 8, 2, 2, 16);
    model::ModelWeights weights(config);
    model::TransformerModel model(config, weights);
    
    std::vector<int> token_ids = {7};
    
    Tensor logits = model.forward(token_ids);
    
    EXPECT_EQ(logits.get_shape()[0], 1);
    EXPECT_EQ(logits.get_shape()[1], 15);
}

TEST(CompleteModel, VaryingSequenceLengths) {
    model::ModelConfig config(20, 50, 8, 2, 2, 16);
    model::ModelWeights weights(config);
    model::TransformerModel model(config, weights);
    
    for (int seq_len : {1, 3, 5, 10}) {
        std::vector<int> token_ids;
        for (int i = 0; i < seq_len; i++) {
            token_ids.push_back(i % 20);
        }
        
        Tensor logits = model.forward(token_ids);
        
        EXPECT_EQ(logits.get_shape()[0], seq_len);
        EXPECT_EQ(logits.get_shape()[1], 20);
    }
}

TEST(CompleteModel, RejectsSequenceTooLong) {
    model::ModelConfig config(10, 5, 8, 2, 2, 16);
    model::ModelWeights weights(config);
    model::TransformerModel model(config, weights);
    
    std::vector<int> token_ids = {1, 2, 3, 4, 5, 6};
    
    EXPECT_THROW(model.forward(token_ids), std::runtime_error);
}

TEST(CompleteModel, RejectsInvalidTokenIDs) {
    model::ModelConfig config(10, 20, 8, 2, 2, 16);
    model::ModelWeights weights(config);
    model::TransformerModel model(config, weights);
    
    std::vector<int> token_ids = {5, 15, 3};
    
    EXPECT_THROW(model.forward(token_ids), std::runtime_error);
}

TEST(CompleteModel, MultipleLayersProcess) {
    for (int num_layers : {1, 2, 4, 8}) {
        model::ModelConfig config(15, 20, 8, num_layers, 2, 16);
        model::ModelWeights weights(config);
        model::TransformerModel model(config, weights);
        
        std::vector<int> token_ids = {2, 5, 8};
        
        Tensor logits = model.forward(token_ids);
        
        EXPECT_EQ(logits.get_shape()[0], 3);
        EXPECT_EQ(logits.get_shape()[1], 15);
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 15; j++) {
                EXPECT_TRUE(std::isfinite(logits.at(i, j)));
            }
        }
    }
}

TEST(CompleteModel, HeadCountValidation) {
    model::ModelConfig config1(10, 20, 8, 2, 2, 16);
    model::ModelWeights weights1(config1);
    EXPECT_NO_THROW(model::TransformerModel(config1, weights1));
    
    model::ModelConfig config2(10, 20, 8, 2, 4, 16);
    model::ModelWeights weights2(config2);
    EXPECT_NO_THROW(model::TransformerModel(config2, weights2));
    
    model::ModelConfig config3(10, 20, 9, 2, 2, 16);
    model::ModelWeights weights3(config3);
    EXPECT_THROW(model::TransformerModel(config3, weights3), std::runtime_error);
}

// ===== SAMPLING TESTS =====
#include "sampling.h"

TEST(Sampling, GreedyPicksHighest) {
    std::vector<float> logits = {1.0, 5.0, 3.0, 2.0};
    
    int sampled = sampling::sample_greedy(logits);
    
    EXPECT_EQ(sampled, 1);
}

TEST(Sampling, GreedyWithTies) {
    std::vector<float> logits = {3.0, 5.0, 5.0, 2.0};
    
    int sampled = sampling::sample_greedy(logits);
    
    EXPECT_EQ(sampled, 1);
}

TEST(Sampling, GreedyWithNegatives) {
    std::vector<float> logits = {-10.0, -5.0, -15.0, -8.0};
    
    int sampled = sampling::sample_greedy(logits);
    
    EXPECT_EQ(sampled, 1);
}

TEST(Sampling, TemperatureDeterministicWithSeed) {
    std::vector<float> logits = {1.0, 2.0, 3.0, 4.0};
    std::mt19937 rng1(42);
    std::mt19937 rng2(42);
    
    int sample1 = sampling::sample_temperature(logits, 1.0f, rng1);
    int sample2 = sampling::sample_temperature(logits, 1.0f, rng2);
    
    EXPECT_EQ(sample1, sample2);
}

TEST(Sampling, LowTemperatureFocuses) {
    std::vector<float> logits = {1.0, 10.0, 2.0, 3.0};
    
    int count_highest = 0;
    int trials = 100;
    
    for (int i = 0; i < trials; i++) {
        std::mt19937 local_rng(i);
        int sample = sampling::sample_temperature(logits, 0.1f, local_rng);
        if (sample == 1) count_highest++;
    }
    
    EXPECT_GT(count_highest, 90);
}

TEST(Sampling, HighTemperatureSpreads) {
    std::vector<float> logits = {1.0, 10.0, 2.0, 3.0};
    
    std::vector<int> counts(4, 0);
    int trials = 1000;
    
    for (int i = 0; i < trials; i++) {
        std::mt19937 local_rng(i);
        int sample = sampling::sample_temperature(logits, 5.0f, local_rng);  // Increased temp
        counts[sample]++;
    }
    
    // With very high temperature, distribution should be more even
    EXPECT_LT(counts[1], 950);  // More lenient
    EXPECT_GT(counts[0], 10);   // Should get some samples
    EXPECT_GT(counts[2], 10);   // Should get some samples
}

TEST(Sampling, TemperatureNearZeroIsGreedy) {
    std::vector<float> logits = {1.0, 5.0, 3.0, 2.0};
    std::mt19937 rng(42);
    
    int sample = sampling::sample_temperature(logits, 1e-7f, rng);
    
    EXPECT_EQ(sample, 1);
}

TEST(Sampling, TopKOnlyConsidersTopK) {
    std::vector<float> logits = {1.0, 2.0, 10.0, 9.0};  // Made index 3 higher
    
    std::vector<int> counts(4, 0);
    int trials = 1000;
    
    for (int i = 0; i < trials; i++) {
        std::mt19937 local_rng(i);
        int sample = sampling::sample_top_k(logits, 2, local_rng);
        counts[sample]++;
    }
    
    // Bottom 2 should never be sampled
    EXPECT_EQ(counts[0], 0);
    EXPECT_EQ(counts[1], 0);
    // Top 2 should be sampled
    EXPECT_GT(counts[2], 100);
    EXPECT_GT(counts[3], 100);
}


TEST(Sampling, TopKWithK1IsGreedy) {
    std::vector<float> logits = {1.0, 5.0, 3.0, 2.0};
    std::mt19937 rng(42);
    
    int sample = sampling::sample_top_k(logits, 1, rng);
    
    EXPECT_EQ(sample, 1);
}

TEST(Sampling, TopKWithKEqualVocabSamplesAll) {
    std::vector<float> logits = {1.0, 2.0, 3.0, 4.0};
    
    std::vector<int> counts(4, 0);
    int trials = 1000;
    
    for (int i = 0; i < trials; i++) {
        std::mt19937 local_rng(i);
        int sample = sampling::sample_top_k(logits, 4, local_rng);
        counts[sample]++;
    }
    
    EXPECT_GT(counts[0], 0);
    EXPECT_GT(counts[1], 0);
    EXPECT_GT(counts[2], 0);
    EXPECT_GT(counts[3], 0);
}

TEST(Sampling, TopPRespectsThreshold) {
    std::vector<float> logits = {10.0, 1.0, 0.5, 0.1};
    
    std::vector<int> counts(4, 0);
    int trials = 1000;
    
    for (int i = 0; i < trials; i++) {
        std::mt19937 local_rng(i);
        int sample = sampling::sample_top_p(logits, 0.95f, local_rng);
        counts[sample]++;
    }
    
    EXPECT_GT(counts[0], 800);
}

TEST(Sampling, TopPWithP1SamplesAll) {
    std::vector<float> logits = {1.0, 2.0, 3.0, 4.0};
    
    std::vector<int> counts(4, 0);
    int trials = 1000;
    
    for (int i = 0; i < trials; i++) {
        std::mt19937 local_rng(i);
        int sample = sampling::sample_top_p(logits, 1.0f, local_rng);
        counts[sample]++;
    }
    
    EXPECT_GT(counts[0], 0);
    EXPECT_GT(counts[1], 0);
    EXPECT_GT(counts[2], 0);
    EXPECT_GT(counts[3], 0);
}

TEST(Sampling, ErrorHandling) {
    std::mt19937 rng(42);
    
    EXPECT_THROW(sampling::sample_greedy({}), std::runtime_error);
    EXPECT_THROW(sampling::sample_temperature({}, 1.0f, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_top_k({}, 1, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_top_p({}, 0.9f, rng), std::runtime_error);
    
    std::vector<float> logits = {1.0, 2.0, 3.0};
    EXPECT_THROW(sampling::sample_temperature(logits, 0.0f, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_temperature(logits, -1.0f, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_top_k(logits, 0, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_top_k(logits, -1, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_top_p(logits, 0.0f, rng), std::runtime_error);
    EXPECT_THROW(sampling::sample_top_p(logits, 1.5f, rng), std::runtime_error);
}

// ===== TEXT GENERATION TESTS =====
#include "generation.h"

TEST(TextGeneration, ConfigDefaults) {
    generation::GenerationConfig config;
    
    EXPECT_EQ(config.max_tokens, 50);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_EQ(config.top_k, 0);
    EXPECT_FLOAT_EQ(config.top_p, 1.0f);
    EXPECT_FALSE(config.use_greedy);
    EXPECT_EQ(config.seed, 42);
}

TEST(TextGeneration, ConfigCustomValues) {
    generation::GenerationConfig config(10, 0.8f, 40, 0.9f, true, 123);
    
    EXPECT_EQ(config.max_tokens, 10);
    EXPECT_FLOAT_EQ(config.temperature, 0.8f);
    EXPECT_EQ(config.top_k, 40);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_TRUE(config.use_greedy);
    EXPECT_EQ(config.seed, 123);
}

TEST(TextGeneration, GenerateWithGreedy) {
    model::ModelConfig model_config(20, 50, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {5, 10};
    generation::GenerationConfig config(3, 1.0f, 0, 1.0f, true);
    
    std::vector<int> result = generation::generate(test_model, prompt, config);
    
    EXPECT_EQ(result.size(), 5);
    EXPECT_EQ(result[0], 5);
    EXPECT_EQ(result[1], 10);
    
    for (int token : result) {
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 20);
    }
}

TEST(TextGeneration, GenerateWithTemperature) {
    model::ModelConfig model_config(15, 30, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {3, 7};
    generation::GenerationConfig config(5, 1.0f, 0, 1.0f, false, 42);
    
    std::vector<int> result = generation::generate(test_model, prompt, config);
    
    EXPECT_EQ(result.size(), 7);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 7);
}

TEST(TextGeneration, GenerateWithTopK) {
    model::ModelConfig model_config(20, 40, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {2};
    generation::GenerationConfig config(10, 1.0f, 5, 1.0f, false, 99);
    
    std::vector<int> result = generation::generate(test_model, prompt, config);
    
    EXPECT_EQ(result.size(), 11);
    EXPECT_EQ(result[0], 2);
}

TEST(TextGeneration, GenerateWithTopP) {
    model::ModelConfig model_config(15, 35, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {1, 4, 7};
    generation::GenerationConfig config(7, 1.0f, 0, 0.9f, false, 77);
    
    std::vector<int> result = generation::generate(test_model, prompt, config);
    
    EXPECT_EQ(result.size(), 10);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 4);
    EXPECT_EQ(result[2], 7);
}

TEST(TextGeneration, DifferentSeedsProduceDifferentResults) {
    model::ModelConfig model_config(20, 40, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {5};
    
    generation::GenerationConfig config1(5, 1.0f, 0, 1.0f, false, 42);
    generation::GenerationConfig config2(5, 1.0f, 0, 1.0f, false, 123);
    
    std::vector<int> result1 = generation::generate(test_model, prompt, config1);
    std::vector<int> result2 = generation::generate(test_model, prompt, config2);
    
    EXPECT_EQ(result1[0], result2[0]);
    
    bool has_difference = false;
    for (size_t i = 1; i < result1.size(); i++) {
        if (result1[i] != result2[i]) {
            has_difference = true;
            break;
        }
    }
    EXPECT_TRUE(has_difference);
}

TEST(TextGeneration, SameSeedIsDeterministic) {
    model::ModelConfig model_config(15, 30, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {3, 8};
    generation::GenerationConfig config(4, 1.0f, 0, 1.0f, false, 42);
    
    std::vector<int> result1 = generation::generate(test_model, prompt, config);
    std::vector<int> result2 = generation::generate(test_model, prompt, config);
    
    EXPECT_EQ(result1.size(), result2.size());
    for (size_t i = 0; i < result1.size(); i++) {
        EXPECT_EQ(result1[i], result2[i]);
    }
}

TEST(TextGeneration, RespectsMaxSeqLen) {
    model::ModelConfig model_config(15, 10, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {1, 2, 3, 4, 5};
    generation::GenerationConfig config(20, 1.0f, 0, 1.0f, true);
    
    std::vector<int> result = generation::generate(test_model, prompt, config);
    
    EXPECT_LE(result.size(), 10);
}

TEST(TextGeneration, TokenCallback) {
    model::ModelConfig model_config(15, 30, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    std::vector<int> prompt = {2};
    generation::GenerationConfig config(10, 1.0f, 0, 1.0f, true);
    
    int callback_count = 0;
    config.token_callback = [&](int token, bool is_final) {
        callback_count++;
        return callback_count < 3;
    };
    
    std::vector<int> result = generation::generate(test_model, prompt, config);
    
    EXPECT_EQ(result.size(), 4);
    EXPECT_EQ(callback_count, 3);
}

TEST(TextGeneration, ErrorHandling) {
    model::ModelConfig model_config(10, 20, 8, 2, 2, 16);
    model::ModelWeights weights(model_config);
    model::TransformerModel test_model(model_config, weights);
    
    EXPECT_THROW(generation::generate(
        test_model, 
        {}, 
        generation::GenerationConfig()
    ), std::runtime_error);
    
    EXPECT_THROW(generation::generate(
        test_model,
        {5},
        generation::GenerationConfig(0)
    ), std::runtime_error);
    
    EXPECT_THROW(generation::generate(
        test_model,
        {5},
        generation::GenerationConfig(-5)
    ), std::runtime_error);
    
    std::vector<int> long_prompt(25, 1);
    EXPECT_THROW(generation::generate(
        test_model,
        long_prompt,
        generation::GenerationConfig()
    ), std::runtime_error);
}

// ===== TOKENIZER TESTS =====
#include "tokenizer.h"

TEST(Tokenizer, VocabularySize) {
    tokenizer::CharTokenizer tok;
    
    EXPECT_EQ(tok.vocab_size(), 132);
}

TEST(Tokenizer, SpecialTokenConstants) {
    EXPECT_EQ(tokenizer::CharTokenizer::PAD_TOKEN, 0);
    EXPECT_EQ(tokenizer::CharTokenizer::UNK_TOKEN, 1);
    EXPECT_EQ(tokenizer::CharTokenizer::BOS_TOKEN, 2);
    EXPECT_EQ(tokenizer::CharTokenizer::EOS_TOKEN, 3);
}

TEST(Tokenizer, EncodeSimpleText) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = tok.encode("Hi");
    
    EXPECT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0], 76);
    EXPECT_EQ(tokens[1], 109);
}

TEST(Tokenizer, EncodeWithSpecialTokens) {
    tokenizer::CharTokenizer tok(true);
    
    std::vector<int> tokens = tok.encode("Hi");
    
    EXPECT_EQ(tokens.size(), 4);
    EXPECT_EQ(tokens[0], 2);
    EXPECT_EQ(tokens[1], 76);
    EXPECT_EQ(tokens[2], 109);
    EXPECT_EQ(tokens[3], 3);
}

TEST(Tokenizer, EncodeEmptyString) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = tok.encode("");
    
    EXPECT_TRUE(tokens.empty());
}

TEST(Tokenizer, EncodeEmptyStringWithSpecialTokens) {
    tokenizer::CharTokenizer tok(true);
    
    std::vector<int> tokens = tok.encode("");
    
    EXPECT_EQ(tokens.size(), 2);
    EXPECT_EQ(tokens[0], 2);
    EXPECT_EQ(tokens[1], 3);
}

TEST(Tokenizer, EncodeNumbersAndSymbols) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = tok.encode("123!@#");
    
    EXPECT_EQ(tokens.size(), 6);
    
    for (int token : tokens) {
        EXPECT_GE(token, 4);
        EXPECT_LT(token, 132);
    }
}

TEST(Tokenizer, EncodeSpacesAndNewlines) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = tok.encode("A B\nC");
    
    EXPECT_EQ(tokens.size(), 5);
}

TEST(Tokenizer, DecodeSimpleTokens) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = {76, 109};
    
    std::string text = tok.decode(tokens);
    
    EXPECT_EQ(text, "Hi");
}

TEST(Tokenizer, DecodeWithSpecialTokens) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = {2, 76, 109, 3};
    
    std::string text = tok.decode(tokens);
    
    EXPECT_EQ(text, "<BOS>Hi<EOS>");
}

TEST(Tokenizer, DecodeEmptyVector) {
    tokenizer::CharTokenizer tok(false);
    
    std::string text = tok.decode({});
    
    EXPECT_EQ(text, "");
}

TEST(Tokenizer, DecodePadAndUnkTokens) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = {0, 1};
    
    std::string text = tok.decode(tokens);
    
    EXPECT_EQ(text, "<PAD><UNK>");
}

TEST(Tokenizer, DecodeInvalidTokens) {
    tokenizer::CharTokenizer tok(false);
    
    std::vector<int> tokens = {-1, 200, 76};
    
    std::string text = tok.decode(tokens);
    
    EXPECT_EQ(text, "<?><?>" + std::string("H"));
}

TEST(Tokenizer, EncodeDecodeRoundtrip) {
    tokenizer::CharTokenizer tok(false);
    
    std::string original = "Hello, World! 123";
    
    std::vector<int> tokens = tok.encode(original);
    std::string decoded = tok.decode(tokens);
    
    EXPECT_EQ(decoded, original);
}

TEST(Tokenizer, EncodeDecodeRoundtripWithSpecialTokens) {
    tokenizer::CharTokenizer tok(true);
    
    std::string original = "Test";
    
    std::vector<int> tokens = tok.encode(original);
    std::string decoded = tok.decode(tokens);
    
    EXPECT_EQ(decoded, "<BOS>Test<EOS>");
}

TEST(Tokenizer, IsSpecialToken) {
    tokenizer::CharTokenizer tok;
    
    EXPECT_TRUE(tok.is_special_token(0));
    EXPECT_TRUE(tok.is_special_token(1));
    EXPECT_TRUE(tok.is_special_token(2));
    EXPECT_TRUE(tok.is_special_token(3));
    
    EXPECT_FALSE(tok.is_special_token(4));
    EXPECT_FALSE(tok.is_special_token(76));
    EXPECT_FALSE(tok.is_special_token(131));
}

TEST(Tokenizer, HandlesAllASCII) {
    tokenizer::CharTokenizer tok(false);
    
    std::string all_ascii;
    for (int i = 32; i < 127; i++) {
        all_ascii += static_cast<char>(i);
    }
    
    std::vector<int> tokens = tok.encode(all_ascii);
    std::string decoded = tok.decode(tokens);
    
    EXPECT_EQ(decoded, all_ascii);
}

TEST(Tokenizer, ConsistentEncoding) {
    tokenizer::CharTokenizer tok(false);
    
    std::string text = "Same text";
    
    std::vector<int> tokens1 = tok.encode(text);
    std::vector<int> tokens2 = tok.encode(text);
    
    EXPECT_EQ(tokens1.size(), tokens2.size());
    for (size_t i = 0; i < tokens1.size(); i++) {
        EXPECT_EQ(tokens1[i], tokens2[i]);
    }
}

// ============================================================================
// GGUF Parser Tests
// ============================================================================

// Helper function to create a minimal valid GGUF file in memory
std::vector<uint8_t> create_mock_gguf_file_gtest() {
    std::vector<uint8_t> data;
    
    auto append_uint32 = [&data](uint32_t value) {
        data.push_back(value & 0xFF);
        data.push_back((value >> 8) & 0xFF);
        data.push_back((value >> 16) & 0xFF);
        data.push_back((value >> 24) & 0xFF);
    };
    
    auto append_uint64 = [&data](uint64_t value) {
        for (int i = 0; i < 8; i++) {
            data.push_back((value >> (i * 8)) & 0xFF);
        }
    };
    
    auto append_string = [&data, &append_uint64](const std::string& str) {
        append_uint64(str.length());
        data.insert(data.end(), str.begin(), str.end());
    };
    
    append_uint32(0x46554747);
    append_uint32(3);
    append_uint64(2);
    append_uint64(3);
    
    append_string("general.architecture");
    append_uint32(8);
    append_string("llama");
    
    append_string("llama.embedding_length");
    append_uint32(4);
    append_uint32(4096);
    
    append_string("llama.block_count");
    append_uint32(4);
    append_uint32(32);
    
    append_string("token_embd.weight");
    append_uint32(2);
    append_uint64(32000);
    append_uint64(4096);
    append_uint32(0);
    append_uint64(0);
    
    append_string("output.weight");
    append_uint32(2);
    append_uint64(32000);
    append_uint64(4096);
    append_uint32(0);
    append_uint64(4096 * 32000 * 4);
    
    while (data.size() % 32 != 0) {
        data.push_back(0);
    }
    
    size_t tensor1_size = 16 * 1024;
    for (size_t i = 0; i < tensor1_size; i++) {
        data.push_back(static_cast<uint8_t>(i % 256));
    }
    
    return data;
}

std::string write_temp_gguf_gtest(const std::vector<uint8_t>& data) {
    std::string temp_path = "test_mock_gtest.gguf";
    std::ofstream file(temp_path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    file.close();
    return temp_path;
}

TEST(GGUFParserTest, FileOpening) {
    EXPECT_THROW({
        gguf::GGUFParser parser("/nonexistent/file.gguf");
    }, std::runtime_error);
}

TEST(GGUFParserTest, HeaderParsing) {
    auto mock_data = create_mock_gguf_file_gtest();
    std::string temp_file = write_temp_gguf_gtest(mock_data);
    
    gguf::GGUFParser parser(temp_file);
    
    EXPECT_FALSE(parser.is_parsed());
    
    EXPECT_NO_THROW(parser.parse());
    EXPECT_TRUE(parser.is_parsed());
    
    EXPECT_EQ(parser.get_version(), 3);
    EXPECT_EQ(parser.get_tensor_count(), 2);
    EXPECT_EQ(parser.get_metadata_count(), 3);
    
    std::remove(temp_file.c_str());
}

TEST(GGUFParserTest, MetadataParsing) {
    auto mock_data = create_mock_gguf_file_gtest();
    std::string temp_file = write_temp_gguf_gtest(mock_data);
    
    gguf::GGUFParser parser(temp_file);
    parser.parse();
    
    EXPECT_EQ(parser.get_string("general.architecture"), "llama");
    EXPECT_EQ(parser.get_string("nonexistent.key", "default"), "default");
    
    EXPECT_EQ(parser.get_uint32("llama.embedding_length"), 4096);
    EXPECT_EQ(parser.get_uint32("llama.block_count"), 32);
    EXPECT_EQ(parser.get_uint32("nonexistent.key", 999), 999);
    
    const gguf::MetadataValue* value = parser.get_metadata("general.architecture");
    EXPECT_TRUE(value != nullptr);
    
    const gguf::MetadataValue* missing = parser.get_metadata("nonexistent");
    EXPECT_TRUE(missing == nullptr);
    
    std::remove(temp_file.c_str());
}

TEST(GGUFParserTest, TensorInfo) {
    auto mock_data = create_mock_gguf_file_gtest();
    std::string temp_file = write_temp_gguf_gtest(mock_data);
    
    gguf::GGUFParser parser(temp_file);
    parser.parse();
    
    const gguf::TensorInfo* info = parser.get_tensor_info("token_embd.weight");
    ASSERT_TRUE(info != nullptr);
    
    EXPECT_EQ(info->shape.size(), 2);
    EXPECT_EQ(info->shape[0], 4096);
    EXPECT_EQ(info->shape[1], 32000);
    EXPECT_EQ(info->type, gguf::TensorType::F32);
    EXPECT_EQ(info->num_elements(), 4096 * 32000);
    
    const gguf::TensorInfo* missing = parser.get_tensor_info("nonexistent.tensor");
    EXPECT_TRUE(missing == nullptr);
    
    auto names = parser.get_tensor_names();
    EXPECT_EQ(names.size(), 2);
    
    bool has_embd = false;
    bool has_output = false;
    for (const auto& name : names) {
        if (name == "token_embd.weight") has_embd = true;
        if (name == "output.weight") has_output = true;
    }
    EXPECT_TRUE(has_embd);
    EXPECT_TRUE(has_output);
    
    std::remove(temp_file.c_str());
}

TEST(GGUFParserTest, TensorDataReading) {
    auto mock_data = create_mock_gguf_file_gtest();
    std::string temp_file = write_temp_gguf_gtest(mock_data);
    
    gguf::GGUFParser parser(temp_file);
    parser.parse();
    
    EXPECT_NO_THROW(parser.read_tensor_data("token_embd.weight"));
    
    EXPECT_THROW({
        parser.read_tensor_data("nonexistent.tensor");
    }, std::runtime_error);
    
    gguf::GGUFParser parser2(temp_file);
    EXPECT_THROW({
        parser2.read_tensor_data("token_embd.weight");
    }, std::runtime_error);
    
    std::remove(temp_file.c_str());
}

TEST(GGUFParserTest, InvalidFiles) {
    {
        std::vector<uint8_t> bad_data;
        for (int i = 0; i < 4; i++) bad_data.push_back(0xFF);
        for (int i = 0; i < 4; i++) bad_data.push_back(0x03);
        
        std::string temp_file = write_temp_gguf_gtest(bad_data);
        gguf::GGUFParser parser(temp_file);
        
        EXPECT_THROW(parser.parse(), std::runtime_error);
        std::remove(temp_file.c_str());
    }
    
    {
        std::vector<uint8_t> bad_data;
        bad_data.push_back(0x47);
        bad_data.push_back(0x47);
        bad_data.push_back(0x55);
        bad_data.push_back(0x46);
        bad_data.push_back(99);
        bad_data.push_back(0);
        bad_data.push_back(0);
        bad_data.push_back(0);
        
        std::string temp_file = write_temp_gguf_gtest(bad_data);
        gguf::GGUFParser parser(temp_file);
        
        EXPECT_THROW(parser.parse(), std::runtime_error);
        std::remove(temp_file.c_str());
    }
}

// ============================================================================
// Quantization Tests
// ============================================================================

TEST(QuantizationTest, FP16ToFP32Conversion) {
    uint16_t h_zero = 0x0000;
    EXPECT_EQ(quantization::fp16_to_fp32(h_zero), 0.0f);
    
    uint16_t h_one = 0x3C00;
    EXPECT_NEAR(quantization::fp16_to_fp32(h_one), 1.0f, 1e-6);
    
    uint16_t h_neg_one = 0xBC00;
    EXPECT_NEAR(quantization::fp16_to_fp32(h_neg_one), -1.0f, 1e-6);
    
    uint16_t h_half = 0x3800;
    EXPECT_NEAR(quantization::fp16_to_fp32(h_half), 0.5f, 1e-6);
}

TEST(QuantizationTest, Q4_0BlockDequantization) {
    quantization::block_q4_0 block;
    block.scale = 0x3C00;
    
    for (int i = 0; i < 16; i++) {
        block.qs[i] = (i * 2 + 1) << 4 | (i * 2);
    }
    
    float output[quantization::QK4_0];
    quantization::dequantize_block_q4_0(&block, output);
    
    EXPECT_NEAR(output[0], -8.0f, 1e-6);
    EXPECT_NEAR(output[8], 0.0f, 1e-6);
    EXPECT_NEAR(output[15], 7.0f, 1e-6);
}

TEST(QuantizationTest, Q4_0BlockWithScale) {
    quantization::block_q4_0 block;
    block.scale = 0x4000;
    
    for (int i = 0; i < 16; i++) {
        block.qs[i] = 0x88;
    }
    
    float output[quantization::QK4_0];
    quantization::dequantize_block_q4_0(&block, output);
    
    for (int i = 0; i < quantization::QK4_0; i++) {
        EXPECT_NEAR(output[i], 0.0f, 1e-6);
    }
}

TEST(QuantizationTest, Q8_0BlockDequantization) {
    quantization::block_q8_0 block;
    block.scale = 0x3C00;
    
    for (int i = 0; i < quantization::QK8_0; i++) {
        block.qs[i] = i - 16;
    }
    
    float output[quantization::QK8_0];
    quantization::dequantize_block_q8_0(&block, output);
    
    EXPECT_NEAR(output[0], -16.0f, 1e-6);
    EXPECT_NEAR(output[16], 0.0f, 1e-6);
    EXPECT_NEAR(output[31], 15.0f, 1e-6);
}

TEST(QuantizationTest, Q8_0BlockWithScale) {
    quantization::block_q8_0 block;
    block.scale = 0x4000;
    
    for (int i = 0; i < quantization::QK8_0; i++) {
        block.qs[i] = 10;
    }
    
    float output[quantization::QK8_0];
    quantization::dequantize_block_q8_0(&block, output);
    
    for (int i = 0; i < quantization::QK8_0; i++) {
        EXPECT_NEAR(output[i], 20.0f, 1e-6);
    }
}

TEST(QuantizationTest, Q4_0TensorSingleBlock) {
    std::vector<uint8_t> data(sizeof(quantization::block_q4_0));
    quantization::block_q4_0* block = reinterpret_cast<quantization::block_q4_0*>(data.data());
    
    block->scale = 0x3C00;
    for (int i = 0; i < 16; i++) {
        block->qs[i] = 0x88;
    }
    
    Tensor result = quantization::dequantize_q4_0(data, 32);
    
    auto shape = result.get_shape();
    EXPECT_EQ(shape[0], 32);
    EXPECT_NEAR(result.at({0}), 0.0f, 1e-6);
    EXPECT_NEAR(result.at({31}), 0.0f, 1e-6);
}

TEST(QuantizationTest, Q4_0TensorMultipleBlocks) {
    size_t n_blocks = 3;
    std::vector<uint8_t> data(n_blocks * sizeof(quantization::block_q4_0));
    
    Tensor result = quantization::dequantize_q4_0(data, 96);
    auto shape = result.get_shape();
    EXPECT_EQ(shape[0], 96);
}

TEST(QuantizationTest, Q4_0TensorPartialBlock) {
    std::vector<uint8_t> data(sizeof(quantization::block_q4_0));
    
    Tensor result = quantization::dequantize_q4_0(data, 20);
    auto shape = result.get_shape();
    EXPECT_EQ(shape[0], 20);
}

TEST(QuantizationTest, Q8_0TensorSingleBlock) {
    std::vector<uint8_t> data(sizeof(quantization::block_q8_0));
    quantization::block_q8_0* block = reinterpret_cast<quantization::block_q8_0*>(data.data());
    
    block->scale = 0x3C00;
    for (int i = 0; i < 32; i++) {
        block->qs[i] = 5;
    }
    
    Tensor result = quantization::dequantize_q8_0(data, 32);
    
    auto shape = result.get_shape();
    EXPECT_EQ(shape[0], 32);
    EXPECT_NEAR(result.at({0}), 5.0f, 1e-6);
    EXPECT_NEAR(result.at({31}), 5.0f, 1e-6);
}

TEST(QuantizationTest, Q8_0TensorMultipleBlocks) {
    size_t n_blocks = 4;
    std::vector<uint8_t> data(n_blocks * sizeof(quantization::block_q8_0));
    
    Tensor result = quantization::dequantize_q8_0(data, 128);
    auto shape = result.get_shape();
    EXPECT_EQ(shape[0], 128);
}