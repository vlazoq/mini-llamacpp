// Define this in exactly ONE cpp file before including doctest.h
// This tells doctest to generate the main() function for us
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Include what we're testing
#include "tensor.h"

// ===== BASIC TENSOR CREATION TESTS =====

// TEST_CASE: Defines a single test with a description
// Each test is independent - they don't share state
TEST_CASE("Tensor creation with shape only (zeros)") {
    // Create a 2x3 tensor (should be filled with zeros)
    Tensor t({2, 3});
    
    // CHECK: Verifies a condition is true
    // If it fails, the test continues (good for checking multiple things)
    // Test that shape was stored correctly
    CHECK(t.get_shape().size() == 2);      // Should be 2D (2 dimensions)
    CHECK(t.get_shape()[0] == 2);          // First dimension: 2 rows
    CHECK(t.get_shape()[1] == 3);          // Second dimension: 3 columns
    CHECK(t.get_total_size() == 6);        // Total elements: 2 * 3 = 6
    
    // Verify elements are initialized to zero
    CHECK(t.at(0, 0) == 0.0f);             // Top-left element
    CHECK(t.at(1, 2) == 0.0f);             // Bottom-right element
}

TEST_CASE("Tensor creation with shape and data") {
    // Create a 2x3 tensor with specific data
    // Data is provided in row-major order: [row0, row1]
    Tensor t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    CHECK(t.get_total_size() == 6);
    
    // Verify each element is stored correctly
    // Matrix layout:
    // [1.0, 2.0, 3.0]
    // [4.0, 5.0, 6.0]
    CHECK(t.at(0, 0) == 1.0f);
    CHECK(t.at(0, 1) == 2.0f);
    CHECK(t.at(0, 2) == 3.0f);
    CHECK(t.at(1, 0) == 4.0f);
    CHECK(t.at(1, 1) == 5.0f);
    CHECK(t.at(1, 2) == 6.0f);
}

// ===== ELEMENT ACCESS TESTS =====

TEST_CASE("Tensor element access and modification") {
    // Create a 2x2 tensor
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Test reading an element
    CHECK(t.at(0, 1) == 2.0f);
    
    // Test modifying an element
    t.set(0, 1, 99.5f);
    CHECK(t.at(0, 1) == 99.5f);          // Should now be 99.5
    
    // Verify other elements weren't changed
    CHECK(t.at(0, 0) == 1.0f);
    CHECK(t.at(1, 0) == 3.0f);
}

// ===== ERROR HANDLING TESTS =====

TEST_CASE("Tensor bounds checking") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // CHECK_THROWS: Verifies that an exception is thrown
    // These should all throw std::runtime_error
    CHECK_THROWS(t.at(2, 0));            // Row 2 doesn't exist (only 0,1)
    CHECK_THROWS(t.at(0, 2));            // Column 2 doesn't exist (only 0,1)
    CHECK_THROWS(t.at(-1, 0));           // Negative indices are invalid
}

TEST_CASE("Tensor data size validation") {
    // Constructor should throw if data size doesn't match shape
    CHECK_THROWS(Tensor({2, 3}, {1.0, 2.0}));                      // Need 6, got 2
    CHECK_THROWS(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0}));      // Need 4, got 5
}

// ===== EDGE CASES =====

TEST_CASE("Tensor with different shapes") {
    // Test with 1x1 tensor (smallest possible)
    Tensor t1({1, 1}, {42.0});
    CHECK(t1.at(0, 0) == 42.0f);
    
    // Test with 4x2 tensor (different dimensions)
    Tensor t2({4, 2});
    CHECK(t2.get_total_size() == 8);
    CHECK(t2.at(3, 1) == 0.0f);          // Last element should be zero
}

// ===== ELEMENT-WISE OPERATIONS TESTS =====

TEST_CASE("Tensor element-wise addition") {
    // Create two 2x2 tensors
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({2, 2}, {5.0, 6.0, 7.0, 8.0});
    
    // Add them element-wise: each position added separately
    Tensor c = Tensor::add(a, b);
    
    // Verify results
    // [1+5, 2+6]   [6,  8]
    // [3+7, 4+8] = [10, 12]
    CHECK(c.at(0, 0) == 6.0f);
    CHECK(c.at(0, 1) == 8.0f);
    CHECK(c.at(1, 0) == 10.0f);
    CHECK(c.at(1, 1) == 12.0f);
}

TEST_CASE("Tensor element-wise multiplication") {
    // Create two 2x2 tensors
    Tensor a({2, 2}, {2.0, 3.0, 4.0, 5.0});
    Tensor b({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Multiply element-wise (Hadamard product)
    // This is NOT matrix multiplication!
    Tensor c = Tensor::multiply(a, b);
    
    // Verify: each element multiplied independently
    // [2*1, 3*2]   [2,  6]
    // [4*3, 5*4] = [12, 20]
    CHECK(c.at(0, 0) == 2.0f);
    CHECK(c.at(0, 1) == 6.0f);
    CHECK(c.at(1, 0) == 12.0f);
    CHECK(c.at(1, 1) == 20.0f);
}

TEST_CASE("Tensor scalar multiplication") {
    // Create a 2x2 tensor
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Multiply every element by 2.5
    Tensor b = Tensor::scale(a, 2.5f);
    
    // Verify: each element scaled by 2.5
    // [1*2.5, 2*2.5]   [2.5,  5.0]
    // [3*2.5, 4*2.5] = [7.5, 10.0]
    CHECK(b.at(0, 0) == 2.5f);
    CHECK(b.at(0, 1) == 5.0f);
    CHECK(b.at(1, 0) == 7.5f);
    CHECK(b.at(1, 1) == 10.0f);
}

TEST_CASE("Tensor operations with mismatched shapes should throw") {
    // Create tensors with different shapes
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});      // 2x2
    Tensor b({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});  // 2x3
    
    // Operations on different shapes should fail
    // Can't add or multiply tensors element-wise if shapes don't match
    CHECK_THROWS(Tensor::add(a, b));
    CHECK_THROWS(Tensor::multiply(a, b));
}

// ===== MATRIX MULTIPLICATION TESTS =====

TEST_CASE("Matrix multiplication basic 2x2") {
    // Simple 2x2 × 2x2 test
    // [1, 2]   [5, 6]
    // [3, 4] × [7, 8]
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({2, 2}, {5.0, 6.0, 7.0, 8.0});
    
    Tensor c = Tensor::matmul(a, b);
    
    // Expected result:
    // [1*5+2*7, 1*6+2*8]   [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    CHECK(c.get_shape()[0] == 2);
    CHECK(c.get_shape()[1] == 2);
    CHECK(c.at(0, 0) == 19.0f);
    CHECK(c.at(0, 1) == 22.0f);
    CHECK(c.at(1, 0) == 43.0f);
    CHECK(c.at(1, 1) == 50.0f);
}

TEST_CASE("Matrix multiplication with different dimensions") {
    // Test (2×3) × (3×2) = (2×2)
    // [1, 2, 3]   [7,  8]
    // [4, 5, 6] × [9, 10]
    //             [11,12]
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b({3, 2}, {7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    
    Tensor c = Tensor::matmul(a, b);
    
    // Result should be 2×2
    CHECK(c.get_shape()[0] == 2);
    CHECK(c.get_shape()[1] == 2);
    
    // Calculate expected values:
    // c[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // c[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // c[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // c[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    CHECK(c.at(0, 0) == 58.0f);
    CHECK(c.at(0, 1) == 64.0f);
    CHECK(c.at(1, 0) == 139.0f);
    CHECK(c.at(1, 1) == 154.0f);
}

TEST_CASE("Matrix multiplication identity") {
    // Any matrix × identity matrix = original matrix
    // [1, 2]   [1, 0]   [1, 2]
    // [3, 4] × [0, 1] = [3, 4]
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor identity({2, 2}, {1.0, 0.0, 0.0, 1.0});
    
    Tensor result = Tensor::matmul(a, identity);
    
    // Should equal original matrix
    CHECK(result.at(0, 0) == 1.0f);
    CHECK(result.at(0, 1) == 2.0f);
    CHECK(result.at(1, 0) == 3.0f);
    CHECK(result.at(1, 1) == 4.0f);
}

TEST_CASE("Matrix multiplication with zeros") {
    // Any matrix × zero matrix = zero matrix
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor zeros({2, 2}, {0.0, 0.0, 0.0, 0.0});
    
    Tensor result = Tensor::matmul(a, zeros);
    
    // All results should be zero
    CHECK(result.at(0, 0) == 0.0f);
    CHECK(result.at(0, 1) == 0.0f);
    CHECK(result.at(1, 0) == 0.0f);
    CHECK(result.at(1, 1) == 0.0f);
}

TEST_CASE("Matrix multiplication dimension mismatch should throw") {
    // (2×3) cannot multiply with (2×2) because inner dimensions don't match
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    CHECK_THROWS(Tensor::matmul(a, b));
}

TEST_CASE("Matrix multiplication requires 2D tensors") {
    // Can't do matmul with non-2D tensors (we'll handle this later for batches)
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b({4}, {1.0, 2.0, 3.0, 4.0});  // 1D tensor
    
    CHECK_THROWS(Tensor::matmul(a, b));
}

// ===== TRANSPOSE TESTS =====

TEST_CASE("Transpose square matrix") {
    // Transpose a 2×2 matrix
    // [1, 2]^T   [1, 3]
    // [3, 4]   = [2, 4]
    Tensor a({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor b = Tensor::transpose(a);
    
    // Check shape is swapped
    CHECK(b.get_shape()[0] == 2);
    CHECK(b.get_shape()[1] == 2);
    
    // Check elements are transposed
    CHECK(b.at(0, 0) == 1.0f);  // Was at (0,0), stays at (0,0)
    CHECK(b.at(0, 1) == 3.0f);  // Was at (1,0), now at (0,1)
    CHECK(b.at(1, 0) == 2.0f);  // Was at (0,1), now at (1,0)
    CHECK(b.at(1, 1) == 4.0f);  // Was at (1,1), stays at (1,1)
}

TEST_CASE("Transpose rectangular matrix") {
    // Transpose a 2×3 matrix to 3×2
    // [1, 2, 3]^T   [1, 4]
    // [4, 5, 6]   = [2, 5]
    //               [3, 6]
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b = Tensor::transpose(a);
    
    // Check shape is swapped: 2×3 becomes 3×2
    CHECK(b.get_shape()[0] == 3);
    CHECK(b.get_shape()[1] == 2);
    
    // Check all elements
    CHECK(b.at(0, 0) == 1.0f);
    CHECK(b.at(0, 1) == 4.0f);
    CHECK(b.at(1, 0) == 2.0f);
    CHECK(b.at(1, 1) == 5.0f);
    CHECK(b.at(2, 0) == 3.0f);
    CHECK(b.at(2, 1) == 6.0f);
}

TEST_CASE("Double transpose returns original") {
    // Property: (A^T)^T = A
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b = Tensor::transpose(a);
    Tensor c = Tensor::transpose(b);
    
    // Should match original shape
    CHECK(c.get_shape()[0] == 2);
    CHECK(c.get_shape()[1] == 3);
    
    // Should match original values
    CHECK(c.at(0, 0) == 1.0f);
    CHECK(c.at(0, 2) == 3.0f);
    CHECK(c.at(1, 1) == 5.0f);
}

// ===== RESHAPE TESTS =====

TEST_CASE("Reshape to same size different dimensions") {
    // Reshape 1×6 to 2×3
    // [1, 2, 3, 4, 5, 6] -> [1, 2, 3]
    //                        [4, 5, 6]
    Tensor a({1, 6}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    Tensor b = Tensor::reshape(a, {2, 3});
    
    // Check new shape
    CHECK(b.get_shape()[0] == 2);
    CHECK(b.get_shape()[1] == 3);
    CHECK(b.get_total_size() == 6);
    
    // Data should be in same order
    CHECK(b.at(0, 0) == 1.0f);
    CHECK(b.at(0, 1) == 2.0f);
    CHECK(b.at(0, 2) == 3.0f);
    CHECK(b.at(1, 0) == 4.0f);
    CHECK(b.at(1, 1) == 5.0f);
    CHECK(b.at(1, 2) == 6.0f);
}

TEST_CASE("Reshape multiple ways") {
    // Same data: [1,2,3,4,5,6] can be viewed as:
    // 1×6, 2×3, 3×2, 6×1
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    // Reshape to 3×2
    Tensor b = Tensor::reshape(a, {3, 2});
    CHECK(b.get_shape()[0] == 3);
    CHECK(b.get_shape()[1] == 2);
    CHECK(b.at(0, 0) == 1.0f);
    CHECK(b.at(2, 1) == 6.0f);
    
    // Reshape to 6×1
    Tensor c = Tensor::reshape(a, {6, 1});
    CHECK(c.get_shape()[0] == 6);
    CHECK(c.get_shape()[1] == 1);
    CHECK(c.at(5, 0) == 6.0f);
}

TEST_CASE("Reshape with wrong size should throw") {
    // 6 elements cannot be reshaped to 2×2 (needs 4 elements)
    Tensor a({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    CHECK_THROWS(Tensor::reshape(a, {2, 2}));  // 4 != 6
    CHECK_THROWS(Tensor::reshape(a, {3, 3}));  // 9 != 6
}

// ===== ACTIVATION FUNCTIONS TESTS =====
#include "activations.h"

TEST_CASE("ReLU activation function") {
    // ReLU zeros out negative values, keeps positive ones
    Tensor input({2, 3}, {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
    
    Tensor output = activations::relu(input);
    
    // Check shape unchanged
    CHECK(output.get_shape()[0] == 2);
    CHECK(output.get_shape()[1] == 3);
    
    // Check values: negatives -> 0, positives unchanged
    CHECK(output.at(0, 0) == 0.0f);   // -2 -> 0
    CHECK(output.at(0, 1) == 0.0f);   // -1 -> 0
    CHECK(output.at(0, 2) == 0.0f);   //  0 -> 0
    CHECK(output.at(1, 0) == 1.0f);   //  1 -> 1
    CHECK(output.at(1, 1) == 2.0f);   //  2 -> 2
    CHECK(output.at(1, 2) == 3.0f);   //  3 -> 3
}

TEST_CASE("ReLU with all negative values") {
    // Edge case: all values should become zero
    Tensor input({2, 2}, {-5.0, -3.0, -1.0, -0.5});
    Tensor output = activations::relu(input);
    
    CHECK(output.at(0, 0) == 0.0f);
    CHECK(output.at(0, 1) == 0.0f);
    CHECK(output.at(1, 0) == 0.0f);
    CHECK(output.at(1, 1) == 0.0f);
}

TEST_CASE("ReLU with all positive values") {
    // Edge case: all values should remain unchanged
    Tensor input({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor output = activations::relu(input);
    
    CHECK(output.at(0, 0) == 1.0f);
    CHECK(output.at(0, 1) == 2.0f);
    CHECK(output.at(1, 0) == 3.0f);
    CHECK(output.at(1, 1) == 4.0f);
}

TEST_CASE("GELU activation function") {
    // GELU is smooth, approximately:
    // - Large negative -> ~0
    // - Zero -> 0
    // - Large positive -> ~x
    Tensor input({1, 5}, {-2.0, -1.0, 0.0, 1.0, 2.0});
    
    Tensor output = activations::gelu(input);
    
    // Check shape
    CHECK(output.get_shape()[0] == 1);
    CHECK(output.get_shape()[1] == 5);
    
    // GELU(-2) should be close to 0 (small negative)
    CHECK(output.at(0, 0) < 0.0f);
    CHECK(output.at(0, 0) > -0.1f);
    
    // GELU(0) = 0
    CHECK(output.at(0, 2) == doctest::Approx(0.0f).epsilon(0.01));
    
    // GELU(1) ≈ 0.84
    CHECK(output.at(0, 3) == doctest::Approx(0.84f).epsilon(0.01));
    
    // GELU(2) ≈ 1.95 (close to 2)
    CHECK(output.at(0, 4) == doctest::Approx(1.95f).epsilon(0.01));
}

TEST_CASE("GELU is smooth (no sharp corners like ReLU)") {
    // Test that GELU transitions smoothly through zero
    // Unlike ReLU which has a sharp corner at 0
    Tensor input({1, 3}, {-0.1, 0.0, 0.1});
    Tensor output = activations::gelu(input);
    
    // All three values should be small but non-zero (except exactly 0)
    // This shows smoothness
    CHECK(output.at(0, 0) < 0.0f);  // Negative but small
    CHECK(output.at(0, 2) > 0.0f);  // Positive but small
}

TEST_CASE("Softmax creates probability distribution") {
    // Softmax should: sum to 1, all positive, larger inputs get more weight
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    Tensor output = activations::softmax(input);
    
    // Check shape
    CHECK(output.get_shape()[0] == 1);
    CHECK(output.get_shape()[1] == 4);
    
    // All values should be positive
    CHECK(output.at(0, 0) > 0.0f);
    CHECK(output.at(0, 1) > 0.0f);
    CHECK(output.at(0, 2) > 0.0f);
    CHECK(output.at(0, 3) > 0.0f);
    
    // Sum should be 1.0
    float sum = output.at(0, 0) + output.at(0, 1) + output.at(0, 2) + output.at(0, 3);
    CHECK(sum == doctest::Approx(1.0f).epsilon(0.001));
    
    // Larger inputs should get more weight
    CHECK(output.at(0, 3) > output.at(0, 2));  // 4 > 3
    CHECK(output.at(0, 2) > output.at(0, 1));  // 3 > 2
    CHECK(output.at(0, 1) > output.at(0, 0));  // 2 > 1
}

TEST_CASE("Softmax with multiple rows (independent distributions)") {
    // Each row should be its own probability distribution
    Tensor input({2, 3}, {1.0, 2.0, 3.0, 
                          4.0, 5.0, 6.0});
    
    Tensor output = activations::softmax(input);
    
    // Row 0 should sum to 1
    float sum_row0 = output.at(0, 0) + output.at(0, 1) + output.at(0, 2);
    CHECK(sum_row0 == doctest::Approx(1.0f).epsilon(0.001));
    
    // Row 1 should sum to 1
    float sum_row1 = output.at(1, 0) + output.at(1, 1) + output.at(1, 2);
    CHECK(sum_row1 == doctest::Approx(1.0f).epsilon(0.001));
}

TEST_CASE("Softmax with equal values gives uniform distribution") {
    // If all inputs are equal, output should be uniform (equal probabilities)
    Tensor input({1, 4}, {2.0, 2.0, 2.0, 2.0});
    
    Tensor output = activations::softmax(input);
    
    // All should be 0.25 (1/4)
    CHECK(output.at(0, 0) == doctest::Approx(0.25f).epsilon(0.001));
    CHECK(output.at(0, 1) == doctest::Approx(0.25f).epsilon(0.001));
    CHECK(output.at(0, 2) == doctest::Approx(0.25f).epsilon(0.001));
    CHECK(output.at(0, 3) == doctest::Approx(0.25f).epsilon(0.001));
}

TEST_CASE("Softmax numerical stability with large values") {
    // This would overflow without the max subtraction trick
    Tensor input({1, 3}, {1000.0, 1001.0, 1002.0});
    
    Tensor output = activations::softmax(input);
    
    // Should still work and sum to 1
    float sum = output.at(0, 0) + output.at(0, 1) + output.at(0, 2);
    CHECK(sum == doctest::Approx(1.0f).epsilon(0.001));
    
    // Largest value should dominate
    CHECK(output.at(0, 2) > output.at(0, 1));
    CHECK(output.at(0, 1) > output.at(0, 0));
}

// ===== LAYER NORMALIZATION TESTS =====
#include "layer_norm.h"

TEST_CASE("Layer norm simple - normalizes to mean=0, variance=1") {
    // Input with known mean and variance
    // Row: [1, 2, 3, 4] -> mean=2.5, variance=1.25
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // Check shape unchanged
    CHECK(output.get_shape()[0] == 1);
    CHECK(output.get_shape()[1] == 4);
    
    // Calculate mean of output (should be ≈0)
    float mean = (output.at(0, 0) + output.at(0, 1) + 
                  output.at(0, 2) + output.at(0, 3)) / 4.0f;
    CHECK(mean == doctest::Approx(0.0f).epsilon(0.01));
    
    // Calculate variance of output (should be ≈1)
    float variance = 0.0f;
    for (int j = 0; j < 4; j++) {
        float diff = output.at(0, j) - mean;
        variance += diff * diff;
    }
    variance /= 4.0f;
    CHECK(variance == doctest::Approx(1.0f).epsilon(0.01));
}

TEST_CASE("Layer norm simple with multiple rows") {
    // Each row should be normalized independently
    Tensor input({2, 3}, {1.0, 2.0, 3.0,    // Row 0: mean=2, var≈0.67
                          10.0, 20.0, 30.0}); // Row 1: mean=20, var≈66.67
    
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // Row 0: check mean ≈ 0
    float mean_row0 = (output.at(0, 0) + output.at(0, 1) + output.at(0, 2)) / 3.0f;
    CHECK(mean_row0 == doctest::Approx(0.0f).epsilon(0.01));
    
    // Row 1: check mean ≈ 0
    float mean_row1 = (output.at(1, 0) + output.at(1, 1) + output.at(1, 2)) / 3.0f;
    CHECK(mean_row1 == doctest::Approx(0.0f).epsilon(0.01));
}

TEST_CASE("Layer norm simple with constant values") {
    // Edge case: all values the same
    // Mean = value, variance = 0
    // Output should be all zeros (after normalization)
    Tensor input({1, 4}, {5.0, 5.0, 5.0, 5.0});
    
    Tensor output = layer_norm::layer_norm_simple(input);
    
    // All outputs should be 0 (or very close, due to epsilon)
    CHECK(output.at(0, 0) == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(output.at(0, 1) == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(output.at(0, 2) == doctest::Approx(0.0f).epsilon(0.01));
    CHECK(output.at(0, 3) == doctest::Approx(0.0f).epsilon(0.01));
}

TEST_CASE("Layer norm with gamma and beta") {
    // Input that will be normalized
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    // Gamma (scale): multiply normalized values by 2
    Tensor gamma({1, 4}, {2.0, 2.0, 2.0, 2.0});
    
    // Beta (shift): add 1 to scaled values
    Tensor beta({1, 4}, {1.0, 1.0, 1.0, 1.0});
    
    Tensor output = layer_norm::layer_norm(input, gamma, beta);
    
    // First normalize (same as simple version)
    Tensor normalized = layer_norm::layer_norm_simple(input);
    
    // Then check that gamma and beta were applied
    // output[j] = gamma[j] * normalized[j] + beta[j]
    for (int j = 0; j < 4; j++) {
        float expected = 2.0f * normalized.at(0, j) + 1.0f;
        CHECK(output.at(0, j) == doctest::Approx(expected).epsilon(0.01));
    }
}

TEST_CASE("Layer norm gamma scales, beta shifts") {
    // Simple input: already normalized (mean=0)
    Tensor input({1, 3}, {-1.0, 0.0, 1.0});
    
    // Gamma: different scales for each feature
    Tensor gamma({1, 3}, {1.0, 2.0, 3.0});
    
    // Beta: different shifts for each feature
    Tensor beta({1, 3}, {0.5, 1.0, 1.5});
    
    Tensor output = layer_norm::layer_norm(input, gamma, beta);
    
    // After normalization (input is already mean=0, var=1):
    // Each value gets: gamma[j] * normalized[j] + beta[j]
    // The exact values depend on normalization, but we can verify the pattern
    
    // Just verify shape for now (detailed math in next test)
    CHECK(output.get_shape()[0] == 1);
    CHECK(output.get_shape()[1] == 3);
}

TEST_CASE("Layer norm with identity gamma and zero beta") {
    // Gamma = 1, Beta = 0 should give same as simple layer norm
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    Tensor gamma({1, 4}, {1.0, 1.0, 1.0, 1.0});  // Identity scale
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});   // Zero shift
    
    Tensor output = layer_norm::layer_norm(input, gamma, beta);
    Tensor expected = layer_norm::layer_norm_simple(input);
    
    // Should be identical (or very close)
    for (int j = 0; j < 4; j++) {
        CHECK(output.at(0, j) == doctest::Approx(expected.at(0, j)).epsilon(0.001));
    }
}

TEST_CASE("Layer norm dimension mismatch throws error") {
    // Gamma/beta size must match input feature dimension
    Tensor input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    // Wrong size gamma (3 instead of 4)
    Tensor gamma_wrong({1, 3}, {1.0, 1.0, 1.0});
    Tensor beta({1, 4}, {0.0, 0.0, 0.0, 0.0});
    
    CHECK_THROWS(layer_norm::layer_norm(input, gamma_wrong, beta));
}

// ===== EMBEDDINGS TESTS =====
#include "embeddings.h"

TEST_CASE("Token embedding basic lookup") {
    // Create a small embedding table: 5 tokens, 3 dimensions each
    // Token 0: [1, 2, 3]
    // Token 1: [4, 5, 6]
    // Token 2: [7, 8, 9]
    // Token 3: [10, 11, 12]
    // Token 4: [13, 14, 15]
    Tensor embedding_table({5, 3}, {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12,
        13, 14, 15
    });
    
    // Look up tokens [1, 3]
    std::vector<int> token_ids = {1, 3};
    
    Tensor result = embeddings::token_embedding(embedding_table, token_ids);
    
    // Should return (2 × 3) tensor with the embeddings for tokens 1 and 3
    CHECK(result.get_shape()[0] == 2);
    CHECK(result.get_shape()[1] == 3);
    
    // Token 1 embedding: [4, 5, 6]
    CHECK(result.at(0, 0) == 4.0f);
    CHECK(result.at(0, 1) == 5.0f);
    CHECK(result.at(0, 2) == 6.0f);
    
    // Token 3 embedding: [10, 11, 12]
    CHECK(result.at(1, 0) == 10.0f);
    CHECK(result.at(1, 1) == 11.0f);
    CHECK(result.at(1, 2) == 12.0f);
}

TEST_CASE("Token embedding with single token") {
    Tensor embedding_table({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });
    
    std::vector<int> token_ids = {2};
    Tensor result = embeddings::token_embedding(embedding_table, token_ids);
    
    CHECK(result.get_shape()[0] == 1);
    CHECK(result.get_shape()[1] == 2);
    CHECK(result.at(0, 0) == 5.0f);
    CHECK(result.at(0, 1) == 6.0f);
}

TEST_CASE("Token embedding with repeated tokens") {
    Tensor embedding_table({3, 2}, {
        1, 2,
        3, 4,
        5, 6
    });
    
    // Same token twice
    std::vector<int> token_ids = {1, 1};
    Tensor result = embeddings::token_embedding(embedding_table, token_ids);
    
    // Should get the same embedding twice
    CHECK(result.at(0, 0) == 3.0f);
    CHECK(result.at(0, 1) == 4.0f);
    CHECK(result.at(1, 0) == 3.0f);
    CHECK(result.at(1, 1) == 4.0f);
}

TEST_CASE("Token embedding out of range throws") {
    Tensor embedding_table({3, 2}, {1, 2, 3, 4, 5, 6});
    
    std::vector<int> token_ids = {5};  // Only have tokens 0-2
    CHECK_THROWS(embeddings::token_embedding(embedding_table, token_ids));
}

TEST_CASE("Positional embedding basic") {
    // Position table: 4 positions, 2 dimensions
    Tensor position_table({4, 2}, {
        0.1, 0.2,   // Position 0
        0.3, 0.4,   // Position 1
        0.5, 0.6,   // Position 2
        0.7, 0.8    // Position 3
    });
    
    // Get embeddings for 3 positions
    int sequence_length = 3;
    Tensor result = embeddings::positional_embedding(position_table, sequence_length);
    
    CHECK(result.get_shape()[0] == 3);
    CHECK(result.get_shape()[1] == 2);
    
    // Should return positions 0, 1, 2
    CHECK(result.at(0, 0) == 0.1f);
    CHECK(result.at(0, 1) == 0.2f);
    CHECK(result.at(1, 0) == 0.3f);
    CHECK(result.at(1, 1) == 0.4f);
    CHECK(result.at(2, 0) == 0.5f);
    CHECK(result.at(2, 1) == 0.6f);
}

TEST_CASE("Positional embedding sequence too long throws") {
    Tensor position_table({3, 2}, {1, 2, 3, 4, 5, 6});
    
    CHECK_THROWS(embeddings::positional_embedding(position_table, 5));
}

TEST_CASE("Combined embedding adds token and position") {
    // Token table: 3 tokens, 2 dims
    Tensor token_table({3, 2}, {
        1, 2,    // Token 0
        3, 4,    // Token 1
        5, 6     // Token 2
    });
    
    // Position table: 3 positions, 2 dims
    Tensor position_table({3, 2}, {
        0.1, 0.2,  // Position 0
        0.3, 0.4,  // Position 1
        0.5, 0.6   // Position 2
    });
    
    // Tokens [0, 2]
    std::vector<int> token_ids = {0, 2};
    
    Tensor result = embeddings::combined_embedding(token_table, position_table, token_ids);
    
    CHECK(result.get_shape()[0] == 2);
    CHECK(result.get_shape()[1] == 2);
    
    // Position 0: token[0] + position[0] = [1,2] + [0.1,0.2] = [1.1, 2.2]
    CHECK(result.at(0, 0) == doctest::Approx(1.1f));
    CHECK(result.at(0, 1) == doctest::Approx(2.2f));
    
    // Position 1: token[2] + position[1] = [5,6] + [0.3,0.4] = [5.3, 6.4]
    CHECK(result.at(1, 0) == doctest::Approx(5.3f));
    CHECK(result.at(1, 1) == doctest::Approx(6.4f));
}