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