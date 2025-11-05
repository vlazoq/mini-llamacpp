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