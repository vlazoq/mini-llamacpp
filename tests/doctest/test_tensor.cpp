// Define this in exactly ONE cpp file before including doctest.h
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

// Include what we're testing
#include "tensor.h"

// TEST_CASE: defines a test with a description
// Each test is independent and runs in isolation

TEST_CASE("Tensor creation with shape only (zeros)") {
    Tensor t({2, 3});
    
    // CHECK: like assert, but continues even if it fails
    CHECK(t.get_shape().size() == 2);
    CHECK(t.get_shape()[0] == 2);
    CHECK(t.get_shape()[1] == 3);
    CHECK(t.get_total_size() == 6);
    
    // All elements should be zero
    CHECK(t.at(0, 0) == 0.0f);
    CHECK(t.at(1, 2) == 0.0f);
}

TEST_CASE("Tensor creation with shape and data") {
    Tensor t({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    
    CHECK(t.get_total_size() == 6);
    
    // Check specific elements
    CHECK(t.at(0, 0) == 1.0f);
    CHECK(t.at(0, 1) == 2.0f);
    CHECK(t.at(0, 2) == 3.0f);
    CHECK(t.at(1, 0) == 4.0f);
    CHECK(t.at(1, 1) == 5.0f);
    CHECK(t.at(1, 2) == 6.0f);
}

TEST_CASE("Tensor element access and modification") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // Read element
    CHECK(t.at(0, 1) == 2.0f);
    
    // Modify element
    t.set(0, 1, 99.5f);
    CHECK(t.at(0, 1) == 99.5f);
    
    // Other elements unchanged
    CHECK(t.at(0, 0) == 1.0f);
    CHECK(t.at(1, 0) == 3.0f);
}

TEST_CASE("Tensor bounds checking") {
    Tensor t({2, 2}, {1.0, 2.0, 3.0, 4.0});
    
    // CHECK_THROWS: verifies that an exception is thrown
    CHECK_THROWS(t.at(2, 0));  // Row out of bounds
    CHECK_THROWS(t.at(0, 2));  // Column out of bounds
    CHECK_THROWS(t.at(-1, 0)); // Negative index
}

TEST_CASE("Tensor data size validation") {
    // Should throw if data size doesn't match shape
    CHECK_THROWS(Tensor({2, 3}, {1.0, 2.0}));  // Too few elements
    CHECK_THROWS(Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0}));  // Too many
}

TEST_CASE("Tensor with different shapes") {
    // 1x1 tensor
    Tensor t1({1, 1}, {42.0});
    CHECK(t1.at(0, 0) == 42.0f);
    
    // 4x2 tensor
    Tensor t2({4, 2});
    CHECK(t2.get_total_size() == 8);
    CHECK(t2.at(3, 1) == 0.0f);  // Should be initialized to zero
}