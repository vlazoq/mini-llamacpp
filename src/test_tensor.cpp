#include <iostream>
#include "tensor.h"

int main() {
    std::cout << "=== Tensor Class Test ===" << std::endl << std::endl;
    
    // Test 1: Create empty tensor (zeros)
    std::cout << "Test 1: Create 2x3 tensor (zeros):" << std::endl;
    Tensor t1({2, 3});
    t1.print();
    std::cout << std::endl;
    
    // Test 2: Create tensor with data
    std::cout << "Test 2: Create 2x3 tensor with data:" << std::endl;
    Tensor t2({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    t2.print();
    std::cout << std::endl;
    
    // Test 3: Access elements
    std::cout << "Test 3: Access element at (1, 2):" << std::endl;
    std::cout << "Value: " << t2.at(1, 2) << std::endl;
    std::cout << std::endl;
    
    // Test 4: Modify elements
    std::cout << "Test 4: Set element at (0, 1) to 99.5:" << std::endl;
    t2.set(0, 1, 99.5);
    t2.print();
    std::cout << std::endl;
    
    // Test 5: Test error handling (uncomment to test)
    // std::cout << "Test 5: Try invalid access (should throw error):" << std::endl;
    // t2.at(5, 5);  // This will throw an error
    
    std::cout << "All tests passed!" << std::endl;
    
    return 0;
}