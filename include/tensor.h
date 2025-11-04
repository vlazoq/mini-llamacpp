// Header guard: prevents this file from being included multiple times
#ifndef TENSOR_H
#define TENSOR_H

// std::vector - dynamic array that can grow/shrink
// Used to store shape [2, 3] and data [1.0, 2.0, ...]
#include <vector>

// std::cout and std::endl for printing
// TODO: Try to remember why not to use namespace std ...
#include <iostream>

// Class for the tensors
class Tensor
{

public: // Everything after this is accessible from outside the class
    // Constructor #1: Create tensor with just shape, filled with zeros
    // Example: Tensor t({2, 3}); creates 2x3 matrix of zeros
    //
    // const std::vector<int>& shape means:
    //   - &: reference (pass by reference, not by copy - more efficient)
    Tensor(const std::vector<int> &shape);

    // Constructor #2: Create tensor with shape AND data
    // Example: Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    //
    // Overload
    Tensor(const std::vector<int> &shape, const std::vector<float> &data);

    // GETTER FUNCTIONS: Read information about the tensor

    // Get the shape vector
    // Example: if shape is [2, 3], this returns that vector
    // const at end means: this function doesn't modify the Tensor object
    // Returns const reference (can't modify the returned shape)
    const std::vector<int> &get_shape() const { return shape; }

    // Get total number of elements
    // Example: [2, 3] shape has 2*3 = 6 elements total
    int get_total_size() const { return total_size; }

    // Get the raw data array
    // Returns const reference (can read but not modify)
    const std::vector<float> &get_data() const { return data; }

    // ACCESS FUNCTIONS: Read/write individual elements (for 2D tensors)

    // Get value at position (row i, column j)
    // Example: at(0, 2) gets element at row 0, column 2
    // const at end = doesn't modify the tensor
    float at(int i, int j) const;

    // Set value at position (row i, column j)
    // Example: set(1, 0, 99.5) sets row 1, column 0 to 99.5
    // No const here because we ARE modifying the tensor
    void set(int i, int j, float value);

    // UTILITY FUNCTIONS

    // Print the tensor to console
    // Useful for debugging
    void print() const;

private: // Everything after this is only accessible inside the class
    // MEMBER VARIABLES: the actual data the tensor stores

    std::vector<int> shape;  // Dimensions, e.g., [2, 3] for 2 rows, 3 columns
    std::vector<float> data; // All elements stored flat: [1, 2, 3, 4, 5, 6]
    int total_size;          // Total elements (product of all dimensions)

    // HELPER FUNCTION: Convert 2D coordinates to flat array index
    // Private because users don't need to call this directly
    // Formula: index = row * num_columns + column
    int get_index(int i, int j) const;

}; // End of class - note the semicolon!

#endif // End of header guard.