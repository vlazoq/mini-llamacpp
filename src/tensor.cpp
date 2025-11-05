// Include our tensor header - this gives us the class declaration
#include "tensor.h"

// stdexcept: for throwing errors (std::runtime_error)
#include <stdexcept>

// iomanip: for formatting output (std::setw, std::setprecision)
#include <iomanip>

// Constructor: create tensor with given shape, fill with zeros
// Example: Tensor t({2, 3}) creates [[0, 0, 0], [0, 0, 0]]
Tensor::Tensor(const std::vector<int>& shape) : shape(shape) {
    // Initializer list above: copies the parameter 'shape' into member variable 'shape'
    // This happens BEFORE the function body runs
    
    // Calculate total number of elements by multiplying all dimensions
    // Example: [2, 3] -> 2 * 3 = 6 elements
    total_size = 1;
    for (int dim : shape) {  // Range-based for loop: iterate through each dimension
        total_size *= dim;
    }
    
    // Resize the data vector to hold all elements, initialize with 0.0
    // resize(size, value) - makes vector 'size' long, fills with 'value'
    data.resize(total_size, 0.0f);  // f means float literal
}

// Constructor: create tensor with given shape and data
// Example: Tensor t({2, 3}, {1, 2, 3, 4, 5, 6})
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) 
    : shape(shape), data(data) {  // Initialize both shape and data from parameters
    
    // Calculate total size
    total_size = 1;
    for (int dim : shape) {
        total_size *= dim;
    }
    
    // Validate: make sure data size matches the shape
    // Example: shape [2, 3] needs exactly 6 data elements
    if (data.size() != total_size) {
        // throw: stops execution and reports an error
        throw std::runtime_error("Data size doesn't match shape");
    }
}

// Convert (row, column) to flat array index
// Formula: index = row * num_columns + column
// Example: in a 3-column matrix, element (1, 2) is at index 1*3 + 2 = 5
//
// Visual:
// Matrix:           Flat array:
// [0, 1, 2]         [a, b, c, d, e, f]
// [3, 4, 5]          0  1  2  3  4  5
// 
// Element (1, 2) = index 5
int Tensor::get_index(int i, int j) const {
    // Safety check: only works for 2D tensors
    if (shape.size() != 2) {
        throw std::runtime_error("at() only works for 2D tensors");
    }
    
    // Bounds checking: make sure indices are valid
    // i must be in [0, num_rows), j must be in [0, num_cols)
    if (i < 0 || i >= shape[0] || j < 0 || j >= shape[1]) {
        throw std::runtime_error("Index out of bounds");
    }
    
    // Calculate flat index
    // shape[0] = number of rows, shape[1] = number of columns
    return i * shape[1] + j;
}

// Get element at (i, j) - read only
float Tensor::at(int i, int j) const {
    return data[get_index(i, j)];  // Convert to flat index and return that element
}

// Set element at (i, j) to a new value
void Tensor::set(int i, int j, float value) {
    data[get_index(i, j)] = value;  // Convert to flat index and modify
}

// Print the tensor in a readable format
void Tensor::print() const {
    // For non-2D tensors, just print basic info
    if (shape.size() != 2) {
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape.size(); i++) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";  // Comma between dimensions
        }
        std::cout << "], size=" << total_size << ")" << std::endl;
        return;  // Exit function early
    }
    
    // For 2D tensors, print as a matrix
    // Output format:
    // [
    //   [1.00, 2.00, 3.00],
    //   [4.00, 5.00, 6.00]
    // ]
    
    std::cout << "[" << std::endl;  // Opening bracket
    
    // Loop through rows
    for (int i = 0; i < shape[0]; i++) {
        std::cout << "  [";  // Start of row (with indent)
        
        // Loop through columns in this row
        for (int j = 0; j < shape[1]; j++) {
            // Format the number nicely:
            // setw(8): use 8 characters width (for alignment)
            // fixed: don't use scientific notation
            // setprecision(2): show 2 decimal places
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                      << at(i, j);
            
            // Add comma between elements (but not after last one)
            if (j < shape[1] - 1) std::cout << ", ";
        }
        
        std::cout << "]";  // End of row
        
        // Add comma after row (but not after last row)
        if (i < shape[0] - 1) std::cout << ",";
        
        std::cout << std::endl;  // New line
    }
    
    std::cout << "]" << std::endl;  // Closing bracket
}


// Add two tensors element-wise
// Both tensors must have identical shapes
Tensor Tensor::add(const Tensor& a, const Tensor& b) {
    // Validation: shapes must match
    if (a.shape != b.shape) {
        throw std::runtime_error("Cannot add tensors with different shapes");
    }
    
    // Create result tensor with same shape as inputs
    Tensor result(a.shape);
    
    // Add corresponding elements
    // Since data is stored flat, we can just iterate through indices
    for (int i = 0; i < a.total_size; i++) {
        result.data[i] = a.data[i] + b.data[i];
    }
    
    return result;
}

// Multiply two tensors element-wise (Hadamard product)
// This is NOT matrix multiplication - just element-by-element multiplication
Tensor Tensor::multiply(const Tensor& a, const Tensor& b) {
    // Validation: shapes must match
    if (a.shape != b.shape) {
        throw std::runtime_error("Cannot multiply tensors with different shapes");
    }
    
    // Create result tensor
    Tensor result(a.shape);
    
    // Multiply corresponding elements
    for (int i = 0; i < a.total_size; i++) {
        result.data[i] = a.data[i] * b.data[i];
    }
    
    return result;
}

// Multiply tensor by a scalar value
// Every element gets multiplied by the same number
Tensor Tensor::scale(const Tensor& t, float scalar) {
    // Create result tensor with same shape
    Tensor result(t.shape);
    
    // Scale each element
    for (int i = 0; i < t.total_size; i++) {
        result.data[i] = t.data[i] * scalar;
    }
    
    return result;
}