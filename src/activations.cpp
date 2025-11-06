#include "activations.h"
#include <algorithm>  // For std::max
#include <cmath>      // For exp, tanh, erf
#include <stdexcept>

// M_PI is not standard C++, so define it if not available
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace activations {

// ReLU: max(0, x)
// Apply element-wise: if value < 0, make it 0; otherwise keep it
Tensor relu(const Tensor& x) {
    // Get shape and create result tensor
    Tensor result(x.get_shape());
    
    // Get raw data for faster access
    const std::vector<float>& x_data = x.get_data();
    std::vector<float> result_data(x.get_total_size());
    
    // Apply ReLU element-wise
    for (int i = 0; i < x.get_total_size(); i++) {
        // max(0, x[i]) - zeros out negatives
        result_data[i] = std::max(0.0f, x_data[i]);
    }
    
    return Tensor(x.get_shape(), result_data);
}

// GELU: Gaussian Error Linear Unit
// Using tanh approximation for speed
// Formula: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
Tensor gelu(const Tensor& x) {
    Tensor result(x.get_shape());
    
    const std::vector<float>& x_data = x.get_data();
    std::vector<float> result_data(x.get_total_size());
    
    // Constants for GELU approximation
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);  // √(2/π) ≈ 0.797 - but use M_PI in case I need to update precision later on.
    const float coeff = 0.044715f;
    
    // Apply GELU element-wise
    for (int i = 0; i < x.get_total_size(); i++) {
        float val = x_data[i];
        
        // Calculate: sqrt(2/π) * (x + 0.044715 * x³)
        float inner = sqrt_2_over_pi * (val + coeff * val * val * val);
        
        // Calculate: 0.5 * x * (1 + tanh(inner))
        result_data[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
    
    return Tensor(x.get_shape(), result_data);
}

// Softmax with numerical stability
// Subtracts max before exp to prevent overflow
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
Tensor softmax_stable(const Tensor& x) {
    // Only works on 2D tensors for now
    if (x.get_shape().size() != 2) {
        throw std::runtime_error("Softmax currently only supports 2D tensors");
    }
    
    int rows = x.get_shape()[0];
    int cols = x.get_shape()[1];
    
    Tensor result(x.get_shape());
    
    // Process each row independently
    // Each row becomes its own probability distribution
    for (int i = 0; i < rows; i++) {
        // Step 1: Find max value in this row (for numerical stability)
        // Without this, exp(large_number) can overflow
        float max_val = x.at(i, 0);
        for (int j = 1; j < cols; j++) {
            max_val = std::max(max_val, x.at(i, j));
        }
        
        // Step 2: Compute exp(x - max) and sum them
        // Subtracting max prevents overflow while giving same result
        float sum = 0.0f;
        std::vector<float> exp_values(cols);
        
        for (int j = 0; j < cols; j++) {
            float val = x.at(i, j) - max_val;  // Subtract max for stability
            exp_values[j] = std::exp(val);
            sum += exp_values[j];
        }
        
        // Step 3: Normalize by sum to get probabilities
        // Now each row sums to 1.0
        for (int j = 0; j < cols; j++) {
            result.set(i, j, exp_values[j] / sum);
        }
    }
    
    return result;
}

// Public softmax function (uses stable version)
Tensor softmax(const Tensor& x) {
    return softmax_stable(x);
}

}  // namespace activations