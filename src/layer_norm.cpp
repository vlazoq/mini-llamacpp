#include "layer_norm.h"
#include <cmath>
#include <stdexcept>

namespace layer_norm {

// Simple layer norm without learnable parameters
// Just normalizes to mean=0, variance=1
Tensor layer_norm_simple(const Tensor& x, float epsilon) {
    // Only works on 2D tensors
    if (x.get_shape().size() != 2) {
        throw std::runtime_error("Layer norm currently only supports 2D tensors");
    }
    
    int rows = x.get_shape()[0];  // Number of samples
    int cols = x.get_shape()[1];  // Number of features
    
    Tensor result(x.get_shape());
    
    // Process each row (sample) independently
    for (int i = 0; i < rows; i++) {
        // Step 1: Calculate mean of this row
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += x.at(i, j);
        }
        mean /= cols;  // Average
        
        // Step 2: Calculate variance of this row
        // variance = average of squared differences from mean
        float variance = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = x.at(i, j) - mean;
            variance += diff * diff;
        }
        variance /= cols;  // Average of squared differences
        
        // Step 3: Normalize each element
        // x_norm = (x - mean) / sqrt(variance + epsilon)
        // The epsilon prevents division by zero if variance is 0
        float std_dev = std::sqrt(variance + epsilon);
        
        for (int j = 0; j < cols; j++) {
            float normalized = (x.at(i, j) - mean) / std_dev;
            result.set(i, j, normalized);
        }
    }
    
    return result;
}

// Full layer norm with learnable scale (gamma) and shift (beta)
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, 
                  float epsilon) {
    // Validate inputs
    if (x.get_shape().size() != 2) {
        throw std::runtime_error("Layer norm currently only supports 2D tensors");
    }
    
    int rows = x.get_shape()[0];
    int cols = x.get_shape()[1];
    
    // Gamma and beta must be 1D with size matching number of features
    if (gamma.get_shape().size() != 2 || gamma.get_shape()[0] != 1 || 
        gamma.get_shape()[1] != cols) {
        throw std::runtime_error("Gamma must be shape (1, num_features)");
    }
    
    if (beta.get_shape().size() != 2 || beta.get_shape()[0] != 1 || 
        beta.get_shape()[1] != cols) {
        throw std::runtime_error("Beta must be shape (1, num_features)");
    }
    
    Tensor result(x.get_shape());
    
    // Process each row independently
    for (int i = 0; i < rows; i++) {
        // Step 1: Calculate mean
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += x.at(i, j);
        }
        mean /= cols;
        
        // Step 2: Calculate variance
        float variance = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = x.at(i, j) - mean;
            variance += diff * diff;
        }
        variance /= cols;
        
        // Step 3: Normalize and apply scale/shift
        float std_dev = std::sqrt(variance + epsilon);
        
        for (int j = 0; j < cols; j++) {
            // Normalize: (x - mean) / std_dev
            float normalized = (x.at(i, j) - mean) / std_dev;
            
            // Apply learnable parameters: gamma * normalized + beta
            // gamma scales the normalized value
            // beta shifts it
            float scaled_shifted = gamma.at(0, j) * normalized + beta.at(0, j);
            
            result.set(i, j, scaled_shifted);
        }
    }
    
    return result;
}

}  // namespace layer_norm