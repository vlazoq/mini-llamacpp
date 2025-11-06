// Activation functions for neural networks
// These add non-linearity to enable learning complex patterns

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"
#include <cmath>  // For exp, tanh, erf functions

// Namespace to group activation functions
namespace activations {

// ReLU (Rectified Linear Unit)
// Formula: f(x) = max(0, x)
// Simply zeros out negative values, keeps positive values
// 
// Example: [-2, -1, 0, 1, 2] -> [0, 0, 0, 1, 2]
// 
// Properties:
// - Very fast to compute
// - Can cause "dying ReLU" problem (neurons that output 0 forever)
// - Used in older transformers, CNNs
Tensor relu(const Tensor& x);

// GELU (Gaussian Error Linear Unit)
// Formula: f(x) = x * Φ(x), where Φ is the cumulative distribution function
// Approximation: f(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
// 
// Smooth, probabilistic activation - "how much to let this value through"
// 
// Properties:
// - Smoother than ReLU (has gradients everywhere)
// - Probabilistically gates inputs based on their value
// - Used in modern transformers (GPT-2, GPT-3, LLaMA)
// - Slightly slower than ReLU but better performance
Tensor gelu(const Tensor& x);

// Softmax
// Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
// Converts a vector of values to a probability distribution
// 
// Example: [1, 2, 3] -> [0.09, 0.24, 0.67] (sums to 1.0)
// 
// Properties:
// - Output values are in range (0, 1) and sum to 1
// - Larger inputs get exponentially larger probabilities
// - Used for: attention weights, final token predictions
// 
// For 2D tensors: applies softmax to each row independently
// (This is what we need for attention: each query gets its own probability distribution)
Tensor softmax(const Tensor& x);

// Softmax with numerical stability
// Subtracts max value before exp to prevent overflow
// Mathematically equivalent but more stable
// This is the version we'll actually use internally
Tensor softmax_stable(const Tensor& x);

}  // namespace activations

#endif