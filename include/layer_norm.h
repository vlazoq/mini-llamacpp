// Layer Normalization
// Normalizes features to have mean=0, variance=1, then applies learned scale/shift
// Critical for stable transformer training

#ifndef LAYER_NORM_H
#define LAYER_NORM_H

#include "tensor.h"

namespace layer_norm {

// Layer Normalization
// For each row (sample) in the input:
//   1. Normalize to mean=0, variance=1
//   2. Apply scale (gamma) and shift (beta)
//
// Args:
//   x: Input tensor (2D: batch_size × features)
//   gamma: Scale parameters (1D: features) - learned during training
//   beta: Shift parameters (1D: features) - learned during training
//   epsilon: Small constant for numerical stability (default: 1e-5)
//
// Returns:
//   Normalized tensor with same shape as input
//
// Example:
//   Input: [1, 2, 3, 4]
//   Mean: 2.5, Variance: 1.25
//   Normalized: [-1.34, -0.45, 0.45, 1.34] (mean≈0, var≈1)
//   With gamma=[2,2,2,2], beta=[1,1,1,1]:
//   Output: [-1.68, 0.1, 1.9, 3.68] (scaled and shifted)
//
// Used in transformers:
//   - Before self-attention
//   - After self-attention
//   - Before feedforward
//   - After feedforward
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, 
                  float epsilon = 1e-5f);

// Simple version without learnable parameters (just normalize)
// Useful for testing and understanding the core normalization
Tensor layer_norm_simple(const Tensor& x, float epsilon = 1e-5f);

}  // namespace layer_norm

#endif