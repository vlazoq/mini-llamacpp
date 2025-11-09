// Feedforward Network (FFN)
// Appears after attention in every transformer layer
// Processes each token independently with two linear transformations

#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "tensor.h"

namespace feedforward {

// Feedforward Network
// Two-layer MLP with expansion in the middle
//
// Architecture:
//   input (embedding_dim)
//     ↓
//   Linear1: W1 × input + b1  →  (intermediate_dim)
//     ↓
//   Activation (GELU)
//     ↓
//   Linear2: W2 × hidden + b2  →  (embedding_dim)
//     ↓
//   output (embedding_dim)
//
// Typical expansion: intermediate_dim = 4 × embedding_dim
// Example: 4096 → 16384 → 4096
//
// Args:
//   input: (seq_len × embedding_dim)
//   W1: First weight matrix (embedding_dim × intermediate_dim)
//   b1: First bias vector (1 × intermediate_dim)
//   W2: Second weight matrix (intermediate_dim × embedding_dim)
//   b2: Second bias vector (1 × embedding_dim)
//
// Returns:
//   output: (seq_len × embedding_dim)
//
// Each token is processed independently - no interaction between tokens
// (That's what attention is for!)
Tensor feedforward(const Tensor& input,
                  const Tensor& W1,
                  const Tensor& b1,
                  const Tensor& W2,
                  const Tensor& b2);

}  // namespace feedforward

#endif