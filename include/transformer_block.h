// Transformer Block: The fundamental building block of transformer models
// Combines attention, feedforward, layer norm, and residual connections

#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "tensor.h"

namespace transformer {

// Complete Transformer Block
// This is what gets stacked 32+ times in models like LLaMA
//
// Architecture:
//   input
//     ↓
//   x = LayerNorm1(input)
//   x = MultiHeadAttention(x)
//   x = input + x  (residual connection)
//     ↓
//   y = LayerNorm2(x)
//   y = Feedforward(y)
//   output = x + y  (residual connection)
//
// Args:
//   input: (seq_len × embedding_dim)
//   
//   Attention parameters:
//   - Wq, Wk, Wv, Wo: Weight matrices for attention
//   - num_heads: Number of attention heads
//   
//   Feedforward parameters:
//   - W1, b1: First layer weights and bias
//   - W2, b2: Second layer weights and bias
//   
//   Layer norm parameters:
//   - gamma1, beta1: For first layer norm
//   - gamma2, beta2: For second layer norm
//
// Returns:
//   output: (seq_len × embedding_dim)
//
// Note: In a real implementation, we'd also have:
//   - Dropout (for training)
//   - Causal masking (for autoregressive generation)
//   - KV caching (for efficient inference)
//   We'll add these later if needed
Tensor transformer_block(
    const Tensor& input,
    // Attention weights
    const Tensor& Wq,
    const Tensor& Wk,
    const Tensor& Wv,
    const Tensor& Wo,
    int num_heads,
    // Feedforward weights
    const Tensor& W1,
    const Tensor& b1,
    const Tensor& W2,
    const Tensor& b2,
    // Layer norm parameters
    const Tensor& gamma1,
    const Tensor& beta1,
    const Tensor& gamma2,
    const Tensor& beta2
);

}  // namespace transformer

#endif