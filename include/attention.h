// Attention Mechanism: The core of transformers
// Allows each token to gather information from other tokens

#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"

namespace attention {

// Scaled Dot-Product Attention (Single Head)
// The fundamental attention operation
//
// Formula: Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
//
// Args:
//   Q: Query matrix (seq_len × head_dim)
//   K: Key matrix (seq_len × head_dim)
//   V: Value matrix (seq_len × head_dim)
//
// Returns:
//   Output matrix (seq_len × head_dim)
//   Each row is the attended representation for that position
//
// Process:
//   1. Compute attention scores: Q × K^T (each query compared to all keys)
//   2. Scale by √head_dim (prevent extreme values)
//   3. Apply softmax (convert to probabilities)
//   4. Multiply by V (weighted sum of values)
//
// Example:
//   Q, K, V are all (3 × 4) - 3 tokens, 4 dimensions
//   scores = Q × K^T → (3 × 3) - each token attends to all tokens
//   attention_weights = softmax(scores / √4) → (3 × 3) probabilities
//   output = attention_weights × V → (3 × 4) - attended representations
Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, 
                                    const Tensor& V);

// Multi-Head Attention
// Runs multiple attention heads in parallel, then combines results
//
// Why multiple heads?
//   - Different heads learn different relationships
//   - Head 1: syntactic patterns (subject-verb)
//   - Head 2: semantic patterns (similar meanings)
//   - Head 3: positional patterns (nearby words)
//   - Etc.
//
// Args:
//   input: Input embeddings (seq_len × embedding_dim)
//   Wq, Wk, Wv: Weight matrices for projecting to Q, K, V
//               Each is (embedding_dim × embedding_dim)
//   Wo: Output projection matrix (embedding_dim × embedding_dim)
//   num_heads: Number of parallel attention heads
//
// Returns:
//   Output (seq_len × embedding_dim)
//
// Process:
//   1. Project input to Q, K, V: input × Wq, input × Wk, input × Wv
//   2. Split into num_heads: reshape from (seq_len × embedding_dim)
//                            to num_heads × (seq_len × head_dim)
//                            where head_dim = embedding_dim / num_heads
//   3. Apply attention for each head in parallel
//   4. Concatenate head outputs
//   5. Project back: concatenated × Wo
//
// Note: For simplicity, we'll implement a version that processes heads
//       sequentially rather than truly in parallel
Tensor multi_head_attention(const Tensor& input,
                            const Tensor& Wq,
                            const Tensor& Wk,
                            const Tensor& Wv,
                            const Tensor& Wo,
                            int num_heads);

}  // namespace attention

#endif