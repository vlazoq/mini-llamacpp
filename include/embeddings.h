// Embeddings: Convert token IDs to dense vectors
// Critical first step in transformer processing

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include "tensor.h"
#include <vector>

namespace embeddings {

// Token Embedding Layer
// Lookup table: token_id → embedding_vector
//
// Args:
//   embedding_table: (vocab_size × embedding_dim)
//     Each row is the embedding for one token
//     Example: row 5 = embedding for token ID 5
//   
//   token_ids: (batch_size × sequence_length)
//     The token IDs to look up
//     Example: [[5, 42, 13]] means look up tokens 5, 42, and 13
//
// Returns:
//   (batch_size × sequence_length × embedding_dim)
//   For now, we'll return (sequence_length × embedding_dim) for simplicity
//
// Example:
//   vocab_size = 1000, embedding_dim = 4
//   embedding_table shape: (1000, 4)
//   token_ids: [5, 42]
//   output: [embedding_table[5], embedding_table[42]]
//          = (2 × 4) tensor
Tensor token_embedding(const Tensor& embedding_table, 
                       const std::vector<int>& token_ids);

// Positional Embedding (Learned)
// Adds position information to embeddings
//
// In transformers, we need to know which position each token is at
// (Position 0, position 1, position 2, ...)
//
// Args:
//   position_table: (max_position × embedding_dim)
//     Learned embeddings for each position
//   
//   sequence_length: How many positions to get
//     We'll return embeddings for positions [0, 1, 2, ..., sequence_length-1]
//
// Returns:
//   (sequence_length × embedding_dim)
//   One position embedding per position
//
// Example:
//   max_position = 2048, embedding_dim = 4
//   sequence_length = 3
//   output: [position_table[0], position_table[1], position_table[2]]
//          = (3 × 4) tensor
Tensor positional_embedding(const Tensor& position_table, 
                            int sequence_length);

// Combined: Token + Position embeddings
// This is what actually gets fed into the transformer
//
// Args:
//   token_table: (vocab_size × embedding_dim)
//   position_table: (max_position × embedding_dim)  
//   token_ids: sequence of token IDs to embed
//
// Returns:
//   (sequence_length × embedding_dim)
//   token_embedding + position_embedding for each position
//
// Example:
//   token_ids = [5, 42, 13]
//   output[0] = token_table[5] + position_table[0]
//   output[1] = token_table[42] + position_table[1]
//   output[2] = token_table[13] + position_table[2]
Tensor combined_embedding(const Tensor& token_table,
                         const Tensor& position_table,
                         const std::vector<int>& token_ids);

}  // namespace embeddings

#endif