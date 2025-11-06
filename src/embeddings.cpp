#include "embeddings.h"
#include <stdexcept>

namespace embeddings {

// Token embedding: lookup vectors for each token ID
Tensor token_embedding(const Tensor& embedding_table, 
                       const std::vector<int>& token_ids) {
    // Validate: embedding_table must be 2D (vocab_size × embedding_dim)
    if (embedding_table.get_shape().size() != 2) {
        throw std::runtime_error("Embedding table must be 2D");
    }
    
    int vocab_size = embedding_table.get_shape()[0];
    int embedding_dim = embedding_table.get_shape()[1];
    int sequence_length = token_ids.size();
    
    // Validate: all token IDs must be valid (in range [0, vocab_size))
    for (int token_id : token_ids) {
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error(
                "Token ID " + std::to_string(token_id) + 
                " out of range [0, " + std::to_string(vocab_size) + ")"
            );
        }
    }
    
    // Create output tensor: (sequence_length × embedding_dim)
    Tensor result({sequence_length, embedding_dim});
    
    // For each token, copy its embedding from the table
    for (int i = 0; i < sequence_length; i++) {
        int token_id = token_ids[i];
        
        // Copy the entire row for this token
        // embedding_table[token_id] → result[i]
        for (int j = 0; j < embedding_dim; j++) {
            result.set(i, j, embedding_table.at(token_id, j));
        }
    }
    
    return result;
}

// Positional embedding: get position vectors for a sequence
Tensor positional_embedding(const Tensor& position_table, 
                            int sequence_length) {
    // Validate: position_table must be 2D
    if (position_table.get_shape().size() != 2) {
        throw std::runtime_error("Position table must be 2D");
    }
    
    int max_position = position_table.get_shape()[0];
    int embedding_dim = position_table.get_shape()[1];
    
    // Validate: sequence length must fit in position table
    if (sequence_length > max_position) {
        throw std::runtime_error(
            "Sequence length " + std::to_string(sequence_length) +
            " exceeds max position " + std::to_string(max_position)
        );
    }
    
    // Create output: (sequence_length × embedding_dim)
    Tensor result({sequence_length, embedding_dim});
    
    // Copy position embeddings for positions [0, 1, 2, ..., sequence_length-1]
    for (int pos = 0; pos < sequence_length; pos++) {
        for (int j = 0; j < embedding_dim; j++) {
            result.set(pos, j, position_table.at(pos, j));
        }
    }
    
    return result;
}

// Combined embedding: token + position
Tensor combined_embedding(const Tensor& token_table,
                         const Tensor& position_table,
                         const std::vector<int>& token_ids) {
    // Get token embeddings
    Tensor token_emb = token_embedding(token_table, token_ids);
    
    // Get position embeddings
    int sequence_length = token_ids.size();
    Tensor pos_emb = positional_embedding(position_table, sequence_length);
    
    // Validate: dimensions must match
    if (token_emb.get_shape()[1] != pos_emb.get_shape()[1]) {
        throw std::runtime_error("Token and position embedding dimensions must match");
    }
    
    // Add them together: element-wise addition
    Tensor result = Tensor::add(token_emb, pos_emb);
    
    return result;
}

}  // namespace embeddings