#include "attention.h"
#include "activations.h"
#include <cmath>
#include <stdexcept>

namespace attention {

// Scaled dot-product attention implementation
Tensor scaled_dot_product_attention(const Tensor& Q, const Tensor& K, 
                                    const Tensor& V) {
    // Validate: all must be 2D with same dimensions
    if (Q.get_shape().size() != 2 || K.get_shape().size() != 2 || 
        V.get_shape().size() != 2) {
        throw std::runtime_error("Q, K, V must be 2D tensors");
    }
    
    int seq_len_q = Q.get_shape()[0];
    int head_dim = Q.get_shape()[1];
    int seq_len_k = K.get_shape()[0];
    int seq_len_v = V.get_shape()[0];
    
    // Validate dimensions match
    if (K.get_shape()[1] != head_dim || V.get_shape()[1] != head_dim) {
        throw std::runtime_error("Q, K, V must have same head_dim");
    }
    
    if (seq_len_k != seq_len_v) {
        throw std::runtime_error("K and V must have same sequence length");
    }
    
    // Step 1: Compute attention scores: Q × K^T
    // Q: (seq_len_q × head_dim)
    // K^T: (head_dim × seq_len_k)
    // scores: (seq_len_q × seq_len_k)
    // Each element [i,j] = how much query i attends to key j
    Tensor K_transpose = Tensor::transpose(K);
    Tensor scores = Tensor::matmul(Q, K_transpose);
    
    // Step 2: Scale by √head_dim
    // Prevents extreme values in softmax
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor scaled_scores = Tensor::scale(scores, scale_factor);
    
    // Step 3: Apply softmax to get attention weights (probabilities)
    // Each row becomes a probability distribution over keys
    // Sum of each row = 1.0
    Tensor attention_weights = activations::softmax(scaled_scores);
    
    // Step 4: Apply attention to values: attention_weights × V
    // attention_weights: (seq_len_q × seq_len_k)
    // V: (seq_len_v × head_dim) where seq_len_v = seq_len_k
    // output: (seq_len_q × head_dim)
    // Each output row is a weighted combination of V rows
    Tensor output = Tensor::matmul(attention_weights, V);
    
    return output;
}

// Multi-head attention implementation
Tensor multi_head_attention(const Tensor& input,
                            const Tensor& Wq,
                            const Tensor& Wk,
                            const Tensor& Wv,
                            const Tensor& Wo,
                            int num_heads) {
    // Validate input
    if (input.get_shape().size() != 2) {
        throw std::runtime_error("Input must be 2D (seq_len × embedding_dim)");
    }
    
    int seq_len = input.get_shape()[0];
    int embedding_dim = input.get_shape()[1];
    
    // Validate weight matrices
    if (Wq.get_shape()[0] != embedding_dim || Wq.get_shape()[1] != embedding_dim ||
        Wk.get_shape()[0] != embedding_dim || Wk.get_shape()[1] != embedding_dim ||
        Wv.get_shape()[0] != embedding_dim || Wv.get_shape()[1] != embedding_dim ||
        Wo.get_shape()[0] != embedding_dim || Wo.get_shape()[1] != embedding_dim) {
        throw std::runtime_error("All weight matrices must be (embedding_dim × embedding_dim)");
    }
    
    // Validate num_heads divides embedding_dim evenly
    if (embedding_dim % num_heads != 0) {
        throw std::runtime_error("embedding_dim must be divisible by num_heads");
    }
    
    int head_dim = embedding_dim / num_heads;
    
    // Step 1: Project input to Q, K, V
    // input: (seq_len × embedding_dim)
    // Wq, Wk, Wv: (embedding_dim × embedding_dim)
    // Q, K, V: (seq_len × embedding_dim)
    Tensor Q = Tensor::matmul(input, Wq);
    Tensor K = Tensor::matmul(input, Wk);
    Tensor V = Tensor::matmul(input, Wv);
    
    // Step 2 & 3: Split into heads and apply attention for each head
    // We'll process heads sequentially and collect outputs
    
    // Storage for all head outputs
    // We'll build a tensor with all head outputs concatenated
    std::vector<float> all_head_outputs;
    all_head_outputs.reserve(seq_len * embedding_dim);
    
    // Process each head
    for (int h = 0; h < num_heads; h++) {
        // Extract this head's portion from Q, K, V
        // Head h uses columns [h*head_dim : (h+1)*head_dim]
        
        // Create tensors for this head's Q, K, V
        Tensor Q_head({seq_len, head_dim});
        Tensor K_head({seq_len, head_dim});
        Tensor V_head({seq_len, head_dim});
        
        int start_col = h * head_dim;
        
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                Q_head.set(i, j, Q.at(i, start_col + j));
                K_head.set(i, j, K.at(i, start_col + j));
                V_head.set(i, j, V.at(i, start_col + j));
            }
        }
        
        // Apply attention for this head
        Tensor head_output = scaled_dot_product_attention(Q_head, K_head, V_head);
        
        // Collect this head's output
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < head_dim; j++) {
                all_head_outputs.push_back(head_output.at(i, j));
            }
        }
    }
    
    // Step 4: Concatenate all heads
    // all_head_outputs is now seq_len × embedding_dim in row-major order
    Tensor concatenated({seq_len, embedding_dim}, all_head_outputs);
    
    // Step 5: Project output: concatenated × Wo
    Tensor output = Tensor::matmul(concatenated, Wo);
    
    return output;
}

}  // namespace attention