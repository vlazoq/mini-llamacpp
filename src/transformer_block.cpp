#include "transformer_block.h"
#include "layer_norm.h"
#include "attention.h"
#include "feedforward.h"
#include <stdexcept>

namespace transformer {

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
) {
    // Validate input
    if (input.get_shape().size() != 2) {
        throw std::runtime_error("Input must be 2D (seq_len × embedding_dim)");
    }
    
    // ===== ATTENTION SUB-LAYER =====
    
    // Step 1: Apply first layer normalization
    // Normalizes the input for stable attention computation
    Tensor norm1 = layer_norm::layer_norm(input, gamma1, beta1);
    
    // Step 2: Apply multi-head attention
    // Allows tokens to gather information from other tokens
    Tensor attn_output = attention::multi_head_attention(
        norm1, Wq, Wk, Wv, Wo, num_heads
    );
    
    // Step 3: Residual connection (add input back)
    // Helps gradients flow and preserves original information
    // attn_residual = input + attn_output
    Tensor attn_residual = Tensor::add(input, attn_output);
    
    // ===== FEEDFORWARD SUB-LAYER =====
    
    // Step 4: Apply second layer normalization
    // Normalizes before feedforward computation
    Tensor norm2 = layer_norm::layer_norm(attn_residual, gamma2, beta2);
    
    // Step 5: Apply feedforward network
    // Processes each token independently with non-linear transformation
    Tensor ff_output = feedforward::feedforward(norm2, W1, b1, W2, b2);
    
    // Step 6: Residual connection (add attention output back)
    // ff_residual = attn_residual + ff_output
    Tensor output = Tensor::add(attn_residual, ff_output);
    
    return output;
}

}  // namespace transformer