// Complete Transformer Model
// Stacks multiple transformer blocks for deep learning

#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include <vector>

namespace model {

// Configuration for a transformer model
// These parameters define the model architecture
struct ModelConfig {
    int vocab_size;        // Number of tokens in vocabulary (e.g., 32000 for LLaMA)
    int max_seq_len;       // Maximum sequence length (e.g., 2048)
    int embedding_dim;     // Dimension of embeddings (e.g., 4096)
    int num_layers;        // Number of transformer blocks (e.g., 32)
    int num_heads;         // Number of attention heads (e.g., 32)
    int intermediate_dim;  // Feedforward intermediate dimension (e.g., 11008)
    
    // Constructor with defaults for tiny test model
    ModelConfig(
        int vocab_size = 100,
        int max_seq_len = 128,
        int embedding_dim = 64,
        int num_layers = 4,
        int num_heads = 4,
        int intermediate_dim = 256
    ) : vocab_size(vocab_size),
        max_seq_len(max_seq_len),
        embedding_dim(embedding_dim),
        num_layers(num_layers),
        num_heads(num_heads),
        intermediate_dim(intermediate_dim) {}
};

// Weights for a single transformer block
// Packages all the weight matrices for one layer
struct BlockWeights {
    // Attention weights
    Tensor Wq;  // Query projection
    Tensor Wk;  // Key projection
    Tensor Wv;  // Value projection
    Tensor Wo;  // Output projection
    
    // Feedforward weights
    Tensor W1;  // First layer
    Tensor b1;  // First bias
    Tensor W2;  // Second layer
    Tensor b2;  // Second bias
    
    // Layer norm parameters
    Tensor gamma1;  // First layer norm scale
    Tensor beta1;   // First layer norm shift
    Tensor gamma2;  // Second layer norm scale
    Tensor beta2;   // Second layer norm shift
    
    // Constructor - creates tensors with given dimensions
    BlockWeights(int embedding_dim, int intermediate_dim);
};

// Complete model weights
// Contains all parameters for the entire model
struct ModelWeights {
    // Embedding tables
    Tensor token_embeddings;      // (vocab_size × embedding_dim)
    Tensor position_embeddings;   // (max_seq_len × embedding_dim)
    
    // Transformer blocks (one per layer)
    std::vector<BlockWeights> blocks;
    
    // Final layer norm (before output projection)
    Tensor final_norm_gamma;
    Tensor final_norm_beta;
    
    // Output projection (embeddings → logits over vocabulary)
    Tensor output_weight;  // (embedding_dim × vocab_size)
    
    // Constructor - creates all weights based on config
    ModelWeights(const ModelConfig& config);
};

// Complete Transformer Model
// This is the main inference class
class TransformerModel {
public:
    // Constructor
    TransformerModel(const ModelConfig& config, const ModelWeights& weights);
    
    // Forward pass: token_ids → logits
    // Args:
    //   token_ids: Input token IDs (vector of integers)
    // Returns:
    //   Tensor of shape (seq_len × vocab_size)
    //   Each row contains logits for that position's next token
    Tensor forward(const std::vector<int>& token_ids);
    
    // Get model configuration
    const ModelConfig& get_config() const { return config; }

private:
    ModelConfig config;
    ModelWeights weights;
};

}  // namespace model

#endif