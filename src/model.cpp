#include "model.h"
#include "embeddings.h"
#include "transformer_block.h"
#include "layer_norm.h"
#include <stdexcept>
#include <random>

namespace model {

// BlockWeights constructor
// Initializes all weight matrices for one transformer block
BlockWeights::BlockWeights(int embedding_dim, int intermediate_dim)
    // Initialize all tensors with random small values
    // In a real model, these would be loaded from a file
    : Wq({embedding_dim, embedding_dim}),
      Wk({embedding_dim, embedding_dim}),
      Wv({embedding_dim, embedding_dim}),
      Wo({embedding_dim, embedding_dim}),
      W1({embedding_dim, intermediate_dim}),
      b1({1, intermediate_dim}),
      W2({intermediate_dim, embedding_dim}),
      b2({1, embedding_dim}),
      gamma1({1, embedding_dim}),
      beta1({1, embedding_dim}),
      gamma2({1, embedding_dim}),
      beta2({1, embedding_dim})
{
    // Initialize with small random values (toy weights for now)
    // We'll use a simple pattern for reproducibility
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    auto fill_random = [&](Tensor& t) {
        int size = t.get_total_size();
        std::vector<float> data(size);
        for (int i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
        t = Tensor(t.get_shape(), data);
    };
    
    // Fill all weight matrices
    fill_random(Wq);
    fill_random(Wk);
    fill_random(Wv);
    fill_random(Wo);
    fill_random(W1);
    fill_random(W2);
    
    // Biases start at zero
    std::vector<float> zeros_b1(intermediate_dim, 0.0f);
    b1 = Tensor({1, intermediate_dim}, zeros_b1);
    
    std::vector<float> zeros_b2(embedding_dim, 0.0f);
    b2 = Tensor({1, embedding_dim}, zeros_b2);
    
    // Layer norm: gamma=1, beta=0 (identity transform)
    std::vector<float> ones(embedding_dim, 1.0f);
    std::vector<float> zeros(embedding_dim, 0.0f);
    gamma1 = Tensor({1, embedding_dim}, ones);
    beta1 = Tensor({1, embedding_dim}, zeros);
    gamma2 = Tensor({1, embedding_dim}, ones);
    beta2 = Tensor({1, embedding_dim}, zeros);
}

// ModelWeights constructor
// Initializes all weights for the complete model
ModelWeights::ModelWeights(const ModelConfig& config)
    : token_embeddings({config.vocab_size, config.embedding_dim}),
      position_embeddings({config.max_seq_len, config.embedding_dim}),
      final_norm_gamma({1, config.embedding_dim}),
      final_norm_beta({1, config.embedding_dim}),
      output_weight({config.embedding_dim, config.vocab_size})
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    auto fill_random = [&](Tensor& t) {
        int size = t.get_total_size();
        std::vector<float> data(size);
        for (int i = 0; i < size; i++) {
            data[i] = dist(rng);
        }
        t = Tensor(t.get_shape(), data);
    };
    
    // Initialize embeddings
    fill_random(token_embeddings);
    fill_random(position_embeddings);
    
    // Create transformer blocks
    for (int i = 0; i < config.num_layers; i++) {
        blocks.emplace_back(config.embedding_dim, config.intermediate_dim);
    }
    
    // Final layer norm
    std::vector<float> ones(config.embedding_dim, 1.0f);
    std::vector<float> zeros(config.embedding_dim, 0.0f);
    final_norm_gamma = Tensor({1, config.embedding_dim}, ones);
    final_norm_beta = Tensor({1, config.embedding_dim}, zeros);
    
    // Output projection
    fill_random(output_weight);
}

// TransformerModel constructor
TransformerModel::TransformerModel(const ModelConfig& config, const ModelWeights& weights)
    : config(config), weights(weights) {
    // Validate configuration
    if (config.embedding_dim % config.num_heads != 0) {
        throw std::runtime_error("embedding_dim must be divisible by num_heads");
    }
    
    if (weights.blocks.size() != static_cast<size_t>(config.num_layers)) {
        throw std::runtime_error("Number of blocks doesn't match num_layers");
    }
}

// Forward pass through the complete model
Tensor TransformerModel::forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    
    // Validate sequence length
    if (seq_len > config.max_seq_len) {
        throw std::runtime_error("Sequence length exceeds max_seq_len");
    }
    
    // Step 1: Embed tokens (token + positional embeddings)
    Tensor x = embeddings::combined_embedding(
        weights.token_embeddings,
        weights.position_embeddings,
        token_ids
    );
    
    // Step 2: Pass through all transformer blocks
    for (int layer = 0; layer < config.num_layers; layer++) {
        const BlockWeights& block = weights.blocks[layer];
        
        x = transformer::transformer_block(
            x,
            block.Wq, block.Wk, block.Wv, block.Wo,
            config.num_heads,
            block.W1, block.b1, block.W2, block.b2,
            block.gamma1, block.beta1, block.gamma2, block.beta2
        );
    }
    
    // Step 3: Final layer normalization
    x = layer_norm::layer_norm(x, weights.final_norm_gamma, weights.final_norm_beta);
    
    // Step 4: Project to vocabulary (get logits)
    // x: (seq_len × embedding_dim)
    // output_weight: (embedding_dim × vocab_size)
    // logits: (seq_len × vocab_size)
    Tensor logits = Tensor::matmul(x, weights.output_weight);
    
    return logits;
}

}  // namespace model