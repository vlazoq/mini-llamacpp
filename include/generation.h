// Text generation using autoregressive sampling
// Generate text token-by-token from a model

#ifndef GENERATION_H
#define GENERATION_H

#include "model.h"
#include "sampling.h"
#include <vector>
#include <random>
#include <functional>

namespace generation {

// Generation configuration
// Controls how text is generated
struct GenerationConfig {
    int max_tokens;           // Maximum tokens to generate
    float temperature;        // Sampling temperature (default: 1.0)
    int top_k;               // Top-k sampling (0 = disabled)
    float top_p;             // Top-p sampling (1.0 = disabled)
    bool use_greedy;         // Use greedy sampling (overrides others)
    unsigned int seed;       // Random seed for reproducibility
    
    // Callback function called after each token generation
    // Args: token_id, is_final_token
    // Return: true to continue, false to stop early
    std::function<bool(int, bool)> token_callback;
    
    // Constructor with sensible defaults
    GenerationConfig(
        int max_tokens = 50,
        float temperature = 1.0f,
        int top_k = 0,
        float top_p = 1.0f,
        bool use_greedy = false,
        unsigned int seed = 42
    ) : max_tokens(max_tokens),
        temperature(temperature),
        top_k(top_k),
        top_p(top_p),
        use_greedy(use_greedy),
        seed(seed),
        token_callback(nullptr) {}
};

// Generate text autoregressively
// Args:
//   model: The transformer model to use
//   prompt_tokens: Initial token sequence to start generation
//   config: Generation configuration
// Returns:
//   Vector of all tokens (prompt + generated)
//
// Process:
//   1. Start with prompt_tokens
//   2. Loop until max_tokens or stop condition:
//      a. Run model forward pass
//      b. Get logits for last position
//      c. Sample next token using config
//      d. Append to sequence
//      e. Call callback if provided
//   3. Return complete sequence
//
// Example:
//   prompt = [15, 42]
//   max_tokens = 3
//   Result might be: [15, 42, 103, 87, 56]
//                     ^^^^^^^^  ^^^^^^^^^^^
//                     prompt    generated
std::vector<int> generate(
    model::TransformerModel& model,
    const std::vector<int>& prompt_tokens,
    const GenerationConfig& config
);

// Sample next token from logits using generation config
// This is a helper that routes to the appropriate sampling strategy
// Args:
//   logits: Model output logits for one position
//   config: Generation configuration
//   rng: Random number generator
// Returns:
//   Sampled token ID
int sample_next_token(
    const std::vector<float>& logits,
    const GenerationConfig& config,
    std::mt19937& rng
);

}  // namespace generation

#endif