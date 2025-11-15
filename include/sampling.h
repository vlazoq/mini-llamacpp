// Sampling strategies for text generation
// Convert model logits into actual token selections

#ifndef SAMPLING_H
#define SAMPLING_H

#include "tensor.h"
#include <vector>
#include <random>

namespace sampling {

// Sample from a probability distribution
// Args:
//   probabilities: Vector of probabilities (must sum to ~1.0)
//   rng: Random number generator
// Returns:
//   Index of sampled element
int sample_from_distribution(const std::vector<float>& probabilities, 
                            std::mt19937& rng);

// Greedy sampling: always pick the most likely token
// Args:
//   logits: Raw model outputs (1D vector for one position)
// Returns:
//   Index of token with highest logit
// 
// Characteristics:
//   - Deterministic (always same output for same input)
//   - Often repetitive and boring
//   - Good for factual tasks
int sample_greedy(const std::vector<float>& logits);

// Temperature sampling: control randomness
// Args:
//   logits: Raw model outputs
//   temperature: 
//     - temperature = 1.0: normal sampling
//     - temperature < 1.0: more focused (less random)
//     - temperature > 1.0: more random (more creative)
//     - temperature → 0: approaches greedy
//   rng: Random number generator
// Returns:
//   Sampled token index
//
// How it works:
//   1. Divide logits by temperature
//   2. Apply softmax to get probabilities
//   3. Sample from the distribution
//
// Examples:
//   logits = [1.0, 2.0, 3.0]
//   temp = 0.5 → focuses on token 2 (highest)
//   temp = 2.0 → spreads probability more evenly
int sample_temperature(const std::vector<float>& logits, 
                      float temperature,
                      std::mt19937& rng);

// Top-k sampling: only consider top k most likely tokens
// Args:
//   logits: Raw model outputs
//   k: Number of top tokens to consider
//   rng: Random number generator
// Returns:
//   Sampled token index
//
// How it works:
//   1. Find k tokens with highest logits
//   2. Set all other logits to -infinity
//   3. Apply softmax and sample
//
// Characteristics:
//   - Prevents sampling very unlikely tokens
//   - k=1 is equivalent to greedy
//   - k=vocab_size is equivalent to temperature sampling
//   - Common values: k=40 to k=100
int sample_top_k(const std::vector<float>& logits,
                int k,
                std::mt19937& rng);

// Top-p (nucleus) sampling: sample from smallest set of tokens 
// whose cumulative probability exceeds p
// Args:
//   logits: Raw model outputs
//   p: Cumulative probability threshold (e.g., 0.9)
//   rng: Random number generator
// Returns:
//   Sampled token index
//
// How it works:
//   1. Sort tokens by probability (descending)
//   2. Keep adding tokens until cumulative probability ≥ p
//   3. Sample only from these tokens
//
// Characteristics:
//   - Dynamic: number of considered tokens varies by context
//   - p=1.0 is equivalent to temperature sampling
//   - Common values: p=0.9 to p=0.95
//   - More adaptive than top-k
int sample_top_p(const std::vector<float>& logits,
                float p,
                std::mt19937& rng);

}  // namespace sampling

#endif