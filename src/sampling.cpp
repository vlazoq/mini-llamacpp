#include "sampling.h"
#include "activations.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace sampling {

// Sample from a discrete probability distribution
int sample_from_distribution(const std::vector<float>& probabilities, 
                            std::mt19937& rng) {
    // Create distribution from probabilities
    std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
    return dist(rng);
}

// Greedy sampling: pick argmax
int sample_greedy(const std::vector<float>& logits) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }
    
    // Find index of maximum value
    int max_idx = 0;
    float max_val = logits[0];
    
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

// Temperature sampling
int sample_temperature(const std::vector<float>& logits, 
                      float temperature,
                      std::mt19937& rng) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }
    
    if (temperature <= 0.0f) {
        throw std::runtime_error("Temperature must be positive");
    }
    
    // Special case: very low temperature → greedy
    if (temperature < 1e-6f) {
        return sample_greedy(logits);
    }
    
    // Apply temperature scaling
    std::vector<float> scaled_logits(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        scaled_logits[i] = logits[i] / temperature;
    }
    
    // Convert to probabilities using softmax
    // Create a 1×n tensor for softmax
    Tensor logits_tensor({1, static_cast<int>(logits.size())}, scaled_logits);
    Tensor probs_tensor = activations::softmax(logits_tensor);
    
    // Extract probabilities
    std::vector<float> probabilities(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        probabilities[i] = probs_tensor.at(0, i);
    }
    
    // Sample from distribution
    return sample_from_distribution(probabilities, rng);
}

// Top-k sampling
int sample_top_k(const std::vector<float>& logits,
                int k,
                std::mt19937& rng) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }
    
    if (k <= 0) {
        throw std::runtime_error("k must be positive");
    }
    
    int vocab_size = logits.size();
    k = std::min(k, vocab_size);  // Can't exceed vocabulary size
    
    // Special case: k=1 is greedy
    if (k == 1) {
        return sample_greedy(logits);
    }
    
    // Create pairs of (logit, index)
    std::vector<std::pair<float, int>> logit_pairs;
    for (int i = 0; i < vocab_size; i++) {
        logit_pairs.push_back({logits[i], i});
    }
    
    // Partial sort to find top k (descending by logit value)
    std::partial_sort(
        logit_pairs.begin(),
        logit_pairs.begin() + k,
        logit_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Extract top k logits
    std::vector<float> top_k_logits(k);
    for (int i = 0; i < k; i++) {
        top_k_logits[i] = logit_pairs[i].first;
    }
    
    // Apply softmax to top k
    Tensor logits_tensor({1, k}, top_k_logits);
    Tensor probs_tensor = activations::softmax(logits_tensor);
    
    std::vector<float> probabilities(k);
    for (int i = 0; i < k; i++) {
        probabilities[i] = probs_tensor.at(0, i);
    }
    
    // Sample from top k
    int top_k_idx = sample_from_distribution(probabilities, rng);
    
    // Return original vocabulary index
    return logit_pairs[top_k_idx].second;
}

// Top-p (nucleus) sampling
int sample_top_p(const std::vector<float>& logits,
                float p,
                std::mt19937& rng) {
    if (logits.empty()) {
        throw std::runtime_error("Cannot sample from empty logits");
    }
    
    if (p <= 0.0f || p > 1.0f) {
        throw std::runtime_error("p must be in (0, 1]");
    }
    
    int vocab_size = logits.size();
    
    // Convert logits to probabilities
    Tensor logits_tensor({1, vocab_size}, logits);
    Tensor probs_tensor = activations::softmax(logits_tensor);
    
    // Create pairs of (probability, index)
    std::vector<std::pair<float, int>> prob_pairs;
    for (int i = 0; i < vocab_size; i++) {
        prob_pairs.push_back({probs_tensor.at(0, i), i});
    }
    
    // Sort by probability (descending)
    std::sort(
        prob_pairs.begin(),
        prob_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );
    
    // Find nucleus: smallest set with cumulative probability ≥ p
    float cumulative_prob = 0.0f;
    int nucleus_size = 0;
    
    for (int i = 0; i < vocab_size; i++) {
        cumulative_prob += prob_pairs[i].first;
        nucleus_size++;
        
        if (cumulative_prob >= p) {
            break;
        }
    }
    
    // Extract nucleus probabilities
    std::vector<float> nucleus_probs(nucleus_size);
    for (int i = 0; i < nucleus_size; i++) {
        nucleus_probs[i] = prob_pairs[i].first;
    }
    
    // Renormalize (should already sum to ~p, but let's be precise)
    float sum = std::accumulate(nucleus_probs.begin(), nucleus_probs.end(), 0.0f);
    for (int i = 0; i < nucleus_size; i++) {
        nucleus_probs[i] /= sum;
    }
    
    // Sample from nucleus
    int nucleus_idx = sample_from_distribution(nucleus_probs, rng);
    
    // Return original vocabulary index
    return prob_pairs[nucleus_idx].second;
}

}  // namespace sampling