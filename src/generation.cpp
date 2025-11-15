#include "generation.h"
#include <stdexcept>
#include <iostream>

namespace generation {

// Sample next token based on generation config
int sample_next_token(
    const std::vector<float>& logits,
    const GenerationConfig& config,
    std::mt19937& rng
) {
    // Route to appropriate sampling strategy based on config
    
    if (config.use_greedy) {
        // Greedy: always pick most likely
        return sampling::sample_greedy(logits);
    }
    
    // Apply top-k if specified
    if (config.top_k > 0 && config.top_k < static_cast<int>(logits.size())) {
        return sampling::sample_top_k(logits, config.top_k, rng);
    }
    
    // Apply top-p if specified
    if (config.top_p < 1.0f) {
        return sampling::sample_top_p(logits, config.top_p, rng);
    }
    
    // Default: temperature sampling
    return sampling::sample_temperature(logits, config.temperature, rng);
}

// Main generation loop
std::vector<int> generate(
    model::TransformerModel& model,
    const std::vector<int>& prompt_tokens,
    const GenerationConfig& config
) {
    // Validate inputs
    if (prompt_tokens.empty()) {
        throw std::runtime_error("Prompt cannot be empty");
    }
    
    if (config.max_tokens <= 0) {
        throw std::runtime_error("max_tokens must be positive");
    }
    
    const auto& model_config = model.get_config();
    
    // Validate prompt doesn't exceed model's max sequence length
    if (static_cast<int>(prompt_tokens.size()) > model_config.max_seq_len) {
        throw std::runtime_error("Prompt exceeds model's max_seq_len");
    }
    
    // Initialize random number generator
    std::mt19937 rng(config.seed);
    
    // Start with prompt tokens
    std::vector<int> generated_tokens = prompt_tokens;
    
    // Generate tokens one at a time
    for (int i = 0; i < config.max_tokens; i++) {
        // Check if we've hit sequence length limit
        if (static_cast<int>(generated_tokens.size()) >= model_config.max_seq_len) {
            std::cerr << "Warning: Reached max_seq_len, stopping generation" << std::endl;
            break;
        }
        
        // Forward pass through model
        // Note: In a real implementation with KV cache, we'd only process the new token
        // For now, we reprocess the entire sequence each time (inefficient but correct)
        Tensor logits = model.forward(generated_tokens);
        
        // Get logits for the last position (next token prediction)
        // logits shape: (seq_len × vocab_size)
        int last_pos = generated_tokens.size() - 1;
        int vocab_size = model_config.vocab_size;
        
        std::vector<float> last_logits(vocab_size);
        for (int j = 0; j < vocab_size; j++) {
            last_logits[j] = logits.at(last_pos, j);
        }
        
        // Sample next token
        int next_token = sample_next_token(last_logits, config, rng);
        
        // Validate token is in vocabulary
        if (next_token < 0 || next_token >= vocab_size) {
            throw std::runtime_error("Sampled token out of vocabulary range");
        }
        
        // Append to sequence
        generated_tokens.push_back(next_token);
        
        // Call callback if provided
        bool is_final = (i == config.max_tokens - 1);
        if (config.token_callback) {
            bool should_continue = config.token_callback(next_token, is_final);
            if (!should_continue) {
                // Callback requested early stop
                break;
            }
        }
        
        // In a real system, we'd check for end-of-sequence token here
        // For now, we just generate until max_tokens
    }
    
    return generated_tokens;
}

}  // namespace generation