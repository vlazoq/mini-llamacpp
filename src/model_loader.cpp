#include "model_loader.h"
#include "quantization.h"
#include <stdexcept>
#include <sstream>
#include <cstring>

// Constructor
ModelLoader::ModelLoader(const std::string& filepath)
    : filepath(filepath), parser(filepath), parsed(false) {
}

// Get model info without full load
ModelLoader::ModelInfo ModelLoader::get_model_info() {
    if (!parsed) {
        parser.parse();
        parsed = true;
    }
    
    ModelInfo info;
    info.architecture = parser.get_string("general.architecture", "unknown");
    
    // Different architectures use different key names
    // Try common variations
    std::string prefix = info.architecture;
    
    info.n_vocab = parser.get_uint32(prefix + ".vocab_size", 
                                     parser.get_uint32("tokenizer.ggml.vocab_size", 32000));
    info.n_embd = parser.get_uint32(prefix + ".embedding_length",
                                    parser.get_uint32(prefix + ".n_embd", 4096));
    info.n_layers = parser.get_uint32(prefix + ".block_count",
                                      parser.get_uint32(prefix + ".n_layers", 32));
    info.n_heads = parser.get_uint32(prefix + ".attention.head_count",
                                     parser.get_uint32(prefix + ".n_heads", 32));
    info.n_ff = parser.get_uint32(prefix + ".feed_forward_length",
                                  parser.get_uint32(prefix + ".n_ff", info.n_embd * 4));
    info.max_seq_len = parser.get_uint32(prefix + ".context_length",
                                         parser.get_uint32(prefix + ".max_seq_len", 2048));
    
    return info;
}

// Extract config from metadata
model::ModelConfig ModelLoader::extract_config() {
    ModelInfo info = get_model_info();
    
    model::ModelConfig config(
        info.n_vocab,
        info.max_seq_len,
        info.n_embd,
        info.n_layers,
        info.n_heads,
        info.n_ff
    );
    
    return config;
}

// Load a tensor by name
Tensor ModelLoader::load_tensor(const std::string& name, const std::vector<int>& expected_shape) {
    if (!parsed) {
        parser.parse();
        parsed = true;
    }
    
    // Get tensor info
    const gguf::TensorInfo* info = parser.get_tensor_info(name);
    if (!info) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    // Validate shape if expected shape is provided
    if (!expected_shape.empty()) {
        if (info->shape.size() != expected_shape.size()) {
            std::ostringstream oss;
            oss << "Tensor " << name << " has wrong number of dimensions. "
                << "Expected " << expected_shape.size() 
                << ", got " << info->shape.size();
            throw std::runtime_error(oss.str());
        }
        
        for (size_t i = 0; i < expected_shape.size(); i++) {
            if (expected_shape[i] > 0 && 
                static_cast<uint64_t>(expected_shape[i]) != info->shape[i]) {
                std::ostringstream oss;
                oss << "Tensor " << name << " dimension " << i << " mismatch. "
                    << "Expected " << expected_shape[i] 
                    << ", got " << info->shape[i];
                throw std::runtime_error(oss.str());
            }
        }
    }
    
    // Read raw tensor data
    std::vector<uint8_t> raw_data = parser.read_tensor_data(name);
    
    // Dequantize if needed
    uint64_t n_elements = info->num_elements();
    
    switch (info->type) {
        case gguf::TensorType::F32:
            // Already float32, just reinterpret
            {
                std::vector<float> float_data(n_elements);
                memcpy(float_data.data(), raw_data.data(), n_elements * sizeof(float));
                
                // Convert shape to int vector
                std::vector<int> shape_int;
                for (uint64_t dim : info->shape) {
                    shape_int.push_back(static_cast<int>(dim));
                }
                
                return Tensor(shape_int, float_data);
            }
            
        case gguf::TensorType::F16:
            // TODO: Implement F16 dequantization
            throw std::runtime_error("F16 tensors not yet supported");
            
        case gguf::TensorType::Q4_0:
            return quantization::dequantize_q4_0(raw_data, n_elements);
            
        case gguf::TensorType::Q8_0:
            return quantization::dequantize_q8_0(raw_data, n_elements);
            
        default:
            throw std::runtime_error("Unsupported tensor type for: " + name);
    }
}

// Load token embedding
Tensor ModelLoader::load_token_embedding() {
    ModelInfo info = get_model_info();
    return load_tensor("token_embd.weight", {static_cast<int>(info.n_vocab), 
                                             static_cast<int>(info.n_embd)});
}

// Load positional embedding
Tensor ModelLoader::load_positional_embedding() {
    ModelInfo info = get_model_info();
    
    // Some models don't have explicit positional embeddings (they use RoPE instead)
    // Try to load it, but don't fail if it doesn't exist
    try {
        return load_tensor("position_embd.weight", {static_cast<int>(info.max_seq_len),
                                                    static_cast<int>(info.n_embd)});
    } catch (...) {
        // Return empty tensor if not found
        return Tensor({static_cast<int>(info.max_seq_len), 
                      static_cast<int>(info.n_embd)});
    }
}

// Load block weights
model::BlockWeights ModelLoader::load_block_weights(int block_idx) {
    ModelInfo info = get_model_info();
    
    std::string prefix = "blk." + std::to_string(block_idx) + ".";
    
    model::BlockWeights weights(info.n_embd, info.n_ff);
    
    // Attention weights
    weights.Wq = load_tensor(prefix + "attn_q.weight", 
                            {static_cast<int>(info.n_embd), 
                             static_cast<int>(info.n_embd)});
    weights.Wk = load_tensor(prefix + "attn_k.weight",
                            {static_cast<int>(info.n_embd),
                             static_cast<int>(info.n_embd)});
    weights.Wv = load_tensor(prefix + "attn_v.weight",
                            {static_cast<int>(info.n_embd),
                             static_cast<int>(info.n_embd)});
    weights.Wo = load_tensor(prefix + "attn_output.weight",
                            {static_cast<int>(info.n_embd),
                             static_cast<int>(info.n_embd)});
    
    // Feed-forward weights
    weights.W1 = load_tensor(prefix + "ffn_gate.weight",
                            {static_cast<int>(info.n_embd),
                             static_cast<int>(info.n_ff)});
    // Note: W1 and b1 might need different naming in GGUF
    weights.W2 = load_tensor(prefix + "ffn_down.weight",
                            {static_cast<int>(info.n_ff),
                             static_cast<int>(info.n_embd)});
    
    // Layer norm weights
    weights.gamma1 = load_tensor(prefix + "attn_norm.weight",
                                {static_cast<int>(info.n_embd), 1});
    weights.gamma2 = load_tensor(prefix + "ffn_norm.weight",
                                {static_cast<int>(info.n_embd), 1});
    
    return weights;
}

// Load final layer norm gamma
Tensor ModelLoader::load_final_norm_gamma() {
    ModelInfo info = get_model_info();
    return load_tensor("output_norm.weight", {static_cast<int>(info.n_embd), 1});
}

// Load final layer norm beta
Tensor ModelLoader::load_final_norm_beta() {
    ModelInfo info = get_model_info();
    // Beta might not exist in all models (often zero)
    try {
        return load_tensor("output_norm.bias", {static_cast<int>(info.n_embd), 1});
    } catch (...) {
        return Tensor({static_cast<int>(info.n_embd), 1});
    }
}

// Load output projection
Tensor ModelLoader::load_output_weight() {
    ModelInfo info = get_model_info();
    return load_tensor("output.weight", {static_cast<int>(info.n_embd),
                                         static_cast<int>(info.n_vocab)});
}

// Main load function
model::TransformerModel ModelLoader::load() {
    // Extract configuration
    model::ModelConfig config = extract_config();
    
    // Create weights structure
    model::ModelWeights weights(config);
    
    // Load embeddings
    weights.token_embeddings = load_token_embedding();
    weights.position_embeddings = load_positional_embedding();
    
    // Load transformer blocks
    for (int i = 0; i < config.num_layers; i++) {
        weights.blocks[i] = load_block_weights(i);
    }
    
    // Load final layers
    weights.final_norm_gamma = load_final_norm_gamma();
    weights.final_norm_beta = load_final_norm_beta();
    weights.output_weight = load_output_weight();
    
    // Create and return model
    return model::TransformerModel(config, weights);
}