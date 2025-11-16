// Model Loader - Loads models from GGUF files
// Ties together GGUF parsing, quantization, and model initialization

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include <string>
#include <memory>
#include "model.h"
#include "gguf_parser.h"

// Load a model from a GGUF file
// This is the main entry point for loading real models
class ModelLoader {
public:
    // Constructor
    // Args:
    //   filepath: Path to .gguf model file
    explicit ModelLoader(const std::string& filepath);
    
    // Load the model
    // This parses the GGUF file, extracts configuration,
    // loads all tensors, dequantizes them, and creates a Model
    // Returns:
    //   Fully initialized TransformerModel ready for inference
    model::TransformerModel load();
    
    // Get model info without loading (quick metadata check)
    // Useful for displaying model details before loading
    struct ModelInfo {
        std::string architecture;  // "llama", "mistral", etc.
        uint32_t n_vocab;         // Vocabulary size
        uint32_t n_embd;          // Embedding dimension
        uint32_t n_layers;        // Number of transformer blocks
        uint32_t n_heads;         // Number of attention heads
        uint32_t n_ff;            // Feed-forward hidden dimension
        uint32_t max_seq_len;     // Maximum sequence length
    };
    
    ModelInfo get_model_info();

private:
    std::string filepath;
    gguf::GGUFParser parser;
    bool parsed;
    
    // Extract model configuration from GGUF metadata
    model::ModelConfig extract_config();
    
    // Load a specific tensor by name
    // Handles both quantized and F32 tensors
    // Args:
    //   name: Tensor name (e.g., "token_embd.weight")
    //   expected_shape: Expected tensor dimensions for validation
    // Returns:
    //   Loaded tensor (dequantized if necessary)
    Tensor load_tensor(const std::string& name, const std::vector<int>& expected_shape);
    
    // Load embedding weights
    Tensor load_token_embedding();
    Tensor load_positional_embedding();
    
    // Load single transformer block weights
    // Args:
    //   block_idx: Block number (0 to n_layers-1)
    // Returns:
    //   BlockWeights for this block
    model::BlockWeights load_block_weights(int block_idx);
    
    // Load final layer norm and output projection
    Tensor load_final_norm_gamma();
    Tensor load_final_norm_beta();
    Tensor load_output_weight();
};

#endif