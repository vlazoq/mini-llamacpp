#include "feedforward.h"
#include "activations.h"
#include <stdexcept>

namespace feedforward {

Tensor feedforward(const Tensor& input,
                  const Tensor& W1,
                  const Tensor& b1,
                  const Tensor& W2,
                  const Tensor& b2) {
    // Validate input
    if (input.get_shape().size() != 2) {
        throw std::runtime_error("Input must be 2D (seq_len × embedding_dim)");
    }
    
    int seq_len = input.get_shape()[0];
    int embedding_dim = input.get_shape()[1];
    
    // Validate W1 and b1
    if (W1.get_shape().size() != 2 || W1.get_shape()[0] != embedding_dim) {
        throw std::runtime_error("W1 must be (embedding_dim × intermediate_dim)");
    }
    
    int intermediate_dim = W1.get_shape()[1];
    
    if (b1.get_shape().size() != 2 || b1.get_shape()[0] != 1 || 
        b1.get_shape()[1] != intermediate_dim) {
        throw std::runtime_error("b1 must be (1 × intermediate_dim)");
    }
    
    // Validate W2 and b2
    if (W2.get_shape().size() != 2 || 
        W2.get_shape()[0] != intermediate_dim ||
        W2.get_shape()[1] != embedding_dim) {
        throw std::runtime_error("W2 must be (intermediate_dim × embedding_dim)");
    }
    
    if (b2.get_shape().size() != 2 || b2.get_shape()[0] != 1 || 
        b2.get_shape()[1] != embedding_dim) {
        throw std::runtime_error("b2 must be (1 × embedding_dim)");
    }
    
    // Step 1: First linear transformation
    // input: (seq_len × embedding_dim)
    // W1: (embedding_dim × intermediate_dim)
    // hidden = input × W1: (seq_len × intermediate_dim)
    Tensor hidden = Tensor::matmul(input, W1);
    
    // Add bias b1 to each row
    // b1: (1 × intermediate_dim)
    // We need to add it to every row of hidden
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < intermediate_dim; j++) {
            float val = hidden.at(i, j) + b1.at(0, j);
            hidden.set(i, j, val);
        }
    }
    
    // Step 2: Apply GELU activation
    // Introduces non-linearity
    hidden = activations::gelu(hidden);
    
    // Step 3: Second linear transformation
    // hidden: (seq_len × intermediate_dim)
    // W2: (intermediate_dim × embedding_dim)
    // output = hidden × W2: (seq_len × embedding_dim)
    Tensor output = Tensor::matmul(hidden, W2);
    
    // Add bias b2 to each row
    // b2: (1 × embedding_dim)
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < embedding_dim; j++) {
            float val = output.at(i, j) + b2.at(0, j);
            output.set(i, j, val);
        }
    }
    
    return output;
}

}  // namespace feedforward