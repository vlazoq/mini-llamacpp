#include "quantization.h"
#include <stdexcept>
#include <cstring>
#include <cmath>

namespace quantization {

// Convert float16 to float32
// IEEE 754 half-precision floating point format:
// - 1 bit sign
// - 5 bits exponent
// - 10 bits mantissa
float fp16_to_fp32(uint16_t h) {
    // Extract components
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    // Handle special cases
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return sign ? -0.0f : 0.0f;
        } else {
            // Denormalized number
            // Convert to normalized float32
            float value = mantissa / 1024.0f;  // 2^10
            value *= std::pow(2.0f, -14.0f);   // Min exponent for fp16
            return sign ? -value : value;
        }
    }
    
    if (exponent == 31) {
        // Infinity or NaN
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    }
    
    // Normalized number
    // Construct float32
    uint32_t f32_sign = sign << 31;
    uint32_t f32_exp = (exponent - 15 + 127) << 23;  // Rebias exponent
    uint32_t f32_mantissa = mantissa << 13;  // Shift mantissa to float32 position
    
    uint32_t f32_bits = f32_sign | f32_exp | f32_mantissa;
    
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Dequantize Q4_0 block
// Q4_0 format: 32 values stored as 4-bit integers (0-15)
// Packed 2 per byte, centered around 8
void dequantize_block_q4_0(const block_q4_0* block, float* output) {
    // Convert scale from float16 to float32
    float scale = fp16_to_fp32(block->scale);
    
    // Dequantize each value
    // Each byte contains 2 values: high 4 bits and low 4 bits
    for (size_t i = 0; i < QK4_0 / 2; i++) {
        uint8_t byte = block->qs[i];
        
        // Extract low 4 bits (first value)
        int8_t q0 = (byte & 0x0F) - 8;  // Center around 0
        output[i * 2] = q0 * scale;
        
        // Extract high 4 bits (second value)
        int8_t q1 = (byte >> 4) - 8;  // Center around 0
        output[i * 2 + 1] = q1 * scale;
    }
}

// Dequantize Q8_0 block
// Q8_0 format: 32 values stored as 8-bit integers (-128 to 127)
void dequantize_block_q8_0(const block_q8_0* block, float* output) {
    // Convert scale from float16 to float32
    float scale = fp16_to_fp32(block->scale);
    
    // Dequantize each value
    for (size_t i = 0; i < QK8_0; i++) {
        output[i] = block->qs[i] * scale;
    }
}

// Dequantize entire Q4_0 tensor
Tensor dequantize_q4_0(const std::vector<uint8_t>& data, size_t n_elements) {
    // Calculate number of blocks
    size_t n_blocks = (n_elements + QK4_0 - 1) / QK4_0;
    
    // Validate input size
    size_t expected_size = n_blocks * sizeof(block_q4_0);
    if (data.size() < expected_size) {
        throw std::runtime_error("Insufficient data for Q4_0 dequantization");
    }
    
    // Create output tensor
    std::vector<float> output(n_elements);
    
    // Dequantize each block
    const block_q4_0* blocks = reinterpret_cast<const block_q4_0*>(data.data());
    
    for (size_t i = 0; i < n_blocks; i++) {
        size_t offset = i * QK4_0;
        size_t n_values = std::min(QK4_0, n_elements - offset);
        
        // Dequantize full block to temporary buffer
        float temp[QK4_0];
        dequantize_block_q4_0(&blocks[i], temp);
        
        // Copy needed values to output
        for (size_t j = 0; j < n_values; j++) {
            output[offset + j] = temp[j];
        }
    }
    
    return Tensor({static_cast<int>(n_elements)}, output);
}

// Dequantize entire Q8_0 tensor
Tensor dequantize_q8_0(const std::vector<uint8_t>& data, size_t n_elements) {
    // Calculate number of blocks
    size_t n_blocks = (n_elements + QK8_0 - 1) / QK8_0;
    
    // Validate input size
    size_t expected_size = n_blocks * sizeof(block_q8_0);
    if (data.size() < expected_size) {
        throw std::runtime_error("Insufficient data for Q8_0 dequantization");
    }
    
    // Create output tensor
    std::vector<float> output(n_elements);
    
    // Dequantize each block
    const block_q8_0* blocks = reinterpret_cast<const block_q8_0*>(data.data());
    
    for (size_t i = 0; i < n_blocks; i++) {
        size_t offset = i * QK8_0;
        size_t n_values = std::min(QK8_0, n_elements - offset);
        
        // Dequantize full block to temporary buffer
        float temp[QK8_0];
        dequantize_block_q8_0(&blocks[i], temp);
        
        // Copy needed values to output
        for (size_t j = 0; j < n_values; j++) {
            output[offset + j] = temp[j];
        }
    }
    
    return Tensor({static_cast<int>(n_elements)}, output);
}

}  // namespace quantization