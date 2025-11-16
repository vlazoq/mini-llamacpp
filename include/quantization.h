// Quantization/Dequantization for GGUF weights
// Handles conversion from quantized formats (Q4_0, Q8_0) to F32

#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <vector>
#include <cstdint>
#include "tensor.h"

namespace quantization {

// Block size for quantization (GGML standard)
constexpr size_t QK4_0 = 32;  // Q4_0 block size
constexpr size_t QK8_0 = 32;  // Q8_0 block size

// Q4_0: 4-bit quantization
// Each block contains:
// - 1 float16 scale factor (2 bytes)
// - 16 bytes of quantized data (32 4-bit values, packed 2 per byte)
// Total: 18 bytes per block of 32 values
struct block_q4_0 {
    uint16_t scale;     // float16 stored as uint16
    uint8_t qs[16];     // Quantized values (2 per byte)
};

// Q8_0: 8-bit quantization
// Each block contains:
// - 1 float16 scale factor (2 bytes)
// - 32 int8 quantized values
// Total: 34 bytes per block of 32 values
struct block_q8_0 {
    uint16_t scale;     // float16 stored as uint16
    int8_t qs[32];      // Quantized values (one per element)
};

// Convert float16 (stored as uint16) to float32
// Uses IEEE 754 half-precision format
float fp16_to_fp32(uint16_t h);

// Dequantize Q4_0 block to float32 values
// Args:
//   block: Pointer to Q4_0 block
//   output: Output array (must have space for QK4_0 floats)
void dequantize_block_q4_0(const block_q4_0* block, float* output);

// Dequantize Q8_0 block to float32 values
// Args:
//   block: Pointer to Q8_0 block
//   output: Output array (must have space for QK8_0 floats)
void dequantize_block_q8_0(const block_q8_0* block, float* output);

// Dequantize entire Q4_0 tensor
// Args:
//   data: Raw Q4_0 data (must be valid block_q4_0 array)
//   n_elements: Total number of float elements to produce
// Returns:
//   Tensor containing dequantized float32 values
Tensor dequantize_q4_0(const std::vector<uint8_t>& data, size_t n_elements);

// Dequantize entire Q8_0 tensor
// Args:
//   data: Raw Q8_0 data (must be valid block_q8_0 array)
//   n_elements: Total number of float elements to produce
// Returns:
//   Tensor containing dequantized float32 values
Tensor dequantize_q8_0(const std::vector<uint8_t>& data, size_t n_elements);

}  // namespace quantization

#endif