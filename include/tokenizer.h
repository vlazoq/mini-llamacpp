// Tokenizer: Convert between text and token IDs
// Phase 3A: Simple character-level tokenizer
// Phase 3B: Will upgrade to BPE tokenizer

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>

namespace tokenizer {

// Simple character-level tokenizer
// Maps each character to a unique token ID
// 
// Vocabulary:
//   - ASCII characters (0-127)
//   - Special tokens: <PAD>, <UNK>, <BOS>, <EOS>
//
// Limitations (for Phase 3A):
//   - Only supports ASCII
//   - One character = one token (inefficient)
//   - Large vocabulary for real text
//
// Phase 3B will add:
//   - BPE (Byte-Pair Encoding)
//   - SentencePiece integration
//   - Unicode support
class CharTokenizer {
public:
    // Special token IDs
    static constexpr int PAD_TOKEN = 0;   // Padding token
    static constexpr int UNK_TOKEN = 1;   // Unknown character
    static constexpr int BOS_TOKEN = 2;   // Beginning of sequence
    static constexpr int EOS_TOKEN = 3;   // End of sequence
    
    // Constructor
    // Args:
    //   add_special_tokens: If true, prepend BOS and append EOS to sequences
    CharTokenizer(bool add_special_tokens = false);
    
    // Encode text to token IDs
    // Args:
    //   text: Input string
    // Returns:
    //   Vector of token IDs
    //
    // Example:
    //   encode("Hi") → [72, 105] (ASCII values for 'H', 'i')
    //   If add_special_tokens=true: [2, 72, 105, 3] (BOS, 'H', 'i', EOS)
    std::vector<int> encode(const std::string& text) const;
    
    // Decode token IDs to text
    // Args:
    //   tokens: Vector of token IDs
    // Returns:
    //   Decoded string
    //
    // Example:
    //   decode([72, 105]) → "Hi"
    //
    // Special tokens are rendered as <TOKEN_NAME>:
    //   decode([2, 72, 3]) → "<BOS>H<EOS>"
    std::string decode(const std::vector<int>& tokens) const;
    
    // Get vocabulary size
    // Returns total number of tokens including special tokens
    int vocab_size() const { return vocab_size_; }
    
    // Check if token is a special token
    bool is_special_token(int token_id) const {
        return token_id == PAD_TOKEN || token_id == UNK_TOKEN ||
               token_id == BOS_TOKEN || token_id == EOS_TOKEN;
    }

private:
    bool add_special_tokens_;
    int vocab_size_;  // Total vocabulary size
    
    // Character to token ID mapping
    // For ASCII: char → token_id = char + 4 (offset for special tokens)
    int char_to_token(char c) const;
    
    // Token ID to character mapping
    char token_to_char(int token_id) const;
};

}  // namespace tokenizer

#endif