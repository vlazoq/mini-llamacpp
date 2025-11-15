#include "tokenizer.h"
#include <stdexcept>
#include <sstream>

namespace tokenizer {

CharTokenizer::CharTokenizer(bool add_special_tokens)
    : add_special_tokens_(add_special_tokens) {
    // Vocabulary:
    // 0-3: Special tokens (PAD, UNK, BOS, EOS)
    // 4-131: ASCII characters (0-127 mapped to 4-131)
    // Total: 132 tokens
    vocab_size_ = 132;
}

int CharTokenizer::char_to_token(char c) const {
    // Convert character to token ID
    // ASCII 0-127 maps to tokens 4-131 (offset by 4 for special tokens)
    unsigned char uc = static_cast<unsigned char>(c);
    
    if (uc > 127) {
        // Non-ASCII character → UNK
        return UNK_TOKEN;
    }
    
    return uc + 4;  // Offset by special tokens
}

char CharTokenizer::token_to_char(int token_id) const {
    // Convert token ID to character
    if (token_id < 4 || token_id >= vocab_size_) {
        return '?';  // Invalid token
    }
    
    return static_cast<char>(token_id - 4);
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    
    // Add BOS token if configured
    if (add_special_tokens_) {
        tokens.push_back(BOS_TOKEN);
    }
    
    // Convert each character to token ID
    for (char c : text) {
        tokens.push_back(char_to_token(c));
    }
    
    // Add EOS token if configured
    if (add_special_tokens_) {
        tokens.push_back(EOS_TOKEN);
    }
    
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<int>& tokens) const {
    std::stringstream result;
    
    for (int token_id : tokens) {
        // Handle special tokens
        if (token_id == PAD_TOKEN) {
            result << "<PAD>";
        } else if (token_id == UNK_TOKEN) {
            result << "<UNK>";
        } else if (token_id == BOS_TOKEN) {
            result << "<BOS>";
        } else if (token_id == EOS_TOKEN) {
            result << "<EOS>";
        } else if (token_id >= 4 && token_id < vocab_size_) {
            // Regular character token
            result << token_to_char(token_id);
        } else {
            // Invalid token
            result << "<?>";
        }
    }
    
    return result.str();
}

}  // namespace tokenizer