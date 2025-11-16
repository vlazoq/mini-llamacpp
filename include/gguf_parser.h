// GGUF File Format Parser
// Loads model weights and metadata from GGUF files

#ifndef GGUF_PARSER_H
#define GGUF_PARSER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <variant>
#include <cstdint>

namespace gguf {

// GGUF version
constexpr uint32_t GGUF_VERSION = 3;

// GGUF magic number "GGUF" in hex
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in little-endian

// Metadata value types
enum class MetadataValueType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

// Tensor data types
enum class TensorType : uint32_t {
    F32 = 0,   // 32-bit float
    F16 = 1,   // 16-bit float
    Q4_0 = 2,  // 4-bit quantized (block format)
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,  // 8-bit quantized
    Q8_1 = 9,
    // Many more quantization formats...
};

// Metadata value (can hold different types)
using MetadataValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool, std::string,
    std::vector<std::string>  // For arrays
>;

// Tensor information from GGUF file
struct TensorInfo {
    std::string name;              // e.g., "token_embd.weight"
    std::vector<uint64_t> shape;   // Dimensions [n_embd, n_vocab]
    TensorType type;               // Data type (F32, Q4_0, etc.)
    uint64_t offset;               // Byte offset in file where data starts
    uint64_t size_bytes;           // Total bytes for this tensor
    
    // Calculate number of elements
    uint64_t num_elements() const {
        uint64_t n = 1;
        for (uint64_t dim : shape) {
            n *= dim;
        }
        return n;
    }
};

// GGUF file parser
class GGUFParser {
public:
    // Constructor
    // Args:
    //   filepath: Path to .gguf file
    explicit GGUFParser(const std::string& filepath);
    
    // Destructor
    ~GGUFParser();
    
    // Parse the GGUF file header and metadata
    // Must be called before accessing metadata or tensors
    void parse();
    
    // Get metadata value by key
    // Returns nullptr if key doesn't exist
    const MetadataValue* get_metadata(const std::string& key) const;
    
    // Get metadata as specific type (helper methods)
    std::string get_string(const std::string& key, const std::string& default_val = "") const;
    uint32_t get_uint32(const std::string& key, uint32_t default_val = 0) const;
    uint64_t get_uint64(const std::string& key, uint64_t default_val = 0) const;
    
    // Get tensor info by name
    const TensorInfo* get_tensor_info(const std::string& name) const;
    
    // Get all tensor names
    std::vector<std::string> get_tensor_names() const;
    
    // Read tensor data (returns raw bytes)
    // Caller must handle dequantization if needed
    std::vector<uint8_t> read_tensor_data(const std::string& name);
    
    // Getters
    uint32_t get_version() const { return version; }
    size_t get_tensor_count() const { return tensors.size(); }
    size_t get_metadata_count() const { return metadata.size(); }
    bool is_parsed() const { return parsed; }

private:
    std::string filepath;
    std::ifstream file;
    
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    uint64_t tensor_data_offset;  // Where tensor data starts in file
    
    std::unordered_map<std::string, MetadataValue> metadata;
    std::unordered_map<std::string, TensorInfo> tensors;
    
    bool parsed;
    
    // Parsing helpers
    void parse_header();
    void parse_metadata();
    void parse_tensor_info();
    
    // Read primitives from file (handles endianness)
    uint8_t read_uint8();
    uint16_t read_uint16();
    uint32_t read_uint32();
    uint64_t read_uint64();
    int32_t read_int32();
    int64_t read_int64();
    float read_float32();
    double read_float64();
    std::string read_string();
    
    // Read metadata value based on type
    MetadataValue read_metadata_value(MetadataValueType type);
};

}  // namespace gguf

#endif