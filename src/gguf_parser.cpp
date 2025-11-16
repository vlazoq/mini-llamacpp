#include "gguf_parser.h"
#include <stdexcept>
#include <cstring>

namespace gguf {

// Constructor
GGUFParser::GGUFParser(const std::string& filepath)
    : filepath(filepath),
      version(0),
      tensor_count(0),
      metadata_kv_count(0),
      tensor_data_offset(0),
      parsed(false) {
    
    // Open file in binary mode
    file.open(filepath, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open GGUF file: " + filepath);
    }
}

// Destructor
GGUFParser::~GGUFParser() {
    if (file.is_open()) {
        file.close();
    }
}

// Read uint8 from file
uint8_t GGUFParser::read_uint8() {
    uint8_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(uint8_t));
    if (!file) {
        throw std::runtime_error("Failed to read uint8 from file");
    }
    return value;
}

// Read uint16 from file
uint16_t GGUFParser::read_uint16() {
    uint16_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(uint16_t));
    if (!file) {
        throw std::runtime_error("Failed to read uint16 from file");
    }
    return value;
}

// Read uint32 from file
uint32_t GGUFParser::read_uint32() {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
    if (!file) {
        throw std::runtime_error("Failed to read uint32 from file");
    }
    return value;
}

// Read uint64 from file
uint64_t GGUFParser::read_uint64() {
    uint64_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(uint64_t));
    if (!file) {
        throw std::runtime_error("Failed to read uint64 from file");
    }
    return value;
}

// Read int32 from file
int32_t GGUFParser::read_int32() {
    int32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
    if (!file) {
        throw std::runtime_error("Failed to read int32 from file");
    }
    return value;
}

// Read int64 from file
int64_t GGUFParser::read_int64() {
    int64_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(int64_t));
    if (!file) {
        throw std::runtime_error("Failed to read int64 from file");
    }
    return value;
}

// Read float32 from file
float GGUFParser::read_float32() {
    float value;
    file.read(reinterpret_cast<char*>(&value), sizeof(float));
    if (!file) {
        throw std::runtime_error("Failed to read float32 from file");
    }
    return value;
}

// Read float64 from file
double GGUFParser::read_float64() {
    double value;
    file.read(reinterpret_cast<char*>(&value), sizeof(double));
    if (!file) {
        throw std::runtime_error("Failed to read float64 from file");
    }
    return value;
}

// Read string from file
// Format: uint64 length, then that many bytes
std::string GGUFParser::read_string() {
    uint64_t length = read_uint64();
    
    if (length == 0) {
        return "";
    }
    
    if (length > 1024 * 1024 * 100) {  // Sanity check: max 100MB string
        throw std::runtime_error("String too long: " + std::to_string(length));
    }
    
    std::vector<char> buffer(length);
    file.read(buffer.data(), length);
    
    if (!file) {
        throw std::runtime_error("Failed to read string of length " + std::to_string(length));
    }
    
    return std::string(buffer.begin(), buffer.end());
}

// Read metadata value based on type
MetadataValue GGUFParser::read_metadata_value(MetadataValueType type) {
    switch (type) {
        case MetadataValueType::UINT8:
            return read_uint8();
        
        case MetadataValueType::INT8:
            return static_cast<int8_t>(read_uint8());
        
        case MetadataValueType::UINT16:
            return read_uint16();
        
        case MetadataValueType::INT16:
            return static_cast<int16_t>(read_uint16());
        
        case MetadataValueType::UINT32:
            return read_uint32();
        
        case MetadataValueType::INT32:
            return read_int32();
        
        case MetadataValueType::UINT64:
            return read_uint64();
        
        case MetadataValueType::INT64:
            return read_int64();
        
        case MetadataValueType::FLOAT32:
            return read_float32();
        
        case MetadataValueType::FLOAT64:
            return read_float64();
        
        case MetadataValueType::BOOL:
            return static_cast<bool>(read_uint8());
        
        case MetadataValueType::STRING:
            return read_string();
        
        case MetadataValueType::ARRAY: {
            // Array format: type, count, then values
            MetadataValueType array_type = static_cast<MetadataValueType>(read_uint32());
            uint64_t array_count = read_uint64();
            
            // For now, only support string arrays (common for tokenizer vocab)
            if (array_type != MetadataValueType::STRING) {
                throw std::runtime_error("Only string arrays are currently supported");
            }
            
            std::vector<std::string> array_values;
            array_values.reserve(array_count);
            
            for (uint64_t i = 0; i < array_count; i++) {
                array_values.push_back(read_string());
            }
            
            return array_values;
        }
        
        default:
            throw std::runtime_error("Unknown metadata value type: " + 
                                   std::to_string(static_cast<uint32_t>(type)));
    }
}

// Parse GGUF header
void GGUFParser::parse_header() {
    // Read magic number
    uint32_t magic = read_uint32();
    if (magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF magic number. Not a valid GGUF file.");
    }
    
    // Read version
    version = read_uint32();
    if (version != GGUF_VERSION) {
        throw std::runtime_error("Unsupported GGUF version: " + std::to_string(version) +
                               " (expected " + std::to_string(GGUF_VERSION) + ")");
    }
    
    // Read counts
    tensor_count = read_uint64();
    metadata_kv_count = read_uint64();
}

// Parse metadata key-value pairs
void GGUFParser::parse_metadata() {
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        // Read key (string)
        std::string key = read_string();
        
        // Read value type
        MetadataValueType value_type = static_cast<MetadataValueType>(read_uint32());
        
        // Read value
        MetadataValue value = read_metadata_value(value_type);
        
        // Store in map
        metadata[key] = value;
    }
}

// Parse tensor information
void GGUFParser::parse_tensor_info() {
    for (uint64_t i = 0; i < tensor_count; i++) {
        TensorInfo info;
        
        // Read tensor name
        info.name = read_string();
        
        // Read number of dimensions
        uint32_t n_dims = read_uint32();
        
        // Read shape (dimensions in reverse order in file)
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            info.shape[n_dims - 1 - d] = read_uint64();  // Reverse order
        }
        
        // Read tensor type
        info.type = static_cast<TensorType>(read_uint32());
        
        // Read offset (where tensor data is in file)
        info.offset = read_uint64();
        
        // Calculate size based on type and shape
        uint64_t n_elements = info.num_elements();
        
        // Estimate size (simplified - real quantization formats are more complex)
        switch (info.type) {
            case TensorType::F32:
                info.size_bytes = n_elements * 4;
                break;
            case TensorType::F16:
                info.size_bytes = n_elements * 2;
                break;
            case TensorType::Q4_0:
            case TensorType::Q4_1:
                // 4-bit: roughly 0.5 bytes per element (plus block metadata)
                info.size_bytes = (n_elements / 2) + ((n_elements / 32) * 4);  // Rough estimate
                break;
            case TensorType::Q8_0:
            case TensorType::Q8_1:
                // 8-bit: 1 byte per element (plus block metadata)
                info.size_bytes = n_elements + ((n_elements / 32) * 4);  // Rough estimate
                break;
            default:
                info.size_bytes = n_elements * 4;  // Assume F32 size for unknown types
                break;
        }
        
        // Store tensor info
        tensors[info.name] = info;
    }
    
    // Calculate alignment and tensor data offset
    // GGUF aligns tensor data to 32-byte boundary
    std::streampos current_pos = file.tellg();
    uint64_t alignment = 32;
    uint64_t offset = static_cast<uint64_t>(current_pos);
    uint64_t padding = (alignment - (offset % alignment)) % alignment;
    
    tensor_data_offset = offset + padding;
}

// Main parse function
void GGUFParser::parse() {
    if (parsed) {
        return;  // Already parsed
    }
    
    // Reset file to beginning
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Parse in order
    parse_header();
    parse_metadata();
    parse_tensor_info();
    
    parsed = true;
}

// Get metadata by key
const MetadataValue* GGUFParser::get_metadata(const std::string& key) const {
    auto it = metadata.find(key);
    if (it == metadata.end()) {
        return nullptr;
    }
    return &it->second;
}

// Get string metadata
std::string GGUFParser::get_string(const std::string& key, const std::string& default_val) const {
    const MetadataValue* value = get_metadata(key);
    if (value == nullptr) {
        return default_val;
    }
    
    if (const std::string* str = std::get_if<std::string>(value)) {
        return *str;
    }
    
    return default_val;
}

// Get uint32 metadata
uint32_t GGUFParser::get_uint32(const std::string& key, uint32_t default_val) const {
    const MetadataValue* value = get_metadata(key);
    if (value == nullptr) {
        return default_val;
    }
    
    if (const uint32_t* val = std::get_if<uint32_t>(value)) {
        return *val;
    }
    
    return default_val;
}

// Get uint64 metadata
uint64_t GGUFParser::get_uint64(const std::string& key, uint64_t default_val) const {
    const MetadataValue* value = get_metadata(key);
    if (value == nullptr) {
        return default_val;
    }
    
    if (const uint64_t* val = std::get_if<uint64_t>(value)) {
        return *val;
    }
    
    return default_val;
}

// Get tensor info by name
const TensorInfo* GGUFParser::get_tensor_info(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
        return nullptr;
    }
    return &it->second;
}

// Get all tensor names
std::vector<std::string> GGUFParser::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors.size());
    
    for (const auto& pair : tensors) {
        names.push_back(pair.first);
    }
    
    return names;
}

// Read tensor data
std::vector<uint8_t> GGUFParser::read_tensor_data(const std::string& name) {
    if (!parsed) {
        throw std::runtime_error("Must call parse() before reading tensor data");
    }
    
    const TensorInfo* info = get_tensor_info(name);
    if (info == nullptr) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    
    // Seek to tensor data location
    uint64_t data_position = tensor_data_offset + info->offset;
    file.seekg(data_position, std::ios::beg);
    
    if (!file) {
        throw std::runtime_error("Failed to seek to tensor data for: " + name);
    }
    
    // Check how much data is actually available
    file.seekg(0, std::ios::end);
    std::streampos file_size = file.tellg();
    file.seekg(data_position, std::ios::beg);
    
    uint64_t available_bytes = static_cast<uint64_t>(file_size) - data_position;
    uint64_t bytes_to_read = std::min(info->size_bytes, available_bytes);
    
    // Read raw bytes
    std::vector<uint8_t> data(bytes_to_read);
    file.read(reinterpret_cast<char*>(data.data()), bytes_to_read);
    
    if (!file) {
        throw std::runtime_error("Failed to read tensor data for: " + name);
    }
    
    return data;
}

}  // namespace gguf