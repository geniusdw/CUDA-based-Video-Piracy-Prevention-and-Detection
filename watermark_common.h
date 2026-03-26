#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

constexpr int kDctMidBandLength = 15;
constexpr uint32_t kWatermarkKey = 42;
constexpr uint16_t kPayloadMagic = 0xABCD;
constexpr std::size_t kPayloadMaxChars = 64;
constexpr std::size_t kPayloadBitCount = 16 + (kPayloadMaxChars * 8);

static constexpr int kMidBandTable[30] = {
    3, 0, 2, 1, 1, 2, 0, 3, 4, 0,
    3, 1, 2, 2, 1, 3, 0, 4, 5, 0,
    4, 1, 3, 2, 2, 3, 1, 4, 0, 5
};

inline uint32_t watermark_mix(uint32_t x) {
    x ^= (x >> 16);
    x *= 0x7FEB352Du;
    x ^= (x >> 15);
    x *= 0x846CA68Bu;
    x ^= (x >> 16);
    return x;
}

inline uint32_t watermark_key_hash(uint32_t key) {
    return watermark_mix(key * 0x9E3779B9u);
}

inline char wm_bit_from_index(uint32_t idx, uint32_t key) {
    return (watermark_mix(idx ^ (key * 0x9E3779B9u)) & 1u) ? 1 : -1;
}

inline std::size_t payload_index_for_slot(std::size_t slot_index, std::size_t payload_size, uint32_t key) {
    if (payload_size == 0) return 0;
    const uint32_t payload_index = static_cast<uint32_t>(slot_index % payload_size);
    const uint32_t shuffled_index = payload_index ^ watermark_key_hash(key);
    return static_cast<std::size_t>(shuffled_index % static_cast<uint32_t>(payload_size));
}

inline signed char payload_bit_for_slot(std::size_t slot_index, const std::vector<signed char>& payload_bits, uint32_t key) {
    if (payload_bits.empty()) return -1;
    return payload_bits[payload_index_for_slot(slot_index, payload_bits.size(), key)];
}

inline void append_lsb_bits(uint32_t value, int bit_count, std::vector<signed char>& bits) {
    for (int bit = 0; bit < bit_count; ++bit) {
        bits.push_back(((value >> bit) & 1u) ? 1 : -1);
    }
}

inline void encode_payload(const std::string& msg, std::vector<signed char>& bits) {
    bits.clear();
    bits.reserve(kPayloadBitCount);
    append_lsb_bits(kPayloadMagic, 16, bits);

    const std::size_t clipped_size = (msg.size() < kPayloadMaxChars) ? msg.size() : kPayloadMaxChars;
    for (std::size_t i = 0; i < kPayloadMaxChars; ++i) {
        const unsigned char value = (i < clipped_size) ? static_cast<unsigned char>(msg[i]) : 0u;
        append_lsb_bits(value, 8, bits);
    }
}

inline std::string decode_payload(const std::vector<float>& soft_bits) {
    if (soft_bits.size() < kPayloadBitCount) return {};

    auto decode_threshold_bit = [&](std::size_t idx) -> uint32_t {
        return soft_bits[idx] > 0.0f ? 1u : 0u;
    };

    uint32_t magic = 0;
    for (int bit = 0; bit < 16; ++bit) {
        magic |= (decode_threshold_bit(static_cast<std::size_t>(bit)) << bit);
    }
    if (magic != kPayloadMagic) return {};

    std::string msg;
    msg.reserve(kPayloadMaxChars);
    for (std::size_t char_index = 0; char_index < kPayloadMaxChars; ++char_index) {
        const std::size_t bit_base = 16 + (char_index * 8);
        unsigned char value = 0;
        for (int bit = 0; bit < 8; ++bit) {
            value |= static_cast<unsigned char>(decode_threshold_bit(bit_base + static_cast<std::size_t>(bit)) << bit);
        }
        if (value == 0u) break;
        msg.push_back(static_cast<char>(value));
    }
    return msg;
}
