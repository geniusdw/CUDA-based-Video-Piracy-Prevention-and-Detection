#pragma once

#include <cstdint>

constexpr int kDctMidBandLength = 15;
constexpr uint32_t kWatermarkKey = 42;

static constexpr int kMidBandTable[30] = {
    3, 0, 2, 1, 1, 2, 0, 3, 4, 0,
    3, 1, 2, 2, 1, 3, 0, 4, 5, 0,
    4, 1, 3, 2, 2, 3, 1, 4, 0, 5
};

inline char wm_bit_from_index(uint32_t idx, uint32_t key) {
    uint32_t x = idx ^ (key * 0x9E3779B9u);
    x ^= (x >> 16);
    x *= 0x7FEB352Du;
    x ^= (x >> 15);
    x *= 0x846CA68Bu;
    x ^= (x >> 16);
    return (x & 1u) ? 1 : -1;
}
