// quantize_fp16_to_int16.cpp
// Build: g++ -O3 -fPIC -shared -std=c++17 quantize_fp16_to_int16.cpp -o libquantize.so

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>

// ---------------------------
// FP16 (IEEE-754 half) -> float32 conversion
// Input fp16 is provided as uint16_t bits.
// This is a common, portable conversion (no compiler-specific half type needed).
// ---------------------------
static inline float fp16_to_fp32(uint16_t h) {
    // Extract sign, exponent, mantissa
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;   // move to float sign bit
    uint32_t exp       = (h >> 10) & 0x1Fu;
    uint32_t mant      = (uint32_t)(h & 0x03FFu);

    uint32_t f;

    if (exp == 0) {
        // Zero or subnormal
        if (mant == 0) {
            // ±0
            f = sign;
        } else {
            // Subnormal: normalize it
            // value = (-1)^sign * 2^(−14) * (mant / 2^10)
            // Convert to float by normalizing mantissa
            exp = 1;
            while ((mant & 0x0400u) == 0) { // while leading bit not set
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu; // remove leading 1
            const uint32_t float_exp  = (exp - 15 + 127); // half bias=15, float bias=127
            const uint32_t float_mant = mant << 13;       // align 10-bit to 23-bit
            f = sign | (float_exp << 23) | float_mant;
        }
    } else if (exp == 31) {
        // Inf or NaN
        const uint32_t float_exp = 255u;
        const uint32_t float_mant = mant ? (mant << 13) : 0u;
        f = sign | (float_exp << 23) | float_mant;
    } else {
        // Normalized
        const uint32_t float_exp  = (exp - 15 + 127);
        const uint32_t float_mant = mant << 13;
        f = sign | (float_exp << 23) | float_mant;
    }

    float out;
    // Safe bit-cast without violating strict aliasing:
    static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit IEEE754");
    std::memcpy(&out, &f, sizeof(float));
    return out;
}

// Round-to-nearest-even is ideal, but in many quant flows "nearest" is okay.
// If you want deterministic banker’s rounding, implement it; here we use nearbyintf.
static inline int32_t round_nearest(float x) {
    return (int32_t)std::nearbyintf(x);
}

// ---------------------------
// C API (extern "C") for shared library use
// ---------------------------
extern "C" {

// Quantize FP16 -> INT16
//
// fp16_in_bits: pointer to FP16 values stored as IEEE half bits (uint16_t each)
// out_i16:      pointer to output int16 tensor
// n:            number of elements
// scale:        quant scale (>0). Typical: real_value / scale -> quant_value
// zero_point:   quant zero point (for asymmetric). For symmetric, pass 0.
// qmin/qmax:    clamp range. For int16 full-range use [-32768, 32767].
//
// Returns: 0 on success, negative on error.
int quantize_fp16_to_int16(
    const uint16_t* fp16_in_bits,
    int16_t* out_i16,
    size_t n,
    float scale,
    int32_t zero_point,
    int32_t qmin,
    int32_t qmax
) {
    if (!fp16_in_bits || !out_i16) return -1;
    if (n == 0) return 0;
    if (!(scale > 0.0f) || std::isnan(scale) || std::isinf(scale)) return -2;
    if (qmin > qmax) return -3;

    // Fast path constants
    const float inv_scale = 1.0f / scale;

    for (size_t i = 0; i < n; ++i) {
        const float x = fp16_to_fp32(fp16_in_bits[i]);

        // Handle NaN/Inf gracefully: clamp them
        float scaled = x * inv_scale;
        if (std::isnan(scaled)) scaled = 0.0f;
        if (scaled >  (float)qmax) scaled = (float)qmax;
        if (scaled <  (float)qmin) scaled = (float)qmin;

        int32_t q = round_nearest(scaled) + zero_point;
        q = std::min(qmax, std::max(qmin, q));
        out_i16[i] = (int16_t)q;
    }

    return 0;
}

// Convenience wrapper: full int16 range clamp
int quantize_fp16_to_int16_fullrange(
    const uint16_t* fp16_in_bits,
    int16_t* out_i16,
    size_t n,
    float scale,
    int32_t zero_point
) {
    return quantize_fp16_to_int16(fp16_in_bits, out_i16, n, scale, zero_point, -32768, 32767);
}

} // extern "C"
