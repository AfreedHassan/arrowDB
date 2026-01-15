#pragma once
#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace hnsw {

/// SimdLevel represents detected SIMD capability.
///
/// Values are NOT directly comparable across architectures:
/// - ARM64: Only NEON or NONE will ever be detected
/// - x86_64: SSE -> AVX -> AVX2 -> AVX512 form a hierarchy
///
/// The >= comparisons in backend_registry.h are safe because
/// platform preprocessor guards (#if defined(__aarch64__), etc.)
/// separate ARM and x86 code paths entirely.
enum class SimdLevel : int {
    NONE = 0,
    NEON,      // ARM64 only
    SSE,       // x86_64: 128-bit baseline
    AVX,       // x86_64: 256-bit
    AVX2,      // x86_64: 256-bit with FMA
    AVX512     // x86_64: 512-bit with FMA
};

struct CPUFeatures {
    SimdLevel level = SimdLevel::NONE;

    static const CPUFeatures& get() {
        static CPUFeatures instance;
        return instance;
    }

private:
    CPUFeatures() { detect(); }

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
#if defined(_MSC_VER)
        __cpuidex(out, eax, ecx);
#else
        __cpuid_count(eax, ecx, out[0], out[1], out[2], out[3]);
#endif
    }

    static uint64_t xgetbv(unsigned int idx) {
#if defined(_MSC_VER)
        return _xgetbv(idx);
#else
        uint32_t eax = 0, edx = 0;
        __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(idx));
        return (static_cast<uint64_t>(edx) << 32) | eax;
#endif
    }
#endif

    void detect() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
        int32_t info[4] = {0};
        cpuid(info, 0, 0);
        const int max_leaf = info[0];
        if (max_leaf < 1) return;

        cpuid(info, 1, 0);
        const bool sse      = (info[3] & (1 << 25)) != 0;
        const bool os_xsave = (info[2] & (1 << 27)) != 0;
        const bool cpu_avx  = (info[2] & (1 << 28)) != 0;
        if (sse) level = SimdLevel::SSE;
        if (!os_xsave || !cpu_avx) return;

        const uint64_t xcr0 = xgetbv(0);
        const bool avx_ok = (xcr0 & 0x6) == 0x6;
        if (!avx_ok) return;
        level = SimdLevel::AVX;

        if (max_leaf >= 7) {
            cpuid(info, 7, 0);
            const bool cpu_avx2   = (info[1] & (1 << 5)) != 0;
            const bool cpu_avx512 = (info[1] & (1 << 16)) != 0;
            constexpr uint64_t AVX512_REQUIRED = (1ULL<<5) | (1ULL<<6) | (1ULL<<7);
            const bool os_avx512  = (xcr0 & AVX512_REQUIRED) == AVX512_REQUIRED;
            if (cpu_avx2) level = SimdLevel::AVX2;
            if (cpu_avx512 && os_avx512) level = SimdLevel::AVX512;
        }

#elif defined(__aarch64__) || defined(_M_ARM64)
        level = SimdLevel::NEON;
#endif
    }
};

} // namespace hnsw
