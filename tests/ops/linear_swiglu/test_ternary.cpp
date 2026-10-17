// Ternary SwiGLU: gate/up contraction through the shared ternary A16 route, then the
// SiLU product. Token cases cover both private ternary routes and the 64-column seam.
#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    try {
        constexpr std::array<std::int32_t, 9> kTokenCases{1, 2, 4, 8, 9, 16, 64, 65, 128};
        int failures = run_profile(
            "LinearSwiGLU Q2_A16",
            {QType::Q2_G128_FP16, 34816, 5120, 17408, 1709U, ActivationCompute::A16}, kTokenCases);
        failures += run_profile(
            "LinearSwiGLU PTQ1_A16",
            {QType::PTQ1_G128_FP16, 34816, 5120, 17408, 1711U, ActivationCompute::A16},
            kTokenCases);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " ternary LinearSwiGLU\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ternary LinearSwiGLU: " << error.what() << '\n';
        return 1;
    }
}
