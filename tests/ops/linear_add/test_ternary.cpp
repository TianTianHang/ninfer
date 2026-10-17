// Ternary LinearAdd: the contraction adds into the existing residual, which is the
// route the dense FFN down projection uses. Token cases cover both private ternary
// routes (masked and exact token groups, then the block contraction) and the
// 64-column block seam.
#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer::test::linear_add;
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        constexpr std::array<std::int32_t, 6> route_starts{3, 5, 9, 17, 65, 129};
        constexpr std::array<std::int32_t, 5> interiors{1, 4, 32, 128, 256};
        constexpr std::array<std::int32_t, 5> graph_tokens{1, 4, 9, 65, 128};
        int failures = 0;
        failures += run_shape("Q2_A16 LinearAdd", WeightFormat::Q2G128F16,
                              {5120, 6144, 701U, route_starts, interiors, graph_tokens});
        failures += run_shape("PTQ1_A16 LinearAdd", WeightFormat::PTQ1G128F16,
                              {5120, 6144, 703U, route_starts, interiors, graph_tokens});
        constexpr std::array<std::int32_t, 4> full_tokens{1, 4, 9, 65};
        failures += run_shape("Q2_A16 LinearAdd full", WeightFormat::Q2G128F16,
                              {128, 256, 707U, {}, full_tokens, {}, true});
        failures += run_shape("PTQ1_A16 LinearAdd full", WeightFormat::PTQ1G128F16,
                              {128, 256, 709U, {}, full_tokens, {}, true});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " ternary LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ternary LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
