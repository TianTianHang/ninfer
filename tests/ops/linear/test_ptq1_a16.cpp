#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

constexpr Invocation graph(std::int32_t t) {
    return {t, CallForm::Policy, ninfer::ops::LinearPolicy::A16Only, true};
}

int ptq1_a16_conformance(std::int32_t selected_n, std::int32_t selected_k) {
    int failures = 0;

    // Full-output oracle covers both private routes, the exact and masked token
    // dispatch boundaries, the warp row grouping, and the 64-column block tiles.
    constexpr std::array kSmallFull{
        a16(1),    convenience(1), a16(2),  a16(3),  a16(4),  a16(5),  a16(6),  a16(7),
        a16(8),    graph(8),       a16(9),  graph(9), graph(16), a16(17), a16(63), graph(64),
        graph(65), a16(127),       a16(128), graph(129),
    };
    if (selected_n == 0 || (selected_n == 128 && selected_k == 512)) {
        failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_g128_fp16_weight,
                              {128, 512, 211U, Comparison::Full, true, kSmallFull});
    }

    // Sampled real geometry: the hot interval, the route seam, and both bulk
    // anchors. Sampled rows/columns keep the FP64 oracle cost bounded.
    constexpr std::array kRealSampled{
        a16(1),    a16(2),    a16(3),    a16(4),   a16(5),   a16(8),   a16(9),   a16(16),
        graph(17), a16(32),   a16(48),   a16(63),  a16(64),  a16(65),  a16(96),  a16(127),
        graph(128), graph(129), a16(256), a16(320), a16(512),
    };
    for (const auto& shape :
         {std::array<std::int32_t, 3>{1024, 5120, 223U},
          std::array<std::int32_t, 3>{5120, 17408, 227U},
          std::array<std::int32_t, 3>{14336, 5120, 229U}}) {
        if (selected_n != 0 && (shape[0] != selected_n || shape[1] != selected_k)) { continue; }
        failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_g128_fp16_weight,
                              {shape[0], shape[1], static_cast<std::uint32_t>(shape[2]),
                               Comparison::Sampled, true, kRealSampled});
    }

    // Full-output oracle at the large gate/up geometry for the route seam and a
    // bulk anchor.
    constexpr std::array kLargeFull{a16(1), a16(4), a16(8), graph(9), a16(65), a16(128)};
    if (selected_n == 0 || (selected_n == 34816 && selected_k == 5120)) {
        failures += run_shape("PTQ1_A16", ActivationCompute::A16, make_ptq1_g128_fp16_weight,
                              {34816, 5120, 233U, Comparison::Full, false, kLargeFull});
    }

    // Partial row tiles: N below, equal to, and just above the warp row grouping
    // and the 64-row block tile.
    constexpr std::array kEdgeCalls{a16(1), a16(3), a16(8), a16(9), a16(64), a16(65)};
    for (const std::int32_t n : {1, 3, 4, 5, 63, 64, 65, 129}) {
        if (selected_n != 0 && n != selected_n) { continue; }
        failures += run_shape("PTQ1_A16 edge", ActivationCompute::A16, make_ptq1_g128_fp16_weight,
                              {n, 256, 239U, Comparison::Sampled, true, kEdgeCalls});
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        std::int32_t n = 0, k = 0;
        if (argc == 4 && std::string_view(argv[1]) == "--shape") {
            n = std::stoi(argv[2]);
            k = std::stoi(argv[3]);
            if (n <= 0 || k <= 0) throw std::invalid_argument("PTQ1 test: positive N/K required");
        } else if (argc != 1) {
            throw std::invalid_argument("usage: ninfer_linear_ptq1_a16_test [--shape N K]");
        }
        const int failures = ptq1_a16_conformance(n, k);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " PTQ1_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "PTQ1_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
