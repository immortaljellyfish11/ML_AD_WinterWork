#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vcnn_tiny_4x4_top.h"
#include "verilated.h"

namespace {

struct CaseData {
    std::array<int8_t, 16> image{};
    std::array<int8_t, 9> kernel{};
    int32_t conv_bias = 0;
    int8_t fc0_weight = 0;
    int8_t fc1_weight = 0;
    int32_t fc0_bias = 0;
    int32_t fc1_bias = 0;
};

using GoldenData = std::array<int32_t, 25>;

int8_t checked_i8(int value, const std::string& name) {
    if (value < -128 || value > 127) {
        throw std::runtime_error(name + " is outside int8 range");
    }
    return static_cast<int8_t>(value);
}

std::vector<CaseData> read_cases(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<CaseData> cases;
    while (file.peek() != EOF) {
        CaseData item;
        int value = 0;
        for (std::size_t i = 0; i < item.image.size(); ++i) {
            if (!(file >> value)) {
                return cases;
            }
            item.image[i] = checked_i8(value, "image");
        }
        for (std::size_t i = 0; i < item.kernel.size(); ++i) {
            file >> value;
            item.kernel[i] = checked_i8(value, "kernel");
        }
        file >> item.conv_bias >> value;
        item.fc0_weight = checked_i8(value, "fc0_weight");
        file >> value;
        item.fc1_weight = checked_i8(value, "fc1_weight");
        file >> item.fc0_bias >> item.fc1_bias;
        if (!file) {
            throw std::runtime_error("truncated tiny CNN case in " + path);
        }
        cases.push_back(item);
        file >> std::ws;
    }
    return cases;
}

std::vector<GoldenData> read_golden(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<GoldenData> golden;
    while (file.peek() != EOF) {
        GoldenData item{};
        for (std::size_t i = 0; i < item.size(); ++i) {
            if (!(file >> item[i])) {
                return golden;
            }
        }
        golden.push_back(item);
        file >> std::ws;
    }
    return golden;
}

uint8_t as_u8(int8_t value) {
    return static_cast<uint8_t>(value);
}

uint32_t as_u32(int32_t value) {
    return static_cast<uint32_t>(value);
}

int8_t from_cdata(CData value) {
    return static_cast<int8_t>(value);
}

int32_t from_idata(IData value) {
    return static_cast<int32_t>(value);
}

void drive(Vcnn_tiny_4x4_top& top, const CaseData& item) {
    top.x00 = as_u8(item.image[0]);
    top.x01 = as_u8(item.image[1]);
    top.x02 = as_u8(item.image[2]);
    top.x03 = as_u8(item.image[3]);
    top.x10 = as_u8(item.image[4]);
    top.x11 = as_u8(item.image[5]);
    top.x12 = as_u8(item.image[6]);
    top.x13 = as_u8(item.image[7]);
    top.x20 = as_u8(item.image[8]);
    top.x21 = as_u8(item.image[9]);
    top.x22 = as_u8(item.image[10]);
    top.x23 = as_u8(item.image[11]);
    top.x30 = as_u8(item.image[12]);
    top.x31 = as_u8(item.image[13]);
    top.x32 = as_u8(item.image[14]);
    top.x33 = as_u8(item.image[15]);

    top.k00 = as_u8(item.kernel[0]);
    top.k01 = as_u8(item.kernel[1]);
    top.k02 = as_u8(item.kernel[2]);
    top.k10 = as_u8(item.kernel[3]);
    top.k11 = as_u8(item.kernel[4]);
    top.k12 = as_u8(item.kernel[5]);
    top.k20 = as_u8(item.kernel[6]);
    top.k21 = as_u8(item.kernel[7]);
    top.k22 = as_u8(item.kernel[8]);

    top.conv_bias = as_u32(item.conv_bias);
    top.fc0_weight = as_u8(item.fc0_weight);
    top.fc1_weight = as_u8(item.fc1_weight);
    top.fc0_bias = as_u32(item.fc0_bias);
    top.fc1_bias = as_u32(item.fc1_bias);
}

GoldenData read_outputs(const Vcnn_tiny_4x4_top& top) {
    return {
        from_cdata(top.conv_y00), from_cdata(top.conv_y01), from_cdata(top.conv_y02), from_cdata(top.conv_y03),
        from_cdata(top.conv_y10), from_cdata(top.conv_y11), from_cdata(top.conv_y12), from_cdata(top.conv_y13),
        from_cdata(top.conv_y20), from_cdata(top.conv_y21), from_cdata(top.conv_y22), from_cdata(top.conv_y23),
        from_cdata(top.conv_y30), from_cdata(top.conv_y31), from_cdata(top.conv_y32), from_cdata(top.conv_y33),
        from_cdata(top.pool_y00), from_cdata(top.pool_y01), from_cdata(top.pool_y10), from_cdata(top.pool_y11),
        from_idata(top.gap_sum), from_cdata(top.gap_avg),
        from_idata(top.logit0), from_idata(top.logit1),
        static_cast<int32_t>(top.class_id),
    };
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto cases = read_cases(data_dir + "cnn_tiny_4x4_cases.txt");
    const auto golden = read_golden(data_dir + "cnn_tiny_4x4_golden.txt");

    if (cases.empty()) {
        throw std::runtime_error("no tiny CNN test cases loaded");
    }
    if (cases.size() != golden.size()) {
        throw std::runtime_error("case count does not match golden count");
    }

    Vcnn_tiny_4x4_top top;
    bool all_pass = true;

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        drive(top, cases[case_index]);
        top.eval();
        const auto rtl = read_outputs(top);

        for (std::size_t i = 0; i < rtl.size(); ++i) {
            if (rtl[i] != golden[case_index][i]) {
                std::cerr << "case " << case_index << " output " << i
                          << " mismatch: rtl=" << rtl[i]
                          << " golden=" << golden[case_index][i] << "\n";
                all_pass = false;
            }
        }
    }

    if (!all_pass) {
        std::cerr << "Tiny CNN 4x4 test FAILED\n";
        return 1;
    }

    std::cout << "Tiny CNN 4x4 test cases: " << cases.size() << "\n";
    std::cout << "PASS\n";
    return 0;
}
