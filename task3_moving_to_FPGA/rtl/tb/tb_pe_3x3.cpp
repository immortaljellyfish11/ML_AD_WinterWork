#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Vpe_3x3.h"
#include "verilated.h"

namespace {

std::array<int8_t, 9> read_int8_vector(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    std::array<int8_t, 9> values{};
    int value = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!(file >> value)) {
            throw std::runtime_error("expected 9 values in " + path);
        }
        if (value < -128 || value > 127) {
            throw std::runtime_error("int8 value out of range in " + path);
        }
        values[i] = static_cast<int8_t>(value);
    }
    return values;
}

int32_t read_i32(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    int32_t value = 0;
    if (!(file >> value)) {
        throw std::runtime_error("expected one int32 value in " + path);
    }
    return value;
}

void write_i32(const std::string& path, int32_t value) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }
    file << value << "\n";
}

uint8_t as_u8(int8_t value) {
    return static_cast<uint8_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto activations = read_int8_vector(data_dir + "pe_3x3_activations.txt");
    const auto weights = read_int8_vector(data_dir + "pe_3x3_weights.txt");
    const int32_t golden = read_i32(data_dir + "pe_3x3_golden.txt");

    Vpe_3x3 top;
    top.act0 = as_u8(activations[0]);
    top.act1 = as_u8(activations[1]);
    top.act2 = as_u8(activations[2]);
    top.act3 = as_u8(activations[3]);
    top.act4 = as_u8(activations[4]);
    top.act5 = as_u8(activations[5]);
    top.act6 = as_u8(activations[6]);
    top.act7 = as_u8(activations[7]);
    top.act8 = as_u8(activations[8]);

    top.wgt0 = as_u8(weights[0]);
    top.wgt1 = as_u8(weights[1]);
    top.wgt2 = as_u8(weights[2]);
    top.wgt3 = as_u8(weights[3]);
    top.wgt4 = as_u8(weights[4]);
    top.wgt5 = as_u8(weights[5]);
    top.wgt6 = as_u8(weights[6]);
    top.wgt7 = as_u8(weights[7]);
    top.wgt8 = as_u8(weights[8]);

    top.eval();

    const int32_t rtl_result = static_cast<int32_t>(top.result);
    write_i32(data_dir + "pe_3x3_rtl_output.txt", rtl_result);

    std::cout << "PE 3x3 RTL result: " << rtl_result << "\n";
    std::cout << "PE 3x3 golden:     " << golden << "\n";

    if (rtl_result != golden) {
        std::cerr << "Mismatch: rtl=" << rtl_result << " golden=" << golden << "\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
