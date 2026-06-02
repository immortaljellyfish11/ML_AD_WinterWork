#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vmaxpool_2x2.h"
#include "verilated.h"

namespace {

using CaseData = std::array<int8_t, 4>;

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
        CaseData item{};
        int value = 0;
        for (std::size_t i = 0; i < item.size(); ++i) {
            if (!(file >> value)) {
                return cases;
            }
            item[i] = checked_i8(value, "maxpool input");
        }
        cases.push_back(item);
        file >> std::ws;
    }
    return cases;
}

std::vector<int8_t> read_golden(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<int8_t> golden;
    int value = 0;
    while (file >> value) {
        golden.push_back(checked_i8(value, "maxpool golden"));
    }
    return golden;
}

uint8_t as_u8(int8_t value) {
    return static_cast<uint8_t>(value);
}

int8_t from_cdata(CData value) {
    return static_cast<int8_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto cases = read_cases(data_dir + "maxpool_2x2_cases.txt");
    const auto golden = read_golden(data_dir + "maxpool_2x2_golden.txt");

    if (cases.empty()) {
        throw std::runtime_error("no MaxPool2x2 test cases loaded");
    }
    if (cases.size() != golden.size()) {
        throw std::runtime_error("case count does not match golden count");
    }

    Vmaxpool_2x2 top;
    bool all_pass = true;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        top.in0 = as_u8(cases[i][0]);
        top.in1 = as_u8(cases[i][1]);
        top.in2 = as_u8(cases[i][2]);
        top.in3 = as_u8(cases[i][3]);
        top.eval();

        const int8_t rtl_out = from_cdata(top.out);
        if (rtl_out != golden[i]) {
            std::cerr << "case " << i << " mismatch: rtl=" << static_cast<int>(rtl_out)
                      << " golden=" << static_cast<int>(golden[i]) << "\n";
            all_pass = false;
        }
    }

    if (!all_pass) {
        std::cerr << "MaxPool2x2 test FAILED\n";
        return 1;
    }

    std::cout << "MaxPool2x2 test cases: " << cases.size() << "\n";
    std::cout << "PASS\n";
    return 0;
}
