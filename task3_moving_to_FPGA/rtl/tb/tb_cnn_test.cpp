#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vcnn_test_top.h"
#include "verilated.h"

namespace {

struct CaseData {
    std::array<int8_t, 9> acts{};
    std::array<int8_t, 9> wgts{};
    int32_t bias = 0;
};

struct GoldenData {
    int32_t conv_raw = 0;
    int32_t conv_bias = 0;
    int32_t conv_quant = 0;
    int32_t conv_relu = 0;
    int8_t conv_out = 0;
};

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

        for (std::size_t i = 0; i < item.acts.size(); ++i) {
            if (!(file >> value)) {
                return cases;
            }
            item.acts[i] = checked_i8(value, "activation");
        }

        for (std::size_t i = 0; i < item.wgts.size(); ++i) {
            file >> value;
            item.wgts[i] = checked_i8(value, "weight");
        }

        file >> item.bias;
        if (!file) {
            throw std::runtime_error("truncated CNN test case in " + path);
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
        GoldenData item;
        int conv_out = 0;
        file >> item.conv_raw >> item.conv_bias >> item.conv_quant >> item.conv_relu >> conv_out;
        if (!file) {
            throw std::runtime_error("truncated CNN golden data in " + path);
        }
        item.conv_out = checked_i8(conv_out, "conv_out");
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

void drive(Vcnn_test_top& top, const CaseData& item) {
    top.act0 = as_u8(item.acts[0]);
    top.act1 = as_u8(item.acts[1]);
    top.act2 = as_u8(item.acts[2]);
    top.act3 = as_u8(item.acts[3]);
    top.act4 = as_u8(item.acts[4]);
    top.act5 = as_u8(item.acts[5]);
    top.act6 = as_u8(item.acts[6]);
    top.act7 = as_u8(item.acts[7]);
    top.act8 = as_u8(item.acts[8]);

    top.wgt0 = as_u8(item.wgts[0]);
    top.wgt1 = as_u8(item.wgts[1]);
    top.wgt2 = as_u8(item.wgts[2]);
    top.wgt3 = as_u8(item.wgts[3]);
    top.wgt4 = as_u8(item.wgts[4]);
    top.wgt5 = as_u8(item.wgts[5]);
    top.wgt6 = as_u8(item.wgts[6]);
    top.wgt7 = as_u8(item.wgts[7]);
    top.wgt8 = as_u8(item.wgts[8]);
    top.bias = as_u32(item.bias);
}

bool check_case(std::size_t index, const Vcnn_test_top& top, const GoldenData& golden) {
    bool pass = true;

    auto report = [&](const std::string& name, auto rtl, auto expected) {
        if (rtl != expected) {
            std::cerr << "case " << index << " mismatch " << name << ": rtl=" << static_cast<int64_t>(rtl)
                      << " golden=" << static_cast<int64_t>(expected) << "\n";
            pass = false;
        }
    };

    report("conv_raw", from_idata(top.conv_raw), golden.conv_raw);
    report("conv_bias", from_idata(top.conv_bias), golden.conv_bias);
    report("conv_quant", from_idata(top.conv_quant), golden.conv_quant);
    report("conv_relu", from_idata(top.conv_relu), golden.conv_relu);
    report("conv_out", from_cdata(top.conv_out), golden.conv_out);
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto cases = read_cases(data_dir + "cnn_test_cases.txt");
    const auto golden = read_golden(data_dir + "cnn_test_golden.txt");

    if (cases.empty()) {
        throw std::runtime_error("no CNN test cases loaded");
    }
    if (cases.size() != golden.size()) {
        throw std::runtime_error("case count does not match golden count");
    }

    Vcnn_test_top top;
    bool all_pass = true;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        drive(top, cases[i]);
        top.eval();
        all_pass = check_case(i, top, golden[i]) && all_pass;
    }

    if (!all_pass) {
        std::cerr << "CNN test top FAILED\n";
        return 1;
    }

    std::cout << "CNN test cases: " << cases.size() << "\n";
    std::cout << "PASS\n";
    return 0;
}
