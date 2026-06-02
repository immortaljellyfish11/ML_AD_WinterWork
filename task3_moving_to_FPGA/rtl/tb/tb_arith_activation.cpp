#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Varith_activation_top.h"
#include "verilated.h"

namespace {

struct CaseData {
    std::array<int8_t, 9> acts{};
    std::array<int8_t, 9> wgts{};
    int32_t acc_in = 0;
    int32_t quant_in = 0;
    int32_t relu_in = 0;
    int32_t sat_in = 0;
};

struct GoldenData {
    int16_t mac_product = 0;
    int32_t mac_acc_out = 0;
    int32_t dot_sum = 0;
    int32_t requantized = 0;
    int32_t relu_out = 0;
    int8_t saturated = 0;
    int8_t requant_relu_int8 = 0;
};

int8_t checked_i8(int value, const std::string& name) {
    if (value < -128 || value > 127) {
        throw std::runtime_error(name + " is outside int8 range");
    }
    return static_cast<int8_t>(value);
}

int16_t checked_i16(int value, const std::string& name) {
    if (value < -32768 || value > 32767) {
        throw std::runtime_error(name + " is outside int16 range");
    }
    return static_cast<int16_t>(value);
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
                if (cases.empty()) {
                    return cases;
                }
                throw std::runtime_error("truncated activations in " + path);
            }
            item.acts[i] = checked_i8(value, "activation");
        }

        for (std::size_t i = 0; i < item.wgts.size(); ++i) {
            file >> value;
            item.wgts[i] = checked_i8(value, "weight");
        }

        file >> item.acc_in >> item.quant_in >> item.relu_in >> item.sat_in;
        if (!file) {
            throw std::runtime_error("truncated case in " + path);
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
        int mac_product = 0;
        int saturated = 0;
        int requant_relu_int8 = 0;
        file >> mac_product >> item.mac_acc_out >> item.dot_sum >> item.requantized >> item.relu_out >> saturated >>
            requant_relu_int8;
        if (!file) {
            throw std::runtime_error("truncated golden data in " + path);
        }
        item.mac_product = checked_i16(mac_product, "mac_product");
        item.saturated = checked_i8(saturated, "saturated");
        item.requant_relu_int8 = checked_i8(requant_relu_int8, "requant_relu_int8");
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

int16_t from_sdata(SData value) {
    return static_cast<int16_t>(value);
}

int32_t from_idata(IData value) {
    return static_cast<int32_t>(value);
}

void drive(Varith_activation_top& top, const CaseData& item) {
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

    top.acc_in = as_u32(item.acc_in);
    top.quant_in = as_u32(item.quant_in);
    top.relu_in = as_u32(item.relu_in);
    top.sat_in = as_u32(item.sat_in);
}

bool check_case(std::size_t index, const Varith_activation_top& top, const GoldenData& golden) {
    bool pass = true;

    const int16_t mac_product = from_sdata(top.mac_product);
    const int32_t mac_acc_out = from_idata(top.mac_acc_out);
    const int32_t dot_sum = from_idata(top.dot_sum);
    const int32_t requantized = from_idata(top.requantized);
    const int32_t relu_out = from_idata(top.relu_out);
    const int8_t saturated = from_cdata(top.saturated);
    const int8_t requant_relu_int8 = from_cdata(top.requant_relu_int8);

    auto report = [&](const std::string& name, auto rtl, auto expected) {
        if (rtl != expected) {
            std::cerr << "case " << index << " mismatch " << name << ": rtl=" << static_cast<int64_t>(rtl)
                      << " golden=" << static_cast<int64_t>(expected) << "\n";
            pass = false;
        }
    };

    report("mac_product", mac_product, golden.mac_product);
    report("mac_acc_out", mac_acc_out, golden.mac_acc_out);
    report("dot_sum", dot_sum, golden.dot_sum);
    report("requantized", requantized, golden.requantized);
    report("relu_out", relu_out, golden.relu_out);
    report("saturated", saturated, golden.saturated);
    report("requant_relu_int8", requant_relu_int8, golden.requant_relu_int8);
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto cases = read_cases(data_dir + "arith_activation_cases.txt");
    const auto golden = read_golden(data_dir + "arith_activation_golden.txt");

    if (cases.empty()) {
        throw std::runtime_error("no arithmetic/activation cases loaded");
    }
    if (cases.size() != golden.size()) {
        throw std::runtime_error("case count does not match golden count");
    }

    Varith_activation_top top;
    bool all_pass = true;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        drive(top, cases[i]);
        top.eval();
        all_pass = check_case(i, top, golden[i]) && all_pass;
    }

    if (!all_pass) {
        std::cerr << "Arithmetic/activation test FAILED\n";
        return 1;
    }

    std::cout << "Arithmetic/activation test cases: " << cases.size() << "\n";
    std::cout << "PASS\n";
    return 0;
}
