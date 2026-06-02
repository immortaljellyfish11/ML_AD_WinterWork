#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vglobal_avg_pool_8x8.h"
#include "verilated.h"

namespace {

using CaseData = std::array<int8_t, 64>;

struct GoldenData {
    int32_t sum = 0;
    int8_t avg = 0;
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
        CaseData item{};
        int value = 0;
        for (std::size_t i = 0; i < item.size(); ++i) {
            if (!(file >> value)) {
                return cases;
            }
            item[i] = checked_i8(value, "global avg pool input");
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
    int avg = 0;
    while (true) {
        GoldenData item;
        if (!(file >> item.sum >> avg)) {
            break;
        }
        item.avg = checked_i8(avg, "global avg pool avg");
        golden.push_back(item);
    }
    return golden;
}

uint8_t as_u8(int8_t value) {
    return static_cast<uint8_t>(value);
}

int8_t from_cdata(CData value) {
    return static_cast<int8_t>(value);
}

int32_t from_idata(IData value) {
    return static_cast<int32_t>(value);
}

void drive(Vglobal_avg_pool_8x8& top, const CaseData& item) {
    top.in00 = as_u8(item[0]);
    top.in01 = as_u8(item[1]);
    top.in02 = as_u8(item[2]);
    top.in03 = as_u8(item[3]);
    top.in04 = as_u8(item[4]);
    top.in05 = as_u8(item[5]);
    top.in06 = as_u8(item[6]);
    top.in07 = as_u8(item[7]);
    top.in08 = as_u8(item[8]);
    top.in09 = as_u8(item[9]);
    top.in10 = as_u8(item[10]);
    top.in11 = as_u8(item[11]);
    top.in12 = as_u8(item[12]);
    top.in13 = as_u8(item[13]);
    top.in14 = as_u8(item[14]);
    top.in15 = as_u8(item[15]);
    top.in16 = as_u8(item[16]);
    top.in17 = as_u8(item[17]);
    top.in18 = as_u8(item[18]);
    top.in19 = as_u8(item[19]);
    top.in20 = as_u8(item[20]);
    top.in21 = as_u8(item[21]);
    top.in22 = as_u8(item[22]);
    top.in23 = as_u8(item[23]);
    top.in24 = as_u8(item[24]);
    top.in25 = as_u8(item[25]);
    top.in26 = as_u8(item[26]);
    top.in27 = as_u8(item[27]);
    top.in28 = as_u8(item[28]);
    top.in29 = as_u8(item[29]);
    top.in30 = as_u8(item[30]);
    top.in31 = as_u8(item[31]);
    top.in32 = as_u8(item[32]);
    top.in33 = as_u8(item[33]);
    top.in34 = as_u8(item[34]);
    top.in35 = as_u8(item[35]);
    top.in36 = as_u8(item[36]);
    top.in37 = as_u8(item[37]);
    top.in38 = as_u8(item[38]);
    top.in39 = as_u8(item[39]);
    top.in40 = as_u8(item[40]);
    top.in41 = as_u8(item[41]);
    top.in42 = as_u8(item[42]);
    top.in43 = as_u8(item[43]);
    top.in44 = as_u8(item[44]);
    top.in45 = as_u8(item[45]);
    top.in46 = as_u8(item[46]);
    top.in47 = as_u8(item[47]);
    top.in48 = as_u8(item[48]);
    top.in49 = as_u8(item[49]);
    top.in50 = as_u8(item[50]);
    top.in51 = as_u8(item[51]);
    top.in52 = as_u8(item[52]);
    top.in53 = as_u8(item[53]);
    top.in54 = as_u8(item[54]);
    top.in55 = as_u8(item[55]);
    top.in56 = as_u8(item[56]);
    top.in57 = as_u8(item[57]);
    top.in58 = as_u8(item[58]);
    top.in59 = as_u8(item[59]);
    top.in60 = as_u8(item[60]);
    top.in61 = as_u8(item[61]);
    top.in62 = as_u8(item[62]);
    top.in63 = as_u8(item[63]);
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto cases = read_cases(data_dir + "global_avg_pool_8x8_cases.txt");
    const auto golden = read_golden(data_dir + "global_avg_pool_8x8_golden.txt");

    if (cases.empty()) {
        throw std::runtime_error("no GlobalAvgPool8x8 test cases loaded");
    }
    if (cases.size() != golden.size()) {
        throw std::runtime_error("case count does not match golden count");
    }

    Vglobal_avg_pool_8x8 top;
    bool all_pass = true;

    for (std::size_t i = 0; i < cases.size(); ++i) {
        drive(top, cases[i]);
        top.eval();

        const int32_t rtl_sum = from_idata(top.sum);
        const int8_t rtl_avg = from_cdata(top.avg);

        if (rtl_sum != golden[i].sum || rtl_avg != golden[i].avg) {
            std::cerr << "case " << i << " mismatch: rtl_sum=" << rtl_sum
                      << " golden_sum=" << golden[i].sum
                      << " rtl_avg=" << static_cast<int>(rtl_avg)
                      << " golden_avg=" << static_cast<int>(golden[i].avg) << "\n";
            all_pass = false;
        }
    }

    if (!all_pass) {
        std::cerr << "GlobalAvgPool8x8 test FAILED\n";
        return 1;
    }

    std::cout << "GlobalAvgPool8x8 test cases: " << cases.size() << "\n";
    std::cout << "PASS\n";
    return 0;
}
