#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Vedge_cnn_top.h"
#include "verilated.h"

namespace {

// 这几个常量来自网络输入尺寸: 3 个颜色通道，每个通道 32x32。
// edge_cnn_top 约定输入 buffer_a 的排列顺序是 CHW:
//   addr = channel * 32 * 32 + row * 32 + col
constexpr std::size_t kInputChannels = 3;
constexpr std::size_t kInputHeight = 32;
constexpr std::size_t kInputWidth = 32;
constexpr std::size_t kInputSize = kInputChannels * kInputHeight * kInputWidth;
constexpr std::size_t kNumClasses = 10;

// 完整 CNN 一次推理大约两千多万拍。这里给 1 亿拍上限，防止 RTL 卡死时 testbench 无限循环。
constexpr std::uint64_t kMaxCyclesPerInference = 100000000ULL;

struct InputCase {
    std::array<int8_t, kInputSize> pixels{};
};

struct GoldenData {
    int label = 0;                  // CIFAR-10 原始标签，只用于打印观察。
    int class_id = 0;               // Python golden 算出的分类结果。
    int32_t max_logit = 0;          // Python golden 算出的最大 logit。
    std::array<int32_t, kNumClasses> logits{};
};

int8_t checked_i8(int value, const std::string& name) {
    if (value < -128 || value > 127) {
        throw std::runtime_error(name + " is outside int8 range");
    }
    return static_cast<int8_t>(value);
}

int checked_class_id(int value, const std::string& name) {
    if (value < 0 || value >= static_cast<int>(kNumClasses)) {
        throw std::runtime_error(name + " is outside class id range");
    }
    return value;
}

// Verilator 的 8 bit 端口类型本质上是 uint8_t。
// 对 signed [7:0] 端口写 -1 时，不能直接写 -1，而要写它的 8 bit 补码 0xff。
// static_cast<uint8_t>(int8_t(-1)) 正好会得到 0xff。
uint8_t as_u8(int8_t value) {
    return static_cast<uint8_t>(value);
}

int32_t from_idata(IData value) {
    return static_cast<int32_t>(value);
}

std::vector<InputCase> read_input_cases(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<InputCase> cases;
    while (true) {
        InputCase item;
        int value = 0;

        if (!(file >> value)) {
            break;
        }
        item.pixels[0] = checked_i8(value, "input pixel");

        for (std::size_t i = 1; i < item.pixels.size(); ++i) {
            if (!(file >> value)) {
                throw std::runtime_error("truncated input case in " + path);
            }
            item.pixels[i] = checked_i8(value, "input pixel");
        }

        cases.push_back(item);
    }

    return cases;
}

std::vector<GoldenData> read_golden(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("failed to open " + path);
    }

    std::vector<GoldenData> golden;
    while (true) {
        GoldenData item;
        if (!(file >> item.label)) {
            break;
        }
        if (!(file >> item.class_id >> item.max_logit)) {
            throw std::runtime_error("truncated golden header in " + path);
        }
        item.class_id = checked_class_id(item.class_id, "golden class_id");

        for (std::size_t i = 0; i < item.logits.size(); ++i) {
            if (!(file >> item.logits[i])) {
                throw std::runtime_error("truncated golden logits in " + path);
            }
        }

        golden.push_back(item);
    }

    return golden;
}

const char* state_name(int state) {
    switch (state) {
        case 0: return "IDLE";
        case 1: return "CONV1";
        case 2: return "POOL1";
        case 3: return "CONV2";
        case 4: return "POOL2";
        case 5: return "CONV3";
        case 6: return "GAP";
        case 7: return "LINEAR";
        case 8: return "ARGMAX";
        case 9: return "DONE";
        default: return "UNKNOWN";
    }
}

void set_idle_inputs(Vedge_cnn_top& top) {
    top.start = 0;
    top.input_we = 0;
    top.input_addr = 0;
    top.input_data = 0;
}

// tick 表示“走一个完整时钟周期”。
// RTL 里的 always @(posedge clk) 只会在 clk 从 0 变 1 的那一瞬间更新寄存器。
// 所以这里先 eval 一次低电平，再把 clk 拉高 eval 一次，这样 Verilator 才会执行 posedge 逻辑。
void tick(Vedge_cnn_top& top, std::uint64_t& cycle) {
    top.clk = 0;
    top.eval();
    top.clk = 1;
    top.eval();
    ++cycle;
}

void reset_dut(Vedge_cnn_top& top, std::uint64_t& cycle) {
    set_idle_inputs(top);
    top.rst_n = 0;

    // 多给几拍复位，写 testbench 时这样做更稳，也更接近真实硬件上电复位。
    for (int i = 0; i < 5; ++i) {
        tick(top, cycle);
    }

    top.rst_n = 1;
    tick(top, cycle);
}

void write_input_image(Vedge_cnn_top& top, std::uint64_t& cycle, const InputCase& input) {
    // edge_cnn_top 的输入不是 ROM，而是外部写口。
    // start 拉高前，testbench 逐地址把 3x32x32 的 INT8 图片写入 buffer_a。
    for (std::size_t addr = 0; addr < input.pixels.size(); ++addr) {
        top.input_we = 1;
        top.input_addr = static_cast<IData>(addr);
        top.input_data = as_u8(input.pixels[addr]);
        tick(top, cycle);
    }

    // 写完后关闭写使能，避免后续推理过程中误写 buffer。
    top.input_we = 0;
    top.input_addr = 0;
    top.input_data = 0;
    tick(top, cycle);
}

void start_one_inference(Vedge_cnn_top& top, std::uint64_t& cycle) {
    // start 是一个“一拍脉冲”: 拉高 1 个周期即可，FSM 会记住这次启动。
    top.start = 1;
    tick(top, cycle);
    top.start = 0;
}

void wait_until_done(Vedge_cnn_top& top, std::uint64_t& cycle) {
    int last_state = -1;
    const std::uint64_t start_cycle = cycle;

    while (!top.done) {
        const int current_state = static_cast<int>(top.state_dbg);
        if (current_state != last_state) {
            std::cout << "  cycle " << cycle << ": state=" << state_name(current_state) << "\n";
            last_state = current_state;
        }

        if (cycle - start_cycle > kMaxCyclesPerInference) {
            throw std::runtime_error("edge_cnn_top timeout: done was not asserted");
        }

        tick(top, cycle);
    }

    std::cout << "  cycle " << cycle << ": state=DONE, inference finished\n";
}

bool check_result(const Vedge_cnn_top& top, const GoldenData& golden, std::size_t case_index) {
    const int rtl_class_id = static_cast<int>(top.class_id);
    const int32_t rtl_max_logit = from_idata(top.max_logit);

    std::cout << "case " << case_index << "\n";
    std::cout << "  CIFAR label  : " << golden.label << "\n";
    std::cout << "  RTL class    : " << rtl_class_id << "\n";
    std::cout << "  Golden class : " << golden.class_id << "\n";
    std::cout << "  RTL max_logit: " << rtl_max_logit << "\n";
    std::cout << "  Golden max   : " << golden.max_logit << "\n";

    bool pass = true;
    if (rtl_class_id != golden.class_id) {
        std::cerr << "  class_id mismatch\n";
        pass = false;
    }
    if (rtl_max_logit != golden.max_logit) {
        std::cerr << "  max_logit mismatch\n";
        pass = false;
    }

    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    const std::string data_dir = "sim/data/";
    const auto input_cases = read_input_cases(data_dir + "edge_cnn_top_input.txt");
    const auto golden = read_golden(data_dir + "edge_cnn_top_golden.txt");

    if (input_cases.empty()) {
        throw std::runtime_error("no edge_cnn_top input cases loaded");
    }
    if (input_cases.size() != golden.size()) {
        throw std::runtime_error("input case count does not match golden count");
    }

    Vedge_cnn_top top;
    bool all_pass = true;

    for (std::size_t case_index = 0; case_index < input_cases.size(); ++case_index) {
        std::uint64_t cycle = 0;

        reset_dut(top, cycle);
        write_input_image(top, cycle, input_cases[case_index]);
        start_one_inference(top, cycle);
        wait_until_done(top, cycle);

        if (!check_result(top, golden[case_index], case_index)) {
            all_pass = false;
        }
    }

    top.final();

    if (!all_pass) {
        std::cerr << "edge_cnn_top full CNN test FAILED\n";
        return 1;
    }

    std::cout << "edge_cnn_top full CNN test cases: " << input_cases.size() << "\n";
    std::cout << "PASS\n";
    return 0;
}
