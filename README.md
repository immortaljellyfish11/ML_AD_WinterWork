# ML_AD_WinterWork

Machine Learning & Autonomous Driving Winter Work project workspace.

This repository currently contains two main lines of work:

- Backpropagation experiments and neural-network basics.
- CIFAR-10 CNN migration toward FPGA RTL implementation.

## Project Structure

```text
ML_AD_WinterWork/
├── README.md
├── Backpropagation/
│   ├── backpropagation_Origin.py
│   ├── demo1.py
│   └── ...
└── task3_moving_to_FPGA/
    ├── main.py
    ├── fixed_point_simulator.py
    ├── accuracy_degradation_curve.py
    ├── outputs/
    │   ├── cifar10_cnn_best.pth
    │   ├── cifar10_cnn_q16.npz
    │   └── accuracy_degradation_curve.png
    └── rtl/
        ├── src/
        ├── sim/
        ├── tb/
        └── build/
```

## Task 1: Backpropagation

This part implements and experiments with basic neural-network training.

Main contents:

- Multi-layer neural network implementation.
- Sigmoid activation and derivative.
- Gradient descent weight updates.
- MSE-based demo training.
- XOR-style small classification experiments.

## Task 3: Moving CNN To FPGA

The current active task is migrating a small CIFAR-10 CNN into an FPGA-friendly RTL flow.

### Network

The CNN structure is:

```text
Input: 3 x 32 x 32

Conv1: 3 -> 32, 3x3, padding=1
ReLU
MaxPool2x2

Conv2: 32 -> 64, 3x3, padding=1
ReLU
MaxPool2x2

Conv3: 64 -> 128, 3x3, padding=1
ReLU
GlobalAvgPool 8x8 -> 1x1

Linear: 128 -> 10
Argmax
```

### Python Side

Relevant files:

- `task3_moving_to_FPGA/main.py`
  - Trains the CIFAR-10 CNN.
  - Saves the floating-point checkpoint.
  - Exports fixed-point parameters.
- `task3_moving_to_FPGA/fixed_point_simulator.py`
  - IImplementation for fixed-point CNN inference.
- `task3_moving_to_FPGA/accuracy_degradation_curve.py`
  - Evaluates accuracy under different fixed-point fractional widths.
  - Exports FPGA/Verilator `.mem` parameter files with INT8.
  - `rtl/sim/data/fpga_params/q07/*.mem`

Current RTL uses `FRAC_BITS = 7`, so `q07` is the directly matched parameter set.

### RTL Side

Important RTL modules:

- `edge_cnn_top.v`: full CNN integration top.
- `cnn_top_fsm.v`: inference-stage controller.
- `conv_layer_engine_sync.v`: convolution layer engine.
- `multi_channel_conv3x3_sync.v`: synchronous multi-channel 3x3 convolution core.
- `pool_layer_2x2_engine_sync.v`: 2x2 maxpool engine.
- `global_avg_pool_layer_engine_sync.v`: 8x8 global average pool engine.
- `linear_128x10_sync.v`: fully connected layer.
- `argmax_10.v`: final class selection.
- `weight_rom.v` / `bias_rom.v`: `.mem` parameter ROMs.
- `feature_buffer_sync.v`: FPGA-style synchronous feature buffer.

### Verilator Status

The project uses Verilator through Cygwin:

Confirmed flow:

- PE 3x3 unit test: PASS
- Arithmetic/ReLU test: PASS
- MaxPool2x2 test: PASS
- GlobalAvgPool8x8 test: PASS
- Single-channel Conv3x3 4x4 test: PASS
- Tiny CNN 4x4 test: PASS
- Full `edge_cnn_top` CNN classification test: PASS

Full CNN testbench:

- Input generator: `rtl/sim/gen_edge_cnn_top_data.py`
- C++ testbench: `rtl/tb/tb_edge_cnn_top.cpp`
- Input data: `rtl/sim/data/edge_cnn_top_input.txt`
- Golden data: `rtl/sim/data/edge_cnn_top_golden.txt`

Example full-CNN simulation:

```bash
cd ./task3_moving_to_FPGA/rtl

python3 sim/gen_edge_cnn_top_data.py --sample-index 0 --num-samples 1 --frac-bits 7

verilator -Wall --cc --top-module edge_cnn_top 
  src/edge_cnn_top.v src/cnn_top_fsm.v src/feature_buffer_sync.v 
  src/weight_rom.v src/bias_rom.v src/conv_layer_engine_sync.v 
  src/conv_controller.v src/multi_channel_conv3x3_sync.v src/conv3x3_addr_gen.v 
  src/requantize.v src/relu.v src/saturate_int8.v 
  src/pool_layer_2x2_engine_sync.v src/global_avg_pool_layer_engine_sync.v 
  src/linear_128x10_sync.v src/argmax_10.v 
  --exe tb/tb_edge_cnn_top.cpp 
  -Mdir build/obj_edge_cnn_top

make -C build/obj_edge_cnn_top -f Vedge_cnn_top.mk Vedge_cnn_top
./build/obj_edge_cnn_top/Vedge_cnn_top
```

Expected current result for sample 0:

```text
RTL class    : 6
Golden class : 6
RTL max_logit: 174
Golden max   : 174
PASS
```

## Next FPGA Bring-Up Work

The CNN core is now functionally verified in Verilator. Work remaining before board testing:

- Add a board-level wrapper, for example `board_top.v`.
- Decide how image input enters FPGA, fixed test image ROM/BRAM initialization will be chosen.
- Connect observable outputs:
- Add FPGA constraint file, such as `.xdc`.
- Run synthesis and implementation on vivado, check for any issues.
- Check timing at a conservative first clock, such as 25 MHz or 50 MHz.
- Run more Verilator samples before bitstream generation.

## Dependencies

FPGA/RTL simulation:

- Cygwin
- Verilator
- GNU Make
- C++ compiler available through Cygwin

