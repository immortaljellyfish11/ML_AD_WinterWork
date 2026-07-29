# D5 RTL Boundary / D5 RTL 边界

Revision `2026-07-28-two-256-256-blocks` retains only two consecutive 256-to-256 blocks. The new `src/d5_mobilenet_05_top.v` is a Verilator-tested external-weight descriptor sequencer for the stem, ten depthwise/pointwise blocks, GAP and FC. It emits the folded layer type, shape, stride and external weight offset to a tiled accelerator.

`sys/run_all.tcl` is a batch Vivado flow for **this descriptor sequencer only**: XSim verifies the ordered handshake `stem -> 10 blocks -> GAP -> FC`; synthesis and implementation stop at `route_design` for `xc7z010clg400-1` with a 100 MHz clock. It writes reports under `rtl/sys/reports/` and never creates a bitstream. It must not be interpreted as a synthesis or implementation result for a full D5 MobileNet accelerator, external weight memory, or convolution datapath.

Run from the repository root with:

```text
D:\\appinstallroot\\XILINX\\Vivado\\2024.2\\bin\\vivado.bat -mode batch -source architectures/d5_mobilenet_05/rtl/sys/run_all.tcl
```

D5 software FP32 and calibrated INT8 are implemented. A complete on-chip ROM design is not claimed because 823,722 INT8 parameters need about 179 BRAM36 tiles while XC7Z010 provides only 60. / D5 的 FP32 软件和校准 INT8 已实现。没有冒充完成片上 ROM RTL：823,722 个 INT8 参数约需要 179 个 BRAM36，而 XC7Z010 只有 60 个。

The intended RTL architecture has independent depthwise 3x3 and pointwise 1x1 engines, shared requantization, and tiled external-weight loading. / 目标 RTL 架构包含独立 depthwise 3x3 和 pointwise 1x1 引擎、共享重定标模块，以及分块的外部权重加载。
