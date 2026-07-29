# CIFAR-10 FPGA Architecture Experiments

This directory keeps the model, quantization artifacts, RTL, simulation collateral, and design record for each experiment together.  `common/` holds shared Python infrastructure only.

| Design | Directory | Network | Current RTL scope |
|---|---|---|---|
| D1 | `d1_baseline/` | 32-64-128-256 CNN | RTL/Verilator/Vivado flow |
| D2 | `d2_small/` | 16-32-64 CNN | RTL/Verilator/Vivado flow |
| D3 | `d3_vgg_like/` | 16-32-64-128 VGG-like CNN | RTL/Verilator/Vivado flow |
| D4 | `d4_resnet20/` | ResNet-20 | software/PTQ; external-memory RTL boundary |
| D5 | `d5_mobilenet_05/` | MobileNetV1 0.5x (reduced 256 stage) | External-weight descriptor RTL/Verilator/XSim/Vivado flow |

## Evidence convention

Every `DESIGN.md` distinguishes three kinds of values:

- **Measured**: produced by the stated Python, Verilator, XSim, or post-route Vivado command.
- **Predicted**: an explicitly labelled estimate used for comparison only.
- **Not run**: no result is claimed.

Do not treat a predicted resource number or an RTL simulation as a physical-board result. No board-level test is claimed by this architecture comparison. Existing image artifacts are retained as historical records; new comparison plots use new file names.

## Qimingxing V2 ZYNQ-7010 target configuration

The target is the ALIENTEK Qimingxing V2 ZYNQ-7010 implementation using `xc7z010clg400-1`. The device budget used by the comparison flows is 17,600 LUTs, 35,200 flip-flops, 80 DSP48E1 slices, and 60 BRAM36 tiles (2,160 Kibit, about 270 KiB raw capacity).

Two clocks are deliberately kept separate:

| Use | Source/constraint | Frequency |
|---|---|---:|
| Board wrapper | PL oscillator on `U18`, `pl_clk_50m`, 20.000 ns XDC constraint | 50 MHz |
| Architecture characterization | Core `clk`, 10.000 ns XDC constraint | 100 MHz |

The board wrapper uses the PL clock and can use the board memory interfaces only after their pinout, controller, and capacity are verified against the board manual. This repository does not assert a DDR or QSPI capacity; D4/D5 full-weight designs therefore document an external-memory/tiled-loading boundary rather than claiming all weights fit in PL BRAM.

## Common commands

Run a design from the workspace root:

```powershell
python architectures/d1_baseline/run.py summary
python architectures/d3_vgg_like/run.py summary
python architectures/d5_mobilenet_05/run.py summary
python architectures/plot_accuracy_comparison.py
```

The per-design `DESIGN.md` files give the exact training, export, simulation, and Vivado commands. Bitstream generation and board programming are intentionally outside this comparison flow.
