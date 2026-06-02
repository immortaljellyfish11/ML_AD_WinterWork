# FPGA CNN 项目进度

> 位置: `task3_moving_to_FPGA/PROGRESS.md`  
> 作用: 记录当前项目状态。每次继续工作前优先阅读本文。  
> 最近更新: 2026-05-28

---

## 当前状态

**项目阶段**: Phase 2 - RTL 设计与 Verilator 单元仿真  
**当前重点**: 先建立可靠的“Python golden -> Verilator RTL -> C++ testbench 对比”闭环，再逐步扩展到池化、卷积层和完整 CNN。

当前已经完成:

- CIFAR-10 CNN 浮点训练与权重保存
- Python 定点推理/量化分析
- Verilator 通过 Cygwin 跑通
- 最小 `pe_3x3` RTL 仿真通过
- 基础算术模块与 ReLU 模块 RTL 仿真通过

---

## 已完成工作

### 1. 算法与量化

位置:

- `task3_moving_to_FPGA/main.py`
- `task3_moving_to_FPGA/fixed_point_simulator.py`
- `task3_moving_to_FPGA/outputs/`

结果:

- 已完成 CIFAR-10 CNN 训练流程
- 已导出训练权重: `outputs/cifar10_cnn_best.pth`
- 已导出定点参数: `outputs/cifar10_cnn_q16.npz`
- 已实现 Python 定点 CNN 推理器
- 已生成精度退化曲线: `outputs/accuracy_degradation_curve.png`

当前硬件侧暂按 **INT8 / Q8.7** 方向推进。注意: 代码中历史导出的 `q16` 文件仍可作为软件定点参考，但 RTL 模块当前正在围绕 INT8 数据路径建立。

### 2. Verilator 仿真环境

当前 Verilator 通过 Cygwin 调用:

```bash
cd /cygdrive/e/ML_AD_WinterWork/task3_moving_to_FPGA/rtl
verilator --version
```

已确认版本:

```text
Verilator 5.049 devel rev v5.048-111-g69b3c5f6d
```

PowerShell 中应通过 Cygwin bash 调用，例如:

```powershell
& 'C:\cygwin64\bin\bash.exe' -lc 'cd /cygdrive/e/ML_AD_WinterWork/task3_moving_to_FPGA/rtl && verilator --version'
```

已知小问题:

- Cygwin 启动时会打印 `/cygdrive/c/Users/ASUS` 权限警告。
- 目前不影响项目目录下的 Verilator 编译和仿真。

### 3. PE 3x3 仿真闭环

相关文件:

- RTL: `rtl/src/pe_3x3.v`
- Python 测试向量: `rtl/sim/gen_pe_3x3_data.py`
- C++ testbench: `rtl/tb/tb_pe_3x3.cpp`
- 数据: `rtl/sim/data/pe_3x3_*.txt`

功能:

- 9 个 INT8 activation
- 9 个 INT8 weight
- 乘法后累加为 INT32

已通过测试:

```text
PE 3x3 RTL result: -6
PE 3x3 golden:     -6
PASS
```

### 4. 基础算术与激活模块

新增 RTL:

- `rtl/src/mac_int8.v`
- `rtl/src/accumulator_9.v`
- `rtl/src/requantize.v`
- `rtl/src/saturate_int8.v`
- `rtl/src/relu.v`
- `rtl/src/arith_activation_top.v`

新增仿真文件:

- `rtl/sim/gen_arith_activation_data.py`
- `rtl/tb/tb_arith_activation.cpp`
- `rtl/sim/data/arith_activation_cases.txt`
- `rtl/sim/data/arith_activation_golden.txt`

覆盖能力:

- INT8 x INT8 乘法
- INT32 累加
- Q8.7 右移量化
- INT8 饱和截断
- ReLU 负数清零
- 量化后 ReLU + saturate 的组合路径

已通过 Verilator 测试:

```text
Arithmetic/activation test cases: 4
PASS
```

---

## 当前 RTL 目录结构

```text
task3_moving_to_FPGA/rtl/
  src/
    pe_3x3.v
    mac_int8.v
    accumulator_9.v
    requantize.v
    saturate_int8.v
    relu.v
    arith_activation_top.v
  tb/
    tb_pe_3x3.cpp
    tb_arith_activation.cpp
  sim/
    gen_pe_3x3_data.py
    gen_arith_activation_data.py
    data/
  build/
  waves/
```

---

## 常用仿真命令

### PE 3x3

```bash
cd /cygdrive/e/ML_AD_WinterWork/task3_moving_to_FPGA/rtl
python3 sim/gen_pe_3x3_data.py
verilator -Wall --cc src/pe_3x3.v --top-module pe_3x3 --exe tb/tb_pe_3x3.cpp -Mdir build/obj_pe_3x3
make -C build/obj_pe_3x3 -f Vpe_3x3.mk Vpe_3x3
./build/obj_pe_3x3/Vpe_3x3
```

### 算术 + 激活

```bash
cd /cygdrive/e/ML_AD_WinterWork/task3_moving_to_FPGA/rtl
python3 sim/gen_arith_activation_data.py
verilator -Wall --cc \
  src/arith_activation_top.v src/mac_int8.v src/accumulator_9.v \
  src/requantize.v src/saturate_int8.v src/relu.v \
  --top-module arith_activation_top \
  --exe tb/tb_arith_activation.cpp \
  -Mdir build/obj_arith_activation
make -C build/obj_arith_activation -f Varith_activation_top.mk Varith_activation_top
./build/obj_arith_activation/Varith_activation_top
```

---

## 重要设计约定

- activation/weight 第一阶段按 INT8 signed 处理。
- 乘积使用 INT16 signed。
- 累加使用 INT32 signed。
- `requantize.v` 当前采用算术右移，等价于向负无穷方向截断的硬件行为。
- `saturate_int8.v` 将结果限制到 `[-128, 127]`。
- RTL 的首要目标是先与 Python fixed-point golden bit-exact，而不是直接对齐浮点 PyTorch。

---

## 下一步计划

优先级从高到低:

1. 写 `maxpool_2x2.v`、Python 测试向量和 C++ testbench。
2. 写 `global_avg_pool.v`，验证 8x8 -> 1x1 的整数平均。
3. 将 `pe_3x3` 与 `requantize/relu/saturate` 组合成卷积后处理路径。
4. 做单通道 `single_channel_conv3x3`，先验证 1 个输入通道的小 feature map。
5. 扩展到多通道卷积，目标先跑通 Conv1 的一个 output channel。
6. 最后再做完整 Conv1 + ReLU + MaxPool。

---

## 风险与注意点

- `-Wall` 下 Verilator 会暴露位宽扩展、空端口连接等问题，建议保持打开。
- signed/unsigned 转换是当前最容易出错的地方。
- 右移、舍入、饱和策略必须和 Python golden 一致。
- 后续 feature map 的展平顺序必须固定，例如 CHW 或 HWC，不能在 Python 和 RTL 中混用。
- 波形目前还未接入；需要时给 C++ testbench 添加 trace dump。

---

## 工作日志

### 2026-05-15

- 完成 CIFAR-10 CNN 浮点训练流程。
- 完成 Python 定点量化分析。
- 确认 INT8 方向可行。

### 2026-05-27

- 通过 Cygwin 调用 Verilator。
- 建立 `pe_3x3` 的 Python -> RTL -> C++ testbench 仿真闭环。
- `pe_3x3` 固定测试点通过，输出 `-6`。

### 2026-05-28

- 新增基础算术模块: `mac_int8`、`accumulator_9`、`requantize`、`saturate_int8`。
- 新增激活模块: `relu`。
- 新增测试顶层: `arith_activation_top`。
- 新增 Python 测试向量与 C++ testbench。
- Verilator 仿真 4 组测试全部通过。
