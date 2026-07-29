# D4 RTL Boundary / D4 RTL 边界

D4 software FP32 and calibrated INT8 are implemented. A complete on-chip ROM design is not claimed because the 272,474 INT8 parameters already require about 59 BRAM36 tiles, leaving no practical space for feature buffers and bias ROMs on XC7Z010. / D4 的 FP32 软件和校准 INT8 已实现。没有冒充完成片上 ROM RTL：272,474 个 INT8 参数约需要 59 个 BRAM36，XC7Z010 无法同时容纳实用的 feature buffer 和 bias ROM。

The intended RTL architecture is a reusable 3x3 engine, 1x1 projection engine, ping-pong feature buffers, residual storage, and scale-aligned addition. The next implementation must stream weights from DDR or load them tile by tile. / 目标 RTL 架构是可复用 3x3 引擎、1x1 projection 引擎、ping-pong feature buffer、残差存储和 scale 对齐加法；下一步需要从 DDR 流式读取或分块加载权重。
