# D4 ResNet-20 Design Record

## Status

The software/PTQ topology is measured. Resource values are **predicted** for a tiled external-weight accelerator because an all-on-chip parameter store is not practical on XC7Z010.

## Network

| Stage | Input | Operation | Output | Parameters | MACs/image |
|---|---|---|---|---:|---:|
| Stem | 3x32x32 | 3x3, 3→16, BN, ReLU | 16x32x32 | 464 | 442,368 |
| Stage 1 | 16x32x32 | 3 basic blocks; each 3x3 16→16, 3x3 16→16, identity residual | 16x32x32 | 14,016 | 14,155,776 |
| Stage 2 | 16x32x32 | 3 blocks; first 16→32 stride 2 + 1x1 shortcut, then 2 identity blocks | 32x16x16 | 51,744 | 13,107,200 |
| Stage 3 | 32x16x16 | 3 blocks; first 32→64 stride 2 + 1x1 shortcut, then 2 identity blocks | 64x8x8 | 204,800 | 13,107,200 |
| GAP | 64x8x8 | global average pool | 64 | 0 | 0 |
| Linear | 64 | fully connected, 64→10 | 10 logits | 650 | 640 |
| **Total** | | | | **272,474** | **40,813,184** |

## Results

| Metric           |                  Value | Evidence                                                          |
| ---------------- | ---------------------: | ----------------------------------------------------------------- |
| Parameters       |                272,474 | measured model summary                                            |
| FP32 accuracy    |                 91.12% | measured CIFAR-10 test set                                        |
| INT8 accuracy    |                 90.21% | measured per-layer PTQ                                            |
| LUT utilization  | 6,800 / 17,600 (38.6%) | predicted tiled/external-weight implementation                    |
| DSP utilization  |          5 / 80 (6.3%) | predicted tiled/external-weight implementation                    |
| BRAM requirement |     59.48 / 60 (99.1%) | all INT8 weights + INT32 biases on chip; excludes feature buffers |

## Quantization

BatchNorm is folded into each preceding convolution. Weights/activations are symmetric INT8; bias and accumulators are INT32. Main and shortcut branches are requantized to the same scale before residual addition.

## RTL mapping

The planned accelerator reuses 3x3 and 1x1 engines with ping-pong feature and residual buffers. The 272,474 parameters alone are about 59 BRAM36 tiles when stored as INT8, before buffers and bias; external streaming or tiling is required.

## Commands

```powershell
python architectures/d4_resnet20/run.py summary
python architectures/d4_resnet20/run.py train
python architectures/d4_resnet20/run.py export-int8
```
