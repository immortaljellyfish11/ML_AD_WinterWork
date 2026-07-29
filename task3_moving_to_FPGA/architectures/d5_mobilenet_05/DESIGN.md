# D5 MobileNetV1 0.5x Design Record

## Status

Current revision retains exactly **two** consecutive 256→256 depthwise-separable blocks. FP32 and INT8 accuracy were measured from the current 150-epoch CIFAR-10 checkpoint. Resource values are explicitly labelled **predicted** until a compatible routed accelerator is available. The RTL descriptor sequencer is structurally verified using data exported from the trained checkpoint; a full-weight image is deliberately external-memory based.

## Network

| Stage | Input | Operation | Output | Parameters | MACs/image |
|---|---|---|---|---:|---:|
| Stem | 3x32x32 | 3x3, 3→16, BN, ReLU | 16x32x32 | 464 | 442,368 |
| Block 1 | 16x32x32 | DW 3x3 + PW 16→32 | 32x32x32 | 752 | 671,744 |
| Block 2 | 32x32x32 | DW 3x3 stride 2 + PW 32→64 | 64x16x16 | 2,528 | 598,016 |
| Block 3 | 64x16x16 | DW 3x3 + PW 64→64 | 64x16x16 | 4,928 | 1,196,032 |
| Block 4 | 64x16x16 | DW 3x3 stride 2 + PW 64→128 | 128x8x8 | 9,152 | 561,152 |
| Block 5 | 128x8x8 | DW 3x3 + PW 128→128 | 128x8x8 | 18,048 | 1,122,304 |
| Block 6 | 128x8x8 | DW 3x3 stride 2 + PW 128→256 | 256x4x4 | 34,688 | 542,720 |
| Block 7 | 256x4x4 | DW 3x3 + PW 256→256 | 256x4x4 | 68,864 | 1,085,440 |
| Block 8 | 256x4x4 | DW 3x3 + PW 256→256 | 256x4x4 | 68,864 | 1,085,440 |
| Block 9 | 256x4x4 | DW 3x3 stride 2 + PW 256→512 | 512x2x2 | 134,912 | 533,504 |
| Block 10 | 512x2x2 | DW 3x3 + PW 512→512 | 512x2x2 | 268,800 | 1,067,008 |
| GAP + Linear | 512x2x2 | global average pool + 512→10 | 10 logits | 5,130 | 5,120 |
| **Total** | | | | **617,130** | **8,910,848** |

The parameter column includes trainable BatchNorm affine terms; the deployment exporter folds BatchNorm into convolution weights/biases. 
## Results

| Metric          |                  Value | Evidence                                       |
| --------------- | ---------------------: | ---------------------------------------------- |
| Parameters      |                617,130 | model summary                                  |
| FP32 accuracy   |                 88.70% | measured CIFAR-10 test set                     |
| INT8 accuracy   |                 88.00% | measured per-layer PTQ                         |
| LUT utilization | 5,600 / 17,600 (31.8%) | predicted tiled/external-weight implementation |
| DSP utilization |          4 / 80 (5.0%) | predicted tiled/external-weight implementation |
| BRAM requirement | 135.64 / 60 (226.1%) | all INT8 weights + INT32 biases on chip; excludes feature buffers |

## Current Vivado verification

The external-weight descriptor sequencer passed XSim and post-route Vivado at a 100 MHz constraint (68 LUT as Logic, 0 DSP48E1, 0 Block RAM Tiles, WNS +4.229 ns). These values cover only layer sequencing and descriptor generation. No bitstream was generated.

## Quantization

Each BatchNorm is folded into its depthwise or pointwise convolution. Weights/activations are symmetric INT8; INT32 is used for bias and accumulation; requantization uses an integer multiplier and shift. Current memories and descriptor vectors are exported from the trained checkpoint used for the reported accuracy.

## RTL mapping

The ten-block descriptor sequencer orders `stem → blocks 1..10 → GAP → FC`. The reduced network still exports 609,248 INT8 weights plus 3,946 INT32 biases (625,032 bytes, about 135.6 BRAM36-equivalents), so external/tiled weights are mandatory. No full all-BRAM D5 mapping or bitstream is claimed.

## Commands

```powershell
python architectures/d5_mobilenet_05/run.py summary
python architectures/d5_mobilenet_05/run.py initialize-checkpoint
python architectures/d5_mobilenet_05/run.py train
python architectures/d5_mobilenet_05/run.py export-int8
```

```bash
bash architectures/d5_mobilenet_05/rtl/sim/build_and_run.sh
```
