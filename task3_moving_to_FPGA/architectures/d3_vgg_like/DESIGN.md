# D3 VGG-like Design Record

## Status

Current revision: `16-32-64-128` (2026-07-28). FP32 and INT8 accuracy below were measured from the current 150-epoch CIFAR-10 checkpoint. 


## Network

| Stage     | Input     | Operation               | Output    | Parameters |     MACs/image |
| --------- | --------- | ----------------------- | --------- | ---------: | -------------: |
| Conv1     | 3x32x32   | 3x3, 3→16, ReLU         | 16x32x32  |        448 |        442,368 |
| Conv2     | 16x32x32  | 3x3, 16→32, ReLU        | 32x32x32  |      4,640 |      4,718,592 |
| Pool1     | 32x32x32  | 2x2 max pool, stride 2  | 32x16x16  |          0 |              0 |
| Conv3     | 32x16x16  | 3x3, 32→64, ReLU        | 64x16x16  |     18,496 |      4,718,592 |
| Conv4     | 64x16x16  | 3x3, 64→128, ReLU       | 128x16x16 |     73,856 |     18,874,368 |
| Pool2     | 128x16x16 | 2x2 max pool, stride 2  | 128x8x8   |          0 |              0 |
| GAP       | 128x8x8   | global average pool     | 128       |          0 |              0 |
| Linear    | 128       | fully connected, 128→10 | 10 logits |      1,290 |          1,280 |
| **Total** |           |                         |           | **98,730** | **28,755,200** |

## Results

| Metric           |                  Value | Evidence                                                          |
| ---------------- | ---------------------: | ----------------------------------------------------------------- |
| Parameters       |                 98,730 | model summary                                                     |
| FP32 accuracy    |                 83.47% | measured CIFAR-10 test set                                        |
| INT8 accuracy    |                 80.43% | measured per-layer PTQ                                            |
| LUT utilization  | 2,604 / 17,600 (14.8%) | measured Vivado post-route, 100 MHz target                        |
| DSP utilization  |          4 / 80 (5.0%) | measured Vivado post-route                                        |
| BRAM requirement |     21.59 / 60 (36.0%) | all INT8 weights + INT32 biases on chip; excludes feature buffers |

## Current Vivado verification

The current trained-parameter XSim completed before synthesis and routing. Vivado post-route used 2,604 LUT as Logic, 4 DSP48E1, and 27 Block RAM Tiles (the latter includes feature buffers and therefore is not substituted into the common weight-plus-bias BRAM comparison). 
## Quantization

Per-layer symmetric INT8 uses INT32 bias/accumulation and multiplier/shift requantization. The current parameter memories and RTL golden vectors were exported from the trained checkpoint used for the reported accuracy.

## RTL mapping

The RTL controller sequence is `Conv1 → Conv2 → Pool1 → Conv3 → Conv4 → Pool2 → GAP → Linear`. The estimate assumes tiled/external weights because a full BRAM-only map is not a practical XC7Z010 deployment.

## Commands

```powershell
python architectures/d3_vgg_like/run.py summary
python architectures/d3_vgg_like/run.py initialize-checkpoint
python architectures/d3_vgg_like/run.py train
python architectures/d3_vgg_like/run.py export-int8
```

```bash
bash architectures/d3_vgg_like/rtl/sim/build_and_run.sh
```
