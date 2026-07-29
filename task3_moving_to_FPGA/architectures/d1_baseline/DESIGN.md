# D1 Baseline+256 Design Record

## Status

Current revision: `32-64-128-256`. FP32 and INT8 accuracy were measured from the current 150-epoch CIFAR-10 checkpoint. Resource values marked **predicted** are planning estimates, not board or post-route results. The current INT8 memories and RTL vectors were exported from that trained checkpoint.

## Network

| Stage     | Input    | Operation               | Output    |  Parameters |     MACs/image |
| --------- | -------- | ----------------------- | --------- | ----------: | -------------: |
| Conv1     | 3x32x32  | 3x3, 3→32, ReLU         | 32x32x32  |         896 |        884,736 |
| Pool1     | 32x32x32 | 2x2 max pool, stride 2  | 32x16x16  |           0 |              0 |
| Conv2     | 32x16x16 | 3x3, 32→64, ReLU        | 64x16x16  |      18,496 |      4,718,592 |
| Pool2     | 64x16x16 | 2x2 max pool, stride 2  | 64x8x8    |           0 |              0 |
| Conv3     | 64x8x8   | 3x3, 64→128, ReLU       | 128x8x8   |      73,856 |      4,718,592 |
| Conv4     | 128x8x8  | 3x3, 128→256, ReLU      | 256x8x8   |     295,168 |     18,874,368 |
| GAP       | 256x8x8  | global average pool     | 256       |           0 |              0 |
| Linear    | 256      | fully connected, 256→10 | 10 logits |       2,570 |          2,560 |
| **Total** |          |                         |           | **390,986** | **29,198,848** |

## Results

| Metric           |                  Value | Evidence                                                          |
| ---------------- | ---------------------: | ----------------------------------------------------------------- |
| Parameters       |                390,986 | model summary                                                     |
| FP32 accuracy    |                 85.29% | measured CIFAR-10 test set                                        |
| INT8 accuracy    |                 81.34% | measured per-layer PTQ                                            |
| LUT utilization  | 2,945 / 17,600 (16.7%) | measured Vivado synthesis; unplaced                               |
| DSP utilization  |          4 / 80 (5.0%) | measured Vivado synthesis; unplaced                               |
| BRAM requirement |    85.17 / 60 (141.9%) | all INT8 weights + INT32 biases on chip; excludes feature buffers |

## Current Vivado verification

The trained-parameter XSim test passed. Synthesis completed with 2,945 LUT as Logic, 4 DSP48E1, and 121.5 Block RAM Tiles; this BRAM value includes feature buffers and is therefore not substituted into the common weight-plus-bias BRAM comparison. Placement did not run because the XC7Z010 has only 60 BRAM36 Tiles (120 RAMB18-equivalent sites) while this top requires 121 BRAM36 and one RAMB18. 

## Quantization

Per-layer symmetric INT8 weights/activations, INT32 bias and accumulation, 99.9th-percentile activation calibration, and integer multiplier/shift requantization are used. The current `.mem` files were exported from the measured 150-epoch checkpoint.

## RTL mapping

The sequential RTL uses ping-pong feature buffers and one reusable 3x3 engine: `Conv1 → Pool1 → Conv2 → Pool2 → Conv3 → Conv4 → GAP → Linear`. The added Conv4 and the classifier now operate on 256 channels. Full D1 weights exceed a practical all-BRAM mapping, so the comparison estimate assumes tiled/external weight loading.

## Commands

```powershell
python architectures/d1_baseline/run.py summary
python architectures/d1_baseline/run.py initialize-checkpoint
python architectures/d1_baseline/run.py train
python architectures/d1_baseline/run.py export-int8
```

Use the RTL simulation/build scripts after export. Vivado characterization stops after simulation, synthesis, and route; do not generate a bitstream.
