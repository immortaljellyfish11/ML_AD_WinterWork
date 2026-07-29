# D2 Small CNN Design Record


## Network

| Stage | Input | Operation | Output | Parameters | MACs/image |
|---|---|---|---|---:|---:|
| Conv1 | 3x32x32 | 3x3, 3→16, ReLU | 16x32x32 | 448 | 442,368 |
| Pool1 | 16x32x32 | 2x2 max pool, stride 2 | 16x16x16 | 0 | 0 |
| Conv2 | 16x16x16 | 3x3, 16→32, ReLU | 32x16x16 | 4,640 | 1,179,648 |
| Pool2 | 32x16x16 | 2x2 max pool, stride 2 | 32x8x8 | 0 | 0 |
| Conv3 | 32x8x8 | 3x3, 32→64, ReLU | 64x8x8 | 18,496 | 1,179,648 |
| GAP | 64x8x8 | global average pool | 64 | 0 | 0 |
| Linear | 64 | fully connected, 64→10 | 10 logits | 650 | 640 |
| **Total** | | | | **24,234** | **2,802,304** |

## Results

| Metric | Value | Evidence |
|---|---:|---|
| Parameters | 24,234 | measured model summary |
| FP32 accuracy | 77.79% | measured CIFAR-10 test set |
| INT8 accuracy | 76.70% | measured per-layer PTQ |
| LUT utilization | 2,296 / 17,600 (13.0%) | measured post-route Vivado |
| DSP utilization | 3 / 80 (3.8%) | measured post-route Vivado |
| BRAM requirement | 5.34 / 60 (8.9%) | all INT8 weights + INT32 biases on chip; excludes feature buffers |

## Quantization

Per-layer symmetric INT8 uses INT32 bias/accumulation and multiplier/shift requantization. Calibration uses 2,048 validation images and a 99.9th-percentile activation threshold.

## RTL mapping

A reusable 3x3 engine and ping-pong feature buffers execute Conv1, Pool1, Conv2, Pool2, Conv3, GAP, and Linear sequentially.

## Commands

```powershell
python architectures/d2_small/run.py summary
python architectures/d2_small/run.py train
python architectures/d2_small/run.py export-int8
```

```bash
bash architectures/d2_small/rtl/sim/build_and_run.sh
```
