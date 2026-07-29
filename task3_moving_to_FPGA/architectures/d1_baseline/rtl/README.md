# D1 RTL link

D1 uses `../../../rtl/src/edge_cnn_top.v` and the per-layer parameter set in
`../../../rtl/sim/data/fpga_params/per_layer_int8/`.

Build and run the full Verilator regression from `rtl/`:

```bash
./sim/build_edge_cnn.sh
./build/obj_edge_cnn_top/Vedge_cnn_top
```

The testbench counts cycles between `start` and `done`. The verified core latency is 21,060,263
cycles. Vivado setup and reproducible batch checks are in `../../../rtl/sys/setup_project.tcl`,
`run_synth_check.tcl`, and `run_impl_check.tcl`.
