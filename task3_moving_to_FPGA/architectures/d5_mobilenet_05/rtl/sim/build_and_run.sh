#!/usr/bin/env bash
set -euo pipefail
DESIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${DESIGN_DIR}"
verilator -Wall --cc --top-module d5_mobilenet_05_top rtl/src/d5_mobilenet_05_top.v \
  --exe rtl/sim/tb_d5_mobilenet_05.cpp -Mdir rtl/sim/obj_d5
make -C rtl/sim/obj_d5 -f Vd5_mobilenet_05_top.mk Vd5_mobilenet_05_top
./rtl/sim/obj_d5/Vd5_mobilenet_05_top
