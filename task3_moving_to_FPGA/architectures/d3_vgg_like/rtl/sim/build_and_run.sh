#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
DESIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${DESIGN_DIR}"
PARAM_DIR="artifacts/int8_params"
read -r M1 S1 M2 S2 M3 S3 M4 S4 < <(python3 -c 'import json; q=json.load(open("artifacts/int8_params/quantization.json")); print(*sum(([q["layers"][n]["requant_multiplier"],q["layers"][n]["requant_shift"]] for n in ("conv1","conv2","conv3","conv4")),[]))')
verilator -Wall --cc --top-module d3_vgg_like_top rtl/src/*.v --exe rtl/sim/tb_d3_vgg_like.cpp -Mdir rtl/sim/obj_d3_d16 \
  -GC1_W="\"${PARAM_DIR}/conv1_weight.mem\"" -GC2_W="\"${PARAM_DIR}/conv2_weight.mem\"" -GC3_W="\"${PARAM_DIR}/conv3_weight.mem\"" \
  -GC4_W="\"${PARAM_DIR}/conv4_weight.mem\"" -GFC_W="\"${PARAM_DIR}/linear_weight.mem\"" \
  -GC1_B="\"${PARAM_DIR}/conv1_bias.mem\"" -GC2_B="\"${PARAM_DIR}/conv2_bias.mem\"" -GC3_B="\"${PARAM_DIR}/conv3_bias.mem\"" \
  -GC4_B="\"${PARAM_DIR}/conv4_bias.mem\"" -GFC_B="\"${PARAM_DIR}/linear_bias.mem\"" \
  -GC1_M=${M1} -GC1_S=${S1} -GC2_M=${M2} -GC2_S=${S2} -GC3_M=${M3} -GC3_S=${S3} \
  -GC4_M=${M4} -GC4_S=${S4}
make -C rtl/sim/obj_d3_d16 -f Vd3_vgg_like_top.mk Vd3_vgg_like_top
./rtl/sim/obj_d3_d16/Vd3_vgg_like_top
