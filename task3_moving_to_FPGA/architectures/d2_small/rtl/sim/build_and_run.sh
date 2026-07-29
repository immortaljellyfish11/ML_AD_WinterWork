#!/usr/bin/env bash
set -euo pipefail
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
DESIGN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${DESIGN_DIR}"
PARAM_DIR="artifacts/int8_params"
read -r M1 S1 M2 S2 M3 S3 < <(python3 -c 'import json; q=json.load(open("artifacts/int8_params/quantization.json")); print(*sum(([q["layers"][n]["requant_multiplier"],q["layers"][n]["requant_shift"]] for n in ("conv1","conv2","conv3")),[]))')
verilator -Wall --cc --top-module d2_small_top rtl/src/*.v --exe rtl/sim/tb_d2_small.cpp \
  -Mdir rtl/sim/obj_d2 -GCONV1_WEIGHT_FILE="\"${PARAM_DIR}/conv1_weight.mem\"" \
  -GCONV2_WEIGHT_FILE="\"${PARAM_DIR}/conv2_weight.mem\"" -GCONV3_WEIGHT_FILE="\"${PARAM_DIR}/conv3_weight.mem\"" \
  -GLINEAR_WEIGHT_FILE="\"${PARAM_DIR}/linear_weight.mem\"" -GCONV1_BIAS_FILE="\"${PARAM_DIR}/conv1_bias.mem\"" \
  -GCONV2_BIAS_FILE="\"${PARAM_DIR}/conv2_bias.mem\"" -GCONV3_BIAS_FILE="\"${PARAM_DIR}/conv3_bias.mem\"" \
  -GLINEAR_BIAS_FILE="\"${PARAM_DIR}/linear_bias.mem\"" -GCONV1_REQUANT_MULTIPLIER=${M1} -GCONV1_REQUANT_SHIFT=${S1} \
  -GCONV2_REQUANT_MULTIPLIER=${M2} -GCONV2_REQUANT_SHIFT=${S2} -GCONV3_REQUANT_MULTIPLIER=${M3} -GCONV3_REQUANT_SHIFT=${S3}
make -C rtl/sim/obj_d2 -f Vd2_small_top.mk Vd2_small_top
./rtl/sim/obj_d2/Vd2_small_top
