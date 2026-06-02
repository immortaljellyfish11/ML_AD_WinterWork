from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sim" / "data"
FRAC_BITS = 7


CASES = [
    {
        "acts": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "wgts": [1, 0, -1, 1, 0, -1, 1, 0, -1],
        "bias": 0,
    },
    {
        "acts": [127, 127, 127, 127, 127, 127, 127, 127, 127],
        "wgts": [127, 127, 127, 127, 127, 127, 127, 127, 127],
        "bias": 0,
    },
    {
        "acts": [-128, -3, 4, 5, -6, 7, 8, -9, 10],
        "wgts": [2, -3, 4, -5, 6, -7, 8, -9, 10],
        "bias": 256,
    },
    {
        "acts": [64, -64, 32, -32, 16, -16, 8, -8, 4],
        "wgts": [-4, 8, -16, 32, -64, 64, -32, 16, -8],
        "bias": -512,
    },
]


def relu(value: int) -> int:
    return max(0, value)


def saturate_int8(value: int) -> int:
    return max(-128, min(127, value))


def golden_for(case: dict[str, object]) -> list[int]:
    conv_raw = sum(int(act) * int(wgt) for act, wgt in zip(case["acts"], case["wgts"]))
    conv_bias = conv_raw + int(case["bias"])
    conv_quant = conv_bias // (1 << FRAC_BITS)
    conv_relu = relu(conv_quant)
    conv_out = saturate_int8(conv_relu)
    return [conv_raw, conv_bias, conv_quant, conv_relu, conv_out]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "cnn_test_cases.txt"
    golden_path = DATA_DIR / "cnn_test_golden.txt"

    with cases_path.open("w", encoding="utf-8") as cases_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for case in CASES:
            values = [*case["acts"], *case["wgts"], case["bias"]]
            cases_file.write(" ".join(str(int(value)) for value in values) + "\n")
            golden_file.write(" ".join(str(value) for value in golden_for(case)) + "\n")

    print(f"Generated {len(CASES)} CNN test cases")
    print(f"frac_bits: {FRAC_BITS}")
    print(f"cases:     {cases_path}")
    print(f"golden:    {golden_path}")


if __name__ == "__main__":
    main()
