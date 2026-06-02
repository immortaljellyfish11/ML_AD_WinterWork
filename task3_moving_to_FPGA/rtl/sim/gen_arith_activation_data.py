from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sim" / "data"
FRAC_BITS = 7


CASES = [
    {
        "acts": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "wgts": [1, 0, -1, 1, 0, -1, 1, 0, -1],
        "acc_in": 10,
        "quant_in": -6 << FRAC_BITS,
        "relu_in": -123,
        "sat_in": 200,
    },
    {
        "acts": [127, -128, 64, -64, 1, -1, 0, 12, -12],
        "wgts": [-1, 1, 2, -2, 127, -128, 7, -3, 3],
        "acc_in": -100,
        "quant_in": 16384,
        "relu_in": 456,
        "sat_in": -300,
    },
    {
        "acts": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "wgts": [127, -128, 3, -3, 5, -5, 9, -9, 11],
        "acc_in": 0,
        "quant_in": 127 << FRAC_BITS,
        "relu_in": 0,
        "sat_in": 127,
    },
    {
        "acts": [-5, -4, -3, -2, -1, 1, 2, 3, 4],
        "wgts": [-9, 8, -7, 6, -5, 4, -3, 2, -1],
        "acc_in": 33,
        "quant_in": -129,
        "relu_in": -1,
        "sat_in": -128,
    },
]


def saturate_int8(value: int) -> int:
    return max(-128, min(127, value))


def relu(value: int) -> int:
    return max(0, value)


def golden_for(case: dict[str, object]) -> list[int]:
    acts = case["acts"]
    wgts = case["wgts"]
    products = [int(act) * int(wgt) for act, wgt in zip(acts, wgts)]
    mac_product = products[0]
    mac_acc_out = int(case["acc_in"]) + mac_product
    dot_sum = sum(products)
    requantized = int(case["quant_in"]) // (1 << FRAC_BITS)
    relu_out = relu(int(case["relu_in"]))
    saturated = saturate_int8(int(case["sat_in"]))
    requant_relu_int8 = saturate_int8(relu(requantized))
    return [
        mac_product,
        mac_acc_out,
        dot_sum,
        requantized,
        relu_out,
        saturated,
        requant_relu_int8,
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "arith_activation_cases.txt"
    golden_path = DATA_DIR / "arith_activation_golden.txt"

    with cases_path.open("w", encoding="utf-8") as cases_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for case in CASES:
            case_values = [
                *case["acts"],
                *case["wgts"],
                case["acc_in"],
                case["quant_in"],
                case["relu_in"],
                case["sat_in"],
            ]
            cases_file.write(" ".join(str(int(value)) for value in case_values) + "\n")
            golden_file.write(" ".join(str(value) for value in golden_for(case)) + "\n")

    print(f"Generated {len(CASES)} arithmetic/activation test cases")
    print(f"frac_bits: {FRAC_BITS}")
    print(f"cases:     {cases_path}")
    print(f"golden:    {golden_path}")


if __name__ == "__main__":
    main()
