from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sim" / "data"
FRAC_BITS = 7


CASES = [
    {
        "image": [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        "kernel": [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ],
        "conv_bias": 0,
        "fc0_weight": 32,
        "fc1_weight": -16,
        "fc0_bias": 0,
        "fc1_bias": 1,
    },
    {
        "image": [
            [127, -128, 64, -64],
            [32, -32, 16, -16],
            [8, -8, 4, -4],
            [2, -2, 1, -1],
        ],
        "kernel": [
            [2, -3, 4],
            [-5, 6, -7],
            [8, -9, 10],
        ],
        "conv_bias": 256,
        "fc0_weight": -8,
        "fc1_weight": 24,
        "fc0_bias": 4,
        "fc1_bias": -2,
    },
]


def relu(value: int) -> int:
    return max(0, value)


def saturate_int8(value: int) -> int:
    return max(-128, min(127, value))


def pixel(image: list[list[int]], row: int, col: int) -> int:
    if row < 0 or row >= 4 or col < 0 or col >= 4:
        return 0
    return image[row][col]


def conv_pixel(case: dict[str, object], row: int, col: int) -> int:
    image = case["image"]
    kernel = case["kernel"]
    total = int(case["conv_bias"])
    for kr in range(3):
        for kc in range(3):
            total += pixel(image, row + kr - 1, col + kc - 1) * kernel[kr][kc]
    return saturate_int8(relu(total // (1 << FRAC_BITS)))


def maxpool2x2(conv: list[list[int]], row: int, col: int) -> int:
    base_r = row * 2
    base_c = col * 2
    return max(conv[base_r][base_c], conv[base_r][base_c + 1], conv[base_r + 1][base_c], conv[base_r + 1][base_c + 1])


def golden_for(case: dict[str, object]) -> list[int]:
    conv = [[conv_pixel(case, row, col) for col in range(4)] for row in range(4)]
    pool = [[maxpool2x2(conv, row, col) for col in range(2)] for row in range(2)]
    gap_sum = sum(value for row in pool for value in row)
    gap_avg = gap_sum // 4
    logit0 = (gap_avg * int(case["fc0_weight"]) // (1 << FRAC_BITS)) + int(case["fc0_bias"])
    logit1 = (gap_avg * int(case["fc1_weight"]) // (1 << FRAC_BITS)) + int(case["fc1_bias"])
    class_id = int(logit1 > logit0)
    return [
        *[value for row in conv for value in row],
        *[value for row in pool for value in row],
        gap_sum,
        gap_avg,
        logit0,
        logit1,
        class_id,
    ]


def flatten_case(case: dict[str, object]) -> list[int]:
    image = [value for row in case["image"] for value in row]
    kernel = [value for row in case["kernel"] for value in row]
    return [
        *image,
        *kernel,
        int(case["conv_bias"]),
        int(case["fc0_weight"]),
        int(case["fc1_weight"]),
        int(case["fc0_bias"]),
        int(case["fc1_bias"]),
    ]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "cnn_tiny_4x4_cases.txt"
    golden_path = DATA_DIR / "cnn_tiny_4x4_golden.txt"

    with cases_path.open("w", encoding="utf-8") as cases_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for case in CASES:
            cases_file.write(" ".join(str(value) for value in flatten_case(case)) + "\n")
            golden_file.write(" ".join(str(value) for value in golden_for(case)) + "\n")

    print(f"Generated {len(CASES)} tiny CNN 4x4 test cases")
    print(f"cases:  {cases_path}")
    print(f"golden: {golden_path}")


if __name__ == "__main__":
    main()
