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
        "bias": 0,
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
        "bias": 256,
    },
    {
        "image": [
            [-10, -9, -8, -7],
            [-6, -5, -4, -3],
            [-2, -1, 0, 1],
            [2, 3, 4, 5],
        ],
        "kernel": [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ],
        "bias": -128,
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


def conv_one(case: dict[str, object], row: int, col: int) -> int:
    image = case["image"]
    kernel = case["kernel"]
    total = 0
    for kr in range(3):
        for kc in range(3):
            total += pixel(image, row + kr - 1, col + kc - 1) * kernel[kr][kc]
    total += int(case["bias"])
    quant = total // (1 << FRAC_BITS)
    return saturate_int8(relu(quant))


def golden_for(case: dict[str, object]) -> list[int]:
    return [conv_one(case, row, col) for row in range(4) for col in range(4)]


def flatten_case(case: dict[str, object]) -> list[int]:
    image = [value for row in case["image"] for value in row]
    kernel = [value for row in case["kernel"] for value in row]
    return [*image, *kernel, int(case["bias"])]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "single_channel_conv3x3_4x4_cases.txt"
    golden_path = DATA_DIR / "single_channel_conv3x3_4x4_golden.txt"

    with cases_path.open("w", encoding="utf-8") as cases_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for case in CASES:
            cases_file.write(" ".join(str(value) for value in flatten_case(case)) + "\n")
            golden_file.write(" ".join(str(value) for value in golden_for(case)) + "\n")

    print(f"Generated {len(CASES)} single-channel Conv3x3 4x4 test cases")
    print(f"frac_bits: {FRAC_BITS}")
    print(f"cases:     {cases_path}")
    print(f"golden:    {golden_path}")


if __name__ == "__main__":
    main()
