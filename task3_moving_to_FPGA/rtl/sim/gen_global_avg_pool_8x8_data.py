from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sim" / "data"


CASES = [
    [0] * 64,
    [1] * 64,
    [-1] * 64,
    list(range(64)),
    [127] * 64,
    [-128] * 64,
    [127, -128] * 32,
    [(i % 17) - 8 for i in range(64)],
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "global_avg_pool_8x8_cases.txt"
    golden_path = DATA_DIR / "global_avg_pool_8x8_golden.txt"

    with cases_path.open("w", encoding="utf-8") as cases_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for case in CASES:
            total = sum(case)
            avg = total // 64
            cases_file.write(" ".join(str(value) for value in case) + "\n")
            golden_file.write(f"{total} {avg}\n")

    print(f"Generated {len(CASES)} GlobalAvgPool8x8 test cases")
    print(f"cases:  {cases_path}")
    print(f"golden: {golden_path}")


if __name__ == "__main__":
    main()
