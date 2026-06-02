from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sim" / "data"


CASES = [
    [1, 2, 3, 4],
    [-1, -2, -3, -4],
    [-128, 127, 0, 64],
    [5, 5, 5, 5],
    [100, -100, 99, -99],
    [-128, -127, -126, -125],
    [0, -1, 1, -2],
    [127, 126, 125, 124],
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cases_path = DATA_DIR / "maxpool_2x2_cases.txt"
    golden_path = DATA_DIR / "maxpool_2x2_golden.txt"

    with cases_path.open("w", encoding="utf-8") as cases_file, golden_path.open("w", encoding="utf-8") as golden_file:
        for case in CASES:
            cases_file.write(" ".join(str(value) for value in case) + "\n")
            golden_file.write(f"{max(case)}\n")

    print(f"Generated {len(CASES)} MaxPool2x2 test cases")
    print(f"cases:  {cases_path}")
    print(f"golden: {golden_path}")


if __name__ == "__main__":
    main()
