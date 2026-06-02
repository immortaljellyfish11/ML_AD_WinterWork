from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "sim" / "data"


def write_vector(path: Path, values: list[int]) -> None:
    path.write_text("\n".join(str(v) for v in values) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    activations = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    weights = [1, 0, -1, 1, 0, -1, 1, 0, -1]
    golden = sum(act * weight for act, weight in zip(activations, weights))

    write_vector(DATA_DIR / "pe_3x3_activations.txt", activations)
    write_vector(DATA_DIR / "pe_3x3_weights.txt", weights)
    (DATA_DIR / "pe_3x3_golden.txt").write_text(f"{golden}\n", encoding="utf-8")

    print("Generated PE 3x3 test vectors")
    print(f"activations: {activations}")
    print(f"weights:     {weights}")
    print(f"golden:      {golden}")


if __name__ == "__main__":
    main()
