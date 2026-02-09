from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data_gens.json_loader import generate_dataset_from_json, load_config


def main(n_samples: int = 32):
    data_dir = Path(__file__).resolve().parents[1] / "code" / "data_gens"
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No domain JSON files found in {data_dir}")

    for path in json_files:
        cfg = load_config(path)
        x, y = generate_dataset_from_json(path, n_samples=n_samples)
        if len(x) != n_samples or len(y) != n_samples:
            raise AssertionError(f"{path.name}: expected {n_samples} samples, got {len(x)}")
        print(f"OK: {path.name} generated {len(x)} samples")


if __name__ == "__main__":
    main()
