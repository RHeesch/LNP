from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data_gens.json_loader import load_config, validate_dataset_config


def main():
    data_dir = Path(__file__).resolve().parents[1] / "code" / "data_gens"
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No domain JSON files found in {data_dir}")
    for path in json_files:
        cfg = load_config(path)
        validate_dataset_config(cfg)
        print(f"OK: {path.name}")


if __name__ == "__main__":
    main()
