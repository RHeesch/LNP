import argparse
import json

from data_module import DataModule
from path_utils import get_exp_setup_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and cache dataset for reproducible training.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached dataset if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)
    path = DataModule.generate_and_store(hparam, force=args.force)
    delta = hparam.get("PREDICT_DELTA", 0) == 1
    print(f"Cached dataset at: {path}")
    print(f"PREDICT_DELTA={int(delta)} (training will use delta targets if 1)")


if __name__ == "__main__":
    main()
