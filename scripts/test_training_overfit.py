import json
from pathlib import Path
from copy import deepcopy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data_module import DataModule
from path_utils import get_exp_setup_dir
from train_module import training


def main():
    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)

    # Minimal overfit settings
    hparam = deepcopy(hparam)
    hparam["EPOCHS"] = 200
    hparam["BATCH_SIZE"] = min(32, hparam.get("BATCH_SIZE", 32))
    hparam["QUIET"] = True

    data_path = DataModule.generate_and_store(hparam, force=True)
    trainer = training(hparam, data_path=data_path)
    results = trainer.train_nn()
    if "test_loss" not in results:
        raise AssertionError("training results missing test_loss")
    print(f"OK: training finished. test_loss={results['test_loss']}")


if __name__ == "__main__":
    main()
