import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from data_module import DataModule
from path_utils import get_exp_setup_dir


def main():
    exp_setup_dir = get_exp_setup_dir()
    hparam_path = exp_setup_dir / "hparams.json"
    with open(hparam_path) as f:
        hparam = json.load(f)

    data_path = DataModule.generate_and_store(hparam, force=True)
    x, y = DataModule(hparam, data_path=data_path).load_cached_data()
    if x.shape[0] != y.shape[0]:
        raise AssertionError("x and y sample sizes differ")
    if x.shape[0] == 0:
        raise AssertionError("empty cached dataset")
    print(f"OK: cached data loaded. x={x.shape} y={y.shape}")


if __name__ == "__main__":
    main()
