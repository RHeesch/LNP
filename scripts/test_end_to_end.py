import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from path_utils import get_exp_setup_dir
from LNP_UPF import main as lnp_main


def main():
    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)
    # This simply runs LNP_UPF main; use --check in CLI for full output.
    lnp_main()


if __name__ == "__main__":
    main()
