import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from LNP_UPF import load_domain_json, step1_load_and_summarize, _build_problem
from path_utils import get_exp_setup_dir


def main():
    exp_setup_dir = get_exp_setup_dir()
    with open(exp_setup_dir / "hparams.json") as f:
        hparam = json.load(f)
    domain = hparam.get("DS_DOMAIN", "drone")

    cfg = step1_load_and_summarize(domain)
    problem = _build_problem(cfg, integrate_ml=True, bounds_cfg=None, blocked_transitions=[], listener_mode=hparam.get("Listener", "full"))

    # Basic sanity: has fluents and actions
    if len(problem.fluents) == 0:
        raise AssertionError("No fluents in problem")
    if len(problem.actions) == 0:
        raise AssertionError("No actions in problem")

    print(f"OK: problem built. fluents={len(problem.fluents)} actions={len(problem.actions)}")


if __name__ == "__main__":
    main()
