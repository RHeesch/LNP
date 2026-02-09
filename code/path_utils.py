from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_exp_setup_dir() -> Path:
    repo_root = get_repo_root()
    candidates = [
        repo_root / "config",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def normalize_log_dir(hparam: dict) -> Path:
    repo_root = get_repo_root()
    raw = hparam.get("LOG_DIR")
    if not raw:
        log_dir = repo_root / "exp" / "log"
    else:
        log_dir = Path(raw).expanduser()
        if log_dir.is_absolute():
            try:
                rel = log_dir.relative_to(repo_root)
                log_dir = repo_root / rel
            except ValueError:
                # Keep it repo-local even if LOG_DIR was absolute elsewhere
                log_dir = repo_root / log_dir.name
        else:
            log_dir = repo_root / log_dir
    log_dir = log_dir.resolve()
    hparam["LOG_DIR"] = to_repo_relative(log_dir)
    return log_dir


def to_repo_relative(path: Path | str) -> str:
    repo_root = get_repo_root()
    p = Path(path).expanduser().resolve()
    try:
        return str(p.relative_to(repo_root))
    except ValueError:
        return str(p)


def get_log_dir(hparam: dict) -> Path:
    return normalize_log_dir(hparam)
