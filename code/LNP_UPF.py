import argparse
import json
from pathlib import Path

from unified_planning.shortcuts import (
    Fluent,
    Problem,
    BoolType,
    RealType,
    And,
    Or,
    GE,
    GT,
    LE,
    LT,
    Equals,
    Iff,
    Real,
    InstantaneousAction,
    OneshotPlanner,
    get_environment,
    Compiler,
    CompilationKind,
    PlanValidator,
    Not,
)
from unified_planning.plans import SequentialPlan, ActionInstance
from train_module import determine_parameter
from fractions import Fraction
from unified_planning.io import ANMLWriter

from data_gens.json_loader import domain_json_path
from path_utils import get_repo_root

CHECK_OUTPUT = False


def _vprint(msg: str):
    if CHECK_OUTPUT:
        print(msg)


def _statusprint(msg: str):
    print(msg)


def load_domain_json(ds_domain: str) -> dict:
    path = domain_json_path(ds_domain)
    if not path.exists():
        raise FileNotFoundError(f"Domain JSON not found at {path}")
    with open(path) as f:
        return json.load(f)


def validate_domain_json(cfg: dict) -> None:
    required = [
        "variables",
        "variable_types",
        "actions",
        "learned_actions",
        "fix_parameters",
        "variable_parameters",
        "subsymbolic_preconditions",
        "listener",
        "problem",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required keys in domain JSON: {missing}")


def step1_load_and_summarize(ds_domain: str) -> dict:
    cfg = load_domain_json(ds_domain)
    validate_domain_json(cfg)
    _vprint(f"Domain: {ds_domain}")
    _vprint(f"Variables: {len(cfg['variables'])}")
    _vprint(f"Actions: {len(cfg['actions'])}")
    _vprint(f"Learned actions: {len(cfg['learned_actions'])}")
    _vprint(f"Fix parameters: {len(cfg['fix_parameters']['names'])}")
    _vprint(f"Variable parameters: {len(cfg['variable_parameters']['names'])}")
    _vprint("Step 1 OK")
    return cfg


def step2_build_fluents(cfg: dict) -> None:
    problem = Problem(f"LNP_{cfg.get('name', 'domain')}")
    fluents = {}
    for v in cfg["variables"]:
        if cfg["variable_types"].get(v) == "boolean":
            fluents[v] = Fluent(v, BoolType())
        else:
            fluents[v] = Fluent(v, RealType())
        problem.add_fluent(fluents[v])
    if CHECK_OUTPUT:
        _vprint("Fluents:")
        for v in cfg["variables"]:
            _vprint(f"  {v}: {fluents[v].type}")
        _vprint("Step 2 OK")
    return problem, fluents


def _as_real(val):
    if isinstance(val, Fraction):
        return val
    return Fraction(str(val))


def _eval_expr(expr: str, fix_params: dict):
    return eval(expr, {"__builtins__": {}}, fix_params)


def _parse_comparison(cell, lhs, fix_params=None, var_params=None):
    if cell is None or str(cell).strip() == "":
        return None
    s = str(cell).strip()
    if s == ">=0.0":
        return None
    def parse_rhs(expr: str):
        if not isinstance(expr, str):
            return expr
        if fix_params:
            try:
                return _eval_expr(expr, fix_params)
            except Exception:
                pass
        if var_params and expr in var_params:
            return var_params[expr]
        return expr

    if s.startswith(">="):
        rhs = parse_rhs(s[2:])
        return GE(lhs, Real(_as_real(rhs)))
    if s.startswith("<="):
        rhs = parse_rhs(s[2:])
        return LE(lhs, Real(_as_real(rhs)))
    if s.startswith(">"):
        rhs = parse_rhs(s[1:])
        return GT(lhs, Real(_as_real(rhs)))
    if s.startswith("<"):
        rhs = parse_rhs(s[1:])
        return LT(lhs, Real(_as_real(rhs)))
    if s.startswith("=="):
        rhs = parse_rhs(s[2:])
        if isinstance(rhs, bool):
            return Iff(lhs, rhs)
        if isinstance(rhs, str) and rhs.lower() in {"true", "false"}:
            return Iff(lhs, rhs.lower() == "true")
        return Equals(lhs, Real(_as_real(rhs)))
    if s.startswith("="):
        rhs = parse_rhs(s[1:])
        if isinstance(rhs, bool):
            return Iff(lhs, rhs)
        if isinstance(rhs, str) and rhs.lower() in {"true", "false"}:
            return Iff(lhs, rhs.lower() == "true")
        return Equals(lhs, Real(_as_real(rhs)))
    if s.lower() in {"true", "false"}:
        return Iff(lhs, s.lower() == "true")
    rhs = parse_rhs(s)
    return Equals(lhs, Real(_as_real(rhs)))


def _parse_numeric_expr(expr: str, fix_params: dict, fluents: dict, var_params: dict | None = None):
    try:
        return Real(_as_real(_eval_expr(expr, fix_params)))
    except Exception:
        pass
    env = {k: Real(_as_real(v)) for k, v in fix_params.items()}
    env.update(fluents)
    if var_params:
        env.update(var_params)
    return eval(expr, {"__builtins__": {}}, env)


def step3_init_and_goals(cfg: dict, problem: Problem, fluents: dict) -> None:
    fix_params = dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"]))
    # Initial state
    for v, init_val in zip(cfg["variables"], cfg["problem"]["init"]):
        if cfg["variable_types"].get(v) == "boolean":
            problem.set_initial_value(fluents[v], bool(init_val))
        else:
            problem.set_initial_value(fluents[v], float(init_val))

    # Goals
    for v, goal_val in zip(cfg["variables"], cfg["problem"]["goal"]):
        cond = _parse_comparison(goal_val, fluents[v], fix_params)
        if cond is not None:
            problem.add_goal(cond)
    _vprint("Step 3 OK")


def step4_symbolic_actions(cfg: dict, problem: Problem, fluents: dict) -> None:
    fix_params = dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"]))
    var_param_names = cfg.get("variable_parameters", {}).get("names", [])
    var_param_down = cfg.get("variable_parameters", {}).get("down", [])
    var_param_top = cfg.get("variable_parameters", {}).get("top", [])
    for act in cfg["actions"]:
        used_var_params = []
        for cell in act["preconditions"] + act["effects"]:
            if cell is None:
                continue
            s = str(cell)
            for name in var_param_names:
                if name in s and name not in used_var_params:
                    used_var_params.append(name)
        param_types = {name: RealType() for name in used_var_params}
        action = InstantaneousAction(act["name"], **param_types)
        var_params = {name: action.parameter(name) for name in used_var_params}
        # Bounds for symbolic variable parameters
        for name in used_var_params:
            if name in var_param_names:
                i = var_param_names.index(name)
                if i < len(var_param_down):
                    action.add_precondition(GE(var_params[name], Real(_as_real(var_param_down[i]))))
                if i < len(var_param_top):
                    action.add_precondition(LE(var_params[name], Real(_as_real(var_param_top[i]))))
        boolean_effects = act.get("boolean_effects", {})
        # Preconditions
        for v, cell in zip(cfg["variables"], act["preconditions"]):
            cond = _parse_comparison(cell, fluents[v], fix_params, var_params)
            if cond is not None:
                action.add_precondition(cond)
        # Effects (numeric only)
        for v, eff in zip(cfg["variables"], act["effects"]):
            if eff is None:
                continue
            if cfg["variable_types"].get(v) == "boolean":
                if v in boolean_effects:
                    if boolean_effects[v] is None:
                        continue
                    action.add_effect(fluents[v], bool(boolean_effects[v]))
                else:
                    if isinstance(eff, bool) and eff:
                        action.add_effect(fluents[v], True)
                continue
            s = str(eff).strip()
            if s == "" or s == "0.0":
                continue
            try:
                expr = _parse_numeric_expr(s, fix_params, fluents, var_params)
                action.add_increase_effect(fluents[v], expr)
            except Exception:
                continue
        problem.add_action(action)
    _vprint("Step 4 OK")


def _strict_anml_check(cfg: dict, anml_text: str, integrate_ml: bool) -> None:
    import re

    actions = {}
    for m in re.finditer(r"action\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)\s*\{", anml_text):
        name = m.group(1)
        sig = m.group(2)
        start = m.end()
        end = anml_text.find("};", start)
        if end == -1:
            continue
        body = anml_text[start:end]
        actions[name] = (sig, body)
    missing_actions = [a["name"] for a in cfg["actions"] if a["name"] not in actions]
    if integrate_ml:
        missing_actions += [a for a in cfg["learned_actions"] if a not in actions]
    if missing_actions:
        raise ValueError(f"ANML strict check: missing actions {missing_actions}")

    for act in cfg["actions"]:
        name = act["name"]
        sig, body = actions[name]
        params = [p.strip() for p in sig.split(",") if p.strip()]
        # expected variable parameters used
        var_param_names = cfg.get("variable_parameters", {}).get("names", [])
        used_var_params = []
        for cell in act["preconditions"] + act["effects"]:
            if cell is None:
                continue
            s = str(cell)
            for n in var_param_names:
                if n in s and n not in used_var_params:
                    used_var_params.append(n)
        if len(params) != len(used_var_params):
            raise ValueError(
                f"ANML strict check: action {name} params mismatch. "
                f"expected {used_var_params}, got {params}"
            )

        # expected preconditions/effects count
        expected_pre = 0
        for cell in act["preconditions"]:
            if cell is None:
                continue
            s = str(cell).strip()
            if s == "" or s == ">=0.0":
                continue
            expected_pre += 1
        # variable parameter bounds add preconditions
        var_param_down = cfg.get("variable_parameters", {}).get("down", [])
        var_param_top = cfg.get("variable_parameters", {}).get("top", [])
        for n in used_var_params:
            if n in var_param_names:
                i = var_param_names.index(n)
                if i < len(var_param_down):
                    expected_pre += 1
                if i < len(var_param_top):
                    expected_pre += 1
        expected_eff = 0
        for v, eff in zip(cfg["variables"], act["effects"]):
            if eff is None:
                continue
            if cfg["variable_types"].get(v) == "boolean":
                if isinstance(eff, bool) and eff:
                    expected_eff += 1
                if isinstance(act.get("boolean_effects", {}).get(v, None), bool):
                    expected_eff += 1
            else:
                s = str(eff).strip()
                if s == "" or s == "0.0":
                    continue
                expected_eff += 1

        actual_pre = len([ln for ln in body.splitlines() if ":increase" not in ln and ":=" not in ln and "[ start ] (" in ln])
        actual_eff = len([ln for ln in body.splitlines() if ":increase" in ln or ":=" in ln])
        if actual_pre != expected_pre:
            raise ValueError(
                f"ANML strict check: action {name} preconditions count mismatch. "
                f"expected {expected_pre}, got {actual_pre}"
            )
        if actual_eff != expected_eff:
            raise ValueError(
                f"ANML strict check: action {name} effects count mismatch. "
                f"expected {expected_eff}, got {actual_eff}"
            )


def _state_from_init(cfg: dict) -> dict:
    return {v: cfg["problem"]["init"][i] for i, v in enumerate(cfg["variables"])}


def _goal_satisfied(cfg: dict, state: dict) -> bool:
    for v, goal_val in zip(cfg["variables"], cfg["problem"]["goal"]):
        cond = _parse_comparison(goal_val, state[v], dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"])))
        if cond is None:
            continue
        # For numeric goal conditions, use string parsing on the state value
        if isinstance(cond, bool):
            if state[v] != cond:
                return False
        else:
            # evaluate manually
            s = str(goal_val).strip()
            if s.startswith(">="):
                if not (state[v] >= float(s[2:])):
                    return False
            elif s.startswith("<="):
                if not (state[v] <= float(s[2:])):
                    return False
            elif s.startswith(">"):
                if not (state[v] > float(s[1:])):
                    return False
            elif s.startswith("<"):
                if not (state[v] < float(s[1:])):
                    return False
            elif s.startswith("==") or s.startswith("="):
                rhs = s.lstrip("=").strip()
                if rhs.lower() in {"true", "false"}:
                    if bool(state[v]) != (rhs.lower() == "true"):
                        return False
                else:
                    if not (state[v] == float(rhs)):
                        return False
            else:
                if s.lower() in {"true", "false"}:
                    if bool(state[v]) != (s.lower() == "true"):
                        return False
                else:
                    if not (state[v] == float(s)):
                        return False
    return True


def _apply_symbolic_effects(cfg: dict, state: dict, action_name: str, params: dict) -> dict:
    new_state = dict(state)
    for act in cfg["actions"]:
        if act["name"] != action_name:
            continue
        for v, eff in zip(cfg["variables"], act["effects"]):
            if eff is None:
                continue
            if cfg["variable_types"].get(v) == "boolean":
                if isinstance(eff, bool):
                    if eff:
                        new_state[v] = True
                elif v in act.get("boolean_effects", {}):
                    val = act["boolean_effects"][v]
                    if val is not None:
                        new_state[v] = bool(val)
                continue
            s = str(eff).strip()
            if s == "" or s == "0.0":
                continue
            try:
                expr = _eval_expr(s, {**dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"])), **params})
            except Exception:
                expr = float(s)
            new_state[v] = float(new_state[v]) + float(expr)
        break
    return new_state


def _apply_learned_effects(cfg: dict, state: dict, action_name: str, params: dict) -> dict:
    new_state = dict(state)
    labels = cfg.get("listener", {}).get("labels", [])
    if action_name not in cfg["learned_actions"]:
        return new_state
    idx = cfg["learned_actions"].index(action_name)
    label_row = labels[idx]
    for v, label in zip(cfg["variables"], label_row):
        if _is_nonzero_label(label) and cfg["variable_types"].get(v) != "boolean":
            delta = params.get(f"delta_{v}", 0.0)
            new_state[v] = float(new_state[v]) + float(delta)
    return new_state


def _action_applicable(cfg: dict, state: dict, action_name: str, params: dict) -> bool:
    fix_params = dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"]))
    def _cmp(val, s):
        s = s.strip()
        for p, pv in fix_params.items():
            s = s.replace(p, str(pv))
        try:
            if s.startswith(">="):
                s = ">=" + str(_eval_expr(s[2:], fix_params))
            elif s.startswith("<="):
                s = "<=" + str(_eval_expr(s[2:], fix_params))
            elif s.startswith("=="):
                s = "==" + str(_eval_expr(s[2:], fix_params))
            elif s.startswith(">"):
                s = ">" + str(_eval_expr(s[1:], fix_params))
            elif s.startswith("<"):
                s = "<" + str(_eval_expr(s[1:], fix_params))
            elif s.startswith("="):
                s = "=" + str(_eval_expr(s[1:], fix_params))
            else:
                s = str(_eval_expr(s, fix_params))
        except Exception:
            pass
        if s.startswith(">="):
            return val >= float(s[2:])
        if s.startswith("<="):
            return val <= float(s[2:])
        if s.startswith("=="):
            rhs = s[2:].strip()
            if rhs.lower() in {"true", "false"}:
                return bool(val) == (rhs.lower() == "true")
            return float(val) == float(rhs)
        if s.startswith(">"):
            return val > float(s[1:])
        if s.startswith("<"):
            return val < float(s[1:])
        if s.startswith("="):
            rhs = s[1:].strip()
            if rhs.lower() in {"true", "false"}:
                return bool(val) == (rhs.lower() == "true")
            return float(val) == float(rhs)
        if s.lower() in {"true", "false"}:
            return bool(val) == (s.lower() == "true")
        return float(val) == float(s)
    # find action preconditions
    for act in cfg["actions"]:
        if act["name"] != action_name:
            continue
        for v, cell in zip(cfg["variables"], act["preconditions"]):
            if cell is None:
                continue
            s = str(cell).strip()
            if s == "" or s == ">=0.0":
                continue
            val = state[v]
            # substitute params if needed
            for p, pv in params.items():
                s = s.replace(p, str(pv))
            if not _cmp(val, s):
                return False
            if s.lower() in {"true", "false"}:
                if bool(val) != (s.lower() == "true"):
                    return False
        return True
    # learned action preconditions
    if action_name in cfg["learned_actions"]:
        idx = cfg["learned_actions"].index(action_name)
        for v, cell in zip(cfg["variables"], cfg["subsymbolic_preconditions"][idx]):
            if cell is None:
                continue
            s = str(cell).strip()
            if s == "" or s == ">=0.0":
                continue
            val = state[v]
            if not _cmp(val, s):
                return False
        return True
    return False


def _placeholder_concretize(s_pre: dict, s_abs_next: dict) -> tuple[list, dict]:
    # Placeholder: return empty params and the abstract successor as concrete successor
    return [], dict(s_abs_next)


def _learned_param_names(cfg: dict, action_name: str) -> list[str]:
    if action_name not in cfg["learned_actions"]:
        return []
    idx = cfg["learned_actions"].index(action_name)
    labels = cfg.get("listener", {}).get("labels", [])
    label_row = labels[idx]
    names = []
    for v, label in zip(cfg["variables"], label_row):
        if _is_nonzero_label(label) and cfg["variable_types"].get(v) != "boolean":
            names.append(f"delta_{v}")
    return names


def _build_ml_vectors(cfg: dict, hparam: dict, state: dict, abs_next: dict, action_name: str):
    dataset = cfg.get("dataset", {})
    if dataset.get("name") != action_name:
        raise ValueError(f"No dataset layout for action {action_name}.")
    in_layout = dataset["input_layout"]
    out_layout = dataset["output_layout"]
    x = []
    bool_vars = {v for v, t in cfg.get("variable_types", {}).items() if t == "boolean"}
    bool_encoding = hparam.get("BOOL_ENCODING", "01")
    def enc(val):
        if bool_encoding == "010":
            return 10.0 if bool(val) else 0.0
        return 1.0 if bool(val) else 0.0
    for v in in_layout["state"]:
        if v in bool_vars:
            x.append(enc(state[v]))
        else:
            x.append(float(state[v]))
    for p in in_layout.get("fix_parameters", []):
        fix_vals = dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"]))
        x.append(float(fix_vals[p]))
    for _ in in_layout.get("variable_parameters", []):
        x.append(0.0)
    y = []
    for v in out_layout["state"]:
        if v in bool_vars:
            y.append(enc(abs_next[v]))
        else:
            y.append(float(abs_next[v]))
    return x, y


def _concretize_learned(cfg: dict, hparam: dict, state: dict, abs_next: dict, action_name: str):
    x, y = _build_ml_vectors(cfg, hparam, state, abs_next, action_name)
    params_list, _loss = determine_parameter(hparam, x, y, action_name)
    param_names = _learned_param_names(cfg, action_name)
    if len(params_list) != len(param_names):
        # fall back to zeros if mismatch
        params_list = [0.0 for _ in param_names]
    params = {name: float(val) for name, val in zip(param_names, params_list)}
    conc_next = _apply_learned_effects(cfg, state, action_name, params)
    return params, conc_next


def _log_line(msg: str, log_path: Path | None):
    _vprint(msg)
    if log_path is None or not CHECK_OUTPUT:
        return
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def _state_distance(cfg: dict, s1: dict, s2: dict) -> float:
    dist = 0.0
    for v in cfg["variables"]:
        if cfg["variable_types"].get(v) == "boolean":
            dist = max(dist, 0.0 if bool(s1[v]) == bool(s2[v]) else 1.0)
        else:
            dist = max(dist, abs(float(s1[v]) - float(s2[v])))
    return dist


def _apply_boolean_thresholds(cfg: dict, hparam: dict, state: dict) -> dict:
    bool_vars = {v for v, t in cfg.get("variable_types", {}).items() if t == "boolean"}
    bool_encoding = hparam.get("BOOL_ENCODING", "01")
    threshold = hparam.get("BOOL_THRESHOLD", 0.5 if bool_encoding == "01" else 5.0)
    new_state = dict(state)
    for v in bool_vars:
        val = new_state[v]
        if isinstance(val, bool):
            continue
        new_state[v] = float(val) >= float(threshold)
    return new_state


def step7_concretize_and_validate(
    cfg: dict, problem: Problem, plan: SequentialPlan, hparam: dict, log_path: Path | None
) -> tuple[bool, list | None]:
    if plan is None:
        _vprint("Step 7 skipped: no plan to concretize.")
        return False, None
    state = _state_from_init(cfg)
    concrete_actions = []
    param_log = []
    threshold = hparam.get("opti_threshold")
    for ai in plan.actions:
        action_name = ai.action.name
        params = {}
        # gather params early
        if ai.actual_parameters:
            param_names = [p.name for p in ai.action.parameters]
            for name, val in zip(param_names, ai.actual_parameters):
                params[name] = float(val.constant_value())
        if not _action_applicable(cfg, state, action_name, params):
            _statusprint(f"Concretization failed: action {action_name} not applicable before concretization.")
            _log_line(f"Step 7 FAILED: action {action_name} not applicable before concretization.", log_path)
            return False, None
        # build abstract successor by applying effects with current params
        if action_name in cfg["learned_actions"]:
            abs_next = _apply_learned_effects(cfg, state, action_name, params)
            _log_line(f"[Concretize] action={action_name} pre_state={state}", log_path)
            _log_line(f"[Concretize] action={action_name} abs_next={abs_next}", log_path)
            params, conc_next = _concretize_learned(cfg, hparam, state, abs_next, action_name)
            conc_next = _apply_boolean_thresholds(cfg, hparam, conc_next)
            _log_line(f"[Concretize] action={action_name} params={params}", log_path)
            _log_line(f"[Concretize] action={action_name} conc_next={conc_next}", log_path)
            if threshold is not None:
                dist = _state_distance(cfg, abs_next, conc_next)
                _log_line(f"[Concretize] action={action_name} dist={dist} threshold={threshold}", log_path)
                if dist > float(threshold):
                    _statusprint("Concretization failed: distance threshold exceeded.")
                    _log_line(
                        f"Step 7 FAILED: concretization distance {dist} exceeds threshold {threshold}.",
                        log_path,
                    )
                    blocked = [{
                        "action": action_name,
                        "s_pre": dict(state),
                        "s_abs_next": dict(abs_next),
                        "radius": float(hparam.get("BLOCK_TRANSITION_RADIUS", 0.0)),
                    }]
                    return False, blocked
            state = conc_next
            param_log.append({"action": action_name, "params": params})
        else:
            # handle symbolic with possible parameters (e.g., recharge energy)
            abs_next = _apply_symbolic_effects(cfg, state, action_name, params)
            conc_next = abs_next
            state = conc_next
            if params:
                param_log.append({"action": action_name, "params": params})
        concrete_actions.append(ActionInstance(ai.action, ai.actual_parameters))

    _log_line(f"[Concretize] final_state={state}", log_path)
    if not _goal_satisfied(cfg, state):
        _statusprint("Concretization failed: goal not satisfied.")
        _log_line("Step 7 FAILED: goal not satisfied after concretization.", log_path)
        return False, None
    validator = PlanValidator(name="sequential_plan_validator")
    result = validator.validate(problem, SequentialPlan(concrete_actions))
    _log_line(f"Step 7 OK: validator status={result.status}", log_path)
    _statusprint("Concrete plan validated. Printing plan and parameter sets:")
    _statusprint(str(SequentialPlan(concrete_actions)))
    if param_log:
        _statusprint(f"Parameters: {param_log}")
    return True, None


def _load_listener_bounds(ds_domain: str, bounds_path: str | None):
    if bounds_path:
        path = Path(bounds_path)
        if not path.exists():
            raise FileNotFoundError(f"Listener bounds file not found at {path}")
        with open(path) as f:
            return json.load(f)

    log_dir = get_repo_root() / "exp" / "log" / ds_domain
    if not log_dir.exists():
        return None
    matches = sorted(log_dir.glob("listener_bounds_*.json"))
    if not matches:
        return None
    if len(matches) > 1:
        _vprint(f"Multiple listener bounds found; using {matches[0].name}")
    with open(matches[0]) as f:
        return json.load(f)


def _is_nonzero_label(label):
    if label is None:
        return False
    if isinstance(label, bool):
        return label
    try:
        return float(label) != 0.0
    except Exception:
        return True


def step5_learned_actions(
    cfg: dict,
    problem: Problem,
    fluents: dict,
    bounds_cfg: dict | None,
    blocked_transitions: list | None,
    listener_mode: str,
) -> None:
    fix_params = dict(zip(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"]))
    labels = cfg.get("listener", {}).get("labels", [])
    if not labels:
        _vprint("Step 5 OK (no listener labels)")
        return
    for idx, action_name in enumerate(cfg["learned_actions"]):
        label_row = labels[idx]
        param_types = {}
        for v, label in zip(cfg["variables"], label_row):
            if _is_nonzero_label(label) and cfg["variable_types"].get(v) != "boolean":
                param_types[f"delta_{v}"] = RealType()
        action = InstantaneousAction(action_name, **param_types)

        # Preconditions (from subsymbolic_preconditions)
        sub_pre = cfg["subsymbolic_preconditions"][idx]
        for v, cell in zip(cfg["variables"], sub_pre):
            cond = _parse_comparison(cell, fluents[v], fix_params)
            if cond is not None:
                action.add_precondition(cond)

        # Effects for labeled variables
        for v, label in zip(cfg["variables"], label_row):
            if not _is_nonzero_label(label):
                continue
            if cfg["variable_types"].get(v) == "boolean":
                continue
            param = action.parameter(f"delta_{v}")
            action.add_increase_effect(fluents[v], param)
            if listener_mode == "full" and bounds_cfg and bounds_cfg.get("ml_action") == action_name:
                lower = bounds_cfg.get("lower", [])
                upper = bounds_cfg.get("upper", [])
                if len(lower) == len(cfg["variables"]) and len(upper) == len(cfg["variables"]):
                    i = cfg["variables"].index(v)
                    action.add_precondition(GE(param, Real(_as_real(lower[i]))))
                    action.add_precondition(LE(param, Real(_as_real(upper[i]))))
        if blocked_transitions:
            for bt in blocked_transitions:
                if bt["action"] != action_name:
                    continue
                s_pre = bt["s_pre"]
                s_abs_next = bt["s_abs_next"]
                radius = float(bt.get("radius", 0.0))
                within_pre = []
                outside_params = []
                for v in cfg["variables"]:
                    if cfg["variable_types"].get(v) == "boolean":
                        within_pre.append(fluents[v] if bool(s_pre[v]) else Not(fluents[v]))
                    else:
                        within_pre.append(GE(fluents[v], Real(_as_real(float(s_pre[v]) - radius))))
                        within_pre.append(LE(fluents[v], Real(_as_real(float(s_pre[v]) + radius))))
                for v, label in zip(cfg["variables"], label_row):
                    if not _is_nonzero_label(label):
                        continue
                    if cfg["variable_types"].get(v) == "boolean":
                        continue
                    param = action.parameter(f"delta_{v}")
                    delta_target = float(s_abs_next[v]) - float(s_pre[v])
                    if radius == 0.0:
                        outside_params.append(Not(Equals(param, Real(_as_real(delta_target)))))
                    else:
                        outside_params.append(LT(param, Real(_as_real(delta_target - radius))))
                        outside_params.append(GT(param, Real(_as_real(delta_target + radius))))
                if outside_params:
                    action.add_precondition(Or(Not(And(*within_pre)), *outside_params))
        problem.add_action(action)
    _vprint("Step 5 OK")


def _register_tempest_engines():
    env = get_environment()
    if "tempest" not in env.factory.engines:
        env.factory.add_engine("tempest", "tempest.engine", "TempestEngine")
    if "tempest-opt" not in env.factory.engines:
        env.factory.add_engine("tempest-opt", "tempest.engine", "TempestOptimal")


def _patch_mpq_autopromote():
    try:
        import gmpy2  # type: ignore
    except Exception:
        return
    mpq_type = gmpy2.mpq
    em = get_environment().expression_manager
    orig = em.auto_promote

    def auto_promote_mpq(*args):
        converted = []
        for e in em._polymorph_args_to_iterator(*args):
            if isinstance(e, mpq_type):
                e = Fraction(int(e.numerator), int(e.denominator))
            converted.append(e)
        return orig(converted)

    em.auto_promote = auto_promote_mpq


def step6_solve(problem: Problem, engine: str, params: dict | None) -> None:
    if engine.startswith("tempest"):
        _register_tempest_engines()
        _patch_mpq_autopromote()
    if engine in {"tamer"}:
        try:
            compiler = Compiler(name="up_grounder")
            problem = compiler.compile(problem).problem
        except Exception as exc:
            try:
                compiler = Compiler(compilation_kind=CompilationKind.GROUNDING)
                problem = compiler.compile(problem).problem
            except Exception as exc2:
                _vprint(f"Step 6 WARNING: grounding failed ({exc2}); trying to solve ungrounded problem.")
    try:
        with OneshotPlanner(name=engine, params=params or {}) as planner:
            result = planner.solve(problem)
    except Exception as exc:
        env = get_environment()
        engines = env.factory.engines
        if isinstance(engines, dict):
            engines = sorted(engines.keys())
        else:
            engines = sorted(engines)
        print(f"Step 6 FAILED: could not solve with engine '{engine}'.")
        print(f"Available engines: {engines}")
        print(f"Error: {exc}")
        return None
    _vprint(f"Step 6 OK (engine={engine}, status={result.status})")
    if result.plan is not None:
        _statusprint("Abstract plan found.")
        _vprint(str(result.plan))
    return result.plan


def parse_args():
    parser = argparse.ArgumentParser(description="LNP UPF builder (stepwise validation).")
    parser.add_argument("--domain", default="drone", help="Domain name (json filename without extension).")
    parser.add_argument("--listener-bounds", default=None, help="Optional path to listener bounds json.")
    parser.add_argument("--engine", default="tempest", help="UPF engine name (default: tempest).")
    parser.add_argument("--solver", default="z3", help="SMT solver name for Tempest (default: z3).")
    parser.add_argument("--horizon", type=int, default=None, help="Max plan length for Tempest.")
    parser.add_argument("--incremental", action="store_true", help="Enable Tempest incremental solving.")
    parser.add_argument(
        "--hparams",
        default="config/hparams.json",
        help="Path to hparams.json used to bound horizon (Max_Step).",
    )
    parser.add_argument("--ground-abstract-step", action="store_true", help="Tempest-opt: ground abstract step.")
    parser.add_argument("--grounder-name", default=None, help="Tempest-opt: UPF grounder name.")
    parser.add_argument("--sat-before-opt", action="store_true", help="Tempest-opt: sat-before-opt.")
    parser.add_argument(
        "--secondary-objective",
        default=None,
        help="Tempest-opt: secondary objective (weighted|lexicographic).",
    )
    parser.add_argument("--max-loops", type=int, default=None, help="Max replanning loops before giving up.")
    parser.add_argument("--check", action="store_true", help="Enable detailed checks/logging output.")
    return parser.parse_args()


def _build_problem(cfg, integrate_ml, bounds_cfg, blocked_transitions, listener_mode):
    problem, fluents = step2_build_fluents(cfg)
    step3_init_and_goals(cfg, problem, fluents)
    step4_symbolic_actions(cfg, problem, fluents)
    if integrate_ml:
        step5_learned_actions(cfg, problem, fluents, bounds_cfg, blocked_transitions, listener_mode)
    else:
        _vprint("Step 5 skipped (Int_ML=0)")
    return problem


def main():
    args = parse_args()
    global CHECK_OUTPUT
    CHECK_OUTPUT = bool(args.check)
    if not CHECK_OUTPUT:
        get_environment().credits_stream = None
    cfg = step1_load_and_summarize(args.domain)
    hparams = {}
    try:
        with open(args.hparams) as f:
            hparams = json.load(f)
    except Exception:
        hparams = {}
    integrate_ml = bool(hparams.get("Int_ML", 1))
    if not CHECK_OUTPUT:
        hparams["QUIET"] = True
    listener_mode = hparams.get("Listener", "full").lower()
    if listener_mode not in {"none", "partial", "full"}:
        raise ValueError(f"Unknown Listener mode: {listener_mode}")
    bounds_cfg = _load_listener_bounds(args.domain, args.listener_bounds)
    max_loops = args.max_loops if args.max_loops is not None else int(hparams.get("MAX_LOOPS", 5))

    params = None
    if args.engine.startswith("tempest"):
        max_step = hparams.get("Max_Step")
        params = {
            "incremental": bool(args.incremental),
            "horizon": args.horizon if args.horizon is not None else max_step,
            "solver_name": args.solver,
        }
        if args.engine.endswith("-opt"):
            params.update(
                {
                    "ground_abstract_step": bool(args.ground_abstract_step),
                    "grounder_name": args.grounder_name,
                    "sat_before_opt": bool(args.sat_before_opt),
                    "secondary_objective": args.secondary_objective,
                }
            )

    log_path = None
    if CHECK_OUTPUT:
        log_path = get_repo_root() / "exp" / "log" / f"{args.domain}_lnp_upf.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
    blocked_transitions = []
    for attempt in range(1, max_loops + 1):
        _log_line(f"Loop {attempt}/{max_loops} with blocked transitions: {blocked_transitions}", log_path)
        if listener_mode != "none" and bounds_cfg is None:
            raise FileNotFoundError(
                f"Listener mode '{listener_mode}' requires listener bounds, but none were found."
            )
        problem = _build_problem(cfg, integrate_ml, bounds_cfg, blocked_transitions, listener_mode)
        if CHECK_OUTPUT:
            anml_path = get_repo_root() / "exp" / "log" / f"{args.domain}_loop{attempt}.anml"
            anml_path.parent.mkdir(parents=True, exist_ok=True)
            if anml_path.exists():
                anml_path.unlink()
            writer = ANMLWriter(problem)
            anml_path.write_text(writer.get_problem(), encoding="utf-8")
            with open(anml_path, "r", encoding="utf-8") as f:
                anml_text = f.read()
            missing_vars = [v for v in cfg["variables"] if v not in anml_text]
            missing_actions = [a["name"] for a in cfg["actions"] if a["name"] not in anml_text]
            if integrate_ml:
                missing_actions += [a for a in cfg["learned_actions"] if a not in anml_text]
            if missing_vars or missing_actions:
                raise ValueError(
                    f"ANML export mismatch. Missing vars: {missing_vars}; missing actions: {missing_actions}"
                )
            _strict_anml_check(cfg, anml_text, integrate_ml)
            _vprint(f"ANML exported to {anml_path}")
        plan = step6_solve(problem, args.engine, params)
        if plan is None:
            break
        ok, blocked = step7_concretize_and_validate(cfg, problem, plan, hparams, log_path)
        if ok:
            break
        if blocked:
            blocked_transitions.extend(blocked)
            _log_line(f"Blocking lemma added: {blocked}", log_path)
        else:
            break


if __name__ == "__main__":
    main()
