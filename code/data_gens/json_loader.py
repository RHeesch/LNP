import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


class input:
    def __init__(self, domain, problem):
        self.domain = domain
        self.problem = problem


class domain:
    def __init__(self, variables, symbolic_actions, learned_actions, fix, var, sym_precon, sym_effect, ml_precon, ml_effects):
        self.variables = variables
        self.symbolic_actions = symbolic_actions
        self.learned_actions = learned_actions
        self.fix_parameters = fix
        self.variable_parameters = var
        self.symbolic_preconditions = sym_precon
        self.symbolic_effects = sym_effect
        self.subsymbolic_preconditions = ml_precon
        self.subsymbolic_effetcs = ml_effects


class df_fix_parameters:
    def __init__(self, parameters, values):
        self.columns = ["limits"]
        self.parameters = parameters
        self.values = pd.Series(values, index=parameters)
        self.df = pd.concat([self.values], axis=1)
        self.df.columns = self.columns


class df_variable_parameters:
    def __init__(self, parameters, upper_bounds, under_bounds):
        self.columns = ["Top_var", "Down_var"]
        self.parameters = parameters
        self.upper_bounds = pd.Series(upper_bounds, index=parameters)
        self.under_bounds = pd.Series(under_bounds, index=parameters)
        self.df = pd.concat([self.upper_bounds, self.under_bounds], axis=1)
        self.df.columns = self.columns


class df_subsymbolic_preconditions:
    def __init__(self, learned_actions, variables, preconditions):
        self.learned_actions = learned_actions
        self.variables = variables
        self.df = pd.concat(preconditions, axis=1)
        self.df.columns = self.learned_actions


class df_symbolic_effects:
    def __init__(self, variables, actions, effects):
        self.variables = variables
        self.actions = actions
        self.df = pd.concat(effects, axis=1)
        self.df.columns = self.actions


class df_subsymbolic_label:
    def __init__(self, variables, ml_actions, label):
        self.variables = variables
        self.columns = ml_actions
        self.df = pd.concat(label, axis=1)
        self.df.columns = self.columns


class df_subsymbolic_under_bound:
    def __init__(self, variables, ml_actions, under_bound):
        self.variables = variables
        self.columns = ml_actions
        self.df = pd.concat(under_bound, axis=1)
        self.df.columns = self.columns


class variable_parameters:
    def __init__(self, ml_actions, variable_parameters, assignment):
        self.variable_parameters = variable_parameters
        self.columns = ml_actions
        self.df = pd.concat(assignment, axis=1)
        self.df.columns = self.columns


class df_subsymbolic_upper_bound:
    def __init__(self, variables, ml_actions, upper_bound):
        self.variables = variables
        self.columns = ml_actions
        self.df = pd.concat(upper_bound, axis=1)
        self.df.columns = self.columns


class subsymbolic_effetcs:
    def __init__(self, labels, under_bounds, upper_bounds, variable_parameters):
        self.labels = labels
        self.under_bounds = under_bounds
        self.upper_bounds = upper_bounds
        self.variable_parameters = variable_parameters


class df_problem:
    def __init__(self, variables, init_values, goal_values):
        self.columns = ["Init", "Goal"]
        self.variables = variables
        self.init_values = pd.Series(init_values, index=variables)
        self.goal_values = pd.Series(goal_values, index=variables)
        self.df = pd.concat([self.init_values, self.goal_values], axis=1)
        self.df.columns = self.columns


class JsonDataSet:
    def __init__(self, setlength, dataset_cfg):
        self.setlength = setlength
        self.dataset_cfg = dataset_cfg
        self.state_vars = dataset_cfg["input_layout"]["state"]
        self.fix_vars = dataset_cfg["input_layout"]["fix_parameters"]
        self.var_vars = dataset_cfg["input_layout"]["variable_parameters"]

        self.features = len(self.state_vars)
        self.number_fix_Parameters = len(self.fix_vars)
        self.number_variable_Parameters = len(self.var_vars)

    def _sample_from(self, spec, values):
        stype = spec.get("type")
        if stype == "fixed":
            return spec["value"]
        if stype == "categorical":
            return random.choice(spec["values"])
        if stype == "range":
            min_val = spec.get("min", 0)
            max_val = spec.get("max", 0)
            if "min_expr" in spec:
                min_val = eval(spec["min_expr"], {"__builtins__": {}}, values)
            if "max_expr" in spec:
                max_val = eval(spec["max_expr"], {"__builtins__": {}}, values)
            step = spec.get("step", 1)
            if spec.get("float"):
                return random.uniform(min_val, max_val)
            return random.randrange(int(min_val), int(max_val), int(step))
        raise ValueError(f"Unknown sampling type: {stype}")

    def _check_constraints(self, values):
        for expr in self.dataset_cfg.get("constraints", []):
            if not eval(expr, {"__builtins__": {}}, values):
                return False
        return True

    def generate(self):
        dimensions_States = (self.setlength, self.features)
        dimensions_fix_Parameters = (self.setlength, self.number_fix_Parameters)
        dimensions_var_Parameters = (self.setlength, self.number_variable_Parameters)

        state_in = np.zeros(dimensions_States)
        state_out = np.zeros(dimensions_States)
        parameters_fix = np.zeros(dimensions_fix_Parameters)
        parameters_var = np.zeros(dimensions_var_Parameters)

        for i in range(self.setlength):
            tries = 0
            while True:
                values = {}
                for v in self.state_vars:
                    values[v] = self._sample_from(self.dataset_cfg["state_sampling"][v], values)
                for v in self.fix_vars:
                    values[v] = self._sample_from(self.dataset_cfg["fix_parameter_sampling"][v], values)
                deferred_var = {}
                for v in self.var_vars:
                    spec = self.dataset_cfg["variable_parameter_sampling"][v]
                    if spec.get("type") == "expr":
                        deferred_var[v] = spec
                    else:
                        values[v] = self._sample_from(spec, values)
                if self._check_constraints(values):
                    break
                tries += 1
                if tries > 1000:
                    raise RuntimeError("Failed to sample a valid data point after 1000 attempts.")

            for idx, v in enumerate(self.state_vars):
                state_in[i][idx] = values[v]
                state_out[i][idx] = values[v]
            for idx, v in enumerate(self.fix_vars):
                parameters_fix[i][idx] = values[v]
            for idx, v in enumerate(self.var_vars):
                if v in values:
                    parameters_var[i][idx] = values[v]

            for eff in self.dataset_cfg.get("effects", []):
                op = eff.get("op")
                if op == "add":
                    tgt = eff["var"]
                    src = eff["src"]
                    state_out[i][self.state_vars.index(tgt)] = state_in[i][self.state_vars.index(tgt)] + values[src]
                elif op == "add_expr":
                    tgt = eff["var"]
                    expr_val = eval(eff["expr"], {"__builtins__": {}}, values)
                    state_out[i][self.state_vars.index(tgt)] = state_in[i][self.state_vars.index(tgt)] + expr_val
                elif op == "set_expr":
                    tgt = eff["var"]
                    expr_val = eval(eff["expr"], {"__builtins__": {}}, values)
                    state_out[i][self.state_vars.index(tgt)] = expr_val
                else:
                    raise ValueError(f"Unknown effect op: {op}")

            if deferred_var:
                state_in_map = {v: state_in[i][self.state_vars.index(v)] for v in self.state_vars}
                state_out_map = {v: state_out[i][self.state_vars.index(v)] for v in self.state_vars}
                ctx = {**values, "state_in": state_in_map, "state_out": state_out_map}
                for v, spec in deferred_var.items():
                    expr_val = eval(spec["expr"], {"__builtins__": {}}, ctx)
                    values[v] = expr_val
                    parameters_var[i][self.var_vars.index(v)] = expr_val

        input_vec = np.concatenate((state_in, parameters_fix, parameters_var), axis=1)
        output_vec = state_out
        return input_vec, output_vec


def load_config(path: Path):
    with open(path) as f:
        return json.load(f)


def domain_json_path(ds_domain: str) -> Path:
    return Path(__file__).resolve().parent / f"{ds_domain}.json"


def build_domain_from_config(cfg):
    def _norm_val(v):
        if v is None:
            return ">=0.0"
        if isinstance(v, bool):
            return "10.0" if v else "0.0"
        return str(v)

    variables = cfg["variables"]
    if "actions" in cfg:
        symbolic_actions = [a["name"] for a in cfg["actions"]]
        sym_pre_rows = [a["preconditions"] for a in cfg["actions"]]
        sym_eff_rows = [a["effects"] for a in cfg["actions"]]
    else:
        symbolic_actions = cfg["symbolic_actions"]
        sym_pre_rows = cfg["symbolic_preconditions"]
        sym_eff_rows = cfg["symbolic_effects"]
    learned_actions = cfg["learned_actions"]

    fix = df_fix_parameters(cfg["fix_parameters"]["names"], cfg["fix_parameters"]["values"])
    var = df_variable_parameters(cfg["variable_parameters"]["names"], cfg["variable_parameters"]["top"], cfg["variable_parameters"]["down"])

    sym_pre = [pd.Series([_norm_val(v) for v in row], index=variables) for row in sym_pre_rows]
    sym_precons = df_subsymbolic_preconditions(symbolic_actions, variables, sym_pre)

    sym_eff = [pd.Series([_norm_val(v) for v in row], index=variables) for row in sym_eff_rows]
    sym_effects = df_symbolic_effects(variables, symbolic_actions, sym_eff)

    sub_pre = [pd.Series(row, index=variables) for row in cfg["subsymbolic_preconditions"]]
    sub_precons = df_subsymbolic_preconditions(learned_actions, variables, sub_pre)

    listener = cfg["listener"]
    var_params = variable_parameters(
        learned_actions,
        cfg["variable_parameters"]["names"],
        [pd.Series(row, index=cfg["variable_parameters"]["names"]) for row in listener["variable_parameters"]],
    )
    labels = df_subsymbolic_label(
        variables, learned_actions, [pd.Series(row, index=variables) for row in listener["labels"]]
    )
    under_rows = listener.get("under_bounds")
    if under_rows is None:
        under_rows = [["0.0" for _ in variables] for _ in learned_actions]
    upper_rows = listener.get("upper_bounds")
    if upper_rows is None:
        upper_rows = [["0.0" for _ in variables] for _ in learned_actions]

    under = df_subsymbolic_under_bound(
        variables, learned_actions, [pd.Series(row, index=variables) for row in under_rows]
    )
    upper = df_subsymbolic_upper_bound(
        variables, learned_actions, [pd.Series(row, index=variables) for row in upper_rows]
    )

    subsym_effects = subsymbolic_effetcs(labels, under, upper, var_params)
    dom = domain(variables, symbolic_actions, learned_actions, fix, var, sym_precons, sym_effects, sub_precons, subsym_effects)

    problem = df_problem(variables, cfg["problem"]["init"], cfg["problem"]["goal"])
    return dom, problem


def load_input_from_json(path: Path):
    cfg = load_config(path)
    dom, problem = build_domain_from_config(cfg)
    return input(dom, problem)


def validate_dataset_config(cfg):
    if "dataset" not in cfg:
        raise ValueError("Missing dataset section in domain JSON.")
    ds = cfg["dataset"]
    layout = ds.get("input_layout", {})
    state = layout.get("state", [])
    fix_params = layout.get("fix_parameters", [])
    var_params = layout.get("variable_parameters", [])

    if state != cfg.get("variables", []):
        raise ValueError("Dataset input_layout.state must match domain variables order.")
    if fix_params != cfg.get("fix_parameters", {}).get("names", []):
        raise ValueError("Dataset input_layout.fix_parameters must match domain fix_parameters names.")
    if var_params != cfg.get("variable_parameters", {}).get("names", []):
        raise ValueError("Dataset input_layout.variable_parameters must match domain variable_parameters names.")


def generate_dataset_from_json(path: Path, n_samples: int):
    cfg = load_config(path)
    validate_dataset_config(cfg)
    dataset = JsonDataSet(n_samples, cfg["dataset"])
    return dataset.generate()
