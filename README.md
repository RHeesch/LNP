# Lazy Neural Planner - LNP

This repository contains the code for the paper _A Lazy Approach to Neural Numerical Planning with Control Parameters_.
The latest update builds the abstract planning problem in the Unified Planning Framework. 

## High-Level Overview
The project implements a **lazy, hierarchical planning pipeline** that combines symbolic planning with neural models for subsymbolic (learned) actions:

1. **Domain definition (JSON)**
   - Each domain is described in a JSON file (state variables, action preconditions/effects, learned actions, listener labels, dataset layout).
2. **Data generation (reproducible cache)**
   - A dataset is generated once and cached. Training and planning only use cached data.
3. **Model training**
   - Neural models learn the transition function for learned actions.
4. **Abstract planning**
   - A symbolic planner searches an abstract plan (ignoring exact control parameters).
5. **Concretization**
   - Given the abstract plan, the system computes control parameters for learned actions (inverse problem): find parameters such that the NN output matches the abstract successor state.
6. **Validation and re-planning**
   - The concrete plan is checked; if it fails, a blocking lemma is added and planning is retried.

There are **two planning front-ends**:
- `code/LNP.py`: original Z3-based planner (baseline and reference implementation).
- `code/LNP_UPF.py`: Unified Planning (UPF) implementation with TemPEST integration, concretization, validation, and blocking.

## External Dependencies
Unified Planning (UPF) installation guide:
```
https://unified-planning.readthedocs.io/en/latest/index.html
```
TemPEST engine repository:
```
https://github.com/fbk-pso/tempest
```

## Configuration
All main settings are in:
- `config/hparams.json`

Key options:
- `DS_DOMAIN`: active domain (e.g. `drone`, `flipsi`, `zeno`, `cashpoint`).
- `Max_Step`: horizon bound for planning.
- `Int_ML`: integrate learned actions (`1`) or symbolic only (`0`).
- `MODEL`: neural architecture (`MLP`, `autoencoder`, `ffn`).
- `PREDICT_DELTA`: train NN on `delta s` instead of absolute `s'`.
- `SCALE`: apply dataset scaling (`0` or `1`).
- `opti_threshold`: concretization distance threshold.
- `MAX_LOOPS`: maximum replanning loops in `LNP_UPF`.
- `BLOCK_TRANSITION_RADIUS`: region radius for transition blocking.
- `Listener`: listener mode (`none`, `partial`, `full`) in `LNP_UPF`.
- Boolean handling:
  - `BOOL_ENCODING`: `01` or `010` (numeric encoding).
  - `BOOL_THRESHOLD`: threshold for boolean inference.
  - `BOOL_LOSS`: `mse` or `bce` for boolean dims.
  - `BOOL_LOSS_WEIGHT`: boolean loss weight (when using MSE).

## Data Generation (Reproducible)
Data is generated once and cached; training and planning use only cached data.

Generate the dataset:
```
python code/generate_data.py
```

Force regeneration:
```
python code/generate_data.py --force
```

Cache location:
- `exp/data_cache/`

## Training
Training is done via:
```
python code/train_models.py
```

Hyperparameter tuning (random search):
```
python code/train_models.py --tune --trials 10
```
You can also adjust early-stop settings:
```
python code/train_models.py --tune --trials 10 --trial-patience 4 --trial-diverge-factor 5 --trial-min-epochs 50
```
Concretization tuning (random search of reconstruction settings):
```
python code/train_models.py --tune-concretization --trials 10 --concretization-samples 64
```
This evaluates reconstruction settings (`REC_LR`, `REC_EPOCHS`, `N_MODELS`, `opti_threshold`) against cached data and the current trained model, and writes `concretization_tuning.json` under `exp/log/<domain>/`.
Add `--auto-write-hparams` to write the best values back into `hparams.json` (atomic replace). This is optional and off by default.

Important training behavior:
- Input = full state + fixed params + control params.
- Output = successor state (`s'`) or delta (`s' - s`) if `PREDICT_DELTA=1`.
- Boolean dims can use weighted MSE or BCE (configurable).
- Concretization/search mode is configurable via `PARAM_SEARCH`:
  - `gradient-descent` (backprop reconstruction)
  - `beam` (beam search within parameter bounds)
  - `finite_diff` (finite-difference search within bounds)
  - Bounds are taken from JSON `variable_parameters.down/top` when available.

Trained models and metrics are saved under:
- `exp/log/<domain>/`

## Planning (UPF + TemPEST)
Main entry:
```
python code/LNP_UPF.py --domain drone --engine tempest
```
By default the script runs quietly. Use `--check` to enable detailed checks, logs, and ANML export:
```
python code/LNP_UPF.py --domain drone --engine tempest --check
```
When a concrete plan validates successfully, the final plan and parameter sets are printed by default.

### Steps in `LNP_UPF.py`
1. **Load JSON domain**
   - Validate schema and required keys.
2. **Build UPF problem**
   - Create fluents for each variable.
   - Add initial state and goal.
   - Add symbolic actions with numeric and boolean effects.
   - Add learned actions with parameters and bounds.
3. **Export ANML**
   - Writes `exp/log/<domain>_loopN.anml` for each loop.
   - Performs a strict consistency check vs JSON.
4. **Solve (TemPEST)**
   - Abstract plan is found without fixing control parameters.
5. **Concretization**
   - For each learned action:
     - Build NN input/output vectors from the abstract plan.
     - Call `determine_parameter` to reconstruct control parameters.
     - Apply learned effects and threshold booleans.
     - Enforce distance threshold (`opti_threshold`).
6. **Validation**
   - Uses UPF `sequential_plan_validator` on the concrete plan.
7. **Blocking + Replanning**
   - If concretization fails, add a transition blocking lemma:
     - Block a predecessor/successor region for the learned action.
   - Retry until `MAX_LOOPS` or success.

### Logging
- Main log: `exp/log/<domain>_lnp_upf.log`
- ANML snapshots: `exp/log/<domain>_loopN.anml`
- Concretization trace includes:
  - pre state, abstract successor, concrete successor
  - parameter values
  - distance vs threshold
  - final state and validation status

## How Components Connect
- **`code/data_gens/*.json`** defines the domain, actions, learned actions, and dataset layout.
- **`code/data_gens/json_loader.py`** loads JSON and builds datasets.
- **`code/data_module.py`** handles dataset caching and loaders.
- **`code/train_models.py`** trains NNs using `code/train_module.py`.
- **`code/train_module.py`** implements training, reconstruction, and parameter inference (`determine_parameter`).
- **`code/LNP.py`** is the Z3-based reference planner.
- **`code/LNP_UPF.py`** is the UPF-based planner with TemPEST integration and concretization.

## Citation
If you use this code, please cite:
```
@incollection{lazyheesch24,
 title={A lazy approach to neural numerical planning with control parameters},
 author={Heesch, René and Cimatti, Alessandro and Ehrhardt, Jonas and Diedrich, Alexander and Niggemann, Oliver},
 booktitle={ECAI 2024},
 year={2024},
 publisher={IOS Press}
}
```

## License
MIT License. See `LICENSE`.
