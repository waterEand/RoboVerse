# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboVerse is a multi-simulator robot learning platform that provides a unified abstraction over Isaac Sim, Isaac Gym, MuJoCo, SAPIEN, PyBullet, Genesis, and others. The three-layer architecture is:

- **`metasim/`** — Simulator-agnostic abstraction layer (handlers, task envs, state types, queries, randomization)
- **`roboverse_pack/`** — Task, robot, scene, and asset definitions (the "content")
- **`roboverse_learn/`** — Learning algorithms: IL (`il/`), RL (`rl/`), VLA (`vla/`)

The main focus for policy development is `roboverse_learn/il/`.

## Common Commands

### Linting
```bash
ruff check .
ruff format .
```
Pre-commit hooks (ruff + trailing whitespace) must pass. Install with `pip install pre-commit && pre-commit install`.

### Testing
```bash
pytest metasim/test
```

### Full IL Pipeline (collect → train → eval)
```bash
# Install policy-specific deps first
cd roboverse_learn/il/policies/dp   # or fm/, vita/
pip install -r requirements.txt
cd ../../../..

# Run the full pipeline
bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name ddpm_dit
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name fm_dit --sim_set mujoco
bash roboverse_learn/il/il_run.sh --task_name_set pick_cube --policy_name vita --demo_num 200
```

Key `il_run.sh` options: `--task_name_set`, `--policy_name`, `--sim_set` (default: `isaacsim`), `--demo_num` (default: 100), `--num_epochs`, `--gpu`, `--dr_level_collect`, `--dr_level_eval`, `--train_enable`, `--eval_enable`.

### Training Only (Hydra)
```bash
export policy_name=ddpm_dit
python roboverse_learn/il/train.py --config-name=default_runner.yaml \
  task_name=close_box \
  "dataset_config.zarr_path=./data_policy/close_boxFrankaL0_isaacsim_obs:joint_pos_act:joint_pos_100.zarr" \
  train_config.training_params.num_epochs=100
```

### Demo Collection
```bash
python scripts/advanced/collect_demo.py \
  --sim=isaacsim --task=close_box --num_envs=1 \
  --headless --num_demo_success=100 --level=0
```

### Convert Demos to Zarr
```bash
python roboverse_learn/il/data2zarr_dp.py \
  --task_name close_boxFrankaL0_isaacsim_obs:joint_pos_act:joint_pos \
  --expert_data_num 100 \
  --metadata_dir ./roboverse_demo/demo_isaacsim/close_box-test/robot-franka/success
```

### Fix Dependency Issues
```bash
bash roboverse_learn/il/il_setup.sh
```

## IL Architecture

### Entry Points
- **`roboverse_learn/il/train.py`** — Hydra entrypoint; instantiates a `BaseRunner` subclass from `cfg._target_` and calls `runner.run()`.
- **`roboverse_learn/il/runners/default_runner.py`** — `DefaultRunner`: full train+eval loop with wandb logging, EMA, checkpointing, and optional domain randomization.

### Configuration System (Hydra)
Config root: `roboverse_learn/il/configs/default_runner.yaml`

Modular config groups composed at launch:
- `dataset_config/robot_image_dataset.yaml` — zarr path, horizon, padding
- `policy_config/<policy_name>.yaml` — selected via `$policy_name` env var
- `train_config/default_train.yaml` — optimizer, LR schedule, epochs, batch size
- `eval_config/default_eval.yaml` — task, sim, max_step, num_envs, DR level

Key top-level fields: `horizon`, `n_obs_steps`, `n_action_steps`, `shape_meta` (obs/action shapes), `train_enable`, `eval_enable`, `eval_path`.

Output structure: `./il_outputs/<policy_name>/<task_name>/checkpoints/<epoch>.ckpt`

Data structure: `./data_policy/<task>FrankaL<dr_level>_<sim>_obs:<obs>_act:<act>_<n>.zarr`

### Adding a New Policy
1. Create a directory under `roboverse_learn/il/policies/` with a `policy.py` implementing `BaseImagePolicy`.
2. Implement `predict_action(obs_dict) -> Dict[str, Tensor]` and `set_normalizer(LinearNormalizer)`.
3. Add a training step method and integrate with `DefaultRunner` (or write a custom `BaseRunner`).
4. Add a policy config YAML to `roboverse_learn/il/configs/policy_config/`.
5. Add a `requirements.txt` for any extra deps.

### Key Abstractions

**`BaseImagePolicy`** (`policies/base_image_policy.py`):
- `predict_action(obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]` — obs keys match `shape_meta.obs`, returns `{"action": B×Ta×Da}`
- `set_normalizer(LinearNormalizer)` — called before training
- `reset()` — for stateful policies (RNNs, etc.)

**`BaseRunner`** (`runners/base_runner.py`):
- `run()` dispatches to `train()` and/or `evaluate()` based on config flags
- `save_checkpoint()` / `load_checkpoint()` — threaded checkpoint management

**`RobotImageDataset`** (`datasets/robot_image_dataset.py`):
- Reads zarr replay buffers; keys used: `head_camera`, `state`, `action`
- Handles sequence sampling with configurable horizon, padding, and train/val split

**`LinearNormalizer`** (`utils/normalizer.py`):
- Normalizes/unnormalizes observation and action tensors; call `normalizer['action'].normalize(x)` / `.unnormalize(x)`

### Supported IL Policies

| Policy name | Algorithm | Backbone |
|---|---|---|
| `ddpm_dit` | Diffusion Policy (DDPM) | DiT |
| `fm_dit` | Flow Matching | DiT |
| `vita` | VITA | MLP |
| `ddpm_unet` | Diffusion Policy (DDPM) | UNet |
| `ddim_unet` | Diffusion Policy (DDIM) | UNet |
| `fm_unet` | Flow Matching | UNet |
| `score_unet` | Score-Based Model | UNet |
| `act` | ACT (separate pipeline via `act_run.sh`) | Transformer+VAE |

1-NFE variants (MeanFlow, iMF, CFM) are also available for flow-matching policies.

## Simulation Layer (`metasim/`)

### Handler Pattern
Simulators are accessed through `BaseSimHandler` subclasses (one per simulator backend in `metasim/sim/`). Handlers are instantiated from `ScenarioCfg` objects.

Core handler interface:
- `simulate()` — advance physics
- `render()` / `refresh_render()` — update visuals
- `get_states()` — returns cached `TensorState` (reset cache with `handler._state_cache = None`)
- `get_extras()` — fetch query outputs
- `close()` — shutdown

### Task Environment
`BaseTaskEnv` (`metasim/task/base.py`) wraps a handler as a gymnasium env. Override `_observation()`, `_reward()`, `_terminated()` to define a task.

### Task Registration
Use the `@register_task("task_name")` decorator on `BaseTaskEnv` subclasses in `roboverse_pack/tasks/`. Tasks are auto-discovered and available via `get_task_class("task_name")` or `gymnasium.make("RoboVerse/<task_name>")`.

### Configuration Objects (`configclass`)
All scenario components (robots, objects, cameras, sim params) are Python dataclasses decorated with `@configclass` from `metasim/utils/configclass.py`. These support `.to_dict()`, `.from_dict()`, and `.replace()`.

Key configs: `ScenarioCfg`, `RobotCfg`, `BaseObjCfg`, `BaseCameraCfg`, `SimParamCfg`.

## Code Style

- Line length: 120
- Docstrings: Google-style (enforced by ruff `D` rules; `roboverse_learn/` is exempt)
- `roboverse_learn/` allows `T20` (print statements) and skips docstring checks
- Use `loguru.logger` (imported as `log`) instead of `print` in `metasim/`
