# Preview: real experiments

**Flight 1 - straight line**

https://github.com/user-attachments/assets/e1ee9e29-17c6-4153-9de2-ada54163c36a

**Flight 2 - along the border**

https://github.com/user-attachments/assets/884575ce-0c42-4f09-85f9-9a640e744112

# EnvMPC MuJoCo demo for the BELKA robot

Minimal reproducible source and numerical results for the article
“Model Predictive Control of a Free-Flying Robot with External Wrench
Estimation by an Extended Kalman Filter”.

The repository compares three controllers in a six-degree-of-freedom MuJoCo
model:

- `ignore`: conventional MPC, which does not use the external-wrench estimate;
- `estimate`: EnvMPC, which supplies the ESKF force and moment estimate to the
  prediction model;
- `oracle`: a reference controller supplied with the true simulated wrench.

Tracking metrics use the true MuJoCo state. Estimated-state tracking metrics
and observer errors are reported separately. The repository contains the exact
robot parameters, actuator allocation, MPC weights and constraints, ESKF
settings, disturbance profiles and fixed experiment matrices used for the
article.

## Quick start

Requirements: Git, CMake, a C/C++ compiler, `make`, Rust/Cargo,
[uv](https://docs.astral.sh/uv/) and Python 3.13. MuJoCo is installed from its
official Python wheel. acados and its template renderer are built from pinned
public submodules with the HPIPM backend used by the controller.

```bash
git clone --recurse-submodules git@git.spacehub.su:Belka/envmpc_demo.git
cd envmpc_demo
./scripts/setup.sh
./scripts/run_reference.sh
```

The reference command repeats `mixed_holdout_103` for MPC, EnvMPC and the
true-wrench controller, then compares position RMSE, orientation RMSE and
control effort with the archived values using a 10% relative tolerance. This
allows for platform-dependent nonlinear-solver trajectories while still
checking all physical metrics and the qualitative ordering. Solver time is
reported but not compared across machines.

Expected truth-based reference metrics:

| Mode | Position RMSE, m | Orientation RMSE, rad | Control effort |
|---|---:|---:|---:|
| MPC (`ignore`) | 0.03922 | 0.08609 | 72.99 |
| EnvMPC (`estimate`) | 0.03034 | 0.03870 | 71.60 |
| True wrench (`oracle`) | 0.02864 | 0.03921 | 69.29 |

acados status 2, which denotes reaching the fixed ten-iteration SQP limit, is
recorded separately from nonstandard/failing statuses. The published cases
have zero nonstandard-status fraction.

## Complete numerical campaigns

Run the original 75 calculations:

```bash
./scripts/with_env.sh python -m \
  experiments.research.paper_1_external_wrench.run \
  --experiment all --output out/baseline_75
```

Run the fixed 36 + 18 + 45 applicability study:

```bash
./scripts/with_env.sh python -m \
  experiments.research.paper_1_external_wrench.applicability \
  --output out/applicability_99
```

Generate the baseline article plots after the 75-case run:

```bash
./scripts/with_env.sh python -m \
  experiments.research.paper_1_external_wrench.analysis \
  --input out/baseline_75 --output out/figures
```

Existing compact results can be checked without acados or MuJoCo:

```bash
python3 scripts/verify_results.py
```

Run the code tests after setup:

```bash
./scripts/with_env.sh pytest -q
```

Generated solvers, snapshots and figures are written below `out/` and are not
tracked. The complete campaign is computationally expensive; the reference
case is the recommended end-to-end reproduction check.

## Scope and provenance

This is a numerical research artifact, not robot flight software. It contains
no ROS integration, stand control, physical-flight records or payload
identification. Numerical and physical results are distinct in the article;
only the former are reproduced here.

The repository has a clean publication history. Source commits and the
deliberate replacement of the internal acados mirror with its public upstream
parent are documented in [PROVENANCE.md](PROVENANCE.md). Compact archived
results and their contents are described in [results/README.md](results/README.md).

## Краткое описание на русском

Репозиторий воспроизводит численную часть статьи о методе EnvMPC для робота
БЕЛКА. В обычном MPC оценка внешних силы и момента не используется; в EnvMPC
она поступает из ESKF в модель прогноза; вариант `oracle` получает истинное
воздействие численной модели и служит эталоном. Ошибка слежения вычисляется по
истинному состоянию MuJoCo отдельно от ошибки оценивания.

Команды `setup.sh` и `run_reference.sh` собирают зависимости и повторяют один
полный подтверждающий случай. Команды раздела Complete numerical campaigns
повторяют все 174 расчёта. Стендовые данные и управление реальным роботом в
этот репозиторий намеренно не включены.

## License

The BELKA publication code is distributed under the MIT License. The pinned
acados submodule retains its own BSD-2-Clause license.
