# Simulation and Evaluation Code

This directory implements the controlled simulation study for **Static and Dynamic Graph Construction for GNNs**. It generates synthetic graph data, trains MLP and static/dynamic GCN and GAT models on paired datasets, estimates data diagnostics, and exports performance comparisons with uncertainty intervals. **The implementation, configurations, and tests are included; generated datasets, trained checkpoints, and original local run directories are not distributed.**

See the [project overview](../README.md) for research questions and findings, the [published report](https://rozias0925.github.io/FY26-GNN-Topology/) for results, and [MECHANISM_V2.md](MECHANISM_V2.md) for the detailed mathematical specification in Traditional Chinese. The current implementation is `src/gnn_mechanisms/`.

## Environment

Run commands from the repository root. The recorded research environment uses Python 3.12 with the versions in [requirements-lock.txt](requirements-lock.txt).

```bash
python -m venv algorithm/.venv
```

Activate the environment with `.\algorithm\.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source algorithm/.venv/bin/activate` on macOS/Linux, then install the dependencies:

```bash
python -m pip install -r algorithm/requirements-lock.txt
```

The main configuration uses CPU training. Device, thread count, training budget, and worker count are explicit configuration fields; record any changes when comparing a new run with the published results.

## Check the implementation

Run the mechanism tests independently, or use the full test suite:

```bash
python -m unittest discover -s algorithm/tests -p test_mechanisms_v2.py -v
python -m unittest discover -s algorithm/tests -v
```

Tests check graph invariants, data boundaries, model calculations, paired comparisons, and pipeline behavior. They do not require a particular model to win or a directional research hypothesis to hold.

Inspect the small-run workload before executing it:

```bash
python algorithm/run_mechanisms.py simulate --config algorithm/configs/mechanism_v2_smoke.json --dry-run
python algorithm/run_mechanisms.py simulate --config algorithm/configs/mechanism_v2_smoke.json --output algorithm/results/readme_smoke
```

This smoke run uses 30 nodes, 20 time steps, two dataset seeds, and an eight-epoch limit. Its outputs demonstrate execution and are separate from the formal research evidence.

## Run the research configuration

The formal suite uses five dataset seeds, equal four-candidate tuning grids, chronological splits, and training-only preprocessing and diagnostics. Primary results use a 200-epoch limit; the 400-epoch sensitivity analysis is kept separate. Dynamic models use the graph at the prediction origin without recurrent temporal memory.

First reproduce the initial Stage A mechanism checks. The default suite references this output for its training-budget comparison:

```bash
python algorithm/run_mechanisms.py simulate --config algorithm/configs/mechanism_v2.json --stages A --dry-run
python algorithm/run_mechanisms.py simulate --config algorithm/configs/mechanism_v2.json --stages A --output algorithm/results/v2_stage_A
```

Then inspect and run the full suite in a separate output directory:

```bash
python algorithm/run_simulation_suite.py --config algorithm/configs/mechanism_v2_suite.json --dry-run
python algorithm/run_simulation_suite.py --config algorithm/configs/mechanism_v2_suite.json --output algorithm/results/reproduced_suite
```

The suite covers one-axis response curves, interaction grids, 200 Latin-hypercube conditions, sample-size experiments, robustness checks, and longer-training sensitivity. Its 7,240 model evaluations are separate from the initial 150 Stage A evaluations. Evaluation counts include reused equivalent computations and must not be described as independent datasets or optimizer fits.

The default suite has two worker processes and two CPU threads per model. Adjust the configuration before launching if resources are shared with other work. For a new experimental design, use a new configuration and output directory. Existing-run continuation verifies configuration and source consistency and uses lock files to prevent concurrent writes. See [SIMULATION_SUITE.md](SIMULATION_SUITE.md) for the scheduler and recovery protocol; its dated progress notes describe earlier execution states.

## Outputs and public artifacts

| Output | Purpose |
| --- | --- |
| `manifest.json` and source snapshots | Record settings, source checksums, and environment metadata |
| `status.json`, logs, and completion markers | Track completed work and failures |
| Per-model checkpoints and predictions | Preserve fitted models and evaluation inputs locally |
| `reports/main/` | Regression scores, paired contrasts, response curves, phase points, and convergence flags |
| `reports/robustness/` | Graph-family, noise, classification, and GATv2 comparisons |
| `reports/convergence/` | Longer-budget scores and paired budget-sensitivity comparisons |

The repository includes the [published CSV tables and provenance](../dist/data/mechanisms-v2/) and [complete website JSON export](../dist/data/mechanism-results-v2.json). These can be inspected without retraining. The original `algorithm/results/` directory and checkpoints remain excluded from Git. Export tools that read those local run directories require a reproduced run or access to the original artifacts.

## Code map and scope

| Module in `src/gnn_mechanisms/` | Responsibility |
| --- | --- |
| `config.py`, `data.py`, `graphs.py` | Conditions, independent random streams, targets, and true/observed topology |
| `models.py`, `training.py` | Model implementations, candidate fitting, and validation-based selection |
| `diagnostics.py` | Feature signal, incremental graph signal, smoothness, relevance, and dynamicity probes |
| `experiment.py`, `suite.py` | Paired execution, workload planning, deduplication, and continuation |
| `reporting.py` | Performance summaries, bootstrap intervals, tables, and figures |

The completed project concerns synthetic-data mechanisms. Next-step regression, same-time denoising, and classification robustness are analyzed separately. Earlier `src/gnn_graph_selection/`, data-download commands, and real-data utilities are retained as historical work; they are not required to reproduce the current simulation study. No real-world external-validation claim is made. See [RESEARCH_SCOPE.md](../RESEARCH_SCOPE.md) for the current scope.
