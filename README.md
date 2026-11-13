# Static and Dynamic Graph Construction for GNNs

This project investigates when graph structure improves prediction beyond a node's own features, when learned neighbor weights are useful, and when changing graph connections carry predictive information. It compares a multilayer perceptron (MLP) with static and dynamic versions of Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), using controlled synthetic experiments to explain their relative performance. **This repository includes the simulation and evaluation source code, experiment configurations, tests, exported results, and a bilingual research website.**

[Research report](https://rozias0925.github.io/FY26-GNN-Topology/) | [Results](https://rozias0925.github.io/FY26-GNN-Topology/#results) | [Simulation code](algorithm/src/gnn_mechanisms/) | [Run the experiments](algorithm/README.md)

## Research approach

A graph represents nodes and the connections through which they exchange information. This study asks which measurable properties of such data explain the benefit of using those connections, learning their importance, or updating them over time.

| Model | Information used in one update |
| --- | --- |
| MLP | The target node's own features |
| Static-GCN | Degree-normalized aggregation over a fixed graph |
| Static-GAT | Learned neighbor weights over a fixed graph |
| Dynamic-GCN | Degree-normalized aggregation over the current graph |
| Dynamic-GAT | Learned neighbor weights over the current graph |

The experiments separately vary graph signal strength, neighbor relevance, signal smoothness, true topology drift, and graph observation noise. This separates changing predictive relationships from noisy observations of an unchanged graph. Dynamic models use successive graph snapshots without recurrent temporal memory; a static GAT can still change its attention weights as node features change.

The main task is next-step regression. A separate smoothness experiment studies same-time denoising, while classification and GATv2 are robustness checks. The main configuration uses 500 nodes, 40 time steps, eight features, and initial degree 10, with specified interventions for graph family and sample size.

## Experimental evidence

Models are compared on the same generated datasets across five independent dataset seeds. Evaluation uses chronological train/validation/test splits, training-only preprocessing and diagnostics, equal hyperparameter-search budgets, and paired bootstrap intervals for performance differences. The primary training budget and longer-training sensitivity analysis are reported separately.

The completed suite contains **7,240 model evaluations**, with **150 initial mechanism-check evaluations** reported separately. These are evaluation counts, not counts of independent datasets or optimizer fits; equivalent static/dynamic computations can reuse training. The study includes five response curves, interaction experiments, 200 sampled multi-axis conditions, sample-size comparisons, and robustness checks. Published artifacts include CSV tables, JSON results, figures, and [provenance checksums](dist/data/mechanisms-v2/provenance.json).

Selected findings within the tested settings:

- **Graph information matters when it adds signal.** MLP remains competitive when the target depends on local features; graph models improve as otherwise inaccessible neighbor signal increases.
- **Selective neighbors favor learned weighting.** In the fixed-topology experiment with $\lambda=0.75$ and $\kappa=4$, Static-GAT exceeds Static-GCN by a mean $\Delta R^2$ of approximately $0.034$, with a 95% paired bootstrap interval of $[0.030,\,0.037]$. Their difference is small under uniform aggregation. The [response-curve table](dist/data/mechanisms-v2/main/response_curves.csv) contains the full comparisons.
- **Useful change and noisy change have different consequences.** Predictive topology drift favors dynamic graphs; observation noise can favor a fixed consensus graph. Greater structural change alone does not establish a benefit from dynamic modeling.
- **Some hypotheses remain unsupported.** Dynamic-model gains are not uniformly monotonic in predictive dynamicity, and the sample-size experiment does not establish a clear small-sample GCN advantage. These findings are retained in the [discussion](https://rozias0925.github.io/FY26-GNN-Topology/#discussion).

The conclusions apply to the specified generators, architectures, and training budgets. Diagnostics depend on their estimation probes, and intervals use five dataset seeds without correction across all comparisons. The project is a simulation study; it does not establish a universal model-selection rule or claim external validation on real-world data.

## Explore or reproduce

The [website](https://rozias0925.github.io/FY26-GNN-Topology/) presents the mathematics, an interactive four-node calculation, experimental design, results, and limitations in English and Traditional Chinese. Select **EN** in the top bar for English. It can also be served locally without installing the research dependencies:

```bash
python -m http.server 4173 --bind 127.0.0.1 --directory dist
```

Open `http://127.0.0.1:4173/`. For the simulations, create and activate a Python 3.12 virtual environment, then run from the repository root:

```bash
python -m pip install -r algorithm/requirements-lock.txt
python -m unittest discover -s algorithm/tests -p test_mechanisms_v2.py -v
python algorithm/run_mechanisms.py simulate --config algorithm/configs/mechanism_v2_smoke.json --output algorithm/results/readme_smoke
```

The smoke configuration checks the pipeline on a small dataset; it does not reproduce the reported research results. The [algorithm guide](algorithm/README.md) gives the full-suite commands and output structure. Generated datasets, trained checkpoints, and the original local run directories are not distributed in this repository; the published evaluation tables and figures are included.

## Repository guide

| Location | Contents |
| --- | --- |
| [algorithm/src/gnn_mechanisms/](algorithm/src/gnn_mechanisms/) | Data generators, graph construction, models, diagnostics, training, and paired evaluation |
| [algorithm/configs/](algorithm/configs/) and [algorithm/tests/](algorithm/tests/) | Experiment settings and implementation checks |
| [algorithm/MECHANISM_V2.md](algorithm/MECHANISM_V2.md) | Detailed mathematical and methodological specification in Traditional Chinese |
| [dist/data/mechanisms-v2/](dist/data/mechanisms-v2/) | Published evaluation tables, figures, and provenance |
| [dist/](dist/) and [scripts/](scripts/) | Static research website, result export, and figure preparation |

The [current research scope](RESEARCH_SCOPE.md) identifies the completed simulation study. Earlier pipelines and dated records are retained for traceability and are separate from the results summarized here.
