# NTG-Bench

NTG-Bench is an implementation of the evaluation framework proposed in the paper _“Generative AI for Synthetic Network Traffic: Methods, Evaluation Metrics, and Practical Insights.”_ The benchmark turns the five complementary evaluation dimensions from the paper into reproducible code so that any synthetic traffic generator can be compared against real data and against other models with the same set of metrics.

## Evaluation axes

The framework evaluates synthetic traffic along five complementary axes:

1. **Distribution Similarity** – Captures how well marginal feature distributions are preserved via Jensen–Shannon divergence (JSD), total variation distance (TVD), Earth mover’s distance (EMD), and the Kolmogorov–Smirnov (KS) statistic.
2. **Protocol Compliance** – Applies the deterministic knowledge checks (DKC) from the paper to flag impossible flows (e.g., TCP on DNS-only ports) and reports the clean-pass ratio alongside the filtered dataset.
3. **Semantic Consistency** – Measures whether inter-feature relationships are retained using the categorical mutual dependence (CMD) metric and Pearson correlation distance (PCD) on continuous features.
4. **Generation Diversity** – Uses the PRDC density/coverage scores over a one-hot + min–max representation to quantify whether the synthetic set covers real operating modes instead of collapsing to a narrow region.
5. **End-to-End Utility** – Benchmarks the usefulness of synthetic traffic for downstream detection by training several classical classifiers (DT, RF, GB, XGB, MLP) in three scenarios: pure real data (baseline), train-on-synthetic-test-on-real (TSTR), mixed synthetic/real ratios, and data augmentation strategies.

All metrics are reported as CSV artifacts that can be version-controlled or ingested into downstream dashboards.

## Repository layout

```
.
├── config.example.json      # Starter configuration file
├── (no top-level script)    # Run via `python -m ntg_bench.cli --config ...`
├── ntg_bench/               # Package with benchmark implementation
│   ├── benchmark.py         # BenchmarkRunner orchestrating every stage
│   ├── cli.py               # CLI helpers
│   ├── config.py            # Dataclass describing configurable knobs
│   ├── data.py              # Data loading utilities
│   ├── metrics.py           # Metric implementations
│   ├── protocol.py          # DKC / protocol compliance checks
│   └── utility.py           # Task utility evaluation helpers
├── paper.pdf                # Reference paper
└── requirements.txt         # Python dependencies
```

## Installation

1. **Python environment**: Python 3.10+ is recommended.
2. **Dependencies**: Install the required packages

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **GPU optional**: XGBoost defaults to `tree_method="auto"`. Switch it to `gpu_hist` in `ntg_bench/utility.py` if you want to force GPU acceleration.

## Data layout

The benchmark expects a directory layout that mirrors the datasets used in the paper:

```
<base_dir>/
├── real/
│   ├── Bruteforce.csv
│   ├── ... other per-class CSVs ...
│   ├── real_train/
│   │   ├── Bruteforce_cleaned.csv
│   │   └── ...
│   └── real_test/
│       ├── Bruteforce_cleaned.csv
│       └── ...
├── NetShare/
│   ├── Bruteforce.csv
│   └── ...
└── OtherModel/
    ├── Bruteforce.csv
    └── ...
```

- Each model (or `real`) gets its own directory containing one CSV per traffic class.
- Optional `_cleaned.csv` variants take precedence when loading data.
- During evaluation, cleaned synthetic CSVs are saved alongside the original files.

## Configuration

All runtime options live in a JSON file to keep experiments reproducible. Start by copying `config.example.json` and adjust the fields:

```json
{
  "base_dir": "/absolute/path/to/data",
  "models": ["NetShare", "FlowGAN"],
  "classes": ["Bruteforce", "WebAttack", "DDoS"],
  "continuous_features": ["td", "pkt", "byt"],
  "categorical_features": ["proto", "service"],
  "label_column": "label",
  "real_data_name": "real",
  "real_train_subdir": "real_train",
  "real_test_subdir": "real_test",
  "output_dir": "outputs",
  "random_state": 42,
  "mixing_rates": [0.0, 0.25, 0.5, 0.75, 1.0],
  "augment_strategy": "self",
  "augment_target_count": 16000
}
```

Key notes:

- `continuous_features` and `categorical_features` must be present in all CSVs.
- `models` is the list of synthetic generators you want to benchmark.
- `mixing_rates` controls the real/synthetic ratios for the mixed-data experiment.
- `augment_strategy` can be `self`, `mean`, or `median`, following the paper.

## Running the benchmark

```bash
python -m ntg_bench.cli --config config.example.json
```

The CLI reports progress in the console and stores every artifact under `output_dir`. Typical outputs:

- `distribution_metrics.csv` – Per-model averages for JSD, TVD, EMD, KS, CMD, PCD, density, coverage, and DKC.
- `feature_similarity.csv` – Per-feature breakdown combining all similarity metrics.
- `baseline_scores.csv` – Real-data-only classifier performance (sanity check / upper bound).
- `tstr_scores.csv` – Train-on-synthetic-test-on-real F1 scores for each generator/model pair.
- `mixing_curve.csv` – Pivot table showing F1 progression as synthetic ratio increases.
- `augmentation_scores.csv` – Pivoted F1 scores for augmentation experiments.

## Extending

- **New metrics**: Add functions under `ntg_bench/metrics.py` and include them inside `_run_distribution_metrics`.
- **Custom classifiers**: Update `ntg_bench/utility.py::get_default_models`.
- **Automation**: Import `BenchmarkConfig` and `BenchmarkRunner` from the package to integrate the benchmark inside CI pipelines.

## Citation

If you build on this code or the accompanying evaluation methodology, please cite:

```
Dongqi Han, Can Zhang, Zhiliang Wang, Linghui Li, Qi Tian.
“Generative AI for Synthetic Network Traffic: Methods, Evaluation Metrics, and Practical Insights.”
```
