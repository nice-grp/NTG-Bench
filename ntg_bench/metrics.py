from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from prdc import compute_prdc
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import MinMaxScaler


@dataclass
class MetricResult:
    value: float
    per_feature: Dict[str, float]


def compute_jsd(real: pd.DataFrame, synthetic: pd.DataFrame) -> MetricResult:
    variables = sorted(set(real.columns).union(set(synthetic.columns)))
    values = []
    per_feature = {}

    for column in variables:
        counts_real = real[column].value_counts()
        counts_syn = synthetic[column].value_counts()
        all_values = sorted(set(counts_real.index).union(counts_syn.index))

        counts_real = counts_real.reindex(all_values, fill_value=0)
        counts_syn = counts_syn.reindex(all_values, fill_value=0)

        jsd = float(jensenshannon(counts_real, counts_syn))
        per_feature[f"JSD_{column}"] = jsd
        values.append(jsd)

    average = float(np.mean(values)) if values else np.nan
    return MetricResult(average, per_feature)


def compute_tvd(real: pd.DataFrame, synthetic: pd.DataFrame) -> MetricResult:
    variables = sorted(set(real.columns).union(set(synthetic.columns)))
    values = []
    per_feature = {}

    for column in variables:
        counts_real = real[column].value_counts().astype(float)
        counts_syn = synthetic[column].value_counts().astype(float)
        all_values = sorted(set(counts_real.index).union(counts_syn.index))

        counts_real = counts_real.reindex(all_values, fill_value=0)
        counts_syn = counts_syn.reindex(all_values, fill_value=0)

        p_real = counts_real / counts_real.sum() if counts_real.sum() else counts_real
        p_syn = counts_syn / counts_syn.sum() if counts_syn.sum() else counts_syn

        tvd = 0.5 * np.abs(p_real - p_syn).sum()
        per_feature[f"TVD_{column}"] = float(tvd)
        values.append(float(tvd))

    average = float(np.mean(values)) if values else np.nan
    return MetricResult(average, per_feature)


def compute_emd(real: pd.DataFrame, synthetic: pd.DataFrame) -> MetricResult:
    values = []
    per_feature = {}

    for column in real.columns:
        real_array = real[column].to_numpy()
        syn_array = synthetic[column].to_numpy()
        hist_real, bins = np.histogram(real_array, bins="fd")
        cdf_real = np.cumsum(hist_real) / len(real_array)
        hist_syn, _ = np.histogram(syn_array, bins=bins)
        cdf_syn = np.cumsum(hist_syn) / len(syn_array)
        score = wasserstein_distance(cdf_real, cdf_syn)
        per_feature[f"EMD_{column}"] = float(score)
        values.append(float(score))

    average = float(np.mean(values)) if values else np.nan
    return MetricResult(average, per_feature)


def compute_ks(real: pd.DataFrame, synthetic: pd.DataFrame) -> MetricResult:
    values = []
    per_feature = {}

    common_columns = real.columns.intersection(synthetic.columns)
    for column in common_columns:
        statistic, _ = ks_2samp(real[column].to_numpy(), synthetic[column].to_numpy())
        per_feature[f"KS_{column}"] = float(statistic)
        values.append(float(statistic))

    average = float(np.mean(values)) if values else np.nan
    return MetricResult(average, per_feature)


def compute_pcd(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    real_values = real.to_numpy()
    synthetic_values = synthetic.to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((real_values, synthetic_values)))
    real_scaled = scaler.transform(real_values)
    synthetic_scaled = scaler.transform(synthetic_values)
    corr_real = np.corrcoef(real_scaled.T)
    corr_syn = np.corrcoef(synthetic_scaled.T)
    corr_real = np.nan_to_num(corr_real, nan=0.0)
    corr_syn = np.nan_to_num(corr_syn, nan=0.0)
    return float(np.linalg.norm(corr_real - corr_syn))


def compute_cmd(real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
    columns = real.columns
    pairs = [(i, j) for i in range(len(columns)) for j in range(i + 1, len(columns))]
    scores = []

    for i, j in pairs:
        col_i = columns[i]
        col_j = columns[j]
        real_table = pd.crosstab(real[col_i], real[col_j], dropna=False, normalize=True)
        syn_table = pd.crosstab(
            synthetic[col_i], synthetic[col_j], dropna=False, normalize=True
        )

        categories_i = sorted(
            set(real[col_i].dropna().unique()).union(synthetic[col_i].dropna().unique())
        )
        categories_j = sorted(
            set(real[col_j].dropna().unique()).union(synthetic[col_j].dropna().unique())
        )

        real_extended = (
            real_table.reindex(index=categories_i, columns=categories_j, fill_value=0)
            if not real_table.empty
            else pd.DataFrame(0, index=categories_i, columns=categories_j)
        )
        syn_extended = (
            syn_table.reindex(index=categories_i, columns=categories_j, fill_value=0)
            if not syn_table.empty
            else pd.DataFrame(0, index=categories_i, columns=categories_j)
        )
        score = np.linalg.norm(real_extended.to_numpy() - syn_extended.to_numpy())
        scores.append(float(score))

    return float(np.mean(scores)) if scores else np.nan


def _one_hot(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    encoded = []
    for column in df.columns:
        unique_count = df[column].nunique(dropna=True)
        if unique_count < threshold or "proto" in column:
            one_hot = pd.get_dummies(
                df[column], prefix=f"{column}_is", prefix_sep="_", dummy_na=False
            )
            encoded.append(one_hot)
        else:
            encoded.append(df[[column]])
    return pd.concat(encoded, axis=1)


def compute_density_coverage(
    real: pd.DataFrame, synthetic: pd.DataFrame, threshold: int = 40, neighbors: int = 5
) -> Tuple[float, float]:
    merged = pd.concat([real, synthetic], axis=0).reset_index(drop=True)
    encoded = _one_hot(merged, threshold).astype(float)
    real_values = encoded.iloc[: len(real)].to_numpy()
    syn_values = encoded.iloc[len(real) :].to_numpy()

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((real_values, syn_values)))
    real_scaled = scaler.transform(real_values)
    syn_scaled = scaler.transform(syn_values)

    prdc_scores = compute_prdc(real_scaled, syn_scaled, neighbors)
    return float(prdc_scores["density"]), float(prdc_scores["coverage"])
