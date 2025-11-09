from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import BenchmarkConfig
from .data import ensure_directory, load_and_label_data
from .metrics import (
    compute_cmd,
    compute_coverage,
    compute_emd,
    compute_jsd,
    compute_ks,
    compute_pcd,
    compute_tvd,
)
from .protocol import compute_protocol_compliance
from .utility import evaluate_classifier, get_default_models

LOGGER = logging.getLogger(__name__)


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.real_dir = self.base_dir / config.real_data_name
        self.output_dir = Path(config.output_dir)
        ensure_directory(self.output_dir)

    def run(self) -> None:
        LOGGER.info("Running NTG-Bench with config: %s", self.config.to_dict())
        dist_df, feature_df = self._run_distribution_metrics()
        dist_path = self.output_dir / "distribution_metrics.csv"
        feature_path = self.output_dir / "feature_similarity.csv"
        dist_df.to_csv(dist_path)
        feature_df.to_csv(feature_path)
        LOGGER.info("Saved distribution metrics to %s", dist_path)
        LOGGER.info("Saved feature similarity metrics to %s", feature_path)

        baseline_df, tstr_df = self._run_task_utility()
        baseline_path = self.output_dir / "baseline_scores.csv"
        tstr_path = self.output_dir / "tstr_scores.csv"
        baseline_df.to_csv(baseline_path)
        tstr_df.to_csv(tstr_path)
        LOGGER.info("Saved baseline utility scores to %s", baseline_path)
        LOGGER.info("Saved task-specific synthetic scores to %s", tstr_path)

        mix_df = self._evaluate_mixed_training()
        mix_path = self.output_dir / "mixing_curve.csv"
        mix_df.to_csv(mix_path)
        LOGGER.info("Saved mixing results to %s", mix_path)

        aug_df = self._evaluate_augmentation()
        aug_path = self.output_dir / "augmentation_scores.csv"
        aug_df.to_csv(aug_path)
        LOGGER.info("Saved augmentation scores to %s", aug_path)

    def _real_class_path(self, class_name: str) -> Path:
        return self.real_dir / f"{class_name}.csv"

    def _synthetic_class_path(self, model_name: str, class_name: str) -> Path:
        return self.base_dir / model_name / f"{class_name}.csv"

    def _cleaned_synthetic_path(self, model_name: str, class_name: str) -> Path:
        return self.base_dir / model_name / f"{class_name}_cleaned.csv"

    def _run_distribution_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_results = []
        feature_similarity: Dict[str, Dict[str, float]] = {}

        for model_name in self.config.models:
            LOGGER.info("Evaluating distribution metrics for %s", model_name)
            per_metric_scores: Dict[str, List[float]] = {
                "JSD": [],
                "EMD": [],
                "TVD": [],
                "KS": [],
                "CMD": [],
                "PCD": [],
                "Coverage": [],
                "DKC": [],
            }
            per_feature_details: List[Dict[str, float]] = []

            for class_name in self.config.classes:
                real_path = self._real_class_path(class_name)
                synthetic_path = self._synthetic_class_path(model_name, class_name)

                if not real_path.exists() or not synthetic_path.exists():
                    LOGGER.warning(
                        "Skipping class '%s' for '%s' (missing files).",
                        class_name,
                        model_name,
                    )
                    continue

                real_df = pd.read_csv(real_path)
                synthetic_df = pd.read_csv(synthetic_path)

                required_columns = set(
                    self.config.continuous_features
                ).union(self.config.categorical_features)
                if not required_columns.issubset(real_df.columns) or not required_columns.issubset(
                    synthetic_df.columns
                ):
                    LOGGER.warning(
                        "Skipping class '%s' for '%s' due to missing columns.",
                        class_name,
                        model_name,
                    )
                    continue

                protocol_result = compute_protocol_compliance(synthetic_df)
                per_metric_scores["DKC"].append(protocol_result.pass_ratio)
                cleaned_path = self._cleaned_synthetic_path(model_name, class_name)
                protocol_result.cleaned_df.to_csv(cleaned_path, index=False)

                real_con = real_df[self.config.continuous_features]
                syn_con = protocol_result.cleaned_df[self.config.continuous_features]
                real_cat = real_df[self.config.categorical_features]
                syn_cat = protocol_result.cleaned_df[self.config.categorical_features]
                real_all = real_df[list(required_columns)]
                syn_all = protocol_result.cleaned_df[list(required_columns)]

                jsd_result = compute_jsd(real_cat, syn_cat)
                tvd_result = compute_tvd(real_cat, syn_cat)
                per_metric_scores["JSD"].append(jsd_result.value)
                per_metric_scores["TVD"].append(tvd_result.value)

                if self.config.continuous_features:
                    emd_result = compute_emd(real_con, syn_con)
                    ks_result = compute_ks(real_con, syn_con)
                    per_metric_scores["EMD"].append(emd_result.value)
                    per_metric_scores["KS"].append(ks_result.value)
                    per_feature_details.append(
                        {
                            **jsd_result.per_feature,
                            **tvd_result.per_feature,
                            **emd_result.per_feature,
                            **ks_result.per_feature,
                        }
                    )
                    per_metric_scores["PCD"].append(compute_pcd(real_con, syn_con))
                else:
                    per_feature_details.append(
                        {**jsd_result.per_feature, **tvd_result.per_feature}
                    )

                per_metric_scores["CMD"].append(compute_cmd(real_cat, syn_cat))
                coverage = compute_coverage(real_all, syn_all)
                per_metric_scores["Coverage"].append(coverage)

            summary = {"Model": model_name}
            for metric, scores in per_metric_scores.items():
                summary[metric] = float(np.nanmean(scores)) if scores else np.nan
            all_results.append(summary)

            feature_df = pd.DataFrame(per_feature_details)
            feature_similarity[model_name] = (
                feature_df.mean().to_dict() if not feature_df.empty else {}
            )

        distribution_df = pd.DataFrame(all_results).set_index("Model")
        feature_df = pd.DataFrame(feature_similarity).T
        return distribution_df, feature_df

    def _load_real_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_path = self.real_dir / self.config.real_train_subdir
        test_path = self.real_dir / self.config.real_test_subdir
        train_df = load_and_label_data(train_path, self.config.classes, self.config.label_column)
        test_df = load_and_label_data(test_path, self.config.classes, self.config.label_column)
        if train_df.empty or test_df.empty:
            raise FileNotFoundError("Real training or testing data is missing.")
        return train_df, test_df

    def _run_task_utility(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        real_train_df, real_test_df = self._load_real_datasets()
        for col in self.config.categorical_features:
            if col in real_train_df.columns:
                real_train_df[col] = real_train_df[col].astype(str)
            if col in real_test_df.columns:
                real_test_df[col] = real_test_df[col].astype(str)
        label_encoder = LabelEncoder()
        label_encoder.fit(
            pd.concat(
                [real_train_df[self.config.label_column], real_test_df[self.config.label_column]],
                ignore_index=True,
            )
        )

        X_real_train = real_train_df.drop(columns=[self.config.label_column])
        X_real_test = real_test_df.drop(columns=[self.config.label_column])
        y_real_train = label_encoder.transform(real_train_df[self.config.label_column])
        y_real_test = label_encoder.transform(real_test_df[self.config.label_column])

        ml_models = get_default_models(self.config.random_state)
        baseline_results: Dict[str, Dict[str, float]] = {}
        tstr_df = pd.DataFrame(index=["real"] + list(self.config.models))

        for name, model in ml_models.items():
            metrics = evaluate_classifier(
                model,
                name,
                X_real_train,
                y_real_train,
                X_real_test,
                y_real_test,
                self.config.continuous_features,
                self.config.categorical_features,
            )
            baseline_results[name] = metrics
            tstr_df.loc["real", f"{name}_F1 Score"] = metrics["F1 Score"]

        for model_name in self.config.models:
            syn_base = self.base_dir / model_name
            syn_df = load_and_label_data(
                syn_base, self.config.classes, self.config.label_column
            )
            if syn_df.empty:
                LOGGER.warning("Skipping TSTR evaluation for %s (no synthetic data).", model_name)
                continue
            sample_size = min(len(real_train_df), len(syn_df))
            gen_train_df = syn_df.sample(n=sample_size, random_state=self.config.random_state)
            X_gen_train = gen_train_df.drop(columns=[self.config.label_column])
            y_gen_train = label_encoder.transform(gen_train_df[self.config.label_column])

            for name, model in ml_models.items():
                metrics = evaluate_classifier(
                    model,
                    name,
                    X_gen_train,
                    y_gen_train,
                    X_real_test,
                    y_real_test,
                    self.config.continuous_features,
                    self.config.categorical_features,
                )
                tstr_df.loc[model_name, f"{name}_F1 Score"] = metrics["F1 Score"]

        baseline_df = pd.DataFrame(baseline_results).T
        return baseline_df, tstr_df

    def _evaluate_mixed_training(self) -> pd.DataFrame:
        real_train_df, real_test_df = self._load_real_datasets()
        label_encoder = LabelEncoder()
        label_encoder.fit(real_train_df[self.config.label_column])

        for col in self.config.categorical_features:
            if col in real_train_df.columns:
                real_train_df[col] = real_train_df[col].astype(str)
            if col in real_test_df.columns:
                real_test_df[col] = real_test_df[col].astype(str)

        X_real_test = real_test_df.drop(columns=[self.config.label_column])
        y_real_test = label_encoder.transform(real_test_df[self.config.label_column])
        ml_models = get_default_models(self.config.random_state)

        results = []
        for model_name in self.config.models:
            syn_base = self.base_dir / model_name
            syn_df = load_and_label_data(
                syn_base, self.config.classes, self.config.label_column
            )
            if syn_df.empty:
                continue

            for col in self.config.categorical_features:
                if col in syn_df.columns:
                    syn_df[col] = syn_df[col].astype(str)

            for rate in self.config.mixing_rates:
                mixed_frames = []
                for class_label in real_train_df[self.config.label_column].unique():
                    real_class = real_train_df[
                        real_train_df[self.config.label_column] == class_label
                    ]
                    syn_class = syn_df[syn_df[self.config.label_column] == class_label]
                    n_total = len(real_class)
                    n_syn = min(int(n_total * rate), len(syn_class))
                    n_real = n_total - n_syn

                    sampled_real = (
                        real_class.sample(n=n_real, random_state=self.config.random_state)
                        if n_real > 0
                        else pd.DataFrame(columns=real_class.columns)
                    )
                    sampled_syn = (
                        syn_class.sample(n=n_syn, random_state=self.config.random_state)
                        if n_syn > 0
                        else pd.DataFrame(columns=syn_class.columns)
                    )
                    mixed_frames.append(pd.concat([sampled_real, sampled_syn], ignore_index=True))

                mixed_df = pd.concat(mixed_frames, ignore_index=True)
                X_mixed = mixed_df.drop(columns=[self.config.label_column])
                y_mixed = label_encoder.transform(mixed_df[self.config.label_column])

                for name, model in ml_models.items():
                    metrics = evaluate_classifier(
                        model,
                        name,
                        X_mixed,
                        y_mixed,
                        X_real_test,
                        y_real_test,
                        self.config.continuous_features,
                        self.config.categorical_features,
                    )
                    results.append(
                        {
                            "Model": model_name,
                            "ML_model": name,
                            "Mixing_rate": rate,
                            "F1_Score": metrics["F1 Score"],
                        }
                    )

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.pivot_table(
            index="Mixing_rate", columns=["Model", "ML_model"], values="F1_Score"
        )

    def _evaluate_augmentation(self) -> pd.DataFrame:
        real_train_df, real_test_df = self._load_real_datasets()
        label_encoder = LabelEncoder()
        label_encoder.fit(real_train_df[self.config.label_column])
        ml_models = get_default_models(self.config.random_state)

        for col in self.config.categorical_features:
            if col in real_train_df.columns:
                real_train_df[col] = real_train_df[col].astype(str)
            if col in real_test_df.columns:
                real_test_df[col] = real_test_df[col].astype(str)

        X_real_test = real_test_df.drop(columns=[self.config.label_column])
        y_real_test = label_encoder.transform(real_test_df[self.config.label_column])

        class_counts = real_train_df[self.config.label_column].value_counts()
        results = []

        if self.config.augment_strategy == "self":
            target_count = self.config.augment_target_count
        elif self.config.augment_strategy == "mean":
            target_count = int(class_counts.mean())
        else:  # median
            target_count = int(class_counts.median())

        for model_name in self.config.models:
            syn_base = self.base_dir / model_name
            syn_df = load_and_label_data(
                syn_base, self.config.classes, self.config.label_column
            )
            if syn_df.empty:
                continue

            for col in self.config.categorical_features:
                if col in syn_df.columns:
                    syn_df[col] = syn_df[col].astype(str)

            augmented_frames = [real_train_df]
            for class_label, current_count in class_counts.items():
                if current_count >= target_count:
                    continue
                needed = target_count - current_count
                syn_class = syn_df[syn_df[self.config.label_column] == class_label]
                if syn_class.empty:
                    continue
                samples = syn_class.sample(
                    n=needed, random_state=self.config.random_state, replace=True
                )
                augmented_frames.append(samples)

            augmented_df = pd.concat(augmented_frames, ignore_index=True)
            X_aug = augmented_df.drop(columns=[self.config.label_column])
            y_aug = label_encoder.transform(augmented_df[self.config.label_column])

            for name, model in ml_models.items():
                metrics = evaluate_classifier(
                    model,
                    name,
                    X_aug,
                    y_aug,
                    X_real_test,
                    y_real_test,
                    self.config.continuous_features,
                    self.config.categorical_features,
                )
                results.append(
                    {
                        "Model": model_name,
                        "ML_model": name,
                        "F1_Score": metrics["F1 Score"],
                    }
                )

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df.pivot_table(columns=["Model", "ML_model"], values="F1_Score")
