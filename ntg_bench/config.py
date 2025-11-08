from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


def _default_mixing_rates() -> List[float]:
    return [round(x * 0.1, 1) for x in range(11)]


@dataclass
class BenchmarkConfig:
    """
    Declarative configuration used by the benchmark runner.

    Attributes mirror the assumptions made in the accompanying paper while
    keeping the structure flexible enough for other datasets.
    """

    base_dir: str
    models: Sequence[str]
    classes: Sequence[str]
    continuous_features: Sequence[str]
    categorical_features: Sequence[str]

    label_column: str = "label"
    real_data_name: str = "real"
    real_train_subdir: str = "real_train"
    real_test_subdir: str = "real_test"
    output_dir: str = "outputs"
    random_state: int = 42

    mixing_rates: Sequence[float] = field(default_factory=_default_mixing_rates)
    augment_strategy: str = "self"  # one of: self, mean, median
    augment_target_count: int = 16000

    def __post_init__(self) -> None:
        strategy = self.augment_strategy.lower()
        if strategy not in {"self", "mean", "median"}:
            raise ValueError(
                "augment_strategy must be one of {'self', 'mean', 'median'}"
            )
        self.augment_strategy = strategy

        if not self.models:
            raise ValueError("At least one model must be provided.")
        if not self.classes:
            raise ValueError("At least one traffic class must be provided.")
        if not (self.continuous_features or self.categorical_features):
            raise ValueError(
                "Provide at least one continuous or categorical feature."
            )

    @classmethod
    def from_json(cls, path: Path | str) -> "BenchmarkConfig":
        loaded = json.loads(Path(path).read_text())
        return cls(**loaded)

    def to_dict(self) -> Dict[str, object]:
        return {
            "base_dir": self.base_dir,
            "models": list(self.models),
            "classes": list(self.classes),
            "continuous_features": list(self.continuous_features),
            "categorical_features": list(self.categorical_features),
            "label_column": self.label_column,
            "real_data_name": self.real_data_name,
            "real_train_subdir": self.real_train_subdir,
            "real_test_subdir": self.real_test_subdir,
            "output_dir": self.output_dir,
            "random_state": self.random_state,
            "mixing_rates": list(self.mixing_rates),
            "augment_strategy": self.augment_strategy,
            "augment_target_count": self.augment_target_count,
        }
