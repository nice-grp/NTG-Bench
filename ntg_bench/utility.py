from __future__ import annotations

import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

LOGGER = logging.getLogger(__name__)

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency
    xgboost_available = False
    xgb = None
    LOGGER.warning(
        "xgboost is not installed; XGB results will be unavailable. "
        "Install it via `pip install xgboost` to run the full benchmark."
    )
else:
    xgboost_available = True


def get_default_models(random_state: int) -> Dict[str, object]:
    models = {
        "DT": DecisionTreeClassifier(random_state=random_state),
        "GB": GradientBoostingClassifier(random_state=random_state),
    }
    if xgboost_available:
        models["XGB"] = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="auto",
        )
    return models


def _prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous: Iterable[str],
    categorical: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray]:
    continuous = list(continuous)
    categorical = list(categorical)

    feature_blocks_train = []
    feature_blocks_test = []

    if continuous:
        scaler = StandardScaler()
        feature_blocks_train.append(scaler.fit_transform(X_train[continuous]))
        feature_blocks_test.append(scaler.transform(X_test[continuous]))

    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        feature_blocks_train.append(encoder.fit_transform(X_train[categorical]))
        feature_blocks_test.append(encoder.transform(X_test[categorical]))

    if feature_blocks_train:
        train_matrix = np.hstack(feature_blocks_train)
        test_matrix = np.hstack(feature_blocks_test)
    else:
        train_matrix = np.empty((len(X_train), 0))
        test_matrix = np.empty((len(X_test), 0))

    return train_matrix, test_matrix


def evaluate_classifier(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    continuous: Iterable[str],
    categorical: Iterable[str],
) -> Dict[str, float]:
    LOGGER.debug("Training %s", model_name)
    train_matrix, test_matrix = _prepare_features(
        X_train, X_test, continuous, categorical
    )
    model.fit(train_matrix, y_train)
    LOGGER.debug("Evaluating %s", model_name)
    predictions = model.predict(test_matrix)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="macro", zero_division=0),
    }
