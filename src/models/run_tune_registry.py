from __future__ import annotations

from typing import Any

import optuna

from tabpfn import TabPFNClassifier

from pytabkit import (
    LGBM_TD_Classifier,
    XGB_TD_Classifier,
    CatBoost_TD_Classifier,
)

# ============================================================
# TABPFN
# ============================================================


def build_tabpfn(params: dict[str, Any], seed: int = 42):
    return TabPFNClassifier(
        random_state=seed,
        device="auto",
        **params,
    )


def tabpfn_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_categorical(
            "n_estimators",
            [8, 16, 32],
        ),
    }


# ============================================================
# LIGHTGBM
# ============================================================


def build_lightgbm(params: dict[str, Any], seed: int = 42):
    return LGBM_TD_Classifier(
        random_state=seed,
        verbose = -1,
        **params,
    )


def lightgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float(
            "learning_rate",
            1e-3,
            0.3,
            log=True,
        ),
        "num_leaves": trial.suggest_int(
            "num_leaves",
            16,
            256,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            12,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
    }


# ============================================================
# XGBOOST
# ============================================================


def build_xgboost(params: dict[str, Any], seed: int = 42):
    return XGB_TD_Classifier(
        random_state=seed,
        **params,
    )


def xgboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float(
            "lr",
            1e-3,
            0.3,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            12,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.5,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.5,
            1.0,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1e-3,
            10.0,
            log=True,
        ),
    }


# ============================================================
# CATBOOST
# ============================================================


def build_catboost(params: dict[str, Any], seed: int = 42):
    return CatBoost_TD_Classifier(
        random_state=seed,
        **params,
    )


def catboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "lr": trial.suggest_float(
            "learning_rate",
            1e-3,
            0.3,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "depth",
            4,
            10,
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg",
            1e-3,
            10.0,
            log=True,
        ),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature",
            0.0,
            10.0,
        ),
    }


# ============================================================
# REGISTRY
# ============================================================


RUN_TUNE_REGISTRY = {
    "tabpfn": {
        "factory": build_tabpfn,
        "search_space": tabpfn_search_space,
    },
    "lightgbm": {
        "factory": build_lightgbm,
        "search_space": lightgbm_search_space,
    },
    "xgboost": {
        "factory": build_xgboost,
        "search_space": xgboost_search_space,
    },
    "catboost": {
        "factory": build_catboost,
        "search_space": catboost_search_space,
    },
}
