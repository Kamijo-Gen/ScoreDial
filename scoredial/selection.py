"""Choosing the regressor's backbone, representation, and ridge alpha by cross-validation.

Two procedures with different jobs:
* ``nested_cv`` estimates how well the whole selection procedure generalizes: inner folds choose
  the candidate and alpha, the untouched outer fold scores the choice (5 repeats of 5 folds);
* ``select`` makes the actual choice with repeated 5-fold CV on all labels, and ``fit_head``
  refits the chosen configuration on all labels.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .regressor import BACKBONES, CANDIDATES, RIDGE_ALPHAS, extract, representation

NESTED_CV_SEEDS = tuple(range(5))
SELECTION_CV_SEEDS = tuple(range(100, 105))


def candidate_features(paths: list[str], device="cuda") -> dict[str, np.ndarray]:
    """Features of the images under every candidate, each encoder loaded once."""
    features = {}
    for family in BACKBONES:
        extracted = extract(family, paths, device=device)
        for candidate in CANDIDATES:
            candidate_family, pooling = candidate.split(":")
            if candidate_family == family:
                features[candidate] = representation(extracted, pooling)
    return {c: features[c] for c in CANDIDATES}


def r2(y, prediction) -> float:
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    return float(1.0 - np.square(y - prediction).sum() / (np.square(y - y.mean()).sum() + 1e-12))


def ridge_predict(X_train, y_train, X_test, alpha):
    # float64: with more feature dimensions than samples, float32 normal equations can overflow.
    X_train, X_test = np.asarray(X_train, np.float64), np.asarray(X_test, np.float64)
    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=alpha).fit(scaler.transform(X_train), np.asarray(y_train, np.float64))
    return model.predict(scaler.transform(X_test))


def cv_r2(X, y, alpha, splits) -> float:
    """R^2 of out-of-fold predictions over the samples the splits test."""
    prediction = np.full(len(y), np.nan)
    tested = set()
    for train, test in splits:
        prediction[test] = ridge_predict(X[train], y[train], X[test], alpha)
        tested.update(np.asarray(test).tolist())
    tested = np.asarray(sorted(tested))
    return r2(y[tested], prediction[tested])


def nested_cv(features: dict, y: np.ndarray) -> dict:
    configs = [(c, a) for c in CANDIDATES for a in RIDGE_ALPHAS]
    scores, choices = [], []
    for repeat in NESTED_CV_SEEDS:
        prediction = np.empty(len(y))
        outer = KFold(5, shuffle=True, random_state=repeat).split(np.arange(len(y)))
        for fold, (train, test) in enumerate(outer):
            inner_splits = KFold(5, shuffle=True, random_state=1000 + repeat * 10 + fold).split(np.arange(len(train)))
            inner = [(train[a], train[b]) for a, b in inner_splits]
            scored = [(cv_r2(features[c], y, a, inner), c, a) for c, a in configs]
            # Highest R^2; ties go to the earlier configuration.
            _, candidate, alpha = max(scored, key=lambda s: (s[0], -configs.index((s[1], s[2]))))
            choices.append(dict(repeat=repeat, fold=fold, candidate=candidate, alpha=alpha))
            prediction[test] = ridge_predict(features[candidate][train], y[train], features[candidate][test], alpha)
        scores.append(r2(y, prediction))
    return dict(r2_mean=float(np.mean(scores)), r2_std=float(np.std(scores)), r2_repeats=scores,
                outer_choices=choices)


def select(features: dict, y: np.ndarray) -> dict:
    grid = []
    for candidate in CANDIDATES:
        for alpha in RIDGE_ALPHAS:
            scores = [cv_r2(features[candidate], y, alpha,
                            list(KFold(5, shuffle=True, random_state=seed).split(np.arange(len(y)))))
                      for seed in SELECTION_CV_SEEDS]
            grid.append(dict(candidate=candidate, alpha=alpha, r2_mean=float(np.mean(scores)),
                             r2_std=float(np.std(scores))))
    # Highest mean R^2; ties go to the earlier candidate, then the smaller alpha.
    winner = max(grid, key=lambda r: (r["r2_mean"], -CANDIDATES.index(r["candidate"]),
                                      -RIDGE_ALPHAS.index(r["alpha"])))
    return dict(winner=winner, grid=grid)


def fit_head(features: dict, y: np.ndarray, candidate: str, alpha: float) -> dict:
    """Refit one configuration on all labels; returns the head that ``Regressor`` loads."""
    X = np.asarray(features[candidate], np.float64)
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=alpha).fit(scaler.transform(X), np.asarray(y, np.float64))
    family, pooling = candidate.split(":")
    return dict(backbone=family, pooling=pooling, alpha=float(alpha),
                feat_mean=scaler.mean_.astype(np.float32), feat_scale=scaler.scale_.astype(np.float32),
                weight=model.coef_.astype(np.float32), bias=float(model.intercept_))
