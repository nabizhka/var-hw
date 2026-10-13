"""Shared helpers: frozen data loading and the empirical VaR / ES estimators."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")


def load_close(name):
    """Frozen Yahoo Finance closes (downloaded 2026-10-12, see README)."""
    return pd.read_csv(os.path.join(DATA, name), index_col=0, parse_dates=True)


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def var_emp(losses, alpha):
    """VaR_alpha(F_n) = l_(ceil(n alpha)), the lecture's order-statistic estimator."""
    l = np.sort(np.asarray(losses))
    k = int(np.ceil(len(l) * alpha - 1e-12))   # guard against n*alpha = 474.99999...
    return l[k - 1]


def es_emp(losses, alpha):
    """ES_alpha(F_n) with the lecture's correction term for non-integer n alpha."""
    l = np.sort(np.asarray(losses))
    n = len(l)
    k = int(np.ceil(n * alpha - 1e-12))
    return (l[k:].sum() + (k - n * alpha) * l[k - 1]) / (n * (1 - alpha))
