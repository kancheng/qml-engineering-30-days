"""Train-only 4D representations feeding the same two-qubit readout."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from articles.day20.iris_models import load_data, split_data, fit_preprocessor, preprocess
from articles.day11.encoding import fit_scaler, transform
from articles.day14.model import Config, predict_batch, reference_prediction
from articles.day15.classifier import metrics, probabilities

COUNTS = {'pca_vqc': 4, 'selection_vqc': 4, 'bottleneck_vqc': 14, 'logistic4': 5}


def matrix(raw):
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or not len(x) or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError('finite nonempty Nx4 input required')
    return x


def fit(raw, labels):
    x = matrix(raw)
    y = np.asarray(labels, dtype=float)
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('both binary classes required for feature selection')
    pca = fit_preprocessor(x)
    z = (x - pca['mean']) / pca['std']
    # Absolute point-biserial (Pearson) correlation; lower column index breaks ties.
    corr = np.mean(z * ((y - y.mean()) / y.std())[:, None], axis=0)
    indices = sorted(range(4), key=lambda j: (-abs(corr[j]), j))[:2]
    return {'pca': pca, 'selected_indices': indices, 'train_correlations': corr.tolist(),
            'selection_scaler': fit_scaler(z[:, indices])}


def representation(model, raw, prep, weights):
    x = matrix(raw)
    if model not in COUNTS:
        raise ValueError('unknown model')
    w = np.asarray(weights, dtype=float)
    if w.shape != (COUNTS[model],) or not np.isfinite(w).all():
        raise ValueError('invalid weights')
    z = (x - prep['pca']['mean']) / prep['pca']['std']
    if model == 'pca_vqc':
        return preprocess(x, prep['pca'])
    if model == 'selection_vqc':
        return transform(z[:, prep['selected_indices']], prep['selection_scaler'])
    if model == 'bottleneck_vqc':
        h = np.tanh(z @ w[:8].reshape(4, 2) + w[8:10])
        return h, np.zeros_like(h, dtype=bool)
    return z, np.zeros_like(z, dtype=bool)


def forward(model, raw, prep, weights, engine='numpy'):
    if engine not in ('numpy', 'cudaq'):
        raise ValueError('unknown engine')
    x, _ = representation(model, raw, prep, weights)
    w = np.asarray(weights, dtype=float)
    if model == 'logistic4':
        return .5 * (1 + np.tanh((x @ w[:4] + w[4]) / 2))
    theta = w[-4:]
    expectation = (predict_batch(x, theta, Config()) if engine == 'cudaq' else
                   [reference_prediction(row, theta, Config()) for row in x])
    return probabilities(expectation)
