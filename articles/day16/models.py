"""Inspect a 2-2-1 NumPy MLP beside the existing two-qubit model."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import cudaq
from articles.day14.model import Config,initialize,predict,reference_prediction
from articles.day15.classifier import probabilities

CONFIG=Config('angle',1)


def mlp_initialize(seed=42):
    return np.random.default_rng(seed).normal(0,.2,9)


def mlp_forward(features,weights):
    x=np.asarray(features,dtype=float);w=np.asarray(weights,dtype=float)
    if x.ndim!=2 or x.shape[0]==0 or x.shape[1]!=2 or not np.isfinite(x).all():
        raise ValueError('expected a nonempty finite batch with two features')
    if w.shape!=(9,) or not np.isfinite(w).all():raise ValueError('expected nine finite MLP parameters')
    hidden=np.tanh(x@w[:4].reshape(2,2)+w[4:6])
    logits=hidden@w[6:8]+w[8]
    return .5*(1+np.tanh(logits/2))


def qnn_probability(features,weights,reference=False):
    fn=reference_prediction if reference else predict
    return float(probabilities([fn(features,weights,CONFIG)])[0])
