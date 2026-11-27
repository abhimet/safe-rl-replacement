"""Helpers shared by the tests."""
from train import train


def tiny(method, out, **kwargs):
    """A 150-step training run on a 3-map pool (seed 0), written to ``out``."""
    kwargs.setdefault("fixed_beta", 2.0 if method == "fabr_fixed" else None)
    return train(method=method, steps=150, seed=0, output=out, pool_size=3, verbose=False, **kwargs)
