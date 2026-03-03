"""
Model Registry
===============
Factory function to create models by name.
"""

from __future__ import annotations
from typing import Any

from src.models.base import BaseReturnPredictor


MODEL_REGISTRY = {
    "xgboost": "src.models.xgboost_model.XGBoostPredictor",
    "random_forest": "src.models.random_forest_model.RandomForestPredictor",
    "neural_network": "src.models.neural_network_model.NeuralNetPredictor",
}


def create_model(
    name: str,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> BaseReturnPredictor:
    """Create a model by name.

    Parameters
    ----------
    name : one of 'xgboost', 'random_forest', 'neural_network'
    config : optional override config (merged with defaults from config.yaml)
    seed : random seed
    """
    if name == "xgboost":
        from src.models.xgboost_model import XGBoostPredictor
        return XGBoostPredictor(config=config, seed=seed)
    elif name == "random_forest":
        from src.models.random_forest_model import RandomForestPredictor
        return RandomForestPredictor(config=config, seed=seed)
    elif name == "neural_network":
        from src.models.neural_network_model import NeuralNetPredictor
        return NeuralNetPredictor(config=config, seed=seed)
    else:
        raise ValueError(
            f"Unknown model: {name!r}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )


def get_all_model_names() -> list[str]:
    return list(MODEL_REGISTRY.keys())
