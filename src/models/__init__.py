"""
Model Registry
===============
Factory function to create models by name.

LiquidityML v3 model set: Elastic-Net, XGBoost, Neural Network.
"""

from __future__ import annotations
from typing import Any

from src.models.base import BaseReturnPredictor


MODEL_REGISTRY = {
    "elastic_net": "src.models.elastic_net_model.ElasticNetPredictor",
    "xgboost": "src.models.xgboost_model.XGBoostPredictor",
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
    name : one of 'elastic_net', 'xgboost', 'neural_network'
    config : optional override config (merged with defaults from config.yaml)
    seed : random seed for reproducibility
    """
    if name == "elastic_net":
        from src.models.elastic_net_model import ElasticNetPredictor
        return ElasticNetPredictor(config=config, seed=seed)
    elif name == "xgboost":
        from src.models.xgboost_model import XGBoostPredictor
        return XGBoostPredictor(config=config, seed=seed)
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
