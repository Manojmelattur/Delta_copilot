# =============================================================================
# strategies/__init__.py - Strategy Registry
# =============================================================================

from strategies.bollinger_bands import BollingerBandsStrategy
from strategies.ema_crossover import EMACrossoverStrategy
from strategies.fv_strategy import FVGStrategy
from strategies.ob_strategy import OrderBlockStrategy
from strategies.smc_strategy import SMCStrategy
from strategies.triple_ema import TripleEMAStrategy
from strategies.triple_ema_vwap import TripleEMAVWAPStrategy
from strategies.triple_ema_vwap_macd_tsl import TripleEmaVwapMacdTsl
from strategies.vwap import VWAPStrategy

STRATEGY_REGISTRY = {
    "ema_crossover": EMACrossoverStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "vwap": VWAPStrategy,
    "triple_ema": TripleEMAStrategy,
    "triple_ema_vwap": TripleEMAVWAPStrategy,
    "triple_ema_vwap_macd_tsl": TripleEmaVwapMacdTsl,
    "fvg": FVGStrategy,
    "ob": OrderBlockStrategy,
    "smc": SMCStrategy,
}


def get_strategy(name: str, **kwargs):
    """
    Instantiate a strategy by name.

    Args:
        name:     Strategy key from STRATEGY_REGISTRY
        **kwargs: Strategy-specific parameters

    Returns:
        Strategy instance

    Raises:
        ValueError if strategy name not found
    """
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_REGISTRY[name](**kwargs)
