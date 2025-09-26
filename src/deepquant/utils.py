import torch
from typing import Callable

def payoff_factory(option_type: str, strike: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a payoff function for a given option type and strike.

    Args:
        option_type: 'put' or 'call'.
        strike: The strike price.

    Returns:
        A callable payoff function.
    """
    if option_type.lower() == 'put':
        return lambda S: torch.clamp(strike - S, min=0)
    elif option_type.lower() == 'call':
        return lambda S: torch.clamp(S - strike, min=0)
    else:
        raise ValueError("option_type must be 'put' or 'call'")