from typing import Dict


class PriceDeducer:
    """
    Deduces a single, actionable price from the results of a PrimalDualEngine.

    This class takes the lower and upper price bounds and calculates the midpoint,
    which serves as the best estimate for the true option price. It also quantifies
    the model's uncertainty as half the duality gap.
    """

    def deduce(self, engine_results: Dict[str, float]) -> Dict[str, float]:
        """
        Calculates the midpoint price and its associated uncertainty.

        Args:
            engine_results (Dict[str, float]): The output dictionary from a
                                               PrimalDualEngine run, containing
                                               'lower_bound' and 'upper_bound'.

        Returns:
            A dictionary containing the 'deduced_price' and its 'uncertainty'.
        """
        lower_bound = engine_results.get("lower_bound", 0.0)
        upper_bound = engine_results.get("upper_bound", 0.0)

        # The best estimate for the price is the midpoint of the bounds.
        deduced_price = (lower_bound + upper_bound) / 2.0

        # The uncertainty is the radius of the price interval.
        uncertainty = (upper_bound - lower_bound) / 2.0

        return {
            "deduced_price": deduced_price,
            "uncertainty": uncertainty
        }