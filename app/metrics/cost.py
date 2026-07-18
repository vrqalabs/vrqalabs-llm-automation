class CostCalculator:
    """
    Calculates estimated request cost.
    """

    @staticmethod
    def estimate_cost(
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        return 0.0
