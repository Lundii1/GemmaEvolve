# EVOLVE-BLOCK-START
def heuristic_score(weight: float, value: float) -> float:
    """
    Assign a priority score to an item based on its weight and value.
    Higher scores will be packed into the knapsack first.
    """
    # Initial naive implementation: prioritize purely by value
    return value
# EVOLVE-BLOCK-END


def evaluate() -> dict[str, float]:
    """Evaluates the heuristic against a simple knapsack environment."""
    items = [
        {"weight": 10.0, "value": 60.0},
        {"weight": 20.0, "value": 100.0},
        {"weight": 30.0, "value": 120.0},
        {"weight": 15.0, "value": 75.0},
        {"weight": 5.0, "value": 40.0},
    ]
    capacity = 50.0

    try:
        items.sort(key=lambda x: heuristic_score(x["weight"], x["value"]), reverse=True)
    except Exception:
        return {"score": 0.0}

    total_value = 0.0
    current_weight = 0.0

    for item in items:
        if current_weight + item["weight"] <= capacity:
            current_weight += item["weight"]
            total_value += item["value"]

    return {"score": total_value}


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
