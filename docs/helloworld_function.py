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
        {"weight": 5.0,  "value": 40.0}
    ]
    capacity = 50.0

    # Sort items using the evolved heuristic
    try:
        items.sort(key=lambda x: heuristic_score(x["weight"], x["value"]), reverse=True)
    except Exception:
        # Penalize code that throws syntax or runtime errors
        return {"score": 0.0} 

    total_value = 0.0
    current_weight = 0.0

    # Simulate packing the knapsack
    for item in items:
        if current_weight + item["weight"] <= capacity:
            current_weight += item["weight"]
            total_value += item["value"]

    # Naive score: 220.0
    # Optimal score (value/weight): 275.0
    return {"score": total_value}

# Entry point for your Docker Sandbox to execute and print the JSON result
if __name__ == "__main__":
    import json
    print(json.dumps(evaluate()))