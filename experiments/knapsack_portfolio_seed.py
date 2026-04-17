# EVOLVE-BLOCK-START
def select_items(items: list[dict[str, int]], capacity: int) -> list[int]:
    """
    Return a list of item indices to pack for a single 0/1 knapsack instance.

    The baseline is intentionally simple: sort by raw value and pack greedily.
    Better solutions can use dynamic programming, branch-and-bound, repair
    passes, or any other deterministic algorithm that returns valid indices.
    """
    order = sorted(range(len(items)), key=lambda idx: items[idx]["value"], reverse=True)
    chosen: list[int] = []
    remaining = capacity
    for idx in order:
        weight = items[idx]["weight"]
        if weight <= remaining:
            chosen.append(idx)
            remaining -= weight
    return chosen
# EVOLVE-BLOCK-END


CASES = (
    {
        "capacity": 50,
        "items": (
            {"weight": 10, "value": 60},
            {"weight": 20, "value": 100},
            {"weight": 30, "value": 120},
            {"weight": 15, "value": 75},
            {"weight": 5, "value": 40},
            {"weight": 24, "value": 90},
            {"weight": 18, "value": 70},
            {"weight": 12, "value": 65},
            {"weight": 7, "value": 45},
            {"weight": 9, "value": 48},
        ),
    },
    {
        "capacity": 60,
        "items": (
            {"weight": 11, "value": 42},
            {"weight": 23, "value": 85},
            {"weight": 17, "value": 68},
            {"weight": 29, "value": 115},
            {"weight": 7, "value": 34},
            {"weight": 13, "value": 51},
            {"weight": 19, "value": 79},
            {"weight": 31, "value": 128},
            {"weight": 5, "value": 23},
            {"weight": 27, "value": 109},
        ),
    },
    {
        "capacity": 75,
        "items": (
            {"weight": 14, "value": 64},
            {"weight": 26, "value": 112},
            {"weight": 33, "value": 146},
            {"weight": 9, "value": 40},
            {"weight": 18, "value": 83},
            {"weight": 21, "value": 89},
            {"weight": 39, "value": 171},
            {"weight": 12, "value": 56},
            {"weight": 16, "value": 72},
            {"weight": 28, "value": 120},
            {"weight": 7, "value": 31},
        ),
    },
    {
        "capacity": 90,
        "items": (
            {"weight": 22, "value": 94},
            {"weight": 35, "value": 143},
            {"weight": 41, "value": 172},
            {"weight": 16, "value": 66},
            {"weight": 27, "value": 117},
            {"weight": 13, "value": 59},
            {"weight": 19, "value": 81},
            {"weight": 24, "value": 104},
            {"weight": 31, "value": 133},
            {"weight": 8, "value": 37},
            {"weight": 11, "value": 50},
        ),
    },
    {
        "capacity": 55,
        "items": (
            {"weight": 6, "value": 18},
            {"weight": 14, "value": 52},
            {"weight": 25, "value": 98},
            {"weight": 19, "value": 77},
            {"weight": 11, "value": 44},
            {"weight": 17, "value": 69},
            {"weight": 21, "value": 86},
            {"weight": 9, "value": 38},
            {"weight": 13, "value": 53},
            {"weight": 28, "value": 108},
            {"weight": 4, "value": 17},
        ),
    },
    {
        "capacity": 80,
        "items": (
            {"weight": 15, "value": 71},
            {"weight": 34, "value": 149},
            {"weight": 27, "value": 118},
            {"weight": 12, "value": 53},
            {"weight": 18, "value": 84},
            {"weight": 23, "value": 96},
            {"weight": 31, "value": 134},
            {"weight": 9, "value": 41},
            {"weight": 14, "value": 64},
            {"weight": 29, "value": 127},
            {"weight": 7, "value": 33},
            {"weight": 20, "value": 87},
        ),
    },
)

EXACT_TARGET_SCORE = 1860


def _score_selection(items: tuple[dict[str, int], ...], capacity: int, chosen_indices) -> int:
    try:
        chosen = [int(idx) for idx in chosen_indices]
    except Exception:
        return 0
    if len(chosen) != len(set(chosen)):
        return 0
    if any(idx < 0 or idx >= len(items) for idx in chosen):
        return 0
    total_weight = sum(items[idx]["weight"] for idx in chosen)
    if total_weight > capacity:
        return 0
    return sum(items[idx]["value"] for idx in chosen)


def evaluate() -> dict[str, float]:
    total_score = 0
    for case in CASES:
        items = tuple({"weight": item["weight"], "value": item["value"]} for item in case["items"])
        try:
            chosen = select_items(list(items), int(case["capacity"]))
        except Exception:
            chosen = []
        total_score += _score_selection(items, int(case["capacity"]), chosen)
    return {"score": float(total_score)}


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
