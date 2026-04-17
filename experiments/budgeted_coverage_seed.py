# EVOLVE-BLOCK-START
def select_projects(
    projects: list[dict[str, object]],
    budget: int,
    element_values: dict[int, int],
) -> list[int]:
    """
    Return the project indices to fund under the budget.

    The baseline uses a static nominal-value-per-cost ratio. Better solutions can
    incorporate marginal gain, lookahead, local repair, or exact subset search.
    """
    order = sorted(
        range(len(projects)),
        key=lambda idx: (
            _nominal_project_value(projects[idx], element_values) / max(int(projects[idx]["cost"]), 1)
        ),
        reverse=True,
    )
    chosen: list[int] = []
    remaining = budget
    for idx in order:
        cost = int(projects[idx]["cost"])
        if cost <= remaining:
            chosen.append(idx)
            remaining -= cost
    return chosen
# EVOLVE-BLOCK-END


CASES = (
    {
        "budget": 15,
        "element_values": {
            0: 9, 1: 7, 2: 8, 3: 10, 4: 6, 5: 11, 6: 5,
            7: 8, 8: 7, 9: 12, 10: 9, 11: 6, 12: 10, 13: 4,
        },
        "projects": (
            {"cost": 4, "elements": (0, 1, 2, 3)},
            {"cost": 5, "elements": (2, 4, 5)},
            {"cost": 6, "elements": (5, 6, 7, 8)},
            {"cost": 3, "elements": (1, 8)},
            {"cost": 7, "elements": (3, 6, 9, 10)},
            {"cost": 4, "elements": (10, 11)},
            {"cost": 5, "elements": (7, 11, 12)},
            {"cost": 6, "elements": (0, 9, 12, 13)},
            {"cost": 2, "elements": (4, 13)},
            {"cost": 4, "elements": (4, 12)},
        ),
    },
    {
        "budget": 16,
        "element_values": {
            0: 8, 1: 6, 2: 9, 3: 7, 4: 10, 5: 5, 6: 8, 7: 11,
            8: 6, 9: 9, 10: 12, 11: 7, 12: 8, 13: 10, 14: 5,
        },
        "projects": (
            {"cost": 5, "elements": (0, 2, 4, 6)},
            {"cost": 4, "elements": (1, 3, 5)},
            {"cost": 6, "elements": (5, 6, 7, 8)},
            {"cost": 3, "elements": (8, 9)},
            {"cost": 7, "elements": (0, 9, 10, 11)},
            {"cost": 5, "elements": (2, 11, 12)},
            {"cost": 4, "elements": (7, 12, 13)},
            {"cost": 6, "elements": (1, 10, 13, 14)},
            {"cost": 3, "elements": (4, 14)},
            {"cost": 5, "elements": (3, 6, 9)},
        ),
    },
    {
        "budget": 14,
        "element_values": {
            0: 10, 1: 5, 2: 8, 3: 9, 4: 7, 5: 6, 6: 11,
            7: 4, 8: 10, 9: 8, 10: 12, 11: 5, 12: 9, 13: 6,
        },
        "projects": (
            {"cost": 4, "elements": (0, 1, 4)},
            {"cost": 5, "elements": (2, 3, 5, 6)},
            {"cost": 3, "elements": (6, 7)},
            {"cost": 6, "elements": (1, 7, 8, 9)},
            {"cost": 4, "elements": (9, 10)},
            {"cost": 5, "elements": (0, 10, 11)},
            {"cost": 6, "elements": (3, 8, 11, 12)},
            {"cost": 2, "elements": (12, 13)},
            {"cost": 5, "elements": (4, 13)},
            {"cost": 4, "elements": (2, 5, 9)},
        ),
    },
    {
        "budget": 18,
        "element_values": {
            0: 7, 1: 8, 2: 6, 3: 10, 4: 9, 5: 5, 6: 11, 7: 7,
            8: 9, 9: 6, 10: 10, 11: 8, 12: 12, 13: 4, 14: 7,
        },
        "projects": (
            {"cost": 6, "elements": (0, 1, 2, 3)},
            {"cost": 5, "elements": (3, 4, 5)},
            {"cost": 7, "elements": (5, 6, 7, 8)},
            {"cost": 4, "elements": (8, 9)},
            {"cost": 8, "elements": (1, 9, 10, 11)},
            {"cost": 5, "elements": (10, 12)},
            {"cost": 6, "elements": (2, 11, 12, 13)},
            {"cost": 3, "elements": (13, 14)},
            {"cost": 4, "elements": (4, 14)},
            {"cost": 5, "elements": (0, 6, 12)},
        ),
    },
)

EXACT_TARGET_SCORE = 343


def _nominal_project_value(project: dict[str, object], element_values: dict[int, int]) -> int:
    return sum(element_values[element] for element in project["elements"])


def _coverage_score(
    projects: tuple[dict[str, object], ...],
    budget: int,
    element_values: dict[int, int],
    chosen_indices,
) -> int:
    try:
        chosen = [int(idx) for idx in chosen_indices]
    except Exception:
        return 0
    if len(chosen) != len(set(chosen)):
        return 0
    if any(idx < 0 or idx >= len(projects) for idx in chosen):
        return 0

    total_cost = sum(int(projects[idx]["cost"]) for idx in chosen)
    if total_cost > budget:
        return 0

    covered: set[int] = set()
    for idx in chosen:
        covered.update(int(element) for element in projects[idx]["elements"])
    return sum(element_values[element] for element in covered)


def evaluate() -> dict[str, float]:
    total_score = 0
    for case in CASES:
        projects = tuple(
            {
                "cost": int(project["cost"]),
                "elements": tuple(int(element) for element in project["elements"]),
            }
            for project in case["projects"]
        )
        element_values = {int(key): int(value) for key, value in case["element_values"].items()}
        try:
            chosen = select_projects(list(projects), int(case["budget"]), dict(element_values))
        except Exception:
            chosen = []
        total_score += _coverage_score(projects, int(case["budget"]), element_values, chosen)
    return {"score": float(total_score)}


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
