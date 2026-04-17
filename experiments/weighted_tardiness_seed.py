# EVOLVE-BLOCK-START
def schedule_jobs(jobs: list[dict[str, int]]) -> list[int]:
    """
    Return a permutation of job indices.

    The objective is to minimize weighted tardiness. The baseline rule uses
    earliest due date, but stronger schedules may use lookahead, scoring rules,
    local search, or exact dynamic programming.
    """
    return sorted(range(len(jobs)), key=lambda idx: jobs[idx]["due"])
# EVOLVE-BLOCK-END


CASES = (
    (
        {"processing": 3, "due": 7, "weight": 4},
        {"processing": 5, "due": 9, "weight": 7},
        {"processing": 2, "due": 6, "weight": 3},
        {"processing": 6, "due": 15, "weight": 8},
        {"processing": 4, "due": 11, "weight": 5},
        {"processing": 7, "due": 18, "weight": 9},
        {"processing": 3, "due": 10, "weight": 4},
        {"processing": 5, "due": 14, "weight": 6},
    ),
    (
        {"processing": 4, "due": 8, "weight": 5},
        {"processing": 6, "due": 12, "weight": 9},
        {"processing": 3, "due": 7, "weight": 4},
        {"processing": 5, "due": 10, "weight": 6},
        {"processing": 2, "due": 5, "weight": 3},
        {"processing": 7, "due": 16, "weight": 8},
        {"processing": 4, "due": 13, "weight": 7},
        {"processing": 6, "due": 18, "weight": 10},
    ),
    (
        {"processing": 5, "due": 11, "weight": 6},
        {"processing": 2, "due": 4, "weight": 2},
        {"processing": 4, "due": 9, "weight": 5},
        {"processing": 7, "due": 17, "weight": 9},
        {"processing": 3, "due": 8, "weight": 4},
        {"processing": 6, "due": 14, "weight": 8},
        {"processing": 5, "due": 13, "weight": 7},
        {"processing": 4, "due": 12, "weight": 6},
    ),
    (
        {"processing": 3, "due": 6, "weight": 3},
        {"processing": 8, "due": 15, "weight": 10},
        {"processing": 4, "due": 9, "weight": 5},
        {"processing": 6, "due": 13, "weight": 7},
        {"processing": 2, "due": 5, "weight": 2},
        {"processing": 5, "due": 12, "weight": 6},
        {"processing": 7, "due": 16, "weight": 8},
        {"processing": 3, "due": 10, "weight": 4},
    ),
)

SCORE_OFFSET = 5000
EXACT_TARGET_SCORE = 3413


def _weighted_tardiness(jobs: tuple[dict[str, int], ...], order) -> int | None:
    try:
        sequence = [int(idx) for idx in order]
    except Exception:
        return None
    if len(sequence) != len(jobs) or len(sequence) != len(set(sequence)):
        return None
    if any(idx < 0 or idx >= len(jobs) for idx in sequence):
        return None

    elapsed = 0
    tardiness = 0
    for idx in sequence:
        job = jobs[idx]
        elapsed += job["processing"]
        tardiness += job["weight"] * max(0, elapsed - job["due"])
    return tardiness


def evaluate() -> dict[str, float]:
    total_tardiness = 0
    for raw_jobs in CASES:
        jobs = tuple(
            {
                "processing": job["processing"],
                "due": job["due"],
                "weight": job["weight"],
            }
            for job in raw_jobs
        )
        try:
            order = schedule_jobs(list(jobs))
        except Exception:
            return {"score": 0.0}
        tardiness = _weighted_tardiness(jobs, order)
        if tardiness is None:
            return {"score": 0.0}
        total_tardiness += tardiness
    return {"score": float(SCORE_OFFSET - total_tardiness)}


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
