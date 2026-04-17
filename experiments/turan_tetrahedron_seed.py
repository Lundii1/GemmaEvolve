from __future__ import annotations

from itertools import combinations


def _balanced_part_assignment(num_vertices: int) -> tuple[int, ...]:
    base_size, remainder = divmod(num_vertices, 3)
    labels: list[int] = []
    for part in range(3):
        labels.extend([part] * (base_size + (1 if part < remainder else 0)))
    return tuple(labels)


def _part_signature(edge: tuple[int, int, int], part_of: tuple[int, ...]) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for vertex in edge:
        counts[part_of[vertex]] += 1
    return counts[0], counts[1], counts[2]


def _normalize_priority(raw_priority: object) -> tuple[float, ...]:
    if isinstance(raw_priority, (int, float)):
        return (float(raw_priority),)
    if isinstance(raw_priority, (tuple, list)):
        try:
            return tuple(float(value) for value in raw_priority)
        except (TypeError, ValueError):
            return (0.0,)
    return (0.0,)


# EVOLVE-BLOCK-START
def edge_priority(
    edge: tuple[int, int, int],
    part_of: tuple[int, ...],
    num_vertices: int,
) -> tuple[float, ...]:
    """
    Return a sortable priority for a candidate triple.

    The fixed builder ignores non-positive priorities and then greedily adds
    triples in descending priority order whenever doing so does not create a
    K_4^3. Better heuristics can use part counts, cyclic structure, or vertex
    positions, but should stay deterministic.
    """
    if _part_signature(edge, part_of) == (1, 1, 1):
        a, b, c = edge
        return (3.0, float(c - a), float(b), float(num_vertices - c))
    return (0.0,)
# EVOLVE-BLOCK-END


CASES = (13, 14, 16, 17, 19, 20, 22, 23, 24)


def _would_create_k4(
    edge: tuple[int, int, int],
    edge_set: set[tuple[int, int, int]],
    num_vertices: int,
) -> bool:
    a, b, c = edge
    for other in range(num_vertices):
        if other in edge:
            continue
        if (
            tuple(sorted((a, b, other))) in edge_set
            and tuple(sorted((a, c, other))) in edge_set
            and tuple(sorted((b, c, other))) in edge_set
        ):
            return True
    return False


def construct_hypergraph(num_vertices: int) -> tuple[tuple[int, int, int], ...]:
    part_of = _balanced_part_assignment(num_vertices)
    candidates: list[tuple[tuple[float, ...], tuple[int, int, int]]] = []
    for edge in combinations(range(num_vertices), 3):
        priority = _normalize_priority(edge_priority(edge, part_of, num_vertices))
        if not priority or priority[0] <= 0:
            continue
        candidates.append((priority, edge))

    candidates.sort(reverse=True)

    chosen: list[tuple[int, int, int]] = []
    edge_set: set[tuple[int, int, int]] = set()
    for _, edge in candidates:
        if _would_create_k4(edge, edge_set, num_vertices):
            continue
        edge_set.add(edge)
        chosen.append(edge)
    return tuple(chosen)


def evaluate() -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_score = 0.0
    for num_vertices in CASES:
        edge_count = float(len(construct_hypergraph(num_vertices)))
        metrics[f"n{num_vertices}_edges"] = edge_count
        total_score += edge_count
    metrics["score"] = total_score
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
