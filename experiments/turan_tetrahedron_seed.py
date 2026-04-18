from __future__ import annotations

from functools import lru_cache
from itertools import combinations

Edge = tuple[int, int, int]
CASES = (13, 14, 16, 17, 19, 20, 22, 23, 24)
MAX_STEPS_BY_N = {
    13: 1_500,
    14: 1_500,
    16: 2_000,
    17: 2_000,
    19: 2_500,
    20: 2_500,
    22: 3_000,
    23: 3_000,
    24: 3_000,
}


def _balanced_part_assignment(num_vertices: int) -> tuple[int, ...]:
    base_size, remainder = divmod(num_vertices, 3)
    labels: list[int] = []
    for part in range(3):
        labels.extend([part] * (base_size + (1 if part < remainder else 0)))
    return tuple(labels)


def _part_signature(edge: Edge, part_of: tuple[int, ...]) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    for vertex in edge:
        counts[part_of[vertex]] += 1
    return counts[0], counts[1], counts[2]


def _signature_family(signature: tuple[int, int, int]) -> str:
    if signature == (1, 1, 1):
        return "111"
    if sorted(signature) == [0, 1, 2]:
        return "210"
    if sorted(signature) == [0, 0, 3]:
        return "300"
    return "other"


@lru_cache(maxsize=None)
def _all_edges(num_vertices: int) -> tuple[Edge, ...]:
    return tuple(combinations(range(num_vertices), 3))


def _seed_edge_priority(
    edge: Edge,
    part_of: tuple[int, ...],
    num_vertices: int,
) -> tuple[float, ...]:
    a, b, c = edge
    signature = _part_signature(edge, part_of)
    if signature == (1, 1, 1):
        return (3.0, float((a + b + c) % num_vertices), float(c - a))
    return (1.0, float(sum(edge) % num_vertices))


@lru_cache(maxsize=None)
def _seed_candidate_order(num_vertices: int) -> tuple[Edge, ...]:
    part_of = _balanced_part_assignment(num_vertices)
    ranked: list[tuple[tuple[float, ...], Edge]] = []
    for edge in _all_edges(num_vertices):
        ranked.append((_seed_edge_priority(edge, part_of, num_vertices), edge))
    ranked.sort(reverse=True)
    return tuple(edge for _, edge in ranked)


def _would_create_k4(
    edge: Edge,
    edge_set: set[Edge],
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


def _blocking_edges(
    edge: Edge,
    edge_set: set[Edge],
    num_vertices: int,
) -> tuple[Edge, ...]:
    a, b, c = edge
    blockers: set[Edge] = set()
    for other in range(num_vertices):
        if other in edge:
            continue
        witness = (
            tuple(sorted((a, b, other))),
            tuple(sorted((a, c, other))),
            tuple(sorted((b, c, other))),
        )
        if all(candidate in edge_set for candidate in witness):
            blockers.update(witness)
    return tuple(sorted(blockers))


def _normalize_edges(raw_edges: object, num_vertices: int) -> tuple[Edge, ...]:
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError("improve_construction must return a sequence of 3-edges.")

    normalized: list[Edge] = []
    seen: set[Edge] = set()
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, (list, tuple)) or len(raw_edge) != 3:
            raise ValueError("Each returned edge must be a 3-tuple of vertices.")
        try:
            vertices = tuple(int(vertex) for vertex in raw_edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("Returned edges must contain integer vertices.") from exc
        if len(set(vertices)) != 3:
            raise ValueError("Returned edges must contain three distinct vertices.")
        if any(vertex < 0 or vertex >= num_vertices for vertex in vertices):
            raise ValueError("Returned vertices must stay within the case range.")
        edge = tuple(sorted(vertices))
        if edge in seen:
            raise ValueError("Returned edge list must not contain duplicates.")
        normalized.append(edge)
        seen.add(edge)
    return tuple(sorted(normalized))


def _validate_construction(edges: tuple[Edge, ...], num_vertices: int) -> None:
    edge_set = set(edges)
    for edge in edges:
        if _would_create_k4(edge, edge_set, num_vertices):
            raise ValueError("Returned construction is not K_4^3-free.")


@lru_cache(maxsize=None)
def _construct_seed_hypergraph(num_vertices: int) -> tuple[Edge, ...]:
    ordered_edges = _seed_candidate_order(num_vertices)
    chosen: list[Edge] = []
    edge_set: set[Edge] = set()
    for edge in ordered_edges:
        if _would_create_k4(edge, edge_set, num_vertices):
            continue
        edge_set.add(edge)
        chosen.append(edge)
    return tuple(chosen)


def _signature_totals(
    edges: tuple[Edge, ...],
    part_of: tuple[int, ...],
) -> dict[str, float]:
    totals = {"111": 0.0, "210": 0.0, "300": 0.0}
    for edge in edges:
        family = _signature_family(_part_signature(edge, part_of))
        if family in totals:
            totals[family] += 1.0
    return totals


# EVOLVE-BLOCK-START
def improve_construction(
    num_vertices: int,
    part_of: tuple[int, ...],
    initial_edges: tuple[Edge, ...],
    max_steps: int,
) -> tuple[Edge, ...]:
    """
    Improve a seeded K_4^3-free construction within a deterministic step budget.

    The seed starts from the known mixed-family basin around the 5/9 construction.
    This default improver first greedily adds any valid high-priority edges, then
    attempts one-for-one swaps for blocked edges, and finally performs one more
    greedy addition pass to capture gains unlocked by those swaps.
    """
    if max_steps <= 0:
        return tuple(sorted(initial_edges))

    def local_priority(edge: Edge) -> tuple[float, ...]:
        a, b, c = edge
        signature = _part_signature(edge, part_of)
        if signature == (1, 1, 1):
            return (3.0, float((a + b + c) % num_vertices), float(c - a))
        elif signature == (2, 1, 0):
            return (2.5, float((a + b + c) % num_vertices), float(c - a))
        elif signature in [(0, 2, 1), (1, 0, 2), (1, 2, 0)]:
            return (2.4, float((a + b + c) % num_vertices), float(c - a))
        elif sorted(signature) == [0, 1, 2]:
            return (2.0, float((a + b + c) % num_vertices), float(c - a))
        else:
            return (1.0, float(sum(edge) % num_vertices))

    candidates = _seed_candidate_order(num_vertices)
    current_set = set(initial_edges)
    best_edges = tuple(sorted(initial_edges))
    best_size = len(best_edges)
    blocked_candidates: list[Edge] = []
    steps = 0

    for edge in candidates:
        if steps >= max_steps:
            break
        if edge in current_set:
            continue
        steps += 1
        if _would_create_k4(edge, current_set, num_vertices):
            blocked_candidates.append(edge)
            continue
        current_set.add(edge)
        if len(current_set) > best_size:
            best_edges = tuple(sorted(current_set))
            best_size = len(best_edges)

    if steps < max_steps:
        blocked_candidates.sort(key=local_priority, reverse=True)
        for edge in blocked_candidates[:max(1, max_steps - steps)]:
            if edge in current_set:
                continue
            blockers = list(_blocking_edges(edge, current_set, num_vertices))
            if not blockers:
                continue
            blockers.sort(key=local_priority)
            for blocker in blockers:
                if blocker not in current_set:
                    continue
                current_set.remove(blocker)
                if _would_create_k4(edge, current_set, num_vertices):
                    current_set.add(blocker)
                    continue
                current_set.add(edge)
                if len(current_set) > best_size:
                    best_edges = tuple(sorted(current_set))
                    best_size = len(best_edges)
                break
            steps += 1
            if steps >= max_steps:
                break

    if steps < max_steps:
        for edge in candidates:
            if steps >= max_steps:
                break
            if edge in current_set:
                continue
            steps += 1
            if _would_create_k4(edge, current_set, num_vertices):
                continue
            current_set.add(edge)
            if len(current_set) > best_size:
                best_edges = tuple(sorted(current_set))
                best_size = len(best_edges)

    return best_edges
# EVOLVE-BLOCK-END


def evaluate() -> dict[str, float]:
    metrics: dict[str, float] = {}
    score = 0.0
    seed_score = 0.0
    signature_totals = {"111": 0.0, "210": 0.0, "300": 0.0}
    seed_signature_totals = {"111": 0.0, "210": 0.0, "300": 0.0}

    for num_vertices in CASES:
        part_of = _balanced_part_assignment(num_vertices)
        initial_edges = _construct_seed_hypergraph(num_vertices)
        improved_edges = improve_construction(
            num_vertices,
            part_of,
            initial_edges,
            MAX_STEPS_BY_N[num_vertices],
        )
        normalized_edges = _normalize_edges(improved_edges, num_vertices)
        _validate_construction(normalized_edges, num_vertices)

        seed_edge_count = float(len(initial_edges))
        edge_count = float(len(normalized_edges))
        delta_edge_count = edge_count - seed_edge_count
        metrics[f"seed_n{num_vertices}_edges"] = seed_edge_count
        metrics[f"n{num_vertices}_edges"] = edge_count
        metrics[f"delta_n{num_vertices}_edges"] = delta_edge_count
        seed_score += seed_edge_count
        score += edge_count

        case_seed_signatures = _signature_totals(initial_edges, part_of)
        case_signatures = _signature_totals(normalized_edges, part_of)
        for family in signature_totals:
            seed_signature_totals[family] += case_seed_signatures[family]
            signature_totals[family] += case_signatures[family]

    metrics["seed_signature_111_edges"] = seed_signature_totals["111"]
    metrics["seed_signature_210_edges"] = seed_signature_totals["210"]
    metrics["seed_signature_300_edges"] = seed_signature_totals["300"]
    metrics["signature_111_edges"] = signature_totals["111"]
    metrics["signature_210_edges"] = signature_totals["210"]
    metrics["signature_300_edges"] = signature_totals["300"]
    metrics["delta_signature_111_edges"] = signature_totals["111"] - seed_signature_totals["111"]
    metrics["delta_signature_210_edges"] = signature_totals["210"] - seed_signature_totals["210"]
    metrics["delta_signature_300_edges"] = signature_totals["300"] - seed_signature_totals["300"]
    metrics["seed_score"] = seed_score
    metrics["delta_score"] = score - seed_score
    metrics["score"] = score
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
