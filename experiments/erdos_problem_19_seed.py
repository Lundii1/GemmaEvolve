from __future__ import annotations

from collections.abc import Iterable

CaseDesign = tuple[tuple[int, ...], ...]
CASES = (10, 11, 12)
MAX_STEPS_BY_N = {
    10: 80,
    11: 96,
    12: 112,
}
COUNTEREXAMPLE_BONUS = 1_000.0
SEED_MAX_BLOCK_SIZE = 4
COLORABILITY_PROBE_INTERVAL = 12


def _normalize_membership(raw_membership: Iterable[int], num_cliques: int) -> tuple[int, ...]:
    membership = tuple(sorted(set(int(clique) for clique in raw_membership)))
    if not membership:
        raise ValueError("Each vertex must belong to at least one clique.")
    if membership[0] < 0 or membership[-1] >= num_cliques:
        raise ValueError("Clique indices must stay within the case range.")
    return membership


def _normalize_design(raw_design: Iterable[Iterable[int]], num_cliques: int) -> CaseDesign:
    memberships = [
        _normalize_membership(raw_membership, num_cliques) for raw_membership in raw_design
    ]
    memberships.sort(key=lambda membership: (len(membership), membership))
    return tuple(memberships)


def _initial_design(num_cliques: int) -> CaseDesign:
    design: list[tuple[int, ...]] = []
    for clique in range(num_cliques):
        for _ in range(num_cliques):
            design.append((clique,))
    return _normalize_design(design, num_cliques)


def _validate_design(design: CaseDesign, num_cliques: int) -> None:
    clique_sizes = [0] * num_cliques
    used_pairs: set[tuple[int, int]] = set()
    for membership in design:
        if membership != _normalize_membership(membership, num_cliques):
            raise ValueError("Each membership must be sorted, unique, and within range.")
        for clique in membership:
            clique_sizes[clique] += 1
        for offset, left in enumerate(membership):
            for right in membership[offset + 1 :]:
                pair = (left, right)
                if pair in used_pairs:
                    raise ValueError("Two cliques may intersect in at most one vertex.")
                used_pairs.add(pair)
    if any(size != num_cliques for size in clique_sizes):
        raise ValueError("Each clique must contain exactly n vertices.")


def _pair_overlap_count(design: CaseDesign) -> int:
    return sum(len(membership) * (len(membership) - 1) // 2 for membership in design)


def _overlap_energy(design: CaseDesign) -> int:
    return sum(len(membership) * len(membership) for membership in design)


def _clique_vertices(design: CaseDesign, num_cliques: int) -> tuple[tuple[int, ...], ...]:
    clique_to_vertices: list[list[int]] = [[] for _ in range(num_cliques)]
    for vertex, membership in enumerate(design):
        for clique in membership:
            clique_to_vertices[clique].append(vertex)
    return tuple(tuple(vertices) for vertices in clique_to_vertices)


def _graph_adjacency(design: CaseDesign, num_cliques: int) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    clique_vertices = _clique_vertices(design, num_cliques)
    adjacency = [0] * len(design)
    for vertices in clique_vertices:
        for offset, left in enumerate(vertices):
            for right in vertices[offset + 1 :]:
                adjacency[left] |= 1 << right
                adjacency[right] |= 1 << left
    return tuple(adjacency), clique_vertices


def _iter_set_bits(mask: int) -> Iterable[int]:
    current = mask
    while current:
        low_bit = current & -current
        yield low_bit.bit_length() - 1
        current ^= low_bit


def _is_singleton_mask(mask: int) -> bool:
    return mask != 0 and (mask & (mask - 1)) == 0


def _restore(
    trail: list[tuple[int, int, int]],
    domains: list[int],
    assigned: list[int],
) -> None:
    for vertex, previous_domain, previous_assignment in reversed(trail):
        domains[vertex] = previous_domain
        assigned[vertex] = previous_assignment


def _assign_color(
    vertex: int,
    color: int,
    domains: list[int],
    assigned: list[int],
    adjacency: tuple[int, ...],
) -> list[tuple[int, int, int]] | None:
    trail: list[tuple[int, int, int]] = []
    stack = [(vertex, color)]
    while stack:
        current_vertex, current_color = stack.pop()
        current_mask = 1 << current_color
        if domains[current_vertex] & current_mask == 0:
            _restore(trail, domains, assigned)
            return None
        if assigned[current_vertex] == current_color:
            continue
        if assigned[current_vertex] != -1 and assigned[current_vertex] != current_color:
            _restore(trail, domains, assigned)
            return None
        if assigned[current_vertex] != current_color or domains[current_vertex] != current_mask:
            trail.append((current_vertex, domains[current_vertex], assigned[current_vertex]))
            assigned[current_vertex] = current_color
            domains[current_vertex] = current_mask
        for neighbor in _iter_set_bits(adjacency[current_vertex]):
            if assigned[neighbor] == current_color:
                _restore(trail, domains, assigned)
                return None
            if domains[neighbor] & current_mask:
                updated_domain = domains[neighbor] & ~current_mask
                if updated_domain != domains[neighbor]:
                    trail.append((neighbor, domains[neighbor], assigned[neighbor]))
                    domains[neighbor] = updated_domain
                    if updated_domain == 0:
                        _restore(trail, domains, assigned)
                        return None
                    if assigned[neighbor] == -1 and _is_singleton_mask(updated_domain):
                        stack.append((neighbor, updated_domain.bit_length() - 1))
    return trail


def _is_n_colorable(design: CaseDesign, num_cliques: int) -> bool:
    adjacency, clique_vertices = _graph_adjacency(design, num_cliques)
    full_domain = (1 << num_cliques) - 1
    domains = [full_domain] * len(design)
    assigned = [-1] * len(design)
    anchor_clique = clique_vertices[0]
    for color, vertex in enumerate(anchor_clique):
        trail = _assign_color(vertex, color, domains, assigned, adjacency)
        if trail is None:
            return False

    def search() -> bool:
        chosen_vertex = -1
        smallest_domain = num_cliques + 1
        largest_degree = -1
        for vertex, color in enumerate(assigned):
            if color != -1:
                continue
            domain_size = domains[vertex].bit_count()
            degree = adjacency[vertex].bit_count()
            if domain_size < smallest_domain or (
                domain_size == smallest_domain and degree > largest_degree
            ):
                chosen_vertex = vertex
                smallest_domain = domain_size
                largest_degree = degree
        if chosen_vertex == -1:
            return True
        available = domains[chosen_vertex]
        while available:
            low_bit = available & -available
            color = low_bit.bit_length() - 1
            trail = _assign_color(chosen_vertex, color, domains, assigned, adjacency)
            if trail is not None:
                if search():
                    return True
                _restore(trail, domains, assigned)
            available ^= low_bit
        return False

    return search()


def _used_clique_pairs(design: CaseDesign) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for membership in design:
        for offset, left in enumerate(membership):
            for right in membership[offset + 1 :]:
                pairs.add((left, right))
    return pairs


def _best_merge_candidate(
    design: CaseDesign,
    max_block_size: int,
) -> tuple[int, int, tuple[int, ...]] | None:
    pairs_used = _used_clique_pairs(design)
    best_candidate: tuple[int, int, tuple[int, ...]] | None = None
    best_key: tuple[int, int, int, tuple[int, ...]] | None = None
    for left_index, left_membership in enumerate(design):
        left_set = set(left_membership)
        for right_index in range(left_index + 1, len(design)):
            right_membership = design[right_index]
            if left_set.intersection(right_membership):
                continue
            merged_size = len(left_membership) + len(right_membership)
            if merged_size > max_block_size:
                continue
            cross_pair_count = 0
            invalid = False
            for left_clique in left_membership:
                for right_clique in right_membership:
                    pair = (
                        (left_clique, right_clique)
                        if left_clique < right_clique
                        else (right_clique, left_clique)
                    )
                    if pair in pairs_used:
                        invalid = True
                        break
                    cross_pair_count += 1
                if invalid:
                    break
            if invalid:
                continue
            merged_membership = tuple(sorted(left_membership + right_membership))
            candidate_key = (
                cross_pair_count,
                merged_size,
                min(len(left_membership), len(right_membership)),
                tuple(-clique for clique in merged_membership),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_candidate = (left_index, right_index, merged_membership)
    return best_candidate


# EVOLVE-BLOCK-START
def improve_design(
    num_cliques: int,
    initial_design: CaseDesign,
    max_steps: int,
) -> CaseDesign:
    """
    Start from disjoint cliques and greedily merge vertices into overlap blocks.

    The objective is not to prove the EFL conjecture directly. Instead this
    bounded improver tries to construct overlap-rich, exact finite instances that
    are harder to color with n colors. Every intermediate design remains a valid
    edge-disjoint union of n copies of K_n by construction.
    """
    if max_steps <= 0:
        return initial_design

    current = list(initial_design)
    for step in range(max_steps):
        candidate = _best_merge_candidate(tuple(current), max_block_size=SEED_MAX_BLOCK_SIZE)
        if candidate is None:
            break
        left_index, right_index, merged_membership = candidate
        current[left_index] = merged_membership
        current.pop(right_index)
        if (step + 1) % COLORABILITY_PROBE_INTERVAL == 0:
            normalized = _normalize_design(current, num_cliques)
            if not _is_n_colorable(normalized, num_cliques):
                return normalized
    return _normalize_design(current, num_cliques)


def _case_metrics(num_cliques: int, design: CaseDesign) -> dict[str, float]:
    normalized = _normalize_design(design, num_cliques)
    _validate_design(normalized, num_cliques)
    vertex_count = len(normalized)
    pair_overlap_count = _pair_overlap_count(normalized)
    overlap_energy = _overlap_energy(normalized)
    max_block_size = max(len(membership) for membership in normalized)
    colorable = _is_n_colorable(normalized, num_cliques)
    counterexample = 0.0 if colorable else 1.0
    average_membership = (num_cliques * num_cliques) / float(vertex_count)
    case_score = (
        COUNTEREXAMPLE_BONUS * counterexample
        + float(pair_overlap_count)
        + 0.01 * float(overlap_energy)
        + 0.001 * float(max_block_size)
    )
    return {
        "score": case_score,
        "counterexample": counterexample,
        "colorable": 1.0 if colorable else 0.0,
        "vertices": float(vertex_count),
        "overlap_pairs": float(pair_overlap_count),
        "overlap_energy": float(overlap_energy),
        "avg_membership": average_membership,
        "max_block_size": float(max_block_size),
    }


def evaluate() -> dict[str, float]:
    metrics: dict[str, float] = {}
    score = 0.0
    seed_score = 0.0
    total_vertices = 0.0
    seed_total_vertices = 0.0
    total_overlap_pairs = 0.0
    seed_total_overlap_pairs = 0.0
    counterexample_cases = 0.0
    seed_counterexample_cases = 0.0
    max_block_size = 0.0
    seed_max_block_size = 0.0

    for num_cliques in CASES:
        initial_design = _initial_design(num_cliques)
        improved_design = improve_design(
            num_cliques,
            initial_design,
            MAX_STEPS_BY_N[num_cliques],
        )
        seed_case = _case_metrics(num_cliques, initial_design)
        final_case = _case_metrics(num_cliques, improved_design)
        case_name = f"n{num_cliques}"

        metrics[f"seed_{case_name}_score"] = seed_case["score"]
        metrics[f"{case_name}_score"] = final_case["score"]
        metrics[f"delta_{case_name}_score"] = final_case["score"] - seed_case["score"]

        metrics[f"seed_{case_name}_counterexample"] = seed_case["counterexample"]
        metrics[f"{case_name}_counterexample"] = final_case["counterexample"]
        metrics[f"seed_{case_name}_colorable"] = seed_case["colorable"]
        metrics[f"{case_name}_colorable"] = final_case["colorable"]

        metrics[f"seed_{case_name}_vertices"] = seed_case["vertices"]
        metrics[f"{case_name}_vertices"] = final_case["vertices"]
        metrics[f"delta_{case_name}_vertices"] = final_case["vertices"] - seed_case["vertices"]

        metrics[f"seed_{case_name}_overlap_pairs"] = seed_case["overlap_pairs"]
        metrics[f"{case_name}_overlap_pairs"] = final_case["overlap_pairs"]
        metrics[f"delta_{case_name}_overlap_pairs"] = (
            final_case["overlap_pairs"] - seed_case["overlap_pairs"]
        )

        metrics[f"seed_{case_name}_overlap_energy"] = seed_case["overlap_energy"]
        metrics[f"{case_name}_overlap_energy"] = final_case["overlap_energy"]
        metrics[f"seed_{case_name}_avg_membership"] = seed_case["avg_membership"]
        metrics[f"{case_name}_avg_membership"] = final_case["avg_membership"]
        metrics[f"seed_{case_name}_max_block_size"] = seed_case["max_block_size"]
        metrics[f"{case_name}_max_block_size"] = final_case["max_block_size"]

        score += final_case["score"]
        seed_score += seed_case["score"]
        total_vertices += final_case["vertices"]
        seed_total_vertices += seed_case["vertices"]
        total_overlap_pairs += final_case["overlap_pairs"]
        seed_total_overlap_pairs += seed_case["overlap_pairs"]
        counterexample_cases += final_case["counterexample"]
        seed_counterexample_cases += seed_case["counterexample"]
        max_block_size = max(max_block_size, final_case["max_block_size"])
        seed_max_block_size = max(seed_max_block_size, seed_case["max_block_size"])

    total_incidence = float(sum(num_cliques * num_cliques for num_cliques in CASES))
    metrics["seed_total_vertices"] = seed_total_vertices
    metrics["total_vertices"] = total_vertices
    metrics["seed_total_overlap_pairs"] = seed_total_overlap_pairs
    metrics["total_overlap_pairs"] = total_overlap_pairs
    metrics["seed_counterexample_cases"] = seed_counterexample_cases
    metrics["counterexample_cases"] = counterexample_cases
    metrics["seed_max_block_size"] = seed_max_block_size
    metrics["max_block_size"] = max_block_size
    metrics["seed_avg_membership"] = total_incidence / max(seed_total_vertices, 1.0)
    metrics["avg_membership"] = total_incidence / max(total_vertices, 1.0)
    metrics["seed_score"] = seed_score
    metrics["score"] = score
    return metrics


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate()))
