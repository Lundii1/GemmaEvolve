from __future__ import annotations

CASES = (10, 11, 12)
COUNTEREXAMPLE_BONUS = 1_000.0
TOTAL_INCIDENCE = float(sum(num_cliques * num_cliques for num_cliques in CASES))
TOTAL_MAX_OVERLAP = float(sum(num_cliques * (num_cliques - 1) // 2 for num_cliques in CASES))


async def evaluate_stage1(context):
    metrics, execution = await context.execute_candidate()
    status = execution.status
    should_continue = status == "success"
    rejection_reason = None if should_continue else status
    return {
        "name": "execute",
        "status": status,
        "metrics": metrics,
        "features": {
            "duration_ms": execution.duration_ms,
            "code_bytes": len(context.program.code.encode("utf-8")),
        },
        "primary_score": metrics.get(context.primary_metric, 0.0),
        "should_continue": should_continue,
        "rejection_reason": rejection_reason,
        "execution": execution.to_dict(),
        "artifacts": [
            {
                "name": "stdout_tail",
                "type": "text",
                "content": execution.stdout[-400:],
                "summary": f"Last stdout length={len(execution.stdout)} chars",
            }
        ],
    }


def evaluate_stage2(context, previous_result):
    metrics = previous_result.metrics
    score = metrics.get(context.primary_metric, 0.0)
    seed_score = metrics.get("seed_score", score)
    delta_score = score - seed_score
    total_vertices = metrics.get("total_vertices", TOTAL_INCIDENCE)
    total_overlap_pairs = metrics.get("total_overlap_pairs", 0.0)
    counterexample_cases = metrics.get("counterexample_cases", 0.0)
    max_block_size = metrics.get("max_block_size", 1.0)

    overlap_ratio = total_overlap_pairs / max(TOTAL_MAX_OVERLAP, 1.0)
    compression_ratio = 1.0 - (total_vertices / max(TOTAL_INCIDENCE, 1.0))
    avg_membership = TOTAL_INCIDENCE / max(total_vertices, 1.0)
    counterexample_case_ratio = counterexample_cases / len(CASES)
    max_block_ratio = max_block_size / max(CASES)

    per_case_scores = {
        f"n{num_cliques}": metrics.get(f"n{num_cliques}_score", 0.0) for num_cliques in CASES
    }
    per_case_deltas = {
        f"n{num_cliques}": metrics.get(f"delta_n{num_cliques}_score", 0.0) for num_cliques in CASES
    }
    per_case_counterexamples = {
        f"n{num_cliques}": metrics.get(f"n{num_cliques}_counterexample", 0.0)
        for num_cliques in CASES
    }
    improved_cases = sum(1 for delta in per_case_deltas.values() if delta > 0.0)
    hardest_case_score = max(per_case_scores.values(), default=0.0)
    counterexample_case_names = [
        case_name for case_name, flag in per_case_counterexamples.items() if flag > 0.0
    ]
    delta_summary = " ".join(
        f"{case_name}={per_case_deltas[case_name]:+.3f}" for case_name in per_case_deltas
    )
    counterexample_summary = ",".join(counterexample_case_names) if counterexample_case_names else "none"

    return {
        "name": "efl_features",
        "status": previous_result.status,
        "metrics": metrics,
        "features": {
            **previous_result.features,
            "score_density": score / max(len(context.program.code.encode("utf-8")), 1),
            "delta_score": delta_score,
            "overlap_ratio": overlap_ratio,
            "compression_ratio": compression_ratio,
            "avg_membership": avg_membership,
            "counterexample_case_ratio": counterexample_case_ratio,
            "max_block_ratio": max_block_ratio,
            "improved_case_ratio": improved_cases / len(CASES),
            "hardest_case_ratio": hardest_case_score / COUNTEREXAMPLE_BONUS,
        },
        "primary_score": score,
        "should_continue": previous_result.status == "success",
        "execution": previous_result.execution.to_dict(),
        "artifacts": [
            {
                "name": "score_report",
                "type": "json",
                "content": {
                    "score": score,
                    "seed_score": seed_score,
                    "delta_score": delta_score,
                    "program_id": context.program.id,
                },
                "summary": f"score={score:.3f} seed={seed_score:.3f} delta={delta_score:+.3f}",
            },
            {
                "name": "behavior_summary",
                "type": "json",
                "content": {
                    "score": score,
                    "seed_score": seed_score,
                    "delta_score": delta_score,
                    "counterexample_cases": counterexample_cases,
                    "overlap_ratio": overlap_ratio,
                    "compression_ratio": compression_ratio,
                    "avg_membership": avg_membership,
                    "max_block_ratio": max_block_ratio,
                    "per_case_scores": per_case_scores,
                    "per_case_counterexamples": per_case_counterexamples,
                },
                "summary": (
                    f"counterexamples={counterexample_summary} "
                    f"overlap={overlap_ratio:.3f} "
                    f"compression={compression_ratio:.3f} "
                    f"avg_mem={avg_membership:.3f}"
                ),
            },
            {
                "name": "delta_report",
                "type": "json",
                "content": per_case_deltas,
                "summary": delta_summary,
            },
        ],
    }
