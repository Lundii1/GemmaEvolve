from __future__ import annotations

CASES = (13, 14, 16, 17, 19, 20, 22, 23, 24)


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
    score = previous_result.metrics.get(context.primary_metric, 0.0)
    seed_score = previous_result.metrics.get("seed_score", score)
    delta_score = score - seed_score
    early_case_edges = sum(
        previous_result.metrics.get(name, 0.0)
        for name in ("n13_edges", "n14_edges", "n16_edges")
    )
    late_case_edges = sum(
        previous_result.metrics.get(name, 0.0)
        for name in ("n22_edges", "n23_edges", "n24_edges")
    )
    signature_111 = previous_result.metrics.get("signature_111_edges", 0.0)
    signature_210 = previous_result.metrics.get("signature_210_edges", 0.0)
    signature_300 = previous_result.metrics.get("signature_300_edges", 0.0)
    safe_score = max(score, 1.0)
    signature_111_ratio = signature_111 / safe_score
    signature_210_ratio = signature_210 / safe_score
    signature_300_ratio = signature_300 / safe_score
    late_case_ratio = late_case_edges / safe_score
    early_case_ratio = early_case_edges / safe_score
    per_case_deltas = {
        f"n{num_vertices}": previous_result.metrics.get(f"delta_n{num_vertices}_edges", 0.0)
        for num_vertices in CASES
    }
    improved_cases = sum(1 for delta in per_case_deltas.values() if delta > 0.0)
    regressed_cases = sum(1 for delta in per_case_deltas.values() if delta < 0.0)
    delta_summary = " ".join(
        f"d{num_vertices}={per_case_deltas[f'n{num_vertices}']:+.0f}"
        for num_vertices in CASES
    )
    behavior_summary = {
        "score": score,
        "seed_score": seed_score,
        "delta_score": delta_score,
        "signature_111_edges": signature_111,
        "signature_210_edges": signature_210,
        "signature_300_edges": signature_300,
        "signature_111_ratio": signature_111_ratio,
        "signature_210_ratio": signature_210_ratio,
        "signature_300_ratio": signature_300_ratio,
        "early_case_ratio": early_case_ratio,
        "late_case_ratio": late_case_ratio,
        "per_case_deltas": per_case_deltas,
    }
    return {
        "name": "score_features",
        "status": previous_result.status,
        "metrics": previous_result.metrics,
        "features": {
            **previous_result.features,
            "score_density": score / max(len(context.program.code.encode("utf-8")), 1),
            "delta_score": delta_score,
            "signature_111_ratio": signature_111_ratio,
            "signature_210_ratio": signature_210_ratio,
            "signature_300_ratio": signature_300_ratio,
            "early_case_ratio": early_case_ratio,
            "late_case_ratio": late_case_ratio,
            "improved_case_ratio": improved_cases / len(CASES),
            "regressed_case_ratio": regressed_cases / len(CASES),
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
                "content": behavior_summary,
                "summary": (
                    f"delta={delta_score:+.0f} "
                    f"mix111={signature_111_ratio:.3f} "
                    f"mix210={signature_210_ratio:.3f} "
                    f"mix300={signature_300_ratio:.3f} "
                    f"late={late_case_ratio:.3f}"
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
