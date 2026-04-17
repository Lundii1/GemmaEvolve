from __future__ import annotations


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
    return {
        "name": "score_features",
        "status": previous_result.status,
        "metrics": previous_result.metrics,
        "features": {
            **previous_result.features,
            "score_density": score / max(len(context.program.code.encode("utf-8")), 1),
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
                    "program_id": context.program.id,
                },
                "summary": f"score={score:.3f}",
            }
        ],
    }
