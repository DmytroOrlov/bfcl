#!/usr/bin/env python3
"""Run EvalScope BFCL-v4 using the native OpenAI-compatible tool-call API.

Usage:
    python run_bfcl_v4.py MODEL_ALIAS [WORK_DIR]

Environment:
    API_URL       OpenAI-compatible endpoint (default: http://127.0.0.1:18002/v1)
    API_KEY       API key (default: EMPTY)
    BFCL_SUBSETS  Comma-separated subset names (default: six subsets listed below)
    BFCL_LIMIT    Maximum examples per subset (default: 2)

Important:
    MODEL_ALIAS must be a safe Python identifier for BFCL multi-turn execution.
    Configure BFCL_UPSTREAM_MODEL on the proxy to send the real model ID to Splash.
    The proxy, not this runner, injects reasoning_effort=xhigh when enabled.
"""

import os
import sys

from evalscope import TaskConfig, run_task


DEFAULT_SUBSETS = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "parallel_multiple",
    "irrelevance",
)


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else "bfcl_next_iq1m"
    work_dir = sys.argv[2] if len(sys.argv) > 2 else f"outputs/bfcl_{model_id}"

    raw_subsets = os.getenv("BFCL_SUBSETS")
    subsets = (
        [s.strip() for s in raw_subsets.split(",") if s.strip()]
        if raw_subsets is not None
        else list(DEFAULT_SUBSETS)
    )
    if not subsets:
        raise ValueError("BFCL_SUBSETS must contain at least one subset")

    limit = int(os.getenv("BFCL_LIMIT", "2"))
    if limit < 1:
        raise ValueError("BFCL_LIMIT must be >= 1")

    task_cfg = TaskConfig(
        model=model_id,
        api_url=os.getenv("API_URL", "http://127.0.0.1:18002/v1"),
        api_key=os.getenv("API_KEY", "EMPTY"),
        eval_type="openai_api",
        datasets=["bfcl_v4"],
        eval_batch_size=1,
        dataset_args={
            "bfcl_v4": {
                "subset_list": subsets,
                "extra_params": {
                    "is_fc_model": True,
                    "underscore_to_dot": True,
                },
            },
        },
        generation_config={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 32768,
            "timeout": 1200,
            "extra_body": {
                "parallel_tool_calls": True,
            },
        },
        limit=limit,
        work_dir=work_dir,
    )

    print(f"BFCL model alias: {model_id}", flush=True)
    print(f"BFCL subsets: {', '.join(subsets)}", flush=True)
    print(f"BFCL limit per subset: {limit}", flush=True)
    print(f"BFCL work directory: {work_dir}", flush=True)
    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
