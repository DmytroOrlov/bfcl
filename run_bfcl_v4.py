import os
import sys
from evalscope import TaskConfig, run_task

model_id = sys.argv[1] if len(sys.argv) > 1 else "qwen36-base"
work_dir = sys.argv[2] if len(sys.argv) > 2 else f"outputs/bfcl_{model_id}"

task_cfg = TaskConfig(
    model=model_id,
    api_url=os.getenv("API_URL", "http://127.0.0.1:8080/v1"),
    api_key=os.getenv("API_KEY", "EMPTY"),
    eval_type="openai_api",
    datasets=["bfcl_v4"],

    # Для локального 35B на M4 Pro держи 1.
    eval_batch_size=1,

    dataset_args={
        "bfcl_v4": {
            "subset_list": [
                # самые дешёвые и полезные для smoke-test
                "simple_python",
                "multiple",
                "parallel",
                "parallel_multiple",
                "irrelevance",

                # уже ближе к agentic/multi-turn
                # "multi_turn_base",
                # "multi_turn_miss_func",
                # "multi_turn_miss_param",
            ],
            "extra_params": {
                # FC mode: EvalScope будет использовать native tools/function calling
                "is_fc_model": True,

                # полезно для моделей, которые плохо любят точки в function names
                "underscore_to_dot": True,
            },
        }
    },

    generation_config={
        "temperature": 0,
        "max_tokens": 1024,
        "timeout": 1200,

        # llama.cpp docs говорят, что parallel tool calls включаются полем parallel_tool_calls в payload.
        # Если endpoint вернёт 400, просто убери extra_body.
        "extra_body": {
            "parallel_tool_calls": True,
        },
    },

    # Для первого запуска. Потом подними до 30/50/100.
    limit=20,

    work_dir=work_dir,
)

run_task(task_cfg=task_cfg)
