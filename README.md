# BFCL v4 EvalScope Runner

A script and environment for a quick local smoke-test comparison of models on **BFCL v4** via `evalscope` and a `llama.cpp` OpenAI-compatible endpoint.

The goal of this setup is to quickly check agentic/tool-calling behavior of local GGUF models without writing custom tasks. The current smoke-test uses only lightweight single-turn BFCL v4 subsets so that a run can finish in tens of minutes on Apple Silicon.

## Tested models and files

The following models were tested in the current comparison:

| Report alias | Actual GGUF file | Notes |
|---|---|---|
| `gemma` | `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` | Gemma 4 26B A4B instruct, IQ4_XS |
| `qwen36-distill` | `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled.i1-IQ4_XS.gguf` | distilled Qwen 3.6 35B A3B, IQ4_XS |
| `qwen36-base` | `Qwen_Qwen3.6-35B-A3B-Q5_K_L.gguf` | clean/base Qwen 3.6 35B A3B, Q5_K_L |


## What `run_bfcl_v4.py` does

The script:

- runs the `bfcl_v4` benchmark through EvalScope;
- uses `eval_type="openai_api"`;
- sends requests to a local OpenAI-compatible endpoint;
- uses native tools/function calling;
- tests lightweight BFCL v4 subsets:
  - `simple_python`
  - `multiple`
  - `parallel`
  - `parallel_multiple`
  - `irrelevance`
- saves results to `outputs/<run_name>/<timestamp>/`;
- lets you specify the model id and output directory via command-line arguments.

## Tested environment

The current runs were done on macOS / Apple Silicon with Python 3.11 and a local `llama.cpp` server.

Recommended Python setup:

```bash
python3.11 -m venv .venvs/bfcl
source .venvs/bfcl/bin/activate
python --version
```

Avoid Python 3.13 for now: some eval/ML dependencies may not have compatible wheels.

## Dependency installation

### Base installation

```bash
python3.11 -m venv .venvs/bfcl
source .venvs/bfcl/bin/activate

pip install -U pip setuptools wheel
pip install evalscope openai pandas datasets
```

### BFCL v4

A normal installation of `bfcl-eval==2025.10.27.1` may fail on macOS/Apple Silicon because of the pinned dependency `faiss-cpu==1.11.0`.

Working option for this smoke-test:

```bash
pip install bfcl-eval==2025.10.27.1 --no-deps
```

FAISS is not needed for the current subsets as long as you do not run memory/vector tasks.

### Dependencies that had to be installed manually

Minimal package set needed during the BFCL/EvalScope run:

```bash
pip install \
  tqdm \
  requests \
  pydantic \
  python-dotenv \
  tenacity
```

BFCL parsers:

```bash
pip install \
  tree_sitter==0.21.3 \
  tree-sitter-java==0.21.0 \
  tree-sitter-javascript==0.21.4
```

Provider SDKs and utility packages that BFCL imports even when using a local OpenAI-compatible endpoint:

```bash
pip install \
  "anthropic>=0.61.0" \
  "cohere==5.18.0" \
  "google-genai==1.24.0" \
  "mistralai==1.7.0" \
  "writer-sdk>=2.1.0" \
  boto3 \
  google-search-results \
  "datamodel-code-generator==0.25.7" \
  qwen-agent \
  "networkx==3.3" \
  "rank_bm25==0.2.2" \
  beautifulsoup4 \
  html2text \
  pathlib
```

`qwen-agent` additionally required an audio package:

```bash
pip install soundfile
```

If `soundfile` does not import on macOS:

```bash
brew install libsndfile
pip install soundfile
```

### Import check

```bash
python - <<'PY'
import evalscope
import openai
import tree_sitter
import tree_sitter_java
import tree_sitter_javascript
import anthropic
import cohere
import google.genai
import mistralai
import boto3
import qwen_agent
import datamodel_code_generator
import soundfile
print("bfcl/evalscope imports ok")
PY
```

### If numpy is too new

If the environment has `numpy 2.x` and older eval packages start failing, pin numpy:

```bash
pip install "numpy==1.26.4"
```

## Running the `llama.cpp` server

You need an OpenAI-compatible endpoint with native tool calling. For `llama.cpp`, make sure to run the server with `--jinja`.

Example:

```bash
llama-server \
  -m /path/to/model.gguf \
  --alias qwen36-base \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096 \
  -ngl 999 \
  --jinja
```

Check that the alias is visible:

```bash
curl http://127.0.0.1:8080/v1/models | jq
```

## Native tool-calling sanity check

Before running BFCL, verify that the endpoint returns actual `tool_calls`, not textual JSON inside `content`.

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen36-base",
    "messages": [
      {"role": "user", "content": "Call the function to add 2 and 3."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "add_numbers",
          "description": "Add two numbers",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {"type": "integer"},
              "b": {"type": "integer"}
            },
            "required": ["a", "b"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0,
    "max_tokens": 256
  }' | jq
```

A good response contains:

```json
"finish_reason": "tool_calls",
"message": {
  "tool_calls": [
    {
      "type": "function",
      "function": {
        "name": "add_numbers",
        "arguments": "{\"a\":2,\"b\":3}"
      }
    }
  ]
}
```

If `tool_calls` is empty and the function call appears only in `content`, the current `is_fc_model=True` mode will produce decode failures.

## Environment variables

The script uses two environment variables:

| Variable | Default value | Description |
|---|---:|---|
| `API_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible API URL |
| `API_KEY` | `EMPTY` | API key for the endpoint |

Example:

```bash
export API_URL="http://127.0.0.1:8080/v1"
export API_KEY="EMPTY"
```

If the endpoint does not require a key, `API_KEY=EMPTY` is usually enough.

## Running the benchmark

```bash
python run_bfcl_v4.py [model_id] [work_dir]
```

| Argument | Required | Default value | Description |
|---|---:|---|---|
| `model_id` | No | `qwen36-base` | Model name passed to the OpenAI API |
| `work_dir` | No | `outputs/bfcl_<model_id>` | Directory where results are saved |

Examples:

```bash
API_URL=http://127.0.0.1:8080/v1 API_KEY=EMPTY \
python run_bfcl_v4.py qwen36-base outputs/bfcl_base
```

```bash
API_URL=http://127.0.0.1:8080/v1 API_KEY=EMPTY \
python run_bfcl_v4.py qwen36-distill outputs/bfcl_distill
```

```bash
API_URL=http://127.0.0.1:8080/v1 API_KEY=EMPTY \
python run_bfcl_v4.py gemma outputs/bfcl_gemma
```

## Benchmark settings

The current configuration runs:

```python
datasets=["bfcl_v4"]
```

Active subsets:

```python
"simple_python"
"multiple"
"parallel"
"parallel_multiple"
"irrelevance"
```

These are suitable for a first smoke-test because they check single-turn tool/function calling without heavier multi-turn cases.

You can test multi-turn separately:

```python
"multi_turn_base"
"multi_turn_miss_func"
"multi_turn_miss_param"
```

Recommended order:

1. run the single-turn subsets first;
2. then run `multi_turn_base` with `limit=3`;
3. only then try `multi_turn_miss_func` and `multi_turn_miss_param`.

## Function-calling mode

Function-calling mode is enabled in `dataset_args`:

```python
"is_fc_model": True
```

This means EvalScope expects native `tool_calls`.

Also enabled:

```python
"underscore_to_dot": True
```

This is useful for models or endpoints that do not handle dots in function names well.

## Generation config

The script uses deterministic generation:

```python
"temperature": 0
```

Limits:

```python
"max_tokens": 1024
"timeout": 1200
```

The payload also includes:

```python
"extra_body": {
    "parallel_tool_calls": True
}
```

If the server returns `400 Bad Request`, try removing or commenting out `extra_body`.

## Number of tasks

The current runs used:

```python
limit=20
```

Because 5 subsets are active, this gives:

```text
5 subsets × 20 tasks = 100 tasks
```

For debugging, use:

```python
limit=2
```

For a more confident comparison, increase to `limit=50`, but local 26B/35B models will take longer.

## Results from current runs

All results below were obtained on the BFCL v4 single-turn smoke-test:

```text
subsets = simple_python, multiple, parallel, parallel_multiple, irrelevance
limit   = 20 per subset
total   = 100 tasks
```

### Summary table

| Model / GGUF | Time | irrelevance | multiple | parallel | parallel_multiple | simple_python | NON_LIVE | HALLUCINATION | OVERALL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled.i1-IQ4_XS.gguf` | 08:07 | 1.00 | 0.95 | 0.85 | 1.00 | 0.95 | 0.9375 | 1.00 | 0.1938 |
| `Qwen_Qwen3.6-35B-A3B-Q5_K_L.gguf` | 35:39 | 1.00 | 0.95 | 0.90 | 1.00 | 0.95 | 0.9500 | 1.00 | 0.1950 |
| `gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` | 10:50 | 0.95 | 0.90 | 0.75 | 0.85 | 0.95 | 0.8625 | 0.95 | 0.1813 |

### Short interpretation

- **Qwen base** had the best quality score: `NON_LIVE = 0.95`, `parallel = 0.90`, but was the slowest: `35:39` for 100 tasks.
- **Qwen distill** was almost unchanged relative to base: `NON_LIVE = 0.9375` vs `0.95`, i.e. one fewer correct task out of 80 non-live cases. The run took `08:07`.
- **Gemma** was noticeably weaker at tool/function calling: `NON_LIVE = 0.8625`, `parallel = 0.75`, `parallel_multiple = 0.85`.
- `OVERALL` in EvalScope/BFCL v4 reports is not the best primary metric for this smoke-test. For comparison, subset scores, `NON_LIVE`, and `HALLUCINATION` are more useful.

### Quick conclusion

For a local agentic/tool-calling smoke-test:

```text
1. Qwen_Qwen3.6-35B-A3B-Q5_K_L.gguf
   Best quality, but the slowest run.

2. Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled.i1-IQ4_XS.gguf
   Almost the same single-turn BFCL quality, much faster in the current run.

3. gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
   Works, but underperforms Qwen on parallel/function-calling tasks.
```

On this slice, there is **no clear agentic regression in Qwen distill relative to Qwen base**. The only quality differences are: `parallel` is lower by 1 task out of 20, and `NON_LIVE` is lower by 1 task out of 80. For a stronger conclusion, run a small multi-turn slice separately.

## Where results are stored

EvalScope writes results to `work_dir` and adds a timestamp:

```text
outputs/bfcl_base/20260506_151612/
outputs/bfcl_distill/20260506_153027/
outputs/bfcl_gemma/20260506_180108/
```

Inside, you usually get:

```text
configs/task_config.yaml
reports/report.html
reports/<model_id>/bfcl_v4.json
predictions/
```

Example JSON report:

```text
outputs/bfcl_gemma/20260506_180108/reports/gemma/bfcl_v4.json
```

Example HTML report:

```text
outputs/bfcl_gemma/20260506_180108/reports/report.html
```

## Renaming incorrect aliases

If a model alias was mixed up, you can rename the directories:

```bash
mkdir -p outputs/renamed

mv outputs/bfcl_base/20260506_151612 \
  outputs/renamed/distill_actual_single_100_20260506_151612

mv outputs/bfcl_distill/20260506_153027 \
  outputs/renamed/base_actual_single_100_20260506_153027
```

To replace the model id inside files on macOS:

```bash
grep -RIl "qwen36-base" outputs/renamed/distill_actual_single_100_20260506_151612 \
  | xargs sed -i '' 's/qwen36-base/qwen36-distill/g'

grep -RIl "qwen36-distill" outputs/renamed/base_actual_single_100_20260506_153027 \
  | xargs sed -i '' 's/qwen36-distill/qwen36-base/g'
```

Safer variant with backup files:

```bash
grep -RIl "qwen36-base" outputs/renamed/distill_actual_single_100_20260506_151612 \
  | xargs sed -i .bak 's/qwen36-base/qwen36-distill/g'

find outputs -name "*.bak" -delete
```

## Troubleshooting

### `faiss-cpu==1.11.0` does not install

Error example:

```text
No matching distribution found for faiss-cpu==1.11.0
```

FAISS is not needed for the current subsets. Use:

```bash
pip install bfcl-eval==2025.10.27.1 --no-deps
```

And do not enable memory/vector subsets:

```text
memory_vector
memory_rec_sum
```

### `zsh: 0.61.0 not found`

In zsh, package specs with `>=` must be quoted:

```bash
pip install "anthropic>=0.61.0" "writer-sdk>=2.1.0"
```

### `ModuleNotFoundError: No module named 'tree_sitter'`

Install:

```bash
pip install \
  tree_sitter==0.21.3 \
  tree-sitter-java==0.21.0 \
  tree-sitter-javascript==0.21.4
```

### `ModuleNotFoundError: No module named 'anthropic'`

BFCL imports provider handlers even for local OpenAI-compatible runs. Install the provider SDKs listed in the installation section.

### `ModuleNotFoundError: No module named 'soundfile'`

Install:

```bash
pip install soundfile
```

If needed:

```bash
brew install libsndfile
pip install soundfile
```

### `Failed to decode the model response`

If this happens a lot, BFCL cannot parse the model response as a tool call.

Check that:

- the server is running with `--jinja`;
- the `curl` sanity check returns `tool_calls`;
- the response does not contain tool-call JSON only inside `content`;
- `is_fc_model=True` is appropriate for the endpoint;
- multi-turn subsets are temporarily disabled.

For debugging, use:

```python
subset_list=["simple_python"]
limit=3
debug=True
```

### Multi-turn runs take too long

Do not run multi-turn together with the main smoke-test. Start separately with:

```python
subset_list=["multi_turn_base"]
limit=3
```

If you see loops like `Turn: 1, Step: 20`, stop the run and inspect raw predictions.

## Cleaning interrupted runs

It is safe to delete timestamp directories under `outputs`:

```bash
rm -rf outputs/bfcl_base/20260506_150802
```

This does not delete the dataset. The dataset cache is stored separately, for example:

```text
~/.cache/modelscope/hub/datasets
```

## Recommended next step

To better check for agentic regressions after the single-turn smoke-test, run a small multi-turn slice:

```python
subset_list=[
    "multi_turn_base",
]
limit=3
```

Then, if it is stable:

```python
subset_list=[
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
]
limit=3
```

Compare not only accuracy, but also:

- total runtime;
- number of decode failures;
- hangs/loops on individual tasks;
- subset-level regressions, especially `parallel` and `parallel_multiple`.
