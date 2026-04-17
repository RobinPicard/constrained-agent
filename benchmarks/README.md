# Benchmarks

## BFCL multi_turn_base

Evaluates whether dynamic tool-parameter constraints help LLM agents on the
[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) `multi_turn_base`
benchmark (200 multi-turn function-calling test cases across 8 API classes).

Each test case is run twice -- once with parameter-narrowing constraints and
once without -- and the results are compared using the official BFCL checker.

### Setup

Install the benchmark dependencies:

```bash
pip install -e ".[bfcl]"
```

You need an OpenAI-compatible API endpoint serving the model (e.g. vLLM, Modal).

### Usage

Run specific test cases by index:

```bash
python -m benchmarks.bfcl.run \
    --model Qwen/Qwen3-5B \
    --base-url http://localhost:8000/v1 \
    --api-key dummy \
    --indices 107 110 \
    --verbose
```

Run all 200 test cases:

```bash
python -m benchmarks.bfcl.run \
    --model Qwen/Qwen3-5B \
    --base-url http://localhost:8000/v1 \
    --all
```

Save results to a JSON file:

```bash
python -m benchmarks.bfcl.run \
    --model Qwen/Qwen3-5B \
    --base-url http://localhost:8000/v1 \
    --all \
    --output results.json
```

You can also set environment variables instead of passing flags:

- `DOTJSON_VLLM_MODEL_NAME` -- model name
- `DOTJSON_VLLM_URL` -- API base URL
- `DOTJSON_VLLM_API_KEY` -- API key

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `$DOTJSON_VLLM_MODEL_NAME` | Model name for the API |
| `--base-url` | `$DOTJSON_VLLM_URL` | OpenAI-compatible API base URL |
| `--api-key` | `$DOTJSON_VLLM_API_KEY` | API key |
| `--indices` | | Test case indices to run |
| `--all` | | Run all 200 test cases |
| `--max-tokens` | 2048 | Max tokens per generation |
| `--max-turns` | 15 | Max agent turns per user message |
| `--verbose` | | Print turn-by-turn debug output |
| `--output` | | Output JSON file for results |

### Structure

```
benchmarks/bfcl/
  run.py          # CLI entrypoint (comparison loop + summary)
  harness.py      # Runs a single test case end-to-end
  constraints/    # Per-API-class parameter-narrowing rules
    trading_bot.py
    ticket_api.py
    twitter_api.py
    message_api.py
    gorilla_fs.py
    vehicle_control.py
    travel_api.py
    math_api.py
```

### How constraints work

Constraints are Python functions with signature
`(session: Session, registry: ToolRegistry) -> None` that narrow tool parameters
based on previous tool results. For example, in the trading bot API:

- After `get_stock_info` returns a price, `place_order.price` is constrained to
  that exact value
- After `place_order` returns an order ID, `cancel_order.order_id` is constrained
  to that ID

These constraints are applied before each generation step. In unconstrained mode,
no constraints are applied and the model sees all tools with their full parameter
ranges.
