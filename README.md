# constrained-agent

An agent framework where tool availability and parameter bounds are enforced at the token level on every turn, based on session history.

## The idea

Most agent frameworks treat tool schemas as static: the model sees the same tools with the same parameters every turn, and workflow compliance is left to the system prompt. This breaks under adversarial prompts, distracted models, or complex multi-step workflows — the model can skip required steps or pass invalid values.

`constrained-agent` takes a different approach: **constraints are declared in a JSON spec that is compiled into schema mutations applied at the start of every turn**. A tool that hasn't been unlocked is absent from the schema — it cannot be called regardless of what the user or model says. Parameter bounds are enforced at the token level via [xgrammar](https://github.com/mlc-ai/xgrammar), so the model cannot generate an out-of-range value even if it tries.

## Install

```bash
pip install -e .
```

Requires Python 3.11+.

## Quick start

Define a spec file describing the model format, tools, and constraint rules:

```json
{
    "format": "qwen3",
    "system_prompt": "You are a helpful banking assistant.",
    "tools": {
        "check_balance": {
            "description": "Check account balance",
            "params": {"account_id": {"type": "string"}},
            "required": ["account_id"]
        },
        "transfer": {
            "description": "Transfer funds to a recipient",
            "params": {
                "account_id": {"type": "string"},
                "recipient_id": {"type": "string"},
                "amount": {"type": "number"}
            },
            "required": ["account_id", "recipient_id", "amount"]
        }
    },
    "rules": [
        {
            "name": "block_transfer_until_balance_checked",
            "if": {"tool": "check_balance", "has_run": false},
            "then": [{"tool": "transfer", "available": false}]
        },
        {
            "name": "cap_transfer_to_balance",
            "if": {"tool": "check_balance", "has_run": true},
            "then": [{
                "tool": "transfer",
                "params": {"amount": {"maximum": {"$from": "check_balance.result.currentBalance"}}}
            }]
        }
    ]
}
```

Then provide only the function implementations:

```python
import outlines
from constrained_agent import Agent

model = outlines.from_transformers(...)

def check_balance(account_id: str) -> dict:
    return {"account_id": account_id, "currentBalance": 500.0}

def transfer(account_id: str, recipient_id: str, amount: float) -> dict:
    return {"status": "success", "amount": amount}

agent = Agent(
    model,
    implementations={"check_balance": check_balance, "transfer": transfer},
    spec="banking_spec.json",
)

response = agent.run("Transfer $600 to ACC-456 from ACC-123.")
```

## Defining tools

Tools can be defined either in the spec JSON (as above) or as Pydantic `BaseModel` classes passed directly to `Agent`. The class name (stripped of common suffixes and converted to snake_case) becomes the tool name; the docstring becomes the description.

```python
from pydantic import BaseModel

class CheckBalance(BaseModel):
    """Check account balance."""
    account_id: str

class Transfer(BaseModel):
    """Transfer funds to a recipient."""
    account_id: str
    recipient_id: str
    amount: float

agent = Agent(
    model,
    implementations={"check_balance": check_balance, "transfer": transfer},
    tools=[CheckBalance, Transfer],
    spec="rules_only_spec.json",  # spec with no tools section
)
```

To control the tool name explicitly, set a `ClassVar` attribute:

```python
from typing import ClassVar

class CheckBalance(BaseModel):
    name: ClassVar[str] = "check_balance"
    """Check account balance."""
    account_id: str
```

## Constraint rules

Rules live in the `"rules"` array of the spec. Each rule has an optional `name`, an `if` condition, and a `then` list of effects.

### Conditions

```json
{"tool": "check_balance", "has_run": false}
```
True when the tool has (or hasn't) been called this session.

```json
{"tool": "process_payment", "result": {"status": "success"}}
```
True when the tool's last result matches all given field values.

```json
{"allOf": [
    {"tool": "calculate_shipping", "has_run": true},
    {"tool": "run_credit_check", "has_run": true}
]}
```
True when all sub-conditions are true.

```json
{"anyOf": [
    {"tool": "collect_symptoms", "has_run": false},
    {"tool": "check_vitals", "has_run": false}
]}
```
True when at least one sub-condition is true.

### Effects

Each item in `then` names a tool and describes how to modify it:

```json
{"tool": "transfer", "available": false, "reason": "check_balance must run first."}
```

```json
{
    "tool": "transfer",
    "params": {
        "amount": {"maximum": 1000, "minimum": 0},
        "currency": {"enum": ["USD", "EUR"], "required": true}
    }
}
```

### `$from` expressions

Any numeric or list value in `params` can reference a previous tool's result:

```json
{"maximum": {"$from": "check_balance.result.currentBalance"}}
{"enum":    {"$from": "get_products.result.items[*].id"}}
```

The expression is evaluated lazily — if the referenced tool hasn't run yet, the constraint is silently skipped.

## Python rules

For patterns not expressible in the JSON DSL, pass Python callables via `rules=`. The function name is the rule name; the docstring is the description.

By default (`rules_mode="replace"`), Python rules replace the spec's rules entirely. Pass `rules_mode="merge"` to run both:

```python
def require_full_inventory_before_shipping(session, registry):
    """check_inventory must be called for every item in the cart."""
    if not session.tool("get_cart").has_run:
        return
    cart_products = {item["product_id"] for item in session.tool("get_cart").last_result["items"]}
    checked = {run.args["product_id"] for run in session.tool("check_inventory").runs}
    missing = cart_products - checked
    if missing:
        registry["calculate_shipping"].available = False
        registry["calculate_shipping"].unavailable_reason = (
            f"check_inventory must be called for every cart item. Missing: {', '.join(sorted(missing))}"
        )

agent = Agent(model, implementations, spec="spec.json", rules=[require_full_inventory_before_shipping], rules_mode="merge")
```

**What rules can modify:**

| Expression | Effect |
|---|---|
| `registry["x"].available = False` | Removes the tool from the schema |
| `registry["x"].unavailable_reason = "..."` | Shown to the model in the system prompt |
| `registry["x"].params["n"].maximum = 500` | Upper bound on a numeric parameter |
| `registry["x"].params["n"].minimum = 0` | Lower bound |
| `registry["x"].params["n"].enum = ["a", "b"]` | Restricts allowed values |
| `registry["x"].params["n"].required = True` | Makes a parameter required |

**What session history exposes:**

```python
session.tool("check_balance").has_run          # bool
session.tool("check_balance").run_count        # int
session.tool("check_balance").last_result      # dict
session.tool("check_balance").runs[-1].args    # dict
session.tool("check_balance").runs[-1].result  # dict
```

## Running the agent

```python
# Single-shot: resets session on each call
response = agent.run("Transfer $200 to ACC-456.")

# Multi-turn: preserves session across calls
response = agent.chat("What is my balance?")
response = agent.chat("Now transfer $100.")
```

## Full `Agent` constructor

```python
Agent(
    model,                               # required — the language model
    implementations={...},               # {tool_name: callable}
    spec="spec.json",                    # path, dict, or AgentSpec — provides defaults
    tools=[PydanticClass, ...],          # overrides spec tools section
    tools_mode="replace",                # "replace" (default) or "merge"
    rules=[python_fn, ...],              # Python constraint callables
    rules_mode="replace",                # "replace" (default) or "merge" with spec rules
    format="qwen3",                      # ModelFormat or name string — overrides spec
    system_prompt="...",                 # overrides spec
    max_turns=10,                        # overrides spec
    inference_kwargs={"max_new_tokens": 512},  # generation kwargs — overrides spec
    max_concurrent_tool_calls=4,         # parallel tool execution
    verbose=True,
)
```

Explicit arguments always override values from the spec.

## Examples

| File | Highlights |
|---|---|
| `example.py` | Basic banking workflow; tools and rules in the JSON spec |
| `example_support.py` | Multi-turn `chat()`; `$from` enum constrained from a prior result |
| `example_medical.py` | `anyOf` condition; `result` value matching; static param bounds; Python rule for negation |
| `example_analytics.py` | No spec file — all tools and rules in Python |
| `example_order.py` | 9 tools, 7 constraints; constrained vs. unconstrained comparison on an adversarial prompt |
