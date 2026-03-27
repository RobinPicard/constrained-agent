# constrained-agent

An agent framework where tool availability and parameter bounds are dynamically enforced based on session history.

## The idea

Most agent frameworks treat tool schemas as static: the model sees the same set of tools with the same parameters on every turn, and compliance with any workflow rules is left to the system prompt. This works until it doesn't — a sufficiently insistent user, a distracted model, or a complex multi-step workflow can cause the agent to skip required steps or call tools with invalid values.

`constrained-agent` takes a different approach: **constraints are expressed as Python functions that mutate tool schemas at the start of each turn**. A tool that hasn't been unlocked yet is simply absent from the schema sent to the model — it cannot be called regardless of what the user asks or how the model reasons.

```python
@constraint
def fraud_check_required(session, tools):
    approved = any(run.result.get("approved") for run in session.tool("run_fraud_check").runs)
    if not approved:
        tools["execute_transfer"].available = False
        tools["execute_transfer"].unavailable_reason = (
            "run_fraud_check must be called and return approved=True before any transfer."
        )
```

At each turn the agent:
1. Resets all tool schemas to their base definitions
2. Evaluates all constraints (which may lock tools, narrow parameter ranges, restrict enums, etc.)
3. Sends only the currently-available tools to the model
4. Executes tool calls, records them in the session, repeats

## Install

```bash
pip install -e .
```

Requires Python 3.11+. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` as appropriate.

## Usage

### Define tools

Tools are defined with a name, description, a Pydantic model or JSON schema for parameters, and a callable:

```python
from pydantic import BaseModel
from constrained_agent import Tool, ToolRegistry

class TransferParams(BaseModel):
    account_id: str
    recipient_id: str
    amount: float

def transfer(account_id: str, recipient_id: str, amount: float) -> dict:
    ...

registry = ToolRegistry([
    Tool("check_balance", "Check account balance.", CheckBalanceParams, check_balance),
    Tool("transfer", "Transfer funds to a recipient.", TransferParams, transfer),
])
```

### Define constraints

Constraints are Python functions that receive the current `session` and `tools` and mutate them in place. They are evaluated in registration order at the start of every turn.

```python
from constrained_agent import constraint, ConstraintEvaluator

@constraint
def require_balance_before_transfer(session, tools):
    if not session.tool("check_balance").has_run:
        tools["transfer"].available = False
        tools["transfer"].unavailable_reason = "check_balance must be called first."

@constraint
def balance_limit(session, tools):
    if session.tool("check_balance").has_run:
        last_balance = session.tool("check_balance").last_result["currentBalance"]
        tools["transfer"].params["amount"].maximum = last_balance

evaluator = ConstraintEvaluator([require_balance_before_transfer, balance_limit])
```

**What constraints can do:**

| Expression | Effect |
|---|---|
| `tools["x"].available = False` | Removes the tool from the schema entirely |
| `tools["x"].unavailable_reason = "..."` | Included in the system prompt so the model knows why |
| `tools["x"].params["amount"].maximum = 500` | Sets an upper bound on a numeric parameter |
| `tools["x"].params["amount"].minimum = 0` | Sets a lower bound |
| `tools["x"].params["status"].enum = ["active"]` | Restricts allowed values |
| `tools["x"].params["reason"].required = True` | Makes a parameter required |

**What session history exposes:**

```python
session.tool("check_balance").has_run          # bool
session.tool("check_balance").run_count        # int
session.tool("check_balance").last_result      # dict
session.tool("check_balance").runs[-1].args    # dict
session.tool("check_balance").runs[-1].result  # dict
```

### Run the agent

```python
from constrained_agent import Agent, OpenAIAdapter

agent = Agent(
    model=OpenAIAdapter(model="gpt-4o"),
    registry=registry,
    evaluator=evaluator,
    system_prompt="You are a helpful banking assistant.",
    verbose=True,
)

# Single-shot (resets session each call)
response = agent.run("Transfer $200 to ACC-456 from ACC-123.")

# Multi-turn (preserves session across calls)
response = agent.chat("What is my balance?")
response = agent.chat("Now transfer $100.")
```

## Supported models

| Provider | Adapter | Notes |
|---|---|---|
| OpenAI | `OpenAIAdapter(model="gpt-4o")` | Structured outputs enforce tool schemas |
| Anthropic | `AnthropicAdapter(model="claude-opus-4-6")` | Tool use with schema validation |

Open-source models (token-level enforcement) are planned.

## Why token-level enforcement matters

For API-based providers, constraints work by updating the tool schema sent to the model each turn. This means:
- Tool availability (`available = False`) is fully enforced — the tool is simply not in the schema
- Parameter constraints (`maximum`, `enum`, etc.) are passed to the provider but may not be enforced at generation time depending on the provider

With open-source models, constraints will be enforced at the token level during generation — the model cannot produce a tool call that violates a constraint regardless of what it was asked to do.

## Example

See `example_order.py` for a full order-fulfillment workflow with 9 tools and 7 constraints, including a comparison between constrained and unconstrained agents given an adversarial prompt that attempts to bypass all checks.
