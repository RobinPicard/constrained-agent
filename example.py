from pydantic import BaseModel
from constrained_agent import (
    Agent, Tool, ToolRegistry,
    Constraint, ConstraintEvaluator, constraint,
    OpenAIAdapter,
)


# --- Tool definitions ---

class CheckBalanceParams(BaseModel):
    account_id: str

class TransferParams(BaseModel):
    account_id: str
    recipient_id: str
    amount: float
    currency: str = "USD"


def check_balance(account_id: str) -> dict:
    return {"account_id": account_id, "currentBalance": 500.0}

def transfer(account_id: str, recipient_id: str, amount: float, currency: str = "USD") -> dict:
    return {"status": "success", "amount": amount, "currency": currency}


tools = ToolRegistry([
    Tool("check_balance", "Check account balance", CheckBalanceParams, check_balance),
    Tool("transfer", "Transfer funds to a recipient", TransferParams, transfer),
])


# --- Constraints ---

@constraint
def balance_limit(session, tools):
    """Cap transfer amount to the last known balance."""
    if session.tool("check_balance").has_run:
        last_balance = session.tool("check_balance").last_result["currentBalance"]
        tools["transfer"].params["amount"].maximum = last_balance

@constraint
def require_balance_before_transfer(session, tools):
    """Disallow transfer until balance has been checked."""
    if not session.tool("check_balance").has_run:
        tools["transfer"].available = False
        tools["transfer"].unavailable_reason = "check_balance must be called first."


evaluator = ConstraintEvaluator([balance_limit, require_balance_before_transfer])


# --- Run ---

agent = Agent(
    model=OpenAIAdapter(),
    registry=tools,
    evaluator=evaluator,
    system_prompt="You are a helpful banking assistant.",
    verbose=True,
)

response = agent.run("Please transfer $600 to recipient ACC-456 from my account ACC-123.")
print(response)
