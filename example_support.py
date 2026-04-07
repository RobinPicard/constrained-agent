"""
Multi-turn support chatbot.

Demonstrates:
- Tool schemas defined in the JSON spec (no Pydantic classes needed)
- agent.chat() for multi-turn conversations (session preserved across calls)
- $from enum: ticket categories are restricted to those returned by lookup_account
"""
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from constrained_agent import Agent


hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="cpu")
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = outlines.from_transformers(hf_model, hf_tokenizer)


# --- Implementations ---

def lookup_account(email: str) -> dict:
    return {
        "account_id": "ACC-789",
        "name": "Alice Smith",
        "plan": "premium",
        "available_categories": ["billing", "technical", "account", "feature_request"],
    }

def get_ticket_history(account_id: str) -> dict:
    return {
        "account_id": account_id,
        "open_tickets": [
            {"ticket_id": "TKT-001", "category": "billing", "status": "open"},
        ],
    }

def create_ticket(account_id: str, category: str, description: str) -> dict:
    return {"ticket_id": "TKT-002", "account_id": account_id, "category": category, "status": "open"}

def escalate_ticket(ticket_id: str, reason: str) -> dict:
    return {"ticket_id": ticket_id, "status": "escalated", "assigned_to": "senior_team"}

def close_ticket(ticket_id: str) -> dict:
    return {"ticket_id": ticket_id, "status": "resolved"}


# --- Run multi-turn ---
# Tool schemas come from the spec; only implementations are provided here.

agent = Agent(
    model,
    implementations={
        "lookup_account": lookup_account,
        "get_ticket_history": get_ticket_history,
        "create_ticket": create_ticket,
        "escalate_ticket": escalate_ticket,
        "close_ticket": close_ticket,
    },
    spec="example_support_spec.json",
    inference_kwargs={"max_new_tokens": 512},
    verbose=True,
)

# Session is preserved between chat() calls.
# On the first call, lookup_account hasn't run yet, so all other tools are
# blocked. By the third call, the session already has lookup_account in history
# and the category enum is fully constrained.
print(agent.chat("Hi, I'm alice@example.com and I have a billing question."))
print(agent.chat("Please open a new ticket about an incorrect charge on my last invoice."))
print(agent.chat("Actually the charge was correct — please close TKT-001."))
