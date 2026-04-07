import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from constrained_agent import Agent


hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="cpu")
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = outlines.from_transformers(hf_model, hf_tokenizer)


# --- Implementations ---

def check_balance(account_id: str) -> dict:
    return {"account_id": account_id, "currentBalance": 500.0}

def transfer(account_id: str, recipient_id: str, amount: float, currency: str = "USD") -> dict:
    return {"status": "success", "amount": amount, "currency": currency}


# --- Run ---

agent = Agent(
    model,
    implementations={"check_balance": check_balance, "transfer": transfer},
    spec="example_spec.json",
    inference_kwargs={"max_new_tokens": 512},
    verbose=True,
)

response = agent.run("Please transfer $600 to recipient ACC-456 from my account ACC-123. You can check first if necessary")
print(response)
