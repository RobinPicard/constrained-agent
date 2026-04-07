"""
Medical triage workflow.

Demonstrates:
- Tool schemas defined in the JSON spec (no Pydantic classes needed)
- anyOf condition: triage blocked until BOTH symptoms AND vitals collected
- result condition: treatment path changes based on triage outcome value
- Static parameter bounds: dosage capped for routine cases
- Python rule for negation: call_ambulance blocked when triage is NOT critical
  (the JSON DSL supports equality matching but not inequality)
"""
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from constrained_agent import Agent, Session, ToolRegistry


hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="cpu")
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = outlines.from_transformers(hf_model, hf_tokenizer)


# --- Implementations ---

def collect_symptoms(patient_id: str, symptoms: str) -> dict:
    return {"patient_id": patient_id, "symptoms": symptoms, "recorded": True}

def check_vitals(patient_id: str) -> dict:
    return {"patient_id": patient_id, "bp": "180/110", "pulse": 112, "temp": 38.9}

def run_triage(patient_id: str) -> dict:
    # Elevated BP + high pulse → critical
    return {"patient_id": patient_id, "urgency": "critical", "notes": "Hypertensive crisis suspected."}

def prescribe_medication(patient_id: str, medication: str, dosage_mg: float) -> dict:
    return {"patient_id": patient_id, "medication": medication, "dosage_mg": dosage_mg, "status": "prescribed"}

def call_ambulance(patient_id: str, location: str) -> dict:
    return {"patient_id": patient_id, "dispatched": True, "eta_minutes": 8}

def schedule_followup(patient_id: str, days_from_now: int) -> dict:
    return {"patient_id": patient_id, "appointment_in_days": days_from_now, "booked": True}


# --- Python rule (negation — not expressible in JSON DSL) ---

def block_ambulance_for_routine_cases(session: Session, registry: ToolRegistry) -> None:
    """call_ambulance is only available when triage result is critical."""
    triage = session.tool("run_triage")
    if not triage.has_run:
        return  # already blocked by spec rule until triage runs
    if triage.last_result.get("urgency") != "critical":
        registry["call_ambulance"].available = False
        registry["call_ambulance"].unavailable_reason = "Only dispatched for critical triage results."


# --- Run ---
# Tool schemas come from the spec; only implementations and the Python rule
# (which can't be expressed in JSON) are provided here.

agent = Agent(
    model,
    implementations={
        "collect_symptoms": collect_symptoms,
        "check_vitals": check_vitals,
        "run_triage": run_triage,
        "prescribe_medication": prescribe_medication,
        "call_ambulance": call_ambulance,
        "schedule_followup": schedule_followup,
    },
    spec="example_medical_spec.json",
    rules=[block_ambulance_for_routine_cases],
    rules_mode="merge",
    inference_kwargs={"max_new_tokens": 512},
    verbose=True,
)

response = agent.run(
    "Patient P-042 just arrived. They report severe chest pain and dizziness. "
    "Please assess and take appropriate action."
)
print(response)
