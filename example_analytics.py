"""
Database analytics pipeline — no spec file.

Demonstrates:
- Tools defined as Pydantic classes, no spec file
- format and system_prompt passed directly to Agent
- All constraints expressed as Python rules
- Dynamic enum: run_query.table is restricted to tables returned by list_tables
"""
import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from constrained_agent import Agent, Session, ToolRegistry


hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="cpu")
hf_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = outlines.from_transformers(hf_model, hf_tokenizer)


# --- Tool schemas ---

class ConnectDatabase(BaseModel):
    """Connect to a database by name."""
    database: str

class ListTables(BaseModel):
    """List all tables available in the connected database."""

class RunQuery(BaseModel):
    """Run a SELECT query on a specific table."""
    table: str   # constrained to tables returned by list_tables
    filters: str

class ExportResults(BaseModel):
    """Export the last query results to a CSV file."""
    filename: str


# --- Implementations ---

def connect_database(database: str) -> dict:
    return {"database": database, "status": "connected", "version": "PostgreSQL 16.1"}

def list_tables() -> dict:
    return {"tables": ["orders", "customers", "products", "inventory"]}

def run_query(table: str, filters: str) -> dict:
    rows = [
        {"id": 1, "customer": "Alice", "amount": 250.0},
        {"id": 2, "customer": "Bob",   "amount": 89.99},
    ]
    return {"table": table, "filters": filters, "row_count": len(rows), "rows": rows}

def export_results(filename: str) -> dict:
    return {"filename": filename, "rows_written": 2, "status": "ok"}


# --- Python rules ---

def require_connection_before_listing(session: Session, registry: ToolRegistry) -> None:
    """list_tables requires an active database connection."""
    if not session.tool("connect_database").has_run:
        registry["list_tables"].available = False
        registry["list_tables"].unavailable_reason = "Connect to a database first."

def require_listing_before_query(session: Session, registry: ToolRegistry) -> None:
    """run_query requires list_tables to have run first."""
    if not session.tool("list_tables").has_run:
        registry["run_query"].available = False
        registry["run_query"].unavailable_reason = "Call list_tables first to see available tables."

def restrict_query_to_listed_tables(session: Session, registry: ToolRegistry) -> None:
    """run_query.table must be a table returned by list_tables."""
    if not session.tool("list_tables").has_run:
        return  # handled by require_listing_before_query
    tables = session.tool("list_tables").last_result.get("tables", [])
    registry["run_query"].params["table"].enum = tables

def require_query_before_export(session: Session, registry: ToolRegistry) -> None:
    """export_results requires at least one successful query."""
    if not session.tool("run_query").has_run:
        registry["export_results"].available = False
        registry["export_results"].unavailable_reason = "Run a query first."


# --- Run (no spec file — format and system_prompt passed directly) ---

agent = Agent(
    model,
    implementations={
        "connect_database": connect_database,
        "list_tables": list_tables,
        "run_query": run_query,
        "export_results": export_results,
    },
    tools=[ConnectDatabase, ListTables, RunQuery, ExportResults],
    rules=[
        require_connection_before_listing,
        require_listing_before_query,
        restrict_query_to_listed_tables,
        require_query_before_export,
    ],
    format="qwen3",
    system_prompt="You are a data analyst assistant. Use the tools to answer questions about the database.",
    inference_kwargs={"max_new_tokens": 512},
    verbose=True,
)

response = agent.run(
    "Connect to the 'sales_db' database, then show me all rows from the orders table and export them."
)
print(response)
