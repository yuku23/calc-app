import ast
import math
import operator
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config (set these as environment variables on your host)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "change-me")          # secret your callers must send
PUBLIC_URL = os.getenv("PUBLIC_URL", "")             # e.g. https://calc-api.onrender.com

app = FastAPI(
    title="Calculator API",
    description="Simple calculator with stored results. Callable from anywhere (e.g. Glean agents).",
    version="1.0.0",
    # Glean/other tools read the `servers` block from the OpenAPI spec
    servers=[{"url": PUBLIC_URL}] if PUBLIC_URL else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# In-memory store (resets on restart). Swap for SQLite/Postgres for persistence.
RESULTS: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Operation(str, Enum):
    add = "add"
    subtract = "subtract"
    multiply = "multiply"
    divide = "divide"
    power = "power"
    modulo = "modulo"


class CalcRequest(BaseModel):
    operation: Operation = Field(..., description="Operation to perform")
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class ExprRequest(BaseModel):
    expression: str = Field(
        ..., description="Math expression, e.g. '(12 + 8) * 3 / 4' or 'sqrt(16) + 2**3'"
    )


class CalcResult(BaseModel):
    id: str
    operation: str
    inputs: dict
    result: float
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
OPS = {
    Operation.add: operator.add,
    Operation.subtract: operator.sub,
    Operation.multiply: operator.mul,
    Operation.divide: operator.truediv,
    Operation.power: operator.pow,
    Operation.modulo: operator.mod,
}

# Safe expression evaluator (no eval())
_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin,
          "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("Unsupported expression")


def _store(operation: str, inputs: dict, result: float) -> dict:
    record = {
        "id": str(uuid.uuid4()),
        "operation": operation,
        "inputs": inputs,
        "result": result,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULTS[record["id"]] = record
    return record


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["meta"], operation_id="health_check")
def health():
    return {"status": "ok"}


@app.post("/calculate", response_model=CalcResult, tags=["calculator"],
          operation_id="calculate", dependencies=[Depends(verify_key)],
          summary="Perform a basic calculation on two numbers")
def calculate(req: CalcRequest):
    try:
        result = OPS[req.operation](req.a, req.b)
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero")
    except OverflowError:
        raise HTTPException(status_code=400, detail="Result too large")
    return _store(req.operation.value, {"a": req.a, "b": req.b}, result)


@app.post("/evaluate", response_model=CalcResult, tags=["calculator"],
          operation_id="evaluate_expression", dependencies=[Depends(verify_key)],
          summary="Evaluate a math expression")
def evaluate(req: ExprRequest):
    try:
        tree = ast.parse(req.expression.strip(), mode="eval")
        result = float(_eval(tree))
    except ZeroDivisionError:
        raise HTTPException(status_code=400, detail="Division by zero")
    except (ValueError, SyntaxError, TypeError, OverflowError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid expression: {exc}")
    return _store("expression", {"expression": req.expression}, result)


@app.get("/results", response_model=List[CalcResult], tags=["results"],
         operation_id="list_results", dependencies=[Depends(verify_key)],
         summary="List recent calculation results (newest first)")
def list_results(limit: int = 20):
    items = sorted(RESULTS.values(), key=lambda r: r["created_at"], reverse=True)
    return items[: max(1, min(limit, 100))]


@app.get("/results/{result_id}", response_model=CalcResult, tags=["results"],
         operation_id="get_result", dependencies=[Depends(verify_key)],
         summary="Fetch one calculation result by id")
def get_result(result_id: str):
    if result_id not in RESULTS:
        raise HTTPException(status_code=404, detail="Result not found")
    return RESULTS[result_id]
