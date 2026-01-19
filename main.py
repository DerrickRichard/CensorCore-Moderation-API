from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import httpx

app = FastAPI(
    title="CensorCore Moderation API",
    description="Semantic moderation backend for CensorCore.",
    version="1.0.0",
)

# -----------------------------
# Request / Response Models
# -----------------------------

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    blocked: bool
    reason: Optional[str]
    confidence: float
    categories: Dict[str, float]


# -----------------------------
# Main Endpoint
# -----------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(payload: AnalyzeRequest):
    text = payload.text.strip()

    # Quick allow for empty or trivial messages
    if not text or len(text) < 3:
        return AnalyzeResponse(
            blocked=False,
            reason=None,
            confidence=0.0,
            categories={}
        )

    # Run semantic model
    result = await run_semantic_moderation(text)

    # Apply your policy thresholds
    blocked, reason, confidence = apply_policy(result)

    return AnalyzeResponse(
        blocked=blocked,
        reason=reason,
        confidence=confidence,
        categories=result.get("categories", {})
    )


# -----------------------------
# Semantic Model (placeholder)
# -----------------------------

async def run_semantic_moderation(text: str) -> dict:
    """
    This function is the ONLY place that talks to an AI model.
    Swap this out for OpenAI, Azure, or any other provider.

    It must return a dict like:
    {
        "categories": {
            "self_harm": 0.92,
            "hate": 0.10,
            "sexual": 0.03,
            ...
        }
    }
    """

    # Placeholder — always harmless
    return {
        "categories": {
            "self_harm": 0.0,
            "hate": 0.0,
            "sexual": 0.0,
            "violence": 0.0
        }
    }


# -----------------------------
# Policy Layer
# -----------------------------

def apply_policy(model_result: dict):
    categories = model_result.get("categories", {})

    thresholds = {
        "self_harm": 0.75,
        "hate": 0.75,
        "sexual": 0.80,
        "violence": 0.85
    }

    worst_category = None
    worst_score = 0.0

    for name, score in categories.items():
        if score > worst_score:
            worst_score = score
            worst_category = name

    if worst_category is None:
        return False, None, 0.0

    threshold = thresholds.get(worst_category, 1.0)
    blocked = worst_score >= threshold

    reason = f"Blocked due to {worst_category} content." if blocked else None

    return blocked, reason, float(worst_score)
