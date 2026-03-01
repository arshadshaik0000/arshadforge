"""
Agent Planner — Rule-based intent classification and tool execution planning.

Classifies user queries into deterministic intents and returns ordered tool plans.
No LLM used for planning — pure keyword/regex matching for reliability.

Intents:
- verification: Fact-check queries (Test 1)
- data_synthesis: Compare structured data (Test 2)
- forecasting: Calculate projections (Test 3)
- general: Fallback for unclassified queries
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

# ─── Arithmetic Detection Helpers ──────────────────────────────────────────

def is_arithmetic_query(query: str) -> bool:
    """Quick heuristic for plain math expressions.

    This covers both symbol-based expressions ("5 + 3") and natural
    language forms ("divide 10 by 2").  It should be evaluated before
    the usual intent rules so that arithmetic queries bypass vector search.
    """
    q = query.lower().strip()

    # direct numeric operations with symbols
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q):
        return True

    # complex expressions with parentheses and exponents: (17333/7351)^(1/9)-1
    if re.search(r"[\(\)]+.*\d+.*[\+\-\*\/\^]", q) and len(re.findall(r"\d+", q)) >= 2:
        return True

    # natural language arithmetic triggers
    # only treat as arithmetic if there is both a numeric operator/verb and
    # at least two numbers present; this avoids catching generic questions like
    # "what is the employment figure?" which happen to mention digits.
    arithmetic_keywords = [
        "divided by",
        "divide",
        "multiply",
        "multiplied by",
        "multiplied",
        "times",
        "plus",
        "minus",
        "product of",
        "increase by",
        "ratio",
    ]

    if any(k in q for k in arithmetic_keywords) and len(re.findall(r"\d+", q)) >= 2:
        return True

    return False

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Ordered list of tools to execute for a given query intent."""
    intent: str
    tools: list[str]
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Intent Classification Rules ──────────────────────────────────────────

INTENT_RULES = [
    {
        "intent": "forecasting",
        "keywords": ["cagr", "compound annual growth", "growth rate", "2030",
                      "projection", "forecast", "baseline", "target"],
        "patterns": [r"cagr", r"growth\s+rate", r"2030.*target", r"compound\s+annual"],
        "min_keyword_matches": 2,
        "min_pattern_matches": 1,
        "tools": ["sql_query", "python_calculator", "vector_retrieval", "citation_validation"],
        "description": "Retrieve baseline/target values from SQL, compute CAGR deterministically",
    },
    {
        "intent": "data_synthesis",
        "keywords": ["compare", "concentration", "pure-play", "south-west",
                      "national average", "regional", "dedicated", "diversified",
                      "versus", "against", "percentage",
                      "foreign-owned", "domestic", "breakdown", "ownership", "supported by"],
        "patterns": [r"compar", r"pure.play", r"south.west", r"national\s+average",
                      r"concentration", r"foreign\-owned", r"domestic", r"breakdown", r"percentage"],
        "min_keyword_matches": 2,
        "min_pattern_matches": 1,
        "tools": ["sql_query", "python_calculator", "vector_retrieval"],
        "description": "Query structured tables, compute comparisons or aggregated percentages numerically",
    },
    {
        "intent": "verification",
        "keywords": ["total number", "how many", "exactly", "stated",
                      "reported", "where", "page", "citation", "jobs",
                      "employment", "firms", "count", "currently",
                      "figure", "work", "sector", "people",
                      "personnel", "headcount", "number of"],
        "patterns": [r"total\s+number", r"where.*stated", r"how\s+many",
                      r"exactly.*stated", r"how\s+much",
                      r"what\s+is\s+the\s+(number|total|figure|employment)",
                      r"(people|employees)\s+(work|in|employed)"],
        "min_keyword_matches": 1,
        "min_pattern_matches": 0,
        "tools": ["vector_retrieval", "sql_query", "citation_validation"],
        "description": "Retrieve facts, validate citations against source text",
    },
    {
        "intent": "general",
        "keywords": [],
        "patterns": [],
        "min_keyword_matches": 0,
        "min_pattern_matches": 0,
        "tools": ["vector_retrieval"],
        "description": "General retrieval query — use vector search with LLM synthesis",
    },
]


def classify_intent(query: str) -> ExecutionPlan:
    """
    Classify a user query into an intent with an ordered tool execution plan.

    The classification is deterministic.  We use several layers:

    * **Arithmetic override** – if the query looks like a math expression, short
      circuit directly to the calculator tool.
    * **Explicit override** – certain schema dimensions (e.g. ownership) always
      imply structured aggregation/percentage logic, so route to
      *data_synthesis* immediately.  This prevents queries about foreign- vs
      domestic-owned firms being treated as simple verification.
    * **Rule-based scoring** – keywords and regex patterns defined in
      `INTENT_RULES` are scored as before, with additional keyword boosts for
      common phrasings to reduce misclassification.
    * **Heuristic fallback/override** – if the query structure suggests multiple
      fields, aggregation or percentages (see `_looks_like_data_synthesis`),
      we bump the intent to *data_synthesis* unless already forecasting.

    Returns the highest-scoring non-general intent, or falls back to 'general'.
    """
    query_lower = query.lower()

    # arithmetic override must come first
    if is_arithmetic_query(query):
        logger.info("Arithmetic query detected")
        return ExecutionPlan(
            intent="arithmetic",
            tools=["python_calculator"],
            description="Direct arithmetic query",
        )

    # keyword score boosts for better routing
    verification_keywords = [
        "how many",
        "what is the number",
        "employment",
        "total",
        "currently",
        "figure",
        "reported",
        "stated",
        "work in",
        "people",
        "sector",
        "number of",
        "headcount",
    ]
    forecast_keywords = [
        "cagr",
        "growth rate",
        "by 2030",
        "projection",
        "target",
    ]
    # we don't need a list for data_synthesis; simple 'compare' check suffices

    score_boosts: dict[str, int] = {"verification": 0, "forecasting": 0, "data_synthesis": 0}
    for k in verification_keywords:
        if k in query_lower:
            score_boosts["verification"] += 5
    for k in forecast_keywords:
        if k in query_lower:
            score_boosts["forecasting"] += 5
    if "compare" in query_lower:
        score_boosts["data_synthesis"] += 5

    # 1. explicit employment-percentage routing (highest priority)
    if "employment" in query_lower and (
        "percent" in query_lower or "percentage" in query_lower or "%" in query_lower
    ):
        # structured lookup from employment_breakdown table
        return ExecutionPlan(
            intent="employment_stats",
            tools=["sql_query", "citation_validation"],
            description="Structured employment percentage lookup",
            metadata={"override": "employment_percentage"},
        )

    # 2. explicit ownership-dimension routing
    ownership_keywords = [
        "foreign-owned",
        "domestic",
        "ownership",
        "supported by",
        "by origin",
        "percentage of total",
    ]
    if any(kw in query_lower for kw in ownership_keywords):
        return ExecutionPlan(
            intent="data_synthesis",
            tools=["sql_query", "python_calculator", "vector_retrieval"],
            description="Forced data_synthesis via ownership keyword override",
            metadata={"override": "ownership_keyword"},
        )

    best_plan: ExecutionPlan | None = None
    best_score = 0

    def _looks_like_data_synthesis(q: str) -> bool:
        """Determine if a query implies multi-field aggregation/percentage work.

        Heuristic criteria:
        * contains explicit terms like 'foreign-owned', 'domestic', 'supported by',
          'breakdown', 'percentage of total', 'by ownership'.
        * or mentions percentage/percent/average along with a field term.
        """
        ql = q.lower()
        special_terms = [
            "foreign-owned", "domestic", "supported by", "percentage of total",
            "breakdown", "by ownership"
        ]
        if any(term in ql for term in special_terms):
            return True
        if re.search(r"\b(percent|percentage|%|average|per|breakdown|compare|versus|vs)\b", ql):
            # require at least one keyword from a likely field set
            if re.search(r"\b(foreign|domestic|employment|employees|firms|offices|jobs)\b", ql):
                return True
        return False

    for rule in INTENT_RULES:
        if rule["intent"] == "general":
            continue

        # Count keyword matches
        keyword_score = sum(
            1 for kw in rule["keywords"]
            if kw.lower() in query_lower
        )

        # Count regex pattern matches
        pattern_score = sum(
            1 for pat in rule["patterns"]
            if re.search(pat, query_lower)
        )

        # Check minimum thresholds
        if (keyword_score >= rule["min_keyword_matches"] and
                pattern_score >= rule["min_pattern_matches"]):
            total_score = keyword_score + pattern_score * 2  # patterns weighted higher
            # apply any global boosts calculated earlier
            total_score += score_boosts.get(rule["intent"], 0)
            if total_score > best_score:
                best_score = total_score
                best_plan = ExecutionPlan(
                    intent=rule["intent"],
                    tools=rule["tools"],
                    description=rule["description"],
                    metadata={
                        "keyword_matches": keyword_score,
                        "pattern_matches": pattern_score,
                        "boost_applied": score_boosts.get(rule["intent"], 0),
                        "total_score": total_score,
                    },
                )

    # If no rule matched, fallback to general
    if best_plan is None:
        general = INTENT_RULES[-1]
        best_plan = ExecutionPlan(
            intent="general",
            tools=general["tools"],
            description=general["description"],
            metadata={"keyword_matches": 0, "pattern_matches": 0, "total_score": 0},
        )

    # Heuristic override: route complex aggregation/percentage queries to data_synthesis
    if _looks_like_data_synthesis(query) and best_plan.intent != "forecasting":
        # find the data_synthesis rule definition
        ds_rule = next((r for r in INTENT_RULES if r["intent"] == "data_synthesis"), None)
        if ds_rule:
            best_plan = ExecutionPlan(
                intent="data_synthesis",
                tools=ds_rule["tools"],
                description=ds_rule["description"],
                metadata={"heuristic": True},
            )

    logger.info(f"Query classified as '{best_plan.intent}' (score: {best_plan.metadata.get('total_score', 0)})")
    logger.info(f"Tool plan: {best_plan.tools}")
    return best_plan
