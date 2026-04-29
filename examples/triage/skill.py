"""Support triage skill: deterministic extraction, scripted LLM steps, repair loop.

Plain module (not a notebook). Imported by `main.py` and `scoring.py`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tk.llmbda import (
    LMCaller,
    Skill,
    SkillContext,
    StepResult,
    lm,
    run_skill,
    strip_fences,
)

NORMALIZE = "λ::normalize"
IDENTIFIERS = "λ::identifiers"
URGENCY = "λ::urgency"
CLASSIFY = "ψ::classify"
DRAFT = "ψ::draft"
REFINE = "ψ::refine"
SUMMARIZE = "λ::summarize"

TICKETS = [
    {
        "id": "SUP-1001",
        "channel": "email",
        "customer_tier": "standard",
        "subject": "Charged twice for the same order",
        "body": (
            "Hi, I was charged twice for order ORD-9982 on account acct_123. "
            "Please refund the duplicate charge. This is blocking month-end "
            "reconciliation."
        ),
    },
    {
        "id": "SUP-1002",
        "channel": "slack",
        "customer_tier": "enterprise",
        "subject": "Production outage for all users",
        "body": (
            "Production is down for all users since 09:10 UTC. Account acct_999 "
            "cannot log in, API calls fail with 503, and our launch is blocked. "
            "We need urgent help."
        ),
    },
    {
        "id": "SUP-1003",
        "channel": "chat",
        "customer_tier": "standard",
        "subject": "Cannot access account",
        "body": (
            "I lost access after changing phones. Password reset emails are not "
            "arriving. I don't know the account id."
        ),
    },
]

_ACCOUNT_RE = re.compile(r"\bacct_[a-z0-9]+\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\bORD-\d+\b", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URGENT_RE = re.compile(
    r"\b(urgent|production|down|outage|blocked|all users|503|cannot log in|fail)\b",
    re.IGNORECASE,
)
_BILLING_RE = re.compile(
    r"\b(refund|charged|charge|invoice|billing|duplicate)\b",
    re.IGNORECASE,
)
_ACCESS_RE = re.compile(
    r"\b(login|log in|password|reset|access|account)\b",
    re.IGNORECASE,
)


def normalize_ticket(ctx: SkillContext) -> StepResult:
    """Normalize the support ticket into a compact text payload."""
    ticket = ctx.entry
    text = f"{ticket['subject']}\n\n{ticket['body']}"
    return StepResult(
        value={
            "id": ticket["id"],
            "channel": ticket["channel"],
            "customer_tier": ticket["customer_tier"],
            "subject": ticket["subject"].strip(),
            "text": re.sub(r"\s+", " ", text).strip(),
        }
    )


def extract_identifiers(ctx: SkillContext) -> StepResult:
    """Extract account, order, and email identifiers from the ticket text."""
    text = ctx.trace[NORMALIZE].value["text"]
    identifiers = {
        "account_ids": sorted({m.group(0).lower() for m in _ACCOUNT_RE.finditer(text)}),
        "order_ids": sorted({m.group(0).upper() for m in _ORDER_RE.finditer(text)}),
        "emails": sorted({m.group(0).lower() for m in _EMAIL_RE.finditer(text)}),
    }
    missing = [name for name, values in identifiers.items() if not values]
    return StepResult(value=identifiers, meta={"missing": missing})


def detect_urgency(ctx: SkillContext) -> StepResult:
    """Detect urgency and coarse keyword features."""
    text = ctx.trace[NORMALIZE].value["text"]
    features = {
        "urgent": bool(_URGENT_RE.search(text)),
        "billing": bool(_BILLING_RE.search(text)),
        "access": bool(_ACCESS_RE.search(text)),
        "enterprise": ctx.entry["customer_tier"] == "enterprise",
    }
    score = sum(
        [
            3 if features["enterprise"] else 0,
            3 if "production" in text.lower() or "outage" in text.lower() else 0,
            2 if features["urgent"] else 0,
            1 if "blocked" in text.lower() else 0,
        ],
    )
    if score >= 6:
        severity = "sev0"
    elif score >= 3:
        severity = "sev1"
    elif features["billing"]:
        severity = "sev2"
    else:
        severity = "sev3"
    return StepResult(value={"features": features, "score": score, "severity": severity})


def _lm_json_call(call: LMCaller, payload: dict[str, Any]) -> tuple[Any, str]:
    """Send a JSON payload to the caller, return (parsed_response, raw_string)."""
    raw = call(messages=[{"role": "user", "content": json.dumps(payload)}])
    return json.loads(strip_fences(raw)), raw


def scripted_support_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    """Scripted LMCaller so this example runs without credentials."""
    system = (
        messages[0]["content"].lower()
        if messages and messages[0]["role"] == "system"
        else ""
    )
    payload = json.loads(strip_fences(messages[-1]["content"]))
    if "intent classifier" in system:
        return json.dumps(_classify_intent(payload))
    if "triage drafter" in system:
        return json.dumps(_draft_triage(payload))
    if "triage repair" in system:
        return json.dumps(_repair_triage(payload))
    msg = "unknown scripted prompt"
    raise ValueError(msg)


def _classify_intent(payload: dict[str, Any]) -> dict[str, Any]:
    text = payload["ticket"]["text"].lower()
    identifiers = payload["identifiers"]
    if any(word in text for word in ["refund", "charged", "billing", "invoice"]):
        intent = "billing_refund"
    elif any(word in text for word in ["outage", "503", "production", "down"]):
        intent = "production_incident"
    elif any(
        word in text for word in ["password", "reset", "access", "login", "log in"]
    ):
        intent = "account_access"
    else:
        intent = "general_support"
    confidence = 0.86
    if intent == "billing_refund" and not identifiers["order_ids"]:
        confidence = 0.62
    if intent == "account_access" and not identifiers["account_ids"]:
        confidence = 0.58
    return {
        "intent": intent,
        "confidence": confidence,
        "signals": {
            "has_account": bool(identifiers["account_ids"]),
            "has_order": bool(identifiers["order_ids"]),
        },
    }


def _draft_triage(payload: dict[str, Any]) -> dict[str, Any]:
    ticket = payload["ticket"]
    identifiers = payload["identifiers"]
    urgency = payload["urgency"]
    intent = payload["intent"]["intent"]
    severity = urgency["severity"]
    missing_info = []
    if not identifiers["account_ids"]:
        missing_info.append("account_id")
    if intent == "billing_refund" and not identifiers["order_ids"]:
        missing_info.append("order_id")
    route = {
        "billing_refund": "billing",
        "production_incident": "support",
        "account_access": "support",
    }.get(intent, "support")
    priority = {
        "sev0": "P0",
        "sev1": "P1",
        "sev2": "P2",
        "sev3": "P3",
    }[severity]
    return {
        "ticket_id": ticket["id"],
        "intent": intent,
        "priority": priority,
        "route": route,
        "summary": ticket["subject"],
        "missing_info": missing_info,
        "customer_reply": _customer_reply(intent, missing_info),
        "internal_note": f"{severity}; drafted route={route}",
    }


def _repair_triage(payload: dict[str, Any]) -> dict[str, Any]:
    draft = dict(payload["draft"])
    issues = payload["issues"]
    if "production incidents must route to incident_commander" in issues:
        draft["route"] = "incident_commander"
        draft["internal_note"] = draft["internal_note"] + "; repaired incident route"
    if "P0 requires explicit urgent customer reply" in issues:
        draft["customer_reply"] = (
            "We are escalating this as a P0 incident now. "
            "An incident commander will coordinate updates and next steps."
        )
    if "missing account id should be requested from customer" in issues:
        draft["customer_reply"] = (
            draft["customer_reply"] + " Please also send the affected account ID."
        )
    return draft


def _customer_reply(intent: str, missing_info: list[str]) -> str:
    if intent == "billing_refund":
        return (
            "Thanks for the details. We will review the duplicate charge and "
            "follow up with refund status."
        )
    if intent == "production_incident":
        return (
            "Thanks for reporting this. We are escalating to the support team "
            "for urgent investigation."
        )
    if intent == "account_access":
        reply = "We can help restore access."
        if "account_id" in missing_info:
            reply += (
                " Please send the affected account ID or the email address on "
                "the account."
            )
        return reply
    return "Thanks for contacting support. We will review and follow up."


CLASSIFY_PROMPT = """\
You are a support intent classifier.
Return ONLY JSON with intent, confidence, and signals.
"""


@lm(scripted_support_model, system_prompt=CLASSIFY_PROMPT)
def classify_intent(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Classify the customer's support intent."""
    parsed, raw = _lm_json_call(
        call,
        {
            "ticket": ctx.trace[NORMALIZE].value,
            "identifiers": ctx.trace[IDENTIFIERS].value,
            "urgency": ctx.trace[URGENCY].value,
        },
    )
    return StepResult(value=parsed, meta={"llm_raw": raw})


DRAFT_PROMPT = """\
You are a support triage drafter.
Return ONLY JSON with ticket_id, intent, priority, route, summary, missing_info,
customer_reply, and internal_note.
"""


@lm(scripted_support_model, system_prompt=DRAFT_PROMPT)
def draft_triage(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Draft a support triage decision from extracted features."""
    parsed, raw = _lm_json_call(
        call,
        {
            "ticket": ctx.trace[NORMALIZE].value,
            "identifiers": ctx.trace[IDENTIFIERS].value,
            "urgency": ctx.trace[URGENCY].value,
            "intent": ctx.trace[CLASSIFY].value,
        },
    )
    return StepResult(value=parsed, meta={"llm_raw": raw})


REPAIR_PROMPT = """\
You are a support triage repair assistant.
Return ONLY the corrected triage JSON. Do not add commentary.
"""


def _validate_draft(draft: dict, urgency: dict, identifiers: dict) -> list[str]:
    """Check a triage draft against support policy. Returns list of issues."""
    issues: list[str] = []
    if (
        draft["intent"] == "production_incident"
        and draft["route"] != "incident_commander"
    ):
        issues.append("production incidents must route to incident_commander")
    if draft["priority"] == "P0" and "escalat" not in draft["customer_reply"].lower():
        issues.append("P0 requires explicit urgent customer reply")
    if (
        not identifiers["account_ids"]
        and "account id" not in draft["customer_reply"].lower()
    ):
        issues.append("missing account id should be requested from customer")
    if urgency["severity"] == "sev0" and draft["priority"] != "P0":
        issues.append("sev0 must map to P0")
    return issues


@lm(scripted_support_model, system_prompt=REPAIR_PROMPT)
def refine_triage(ctx: SkillContext, call: LMCaller) -> StepResult:
    """Validate and repair the triage draft, looping up to 2 times."""
    draft = ctx.prev.value
    urgency = ctx.trace[URGENCY].value
    identifiers = ctx.trace[IDENTIFIERS].value
    for _ in range(2):
        issues = _validate_draft(draft, urgency, identifiers)
        if not issues:
            return StepResult(value=draft, meta={"valid": True, "issues": []})
        draft, _ = _lm_json_call(
            call,
            {
                "ticket": ctx.trace[NORMALIZE].value,
                "draft": draft,
                "issues": issues,
            },
        )
    issues = _validate_draft(draft, urgency, identifiers)
    return StepResult(value=draft, meta={"valid": not issues, "issues": issues})


def summarize(ctx: SkillContext) -> StepResult:
    """Attach final status based on validation outcome."""
    triage = ctx.prev.value
    valid = ctx.prev.meta.get("valid", False)
    status = "validated" if valid else "needs_review"
    return StepResult(value={**triage, "status": status}, meta=ctx.prev.meta)


support_triage = Skill(
    name="support_triage",
    steps=[
        Skill(NORMALIZE, fn=normalize_ticket),
        Skill(IDENTIFIERS, fn=extract_identifiers),
        Skill(URGENCY, fn=detect_urgency),
        Skill(CLASSIFY, fn=classify_intent),
        Skill(DRAFT, fn=draft_triage),
        Skill(REFINE, fn=refine_triage),
        Skill(SUMMARIZE, fn=summarize),
    ],
)

__all__ = [
    "CLASSIFY",
    "DRAFT",
    "IDENTIFIERS",
    "NORMALIZE",
    "REFINE",
    "SUMMARIZE",
    "TICKETS",
    "URGENCY",
    "run_skill",
    "support_triage",
]
