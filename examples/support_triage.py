# %% [markdown]
# # Support triage with deterministic extraction, scripted LLM steps, and repair loop
#
# This notebook models a small but realistic support-ticket triage pipeline.
#
# - Deterministic steps extract account/order IDs and urgency signals.
# - Scripted LLM steps classify intent and draft a triage decision.
# - A validation/repair loop catches policy violations and asks for a corrected draft.
# - The final markdown cell records ergonomic observations from building this example.

# %%
from __future__ import annotations

import json
import re
from typing import Any

from tk.llmbda import (
    LMCaller,
    Skill,
    Step,
    StepContext,
    StepResult,
    lm,
    loop,
    run_skill,
    strip_fences,
)

# %% [markdown]
# ## Sample tickets
#
# The inputs are intentionally varied:
#
# - a duplicate billing/refund case with both account and order identifiers
# - a production outage that should be escalated
# - an account-access case missing an account identifier

# %%
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

# %% [markdown]
# ## Deterministic feature extraction

# %%
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


def ticket_text(ctx: StepContext) -> str:
    ticket = ctx.entry
    return f"{ticket['subject']}\n\n{ticket['body']}"


def normalize_ticket(ctx: StepContext) -> StepResult:
    """Normalize the support ticket into a compact text payload."""
    ticket = ctx.entry
    text = ticket_text(ctx)
    normalized = {
        "id": ticket["id"],
        "channel": ticket["channel"],
        "customer_tier": ticket["customer_tier"],
        "subject": ticket["subject"].strip(),
        "text": re.sub(r"\s+", " ", text).strip(),
    }
    return StepResult(normalized, resolved=False)


def extract_identifiers(ctx: StepContext) -> StepResult:
    """Extract account, order, and email identifiers from the ticket text."""
    text = ctx.prior["λ::normalize"].value["text"]
    identifiers = {
        "account_ids": sorted({m.group(0).lower() for m in _ACCOUNT_RE.finditer(text)}),
        "order_ids": sorted({m.group(0).upper() for m in _ORDER_RE.finditer(text)}),
        "emails": sorted({m.group(0).lower() for m in _EMAIL_RE.finditer(text)}),
    }
    missing = [name for name, values in identifiers.items() if not values]
    return StepResult(identifiers, {"missing": missing}, resolved=False)


def detect_urgency(ctx: StepContext) -> StepResult:
    """Detect urgency and coarse keyword features."""
    text = ctx.prior["λ::normalize"].value["text"]
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
    return StepResult(
        {"features": features, "score": score, "severity": severity},
        resolved=False,
    )


# %% [markdown]
# ## Scripted model
#
# The model is deterministic so this notebook is runnable without credentials.
# It inspects the bound system prompt to decide which role it is playing.

# %%
def _read_json_user_message(messages: list[dict[str, str]]) -> dict[str, Any]:
    return json.loads(strip_fences(messages[-1]["content"]))


def scripted_support_model(*, messages: list[dict[str, str]], **_kw: Any) -> str:
    """OpenAI-shaped deterministic caller for examples."""
    system = (
        messages[0]["content"].lower()
        if messages and messages[0]["role"] == "system"
        else ""
    )
    payload = _read_json_user_message(messages)
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
        word in text
        for word in ["password", "reset", "access", "login", "log in"]
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

# %% [markdown]
# ## LLM-style classification and drafting steps

# %%
CLASSIFY_PROMPT = """\
You are a support intent classifier.
Return ONLY JSON with intent, confidence, and signals.
"""


@lm(scripted_support_model, system_prompt=CLASSIFY_PROMPT)
def classify_intent(ctx: StepContext, call: LMCaller) -> StepResult:
    """Classify the customer's support intent."""
    payload = {
        "ticket": ctx.prior["λ::normalize"].value,
        "identifiers": ctx.prior["λ::identifiers"].value,
        "urgency": ctx.prior["λ::urgency"].value,
    }
    raw = call(messages=[{"role": "user", "content": json.dumps(payload)}])
    return StepResult(json.loads(strip_fences(raw)), {"llm_raw": raw}, resolved=False)


DRAFT_PROMPT = """\
You are a support triage drafter.
Return ONLY JSON with ticket_id, intent, priority, route, summary, missing_info,
customer_reply, and internal_note.
"""


@lm(scripted_support_model, system_prompt=DRAFT_PROMPT)
def draft_triage(ctx: StepContext, call: LMCaller) -> StepResult:
    """Draft a support triage decision from extracted features."""
    payload = {
        "ticket": ctx.prior["λ::normalize"].value,
        "identifiers": ctx.prior["λ::identifiers"].value,
        "urgency": ctx.prior["λ::urgency"].value,
        "intent": ctx.prior["ψ::classify"].value,
    }
    raw = call(messages=[{"role": "user", "content": json.dumps(payload)}])
    return StepResult(json.loads(strip_fences(raw)), {"llm_raw": raw}, resolved=False)

# %% [markdown]
# ## Policy validation and repair loop
#
# This loop validates the latest draft. If invalid, it asks the scripted model
# for a repaired draft and validates again.

# %%
def _latest_draft(ctx: StepContext) -> dict[str, Any]:
    repaired = ctx.prior.get("ψ::repair")
    if repaired:
        return repaired.value
    return ctx.prior["ψ::draft"].value


def validate_triage(ctx: StepContext) -> StepResult:
    """Validate the latest triage draft against support policy."""
    draft = _latest_draft(ctx)
    urgency = ctx.prior["λ::urgency"].value
    identifiers = ctx.prior["λ::identifiers"].value
    issues = []
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
    metadata = {"valid": not issues, "issues": issues}
    return StepResult(draft, metadata, resolved=not issues)


REPAIR_PROMPT = """\
You are a support triage repair assistant.
Return ONLY the corrected triage JSON. Do not add commentary.
"""


@lm(scripted_support_model, system_prompt=REPAIR_PROMPT)
def repair_triage(ctx: StepContext, call: LMCaller) -> StepResult:
    """Repair the latest triage draft using validation issues."""
    payload = {
        "ticket": ctx.prior["λ::normalize"].value,
        "draft": _latest_draft(ctx),
        "issues": ctx.prior["λ::validate"].metadata["issues"],
    }
    raw = call(messages=[{"role": "user", "content": json.dumps(payload)}])
    return StepResult(json.loads(strip_fences(raw)), {"llm_raw": raw}, resolved=False)


support_triage = Skill(
    name="support_triage",
    steps=[
        Step("λ::normalize", normalize_ticket),
        Step("λ::identifiers", extract_identifiers),
        Step("λ::urgency", detect_urgency),
        Step("ψ::classify", classify_intent),
        Step("ψ::draft", draft_triage),
        loop(
            Step("λ::validate", validate_triage),
            Step("ψ::repair", repair_triage),
            name="refine_triage",
            max_iter=2,
            until=lambda ctx: bool(
                ctx.prior.get("λ::validate")
                and ctx.prior["λ::validate"].metadata.get("valid")
            ),
        ),
    ],
)

# %% [markdown]
# ## Run the triage skill

# %%
for ticket in TICKETS:
    result = run_skill(support_triage, ticket)
    print(f"\n{ticket['id']} · {ticket['subject']}")
    print(f"resolved_by: {result.resolved_by}")
    print(json.dumps(result.value, indent=2))
    print(f"validation:  {result.metadata}")

# %% [markdown]
# ## Inspect one trace

# %%
result = run_skill(support_triage, TICKETS[1])
for name, step_result in result.trace.items():
    print(f"\n{name}")
    print(f"value:    {json.dumps(step_result.value, indent=2)}")
    print(f"metadata: {json.dumps(step_result.metadata, indent=2)}")

# %% [markdown]
# ## Ergonomics observations from this example
#
# What works well:
#
# - `Step` and `StepResult` make a heterogeneous pipeline easy to read.
# - Plain Python composition is enough for realistic routing policy.
# - Scripted `LMCaller` fakes make LLM paths testable without mocks or network calls.
# - `Step.description` via docstrings is useful when serialising context for
#   later model calls.
# - `loop(...)` composes as a normal step, so retry/repair logic stays local.
#
# Issues and non-ergonomic spots:
#
# - `StepResult.resolved=True` is a footgun for intermediate steps; most
#   steps in this notebook need `resolved=False`.
# - A successful loop resolves the whole skill, so post-loop finalisation must
#   be inside the loop or omitted.
# - `until` is checked only after all inner loop steps run; to skip repair
#   after successful validation, validation must return `resolved=True`.
# - `resolved=True` means both "this step is valid" and "stop the skill",
#   which is awkward for validators.
# - `ctx.prior` stores only the latest value per step name, so retry history
#   is lost.
# - Inner loop step names share the global `ctx.prior` namespace, making
#   collisions easy in larger skills.
# - `ctx.steps` contains the outer skill plan, not the loop's inner steps, so
#   loop children are harder to introspect.
# - LLM steps are detected indirectly by function attributes; a first-class
#   step kind would make compile/export easier.
# - There is no structured way to express required inputs/outputs for a step,
#   so validation uses ad-hoc dictionary keys.
# - The loop metadata exposed for compile currently omits the `until`
#   predicate, so generated docs cannot fully explain stop conditions.
#
# Design pressure suggested by this notebook:
#
# - Consider `StepResult(stop=False)` or separate `status` from `resolved`.
# - Consider preserving per-iteration loop traces.
# - Consider typed loop metadata instead of ad-hoc function attributes.
# - Consider helper APIs for "latest prior from any of these names" and
#   "serialise prior payload".
