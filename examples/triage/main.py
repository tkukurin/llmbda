# %% [markdown]
# # Support triage: deterministic extraction, scripted LLM steps, repair loop
#
# Runs `support_triage` from `skill.py` on the bundled tickets. See
# `scoring.py` for the Inspect AI evaluation of the same skill.

# %%
import json

from skill import TICKETS, run_skill, support_triage

# %% [markdown]
# ## Run the skill on every ticket

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
