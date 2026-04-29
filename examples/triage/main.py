# %% [markdown]
# # Support triage: deterministic extraction, scripted LLM steps, repair loop
#
# Runs `support_triage` from `skill.py` on the bundled tickets. See
# `scoring.py` for the Inspect AI evaluation of the same skill.

# %%
import json

from skill import TICKETS, run_skill, support_triage

from tk.llmbda import last

# %% [markdown]
# ## Run the skill on every ticket

# %%
for ticket in TICKETS:
    trace = run_skill(support_triage, ticket)
    result = last(trace)
    print(f"\n{ticket['id']} · {ticket['subject']}")
    print(json.dumps(result.value, indent=2))
    print(f"meta: {result.meta}")

# %% [markdown]
# ## Inspect one trace

# %%
trace = run_skill(support_triage, TICKETS[1])
for name, step_result in trace.items():
    print(f"\n{name}")
    print(f"value: {json.dumps(step_result.value, indent=2)}")
    print(f"meta:  {json.dumps(step_result.meta, indent=2)}")
