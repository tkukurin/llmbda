# AGENTS.md

## Style

- **No decorative comments.** No ASCII banners, section dividers, or `# ---` separators. Let code structure speak for itself.
- **Inline comments.** If a comment is needed, prefer to put it on the same line as the code if line length is short.
- **Compact docstrings.** Prefer `attr: description` inline style. Never use NumPy/Sphinx `Attributes\n----------` blocks.
- **Factual docstrings.** State what the function does, nothing more.
- **Minimal vertical whitespace.** No blank lines between a class docstring and its first member. No blank lines between a protocol docstring and `def __call__`.
- **Concise examples.** Keep docstring examples short and scannable — one-liners where possible.

## Code

- **Deep modules**. Indirections are rarely worth it. A function needs to justify its existence
- **YAGNI.** Avoid convenience wrappers or abstractions. Reduce speculative code.
- **Flat over nested.** Minimise nesting. Especially: no `try`/`except` for control flow when a conditional check would do.
- **Modern Python.** Use `X | Y` unions, not `Optional[X]`. Always `from __future__ import annotations`.

## Git commits & documentation

- **Bullet points, not prose.** Commit bodies, docstrings beyond one line, and README sections are bulleted.
  - one fact per bullet. No narrative, no restating diffs, no justification paragraphs.
- **Subject line: `type(scope): summary`.** Imperative, lowercase, no trailing period.
- **Breaking changes:** we do not care. Prefer improving the architecture and reducing LOCs.
- **No marketing.** Don't explain why something is "clean" or "elegant" — state what changed.

## Assistant responses

- **~10-line budget.** Default ceiling unless asked for depth. Prefer bullets over paragraphs.
- **Never restate tool output.** If a tool just showed a diff, status, or test result, don't summarise it.
- **No "summary of changes" sections.** The diff is the summary.
- **No self-narration.** Skip "all tests pass", "working tree clean", "over to you" — state facts only when load-bearing.
- **One proposal per turn.** If the user asks for options, list them terse; don't pre-argue.

## Testing

- **Run the real example.** After fixing a bug, run the actual failing scenario . Make sure it tests intent. Never write ad-hoc `python -c` script to "prove".
- pytest, plain functions or classes (no unittest).
- Test files live in `tests/` and mirror source structure.
- Prefer inline lambdas or closures for fakes over mock libraries.

## Project conventions

- `uv` for dependency management.
- **Config as code.** No configuration file formats (YAML, TOML, JSON) for runtime behaviour. Skills and pipelines are Python objects composed with plain Python.
- **Notebooks are Jupytext `# %%` scripts, never `.ipynb`.** Keep notebooks as plain `.py` files under `examples/`, using the percent cell format (`# %%` for code, `# %% [markdown]` for prose).
