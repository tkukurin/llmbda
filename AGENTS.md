# AGENTS.md

## Style

- **No decorative comments.** No ASCII banners, section dividers, or `# ---` separators. Let code structure speak for itself.
- **Inline comments over standalone lines.** If a comment is needed, prefer to put it on the same line as the code if line length is short.
- **Compact docstrings.** Prefer `attr: description` inline style. Never use NumPy/Sphinx `Attributes\n----------` blocks.
- **Factual docstrings.** State what the function does, nothing more.
- **Minimal vertical whitespace.** No blank lines between a class docstring and its first member. No blank lines between a protocol docstring and `def __call__`.
- **Concise examples.** Keep docstring examples short and scannable — one-liners where possible.

## Code

- **Deep modules**. Indirections are rarely worth it. A function needs to justify its existence
- **YAGNI.** Don't build convenience wrappers or abstractions until there's a concrete caller. No speculative code.
- **Flat over nested.** Minimise nesting. Especially: no `try`/`except` for control flow when a conditional check would do.
- **Modern Python.** Use `X | Y` unions, not `Optional[X]`. Always `from __future__ import annotations`.

## Testing

- pytest, plain functions or classes (no unittest).
- Test files live in `tests/` and mirror source structure.
- Prefer inline lambdas or closures for fakes over mock libraries.

## Project conventions

- `uv` for dependency management.
- **Config as code.** No configuration file formats (YAML, TOML, JSON) for runtime behaviour. Skills and pipelines are Python objects composed with plain Python.
