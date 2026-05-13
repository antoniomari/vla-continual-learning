# Project Codex Instructions

This directory contains project-local instructions for generating and updating Markdown documentation in this repository.

## Scope

- Apply these rules when creating or editing `.md` files in `vla-continual-learning`.
- Prefer local project rules over global defaults when they conflict.

## Documentation Style

- Keep a practical, research-engineering tone: concise, precise, reproducible.
- Use clear section headers in sentence case.
- Start docs with a short purpose statement before deep detail.
- Prefer short paragraphs and focused bullet lists.
- Use fenced code blocks with language tags (`bash`, `yaml`, `python`, etc.).
- Use inline backticks for commands, paths, flags, env vars, and identifiers.
- Avoid marketing language and unsupported claims.

## Repository-Specific Conventions

- Assume Linux-first workflows unless a file explicitly needs cross-platform steps.
- Prefer `uv` for Python environment and package commands in examples.
- Keep paths repo-relative where possible (for example: `examples/`, `config/`, `model/`).
- When documenting experiments, include:
  - prerequisites,
  - exact launch command,
  - key configuration knobs,
  - expected outputs/artifacts.
- When adding troubleshooting notes, include symptom, likely cause, and concrete fix.

## README and Guide Structure

When writing a new guide, use this baseline order unless the user asks otherwise:

1. What this guide is for
2. Prerequisites
3. Setup or inputs
4. Run commands
5. Verify results
6. Common issues
7. Related files/links

## Editing Rules

- Preserve existing technical meaning; do not change command semantics unless requested.
- Do not silently remove important warnings, caveats, or safety notes.
- If information is uncertain, mark assumptions explicitly.
- Keep examples copy-paste ready.

## File Naming

- Use descriptive lowercase kebab-case for new docs when possible.
- Keep `README.md` for directory entrypoints; use topic-specific names for deep dives.

## Done Criteria For New Docs

A doc is considered complete when it includes:

- purpose and context,
- runnable examples,
- verification steps,
- references to related config/scripts in this repository.
