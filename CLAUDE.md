# CLAUDE.md

> **Note:** Core project architecture and coding standards are defined in [AGENTS.md](./AGENTS.md).

@AGENTS.md

This file holds what is specific to Claude Code: how to work in this
repository. Commands and everything about the code itself are in
AGENTS.md.

## Workflow

- Run single test files while iterating; run `pixi run test` before
  calling a change done.
- After changing doc strings, run `pixi run doc-check`.
- After changing a public signature, check MojoDEC's forwarders
  (`../MojoDEC/mojodec/routines/{linalg,lanczos,sparse_ldlt}.mojo`) and
  the other consumers (`../mogeometry`, `../minigrad`).
- Keep `commit_msg.txt` current: under 20 words, covering exactly the
  uncommitted changes. Never add a Co-Authored-By line.
