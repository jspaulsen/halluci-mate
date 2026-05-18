---
name: grill-me
description: This skill should be used at the start of a non-trivial coding task, before any implementation, to interview the user one question at a time and resolve design decisions. Triggers on phrases like "grill me", "design review", "before we code", "what do I need to decide", or any greenfield/large-refactor request. Walks down the design tree until intent, constraints, and acceptance criteria are concrete.
allowed-tools: Read, Glob, Grep
license: MIT
---

Interview the user before writing any code, until intent, constraints, and acceptance criteria are concrete enough to implement against. Vendored from Matt Pocock's `skills` repo (MIT). Used to satisfy the "state intent + constraints + acceptance + file locations" requirement in `CLAUDE.md`.

**Tradeoff:** Slower start, far fewer wrong inferences. Skip for trivial tasks (one-line fix, rename, doc tweak).

## Protocol

1. **Walk down the design tree.** Treat the task as a tree of decisions: schema → API → handler → wire format → error shape → tests. Resolve parents before children. Brooks (*The Design of Design*) calls the resolved root the "design concept"; everything else hangs off it.

2. **One question at a time.** Ask a single question, propose a recommended answer with one-line rationale, wait for feedback. Do not batch multiple questions in one turn. Do not present a numbered list of open questions — that defers the work back to the user.

3. **Recommend, don't just ask.** Every question carries your recommended answer. The user is reviewing recommendations, not generating them.

4. **Explore the codebase instead of asking.** If a question is answerable from existing code (current type, current import path, current test pattern), use Read/Glob/Grep first. Only ask the user for things only the user knows (intent, priority, business rule).

5. **Stop when you can write the acceptance criteria yourself.** Once intent + constraints + acceptance criteria + relevant files are concrete, exit the loop and either propose a plan or hand back to the user.

## What "concrete" means here

- **Intent**: one sentence the user would sign off on.
- **Constraints**: what must NOT change (module boundaries, dep list, public API). Reference `.claude/rules/architecture.md` and the highest-stakes invariants in `CLAUDE.md`.
- **Acceptance criteria**: a test or observable behaviour that flips from failing to passing. Per `karpathy-guidelines` §4.
- **Relevant files**: actual paths, found by exploring — not guessed.

## What this skill is NOT

- Not a planning skill. Hand off to plan mode (`defaultMode: "plan"` is on) once aligned.
- Not a review skill. Use `plan-reviewer` (subagent) for adversarial review of the resulting plan.
- Not a substitute for `karpathy-guidelines`. That skill governs *how* to code; this one governs *what* to code.

## When NOT to invoke

- Trivial edits (one-line fix, typo, rename within a file).
- Tasks where the user has already supplied all four of intent / constraints / acceptance / files.
- During `commit-push-pr` or `/go` flows — those are post-implementation.
