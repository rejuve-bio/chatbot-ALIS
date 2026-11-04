# Native feedback runtime

Unmodified upstream calculation sources used by the clinician-review CLI:

- Omega `lib_pln.metta`: commit `b78c774061f559e62b5b4a2a40e39c89cd1aa403`.
- PeTTa minimal Prolog interpreter sources: commit
  `43705f5d9ff8958ffe7f0aa6777fb8477f2401f2`.

Repository URLs, hashes and provenance are in `manifest.json`. Upstream licenses
are preserved under `vendor/Omega/LICENSE` and `vendor/PeTTa/LICENSE`. The CLI
checks these hashes and executes the native `Truth__w2c` and `Truth__Revision`
functions through system SWI-Prolog 9.3+. Python validates inputs and stores the
results; no fallback arithmetic or generic STV placeholder is used.

Each authenticated clinician contributes one current decisive judgment per entry
and evidence version: Support contributes strength 1, Oppose contributes 0, and
each has one declared unit of count weight. Uncertain stays visible without a
positive or negative premise. With N decisive reviewers, confidence is N/(N+1).
These numbers describe recorded judgments, not treatment effectiveness or
calibrated scientific certainty. Optional reasons are shared notes, not weights.

## Omega profiles

These chatbot-owned profiles execute the unmodified native Omega Core already
installed in the configured container. Clients check its pinned commits and the
isolated runners verify its upstream source hashes. The existing container and
its provider configuration are required separately; the vendored files here
provide the local numerical interpreter, not a complete container installation.

- `atlas/`: reads saved reviews, applies the declared research-list policy and
  assembles the next reviewed atlas using Core's loop on PeTTa. A local symbolic
  planner selects the tools. It makes zero LLM calls.
- `brief/`: optional read-only explanation of selected evidence and review notes.
  It uses the configured provider for at most one model call, checks quotations,
  caches the result and cannot change reviews, scores or rankings.
- `agent/`: source-assessment/reason tools retained for the advanced interface.
  This is separate from the ordinary judgment-and-note workflow.

External text export requires authorisation for the selected handoff. Profiles
are request-isolated and do not change the Core source or container image. See
[the CLI guide](../../docs/feedback-cli.md) for setup and numerical semantics.
