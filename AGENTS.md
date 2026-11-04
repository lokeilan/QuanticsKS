# Project reporting conventions

- Apply these rules to chat updates, completion messages, README files, reports,
  generated summaries, and report templates.
- Describe the behavior checked, its outcome, and relevant failures or limitations.
  Do not report totals of passed tests, assertions, checks, or test suites unless
  the user explicitly requests those totals.
- Do not use SHA-256 generation, checksum manifests, or checksum comparisons as
  a validation or acceptance requirement. Do not add them to reports or reporting
  workflows unless explicitly requested by the user.
- Keep scientific measurements such as residuals, errors, timings, SCF iterations,
  and TCI operation counts. Distinguish energy stopping from physical acceptance.
- When correcting a report, update its generator as well. Preserve raw logs,
  checkpoints, frozen source snapshots, and historical receipts.
- Use concise English code comments. Do not generate a code package unless asked.

# Reasoning and prioritization

- Start from the algorithm's natural workflow and each operation's intended
  responsibility. Ask why an operation is needed and whether an earlier step
  has already done that work before defending the current implementation.
- State the main problem, the main judgment, and the order of work first.
  Discuss technical details and edge cases when they materially change the
  decision; do not let them obscure the main issue.
- Do not treat a mathematically possible use for an operation as evidence that
  adding it unconditionally was a sound design. Recognize duplicated work and
  design mistakes directly instead of constructing increasingly detailed
  defenses of the existing code.
- Use validation to check fixes and relevant numerical requirements. Missing
  validation or hypothetical risks do not establish that existing redundant
  work is necessary. Keep required correctness constraints explicit without
  using them to postpone the main engineering judgment.
- When corrected, acknowledge the concrete mistake, update the approach, and
  proceed. Avoid repeated qualifications that amount to defending the mistake.
- In QTT/TCI work, review construction, rank control, canonicalization, and
  consumer needs as separate responsibilities. Do not mechanically append a
  full SVD rounding to every TCI/prrLU output; any additional operation should
  have an explicit consumer requirement and a clear place in the workflow.
