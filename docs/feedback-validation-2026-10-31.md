# Feedback calculation verification — 31 October 2026

Historical verification of the earlier reference-group policy. This is not the
current calculation contract. The current reviewer-count policy and guided
workflow are described in [Clinician review](feedback-cli.md#what-the-numbers-mean).

The policy tested on this date was `reason-grounded-appraisal-v1`. It measures referenced
clinician support for research retention, with provenance-group count confidence.
It does not claim calibrated scientific confidence or learn numerical reviewer
reliability. The reference-group results below are retained as historical verification;
current reviews count decisive clinicians rather than reference groups.

## Actual runs

- Imported the supplied `full_temp.json`: 36 intervention entries across three
  conditions. No intervention or condition was special-cased in the scorer.
- Used the actual terminal CLI with two separately authenticated local test
  accounts and an isolated database. The second account opened the same handoff
  and saw the first account's review before submitting its own.
- A referenced Support review initialized appraisal strength 1.0, count
  confidence 0.5. Opposition on the same reference changed strength to 0.5 while
  confidence remained 0.5. The CLI displayed disagreement and revised ranking.
  Withdrawal restored the previous aggregate. `trace` displayed actual native
  MeTTa inputs/output; `learning` displayed the successive belief snapshots.
- Ran one **fresh**, bounded native Omega/MiniMax interpretation of a public
  Captopril/Cancer evidence-appraisal reason. Runtime Omega commit was
  `b78c774061f559e62b5b4a2a40e39c89cd1aa403`; configured model was
  `minimax/minimax-m3` on ASI Cloud. Usage was one model call, 2,442 prompt tokens,
  1,058 completion tokens, 26.06 seconds. No patient/account identifiers or login
  credentials were included in the outbound packet.
- Omega proposed opposition to evidence sufficiency and human applicability;
  benefit and safety remained uncertain. Confirming that interpretation in the
  isolated test database changed overall strength from 0.5 to 0.0, with count
  confidence still 0.5. Scoped aspect beliefs were calculated by native PLN.
- Backed up the existing review database read-only, migrated the isolated copy
  to schema version 7, and recomputed current appraisals. All 10 existing review
  events were preserved. Row-content hashes for accounts, reviews, saved Omega
  interpretations, source events/scores, corrections and reusable knowledge were
  unchanged. The original database was not migrated or used for test judgments.

## Regression checks

The full 70-check feedback suite passed, including the localhost ALIS
signup/login protocol fixture. Two additional regression checks cover uncited
judgments receiving no invented numerical weight and ranking by strength when a
confidence product would reverse the order. The resulting 10-check appraisal
module passed against the native runtime.

Other checked behaviors include transitive source grouping, shared study IDs,
replacement, withdrawal, restart/export, changed-evidence invalidation, exclusion
of unauthenticated legacy reviews, scoped reason assessments, unknown effects,
and transactional rollback on native calculation failure.

The actual model run and terminal checks complement the automated checks. The
local test identities do not verify professional qualifications; no real
clinician password was used for a live ALIS login. Existing ALIS authentication
and file/patient access controls remain in place.

## Preserved boundaries

The original JSON, provider settings, pinned Omega/PeTTa sources and their
licenses are unchanged. Historical source-statistics results are preserved but
no longer determine the displayed ranking. New appraisals are versioned in a
separate table; startup explains the policy migration.
