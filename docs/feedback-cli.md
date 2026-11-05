# Clinician review

Run from the repository root with your authorised intervention handoff:

```bash
python3 scripts/feedback --input 'path/to/interventions.json'
```

Sign in, choose an intervention number, read the evidence and current assessments,
then select **1 Support**, **2 Oppose** or **3 Uncertain**. Add an optional note or
press Enter. Native PLN updates the appraisal. Omega then reads the current
reviews and assembles the next **reviewed intervention atlas** automatically.

## See the feedback change the next atlas

The normal workflow is **read evidence → give judgment → see the update**.
After saving, the screen shows the previous and current strength/confidence,
the research-list status and order, and the specific review event responsible.
If a note changes without changing the appraisal, the screen says so. Other
entries moving down because one moved up are summarized separately.

The list on login is read from this reviewed atlas. **r Reviewed atlas / changes**
shows its latest version, reviewer history, changes, an atlas export and a plain
explanation of the scores. Entry numbers remain fixed even when the research
order changes. Runtime details and native calculations are available in the
advanced interface.

Changing your assessment, withdrawing it, reporting a concern or clearing your
own concern all refresh the atlas. A later login loads the saved result. If
another reviewer has changed the inputs, the next refresh rebuilds it. An
unchanged request reuses its verified atlas without another runtime execution.

The exported atlas preserves the supplied evidence and adds current feedback,
source/evidence version, research order and an audit trail. It is a new file;
the supplied JSON is never overwritten. Reviews apply only to the matching
patient, condition, intervention and evidence/statement version.

## What executes

The pinned **native Omega Core loop** executes `read-current-feedback`,
`apply-review-policy` and `finish-reviewed-atlas` on PeTTa. Its local symbolic
workflow planner uses Core's provider interface to select these three tools.
There is no LLM request, text-to-score inference or remote model decision in
this path. The tools consume the native PLN results saved with the review.

The record/ordering policy is application code; the numerical review revision
is the unmodified native `Truth__w2c` and `Truth__Revision`. The screen and stored
trace distinguish these operations. SQLite stores versioned appraisals and
atlas snapshots, so feedback changes subsequent outputs across sessions.

This completes the post-intervention **CLI** feedback-to-atlas workflow. It
starts with the intervention JSON; upstream discovery and the legacy web atlas
endpoint are separate. Learned reviewer/source reliability weights and LLM
training are not part of this count-based revision model.

The existing Omega container must be running (`chatbot-alis-omega-mapping-1`,
or `OMEGA_FEEDBACK_CONTAINER`). If assembly fails, the review stays saved, the
screen marks the atlas as pending, and **r** retries the local assembly. A failed
or stale runtime result is never presented as a freshly updated atlas.

## What the numbers mean

Strength is the fraction of **decisive** current judgments that support research
retention. Review-count confidence is N/(N+1), where N = Support + Oppose.
Uncertain is saved, visible and counted separately. It adds neither a positive
nor negative premise and is never converted into an invented 0.5 vote. All
Uncertain means unknown strength/confidence, not opposition. The display shows
how many of all reviewers made a decisive judgment.

These are clinician-appraisal counts, not scientific or treatment-certainty
probabilities. Reasons are shared notes, not numerical weights.

## Useful review status and filters

- **More support:** Support exceeds Oppose among decisive reviewers.
- **More opposition:** Oppose exceeds Support; deprioritize for research review.
- **Needs review:** a recorded entry concern, no decisive responses, uncertainty
  is the largest category (including ties), tied Support/Oppose, or insufficient
  coverage for the minimum you selected.
- **Not reviewed:** no current assessments.

The status routes the review work; it is not another PLN truth value. Concerns
and uncertainty remain visible even when the decisive-support score is high.
They do not erase other clinicians' judgments. Within a status, entries are
ordered by strength then review-count confidence. Equal support is ordered by
review coverage, without multiplying strength and confidence together.

Use **v** on the intervention list to view these groups or set a minimum number
of decisive reviews. The default minimum is zero: the application does not invent
an institutional approval threshold. Entries below a chosen minimum move to
Needs review. Filters only change the display; no record is deleted.

## Corrections concern the assessment or intervention record

Choose Support/Oppose/Uncertain again to correct **your own assessment**. It
replaces your prior contribution; history is preserved. A decisive-to-Uncertain
change removes that person's previous positive/negative contribution and rebuilds
native PLN. A new decisive judgment replaces Uncertain without adding a duplicate.

**5 Report or clear a correction / concern** records an attributed issue about
mismatched evidence, entry information, patient applicability, safety/interaction
context or another record concern. The author supplies what needs checking.
It is a clinician-reported issue, not an automatically verified scientific fact.
The entry becomes Needs review while its score and other assessments stay visible.
A clinician can clear their own concern after it has been addressed; earlier
versions and other clinicians' concerns remain. Clearing a concern does not
change Support/Oppose or rewrite the publication. Ordinary review changes carry
an outstanding concern until its author explicitly clears or withdraws it.

**4 More evidence / history** opens the complete source, other linked studies,
current clinician responses and previous assessment/correction events. Statement
rewriting is no longer the normal correction path. The earlier technical tools
remain available under `--advanced`.

## Accounts, scope and persistence

**a** switches reviewer for the selected file using existing signup/login;
**f** opens another handoff; **q** exits. Reviews remain tied to authenticated
clinician IDs and the patient/condition/evidence version. Patient-specific
concerns are not silently shared as general drug facts. A new evidence version
requires its own assessments; earlier events remain in history.

This is the clinician's research review of a patient's candidate list. A More
support status does not establish patient eligibility or become a treatment
recommendation. No patient portal or automated prescribing is introduced.

For separate test identities use `--auth local --db PATH`. The technical
[command reference](feedback-cli-advanced.md) is optional. No model call is made
for review, correction flags, scoring or filtering; retrieving a linked public
abstract sends only its publication ID to Europe PMC.

## Optional Omega review brief

Open an intervention, choose **4 More evidence / history**, then **5 Omega review
brief**. Omega explains what the displayed abstract reports, summarizes the
reasons actually expressed in selected reviewer notes, and suggests up to two
questions for further review. Each explanation includes a checked source quote
or review ID. This is a draft interpretation for inspection.

This feature is on request and separate from saving, native PLN revision and
atlas ordering. It makes at most one configured-model call and never changes
judgments, source facts, scores, concerns or ranking. Reading an unchanged brief
uses the saved result without another model call. Changes to the inspected
evidence or review context invalidate that brief. If a request fails, the
ordinary loop keeps working; another paid attempt requires an explicit retry.

The brief uses the existing native Omega Core loop with a dedicated profile and
the configured provider/model. It sends the displayed linked abstract and up to
two latest nonempty notes per judgment category. Counts and selection coverage
remain visible. Patient identifiers, clinician account details and known names
are removed/redacted. Existing external-text authorization is reused; otherwise
the clinician authorizes this optional export before the model is called.

## Terminal layout

The guided interface uses a compact table, separate evidence/appraisal/note
sections and a persistent before/after panel after saving. List pages adapt to
the terminal height; **n/p** moves between pages. Any stable entry ID can be
opened from any page. Your current response remains visible on the review card.

Full abstracts, reviewer history and Omega briefs use a reader
with **Enter/n** for the next page, **p** for the previous page and **0** to return.
The review choices and evidence/history menu keep their existing numbers.

Colors are automatic on supported terminals. Set **NO_COLOR=1** to disable
color, or **FEEDBACK_NO_CLEAR=1** to preserve screen output while debugging.
Piped output is plain and includes the complete requested document. No extra
Python packages or Docker rebuild are needed. The data and numerical rules are
unchanged.
