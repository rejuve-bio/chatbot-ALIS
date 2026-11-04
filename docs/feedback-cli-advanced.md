# Technical feedback interface

This CLI starts at the intervention JSON handoff. It runs independently of the
chatbot API and Qdrant. ALIS login uses the existing backend. The native Omega
container is available for explicit source inspection with `assess N`. Reviews
make no model calls: reasons are optional shared notes. `--omega off` disables
source-inspection model access too. Separate local test accounts work offline. It imports both the fuller
`treatments` / `investigational_evidence` format and the earlier `evidence` format.

## Start

Requires Python 3.9+ and SWI-Prolog 9.3+ (`brew install swi-prolog` on macOS).
The CLI uses only the Python standard library. The existing native Omega
container holds its configured model and credentials; the CLI does not copy its
API key. Manual mode needs no model key or container.

```bash
python3 scripts/feedback --advanced --input 'path/to/interventions.json'
```

The ALIS account menu is the same flow as the earlier CLI:

1. Log in as a clinician.
2. Log in as a clinic.
3. First-time setup: create a clinic and its first clinician.
4. Add a clinician to an existing clinic (requires that clinic administrator's login).

If you have never registered with this ALIS backend, choose **3**. Option **4**
asks for an existing clinic administrator's credentials before it collects the
new clinician's details. A clinician's own login belongs in option **1**.

Creating a clinic guides you directly into creating its first clinician, using
the normal clinic login token returned by ALIS. A clinic login can create more
clinicians; only authenticated clinician accounts submit reviews. Successful
signup prints the account name, email and ID, separately from login status.

Log in with an ALIS **clinician** account. The API address comes from `--api-url`,
`ALIS_API_URL` or the repository's `.env`, defaulting to `http://127.0.0.1:5001`.
Clinic administrator accounts can create clinician accounts through the login
menu, but cannot submit clinician reviews themselves. The CLI gets clinician
identity from the trusted login response. In the default file-review mode,
`--input FILE.json` or `import "FILE.json"` explicitly opens the supplied handoff
for the signed-in clinician. The file's patient ID scopes the reviews; no live
patient record is requested and no clinic assignments are changed.
Remote login requires HTTPS; HTTP is accepted only on loopback.
Before asking for credentials, the CLI checks ALIS health. For this installed
localhost:5001 stack, it can start Docker Desktop and resume the existing API
and database containers. It never rebuilds the older backend checkout or
recreates databases/accounts. Use `--no-start-backend` to perform only the health
check. Remote/custom backends must already be running.

## Sign up multiple clinicians using ALIS

Choose **Sign up a clinician** in the login menu, or start directly with:

```bash
python3 scripts/feedback --advanced --signup --input 'path/to/interventions.json'
```

This uses the same existing ALIS endpoints as its frontend: clinic administrator
login at `POST /clinics/login`, clinician creation at `POST /clinicians`, and
clinician sign-in at `POST /clinicians/login`. The clinician is created in ALIS,
not in a replacement local account system. A clinic administrator's credentials
are required by ALIS's existing registration flow.

Inside the CLI, `signup` opens registration for another clinician. Each account
uses its own email and the ID returned by ALIS. `logout` and `login` switch between
the registered accounts. Reviews remain associated with their original author.

Creation outcomes are explicit:

- **CREATED successfully**, with email and clinician ID: registration completed.
- **ALREADY EXISTS**: no additional account was created; use login.
- **NOT completed**: ALIS rejected registration.
- **Could NOT BE CONFIRMED**: a connection/response problem means the server's
  final state is unknown; try login before repeating registration.

Account creation is confirmed before automatic sign-in. If sign-in fails after
creation, the CLI explicitly says that the account exists and should not be
created again. `--register` remains an alias for `--signup`.

The default SQLite database is `.local/feedback/reviews.sqlite3`, ignored by Git.
Reopen it and log in to see previous reviews for a file you have already opened:

```bash
python3 scripts/feedback --advanced
```

When the backend is offline, separate password-protected local test accounts are
available explicitly:

```bash
python3 scripts/feedback --advanced --auth local --input 'path/to/interventions.json'
```

On the first run, an empty account database opens account creation directly.
Enter your name, email and a password of at least 10 characters, then confirm
the hidden password. No test accounts are preloaded. Subsequent sessions use
the same email/password and the same database. To create another account directly:

```bash
python3 scripts/feedback --advanced --auth local --signup
```

The login screen shows which account database is in use. An unavailable ALIS
backend is checked and the installed local stack resumed before credentials are
requested. Invalid credentials allow at most three attempts
and then stop with an actionable message, rather than repeating the menu forever.
Local
accounts use `.local/feedback/local-reviews.sqlite3`; they do not create ALIS
accounts or verify professional credentials. Passwords are stored only as salted
PBKDF2-HMAC-SHA256 hashes (600,000 iterations, random per-account salts). Login
tokens and plaintext passwords are never saved. ALIS and
local accounts cannot be mixed in the same review database.

File review access is stored per clinician and exact handoff content, alongside
the source filename and time. There is no automatic access to every saved file.
For a second clinician, run the same command with `--input`, or log in and use
`import "FILE.json"`. They can then see the existing reviews of that evidence.
An updated file must be opened by each clinician; previous reviews and history
remain preserved. Only open files you are authorised to review.

For deployments that require live ALIS patient permissions as well, use:

```bash
python3 scripts/feedback --advanced --access patient --input 'path/to/interventions.json'
```

This mode checks patient access with ALIS before import, display and review. A
new clinic does not inherit another clinic's patient access. A rejected live
patient check does not grant file access or change backend permissions.

## Improve the claim and reuse the improvement

`claim N` proposes precise review wording with an exact quote from a paper
already linked to that entry. It uses a saved abstract or retrieves the exact
PMID from Europe PMC; no model call or patient information is involved in that
retrieval. The quoted text is checked against the saved source. A different
clinician approves or rejects the wording using `claims N` and `accept ID` or
`reject ID`. Quote matching does not establish that the interpretation is sound;
that is the explicit human approval step.

Approved wording becomes the actual statement shown by `show N` and `review N`.
The new claim has its own review version. Earlier judgments stay in history.
The wording, quote, public-source snapshot and approval identities are reused
for later handoffs with the exact same intervention, condition and upstream
evidence entry. Patient-specific reviews are kept separate. Original files are
preserved. A changed evidence snapshot needs a new approval.

See [the claim-review walkthrough](feedback-claim-demo.md) for a short, complete
two-account demonstration. Ordinary review notes remain display-only.

## Try a complete loop

Choose any intervention number from `list`. The following uses entry 1 only as
an example; no intervention, condition or patient is hardcoded.

```text
show 1
review 1
```

`review N` first displays the intervention, the review statement, the current
score and the saved clinician responses, including names, dates, reasons and
references. Disagreement is shown before the question. Your own existing
response is marked; changing it replaces that contribution rather than adding
another reviewer. `show N` also includes the score and clinician responses.

The CLI then asks for support, opposition or uncertainty, an optional note for
other clinicians and a separate optional reference. No reason interpretation,
consent request or model call occurs during `review N`. You are reviewing this explicit statement:

> The supplied evidence supports retaining this intervention as a research
> candidate for the stated condition.

This is an assessment of a research candidate. Patient eligibility and treatment
decisions are separate. Supplied approval labels and summaries remain attributed
to the upstream service; the CLI does not verify or endorse them.
`details N` displays the full supplied rationale, interaction/safety text and
gene annotations when those fields exist.

To inspect a second review in a local demonstration:

```text
logout
login
import "/path/to/the-same-handoff.json"
review 1
reviews 1
history 1
trace 1
```

`login` requires authentication as the other account; a name cannot switch the
review author. `whoami` shows the authenticated clinician ID, name and email.
Each review and correction links to its author's stable account ID. Display
names can be identical without merging accounts. One account's repeated reviews
replace its own contribution. Existing name-only reviews remain in history as
unverified and no longer contribute to scores or correction approvals.

All authorised reviewers using the same database can see saved responses.
Multiple computers need a shared application
service rather than independent database copies; SQLite files should not be put
on a network filesystem.

## Clinician judgments and shared notes

The primary score now describes **clinician support for retaining the supplied
intervention/condition research claim**. It is an experimental appraisal measure,
not treatment effectiveness, patient eligibility or calibrated clinical certainty.

The original handoff is preserved. It contains no initial PLN truth value. Its
labels (`human_and_animal`, collection names and gene/orthology annotations) are
not converted into quality, applicability or confidence defaults. Entries start
unreviewed. A referenced clinician appraisal establishes the first numerical input.

### One review

```text
show 7
review 7
reviews 7
trace 7
rank Cancer
```

`review N` shows the existing score and current clinician responses before asking
for Support/Oppose/Uncertain. An optional reason is a shared note. An optional
public reference records what the reviewer consulted. Neither changes numerical
weights or source facts; neither is required to save or count a judgment.
Reasons are not sent to a model or interpreted during review. A note cannot
override the selected judgment. Existing interpreted reviews remain in history;
current appraisals use the saved clinician selections, including for older reviews.

### The numerical contract

The observation is a clinician's selected judgment about retaining the supplied
intervention as a research candidate for the stated condition. It is not a
participant's treatment response or a new clinical study.

- Support contributes the binary value 1; Oppose contributes 0.
- Each authenticated clinician has one current judgment and one count unit.
- Strength is Support / (Support + Oppose).
- **Review-count confidence is N/(N+1)**, where N = Support + Oppose.
- Uncertain is visible but contributes no binary evidence. With no decisive
  reviews, both strength and confidence remain unknown.
- Changing a judgment replaces the old contribution. Withdrawal removes it.
  Repeating a submission from the same clinician never increases the count.
- Citations remain visible as provenance. Reviews of the same paper still count
  as distinct clinician judgments; they are not counted as additional studies.

The pinned native Omega PLN `Truth__w2c` uses k = 1 to convert a unit weight into
confidence. Native `Truth__Revision` combines the binary appraisal inputs. The
application verifies and stores the actual native result and calculation trace.
The one-unit convention measures review volume; it is not a learned reliability
weight. The interface calls it **review-count confidence** because additional
accounts do not establish independent expertise or scientific certainty.
`calibrated` is false and `scientific_confidence` remains null.

| Current judgments | Strength | Review-count confidence |
| --- | ---: | ---: |
| 1 Support | 1.000 | 0.500 |
| 2 Support | 1.000 | 0.667 |
| 3 Support | 1.000 | 0.750 |
| 2 Support, 1 Oppose | 0.667 | 0.750 |
| 2 Support, 1 Uncertain | 1.000 | 0.667 |

The confidence column counts decisive judgments; it can rise while disagreement
remains. The Support/Oppose counts and disagreement marker must be read alongside
it. It does not resolve which reviewer is correct.

### Ranking, persistence and learning

`list` and `rank` order by **strength**, using review-count confidence to break ties.
The visible 0–100 score is `100*strength`. Equal strength/confidence pairs share a rank; stable
entry IDs remain the arguments to `show N` and `review N` after rank changes.
The former `100*strength*confidence` source-statistics product is not used for
ranking: an opposing input could increase that product as confidence rose.

A new selected judgment rebuilds the aggregate from current clinician judgments. A changed note alone leaves the numerical result unchanged.
Replacement and withdrawal remove the superseded contribution; no old aggregate
is revised with itself. Disagreement remains visible.

Versioned appraisal snapshots include the input review IDs, public-reference
groups, rule/profile hashes and actual native MeTTa program/output. `trace N`
shows the calculation. `history N` and `learning N` show belief changes. Existing
`teach`/approval/reuse operations provide a second learning path for source-cited
knowledge corrections; they do not train numerical weights or a language model.

The versioned SQLite migrations preserve existing reviews, source-statistics history,
accounts, corrections and prior appraisal snapshots. Current authenticated
reviews are recalculated with the reviewer-count policy. Earlier reference-group
confidence values remain labelled as the previous policy in history. Changed upstream evidence gets a new
claim version and requires its own reviews. Legacy name-only reviews stay excluded.
The JSON itself is not overwritten.

### Source facts and bounded model use

`assess N`, `evidence N`, `source-text N PMID:ID`, `correct-source N` and
`restore-evidence N EVENT "reason"` remain available to inspect and correct the
source record. Comparative estimates and confidence intervals are shown as study
facts. The earlier source-statistics adapter remains a separate experimental
calculation and historical export field; it does not initialize the appraisal or
supply its clinical confidence. Known study IDs keep source overlap visible as provenance; they do not change
the number of clinician appraisals. No extra scientific-study weight is created
by recording another review.

Reviewing, listing, ranking and reading traces make no model calls. Only an
explicit source inspection (`assess N`) sends its bounded source-assessment
packet to Omega after consent and identifier redaction. Unchanged inspections
reuse the saved result; `assess N --retry` explicitly retries a failed inspection.
Source inspection is not required to save a clinician judgment and note.
A native PLN failure rolls back the attempted review and snapshot together.

For a clean numerical test without a paid model call:

```bash
python3 scripts/feedback --advanced --auth local --omega off \
  --db .local/feedback/appraisal-demo.sqlite3 \
  --input path/to/interventions.json
```

Create a local test account, submit a referenced review, then use `signup` or
`login` for another account. Each account opens the authorized file with
`import "path/to/interventions.json"` (or by restarting with `--input`).
It then sees the current score and the earlier clinician's response. Local
accounts are test identities and do not verify clinical credentials. For ALIS
accounts use the normal command without `--auth local`; existing signup/login
and file/patient access controls are preserved.

## Correct an assessment

```text
correct 1
corrections
logout
login
accept 1
```

Text corrections are proposed first. A different authenticated account must explicitly
accept or reject them. An accepted correction creates a new local evidence
version; the original JSON remains unchanged. Previous reviews are retained in
history, and the new version starts unassessed until it is reviewed. This avoids
pretending that a text correction supplies a new numerical scientific premise.

The correction fields are `why`, `note`, `warning`, `evidence` and `evidence_tier`.
They cannot be used to edit an intervention identity or to execute source text.
Only text-valued fields can be corrected through this first CLI.

## Learn a reusable evidence correction

`correct N` remains local to that patient/condition entry. `teach N` explicitly
proposes a general correction to the supplied evidence explanation (`why` or
`note`) for reuse. It requires a public source reference and approval by a
different authenticated clinician. Ordinary support/opposition votes do not
create reusable knowledge.

```text
teach 1
```

Enter the field, corrected explanation, reason and public PMID, NCT, DOI or HTTPS
reference. Include evidence-level facts, not patient observations or eligibility
decisions. The proposed correction has no effect until another clinician opens
the same handoff, inspects `corrections`, and accepts its correction ID:

```text
accept 1
learning 1
```

The acceptance view shows the proposed text, reason, reference and reuse scope.
The accepted correction becomes a versioned knowledge record with both clinician
identities and their reasons. The reference is recorded for review; the CLI does
not automatically verify the paper's contents.

When a later handoff has the same intervention, condition, section, drugs and
exact upstream evidence entry, the correction is applied automatically as a
local overlay. The original JSON is unchanged. `list`, `show N` and `review N`
identify the learned knowledge; `learning N` shows the original and corrected
text, source, approval and match decision. The later record keeps its own
patient-specific clinician judgments. They are not copied from the earlier case.

Changed sources or claim context prevent automatic reuse. An explicit local
correction is preserved rather than overwritten by shared knowledge. A new
approved teaching correction can supersede an earlier one; both remain in the
audit history. The original approver can stop reuse with:

```text
unlearn 1 "Reason for withdrawing this shared evidence correction"
```

Retirement updates derived assessments and preserves history. Explicit local
corrections remain in place. A corrected or restored evidence version needs its
own reviews. This feature learns reusable knowledge; it does not train numerical
weights or the language model. Scientific confidence is still not supplied by
the handoff, and no numerical evidence prior is invented.

## New handoffs and exports

```text
import "/path/to/new-handoff.json"
export "/path/to/new-feedback-export.json"
```

Identical source entries retain their reviews and accepted local corrections.
Changed source entries get a new evidence version and require fresh review. A
pending correction against old evidence cannot be accepted. Removed entries stay
in the database history but disappear from the active handoff list.

Stable identities use patient, condition, section and upstream intervention ID
when available, otherwise its normalised name. Upstream IDs should be supplied
when names can change. Drug classes and individually named compounds remain
separate review records. This version scopes all reviews to their handoff's
patient/condition entry. Only explicitly taught, source-cited corrections approved
by a different clinician can be reused across matching handoffs. Patient-specific
votes, eligibility and local corrections remain scoped to their own records.

Exports contain author account IDs and public account profiles, the original handoff, effective assessment versions, clinician
responses, correction decisions, learned knowledge and its approval/retirement
history, score history and native calculation traces.
Passwords, password hashes and tokens are excluded from exports. Export refuses to overwrite an existing file. The local database and new export
files are created with private permissions.

## Native execution and failures

`runtime/feedback/manifest.json` pins the upstream Omega rule file and a minimal
PeTTa interpreter source snapshot. Upstream sources and licenses are included
unchanged. Their hashes are checked before every numerical calculation. The
rules are executed by SWI-Prolog; Python handles storage, validation and display.
Only generated numerical judgments enter MeTTa, never clinical free text.

Every calculation stores its MeTTa program, native stdout, rule hash, pinned
versions and input review IDs. `trace N` displays this saved trace; submitting a
new or edited referenced review performs a live native calculation. Numerical
recalculation makes no LLM call; Omega interpretation, when enabled, is its
separate bounded request.

If the runtime fails or times out, the transaction rolls back: neither the new
review nor a fabricated replacement score is saved. The original handoff remains
available. On macOS, the child process uses the installed CommandLineTools Python
framework when necessary for Homebrew's Janus dependency; no system files change.

Lifecycle checks:

```bash
python3 -m unittest discover -s tests -p 'test_feedback.py' -v
```

The implementation was also exercised through actual interactive CLI sessions
with the supplied full JSON. Demonstration judgments are isolated in a separate
verification database, never the default review database.
