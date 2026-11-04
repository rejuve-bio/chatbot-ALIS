"""Native PLN calculation for clinician support and confirmed reason-specific claims.

This adapter measures recorded judgments, not experimental treatment evidence.
One distinct reviewer contributes one current decisive judgment. Its unit weight
is a declared counting convention, not a learned reviewer-reliability estimate.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from app.services.feedback_reason_schema import ASPECTS

ROOT = Path(__file__).resolve().parents[2] / "runtime" / "feedback"
ADAPTER = "clinician-review-consensus-v1"
SEMANTICS = (
    "Strength is the proportion of current decisive reviewer judgments supporting "
    "retaining this intervention as a research candidate for the stated condition. "
    "Confidence is the native PLN count-weight quantity n/(n+1), using one unit "
    "per distinct decisive reviewer. It measures recorded review volume, not "
    "clinical certainty, evidence quality, treatment efficacy or patient suitability. "
    "Reviewer judgments may be correlated; this adapter does not estimate independence."
)


class NativePLNError(RuntimeError):
    pass


def fingerprint() -> dict:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise NativePLNError("Native runtime checksum mismatch: " + relative)
    return manifest


def execute_native(program: str) -> dict:
    """Execute generated numeric-only code against the pinned, unmodified PLN."""
    manifest=fingerprint()
    executable=shutil.which('swipl')
    if not executable:raise NativePLNError('SWI-Prolog is required; no fallback score was saved.')
    env=dict(os.environ)
    framework=Path('/Library/Developer/CommandLineTools/Library/Frameworks')
    if sys.platform=='darwin' and (framework/'Python3.framework').exists():
        env['DYLD_FRAMEWORK_PATH']=str(framework)+(':'+env['DYLD_FRAMEWORK_PATH'] if env.get('DYLD_FRAMEWORK_PATH') else '')
    try:
        with tempfile.TemporaryDirectory(prefix='alis-source-pln-') as directory:
            tmp=Path(directory)
            shutil.copyfile(ROOT/'vendor/Omega/lib_pln.metta',tmp/'lib_pln.metta')
            request=tmp/'source.metta';request.write_text(program)
            p=subprocess.run([executable,'-q','-s',str(ROOT/'vendor/PeTTa/src/main.pl'),'--','--silent',str(request)],
                             cwd=tmp,env=env,capture_output=True,text=True,timeout=20)
    except subprocess.TimeoutExpired:
        raise NativePLNError('Native source assessment timed out. No replacement score was saved.') from None
    if p.returncode or re.search(r'\bERROR\b|Exception|failed to process',p.stdout+p.stderr):
        raise NativePLNError('Native source assessment failed; no fallback score was saved. '+(p.stderr or p.stdout)[-800:])
    return {'engine':'Omega PLN on PeTTa / SWI-Prolog','omega_commit':manifest['omega_commit'],
        'petta_commit':manifest['petta_commit'],'rule_sha256':manifest['files']['vendor/Omega/lib_pln.metta'],
        'program':program,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),
        'stdout':p.stdout,'stderr':p.stderr,'exit_code':p.returncode}


def assess(reviews: list[dict]) -> dict:
    """Rebuild from current judgments; never revise an aggregate with itself."""
    support = sum(r["decision"] == "support" for r in reviews)
    oppose = sum(r["decision"] == "oppose" for r in reviews)
    uncertain = sum(r["decision"] == "uncertain" for r in reviews)
    decisive = [r for r in reviews if r["decision"] in ("support", "oppose")]
    identities = [r["reviewer_key"] for r in reviews]
    if len(set(identities)) != len(identities):
        raise ValueError("Only one current judgment per reviewer may enter an assessment.")
    result = {
        "adapter": ADAPTER, "semantics": SEMANTICS,
        "support": support, "oppose": oppose, "uncertain": uncertain,
        "strength": None, "confidence": None,
        "conflict": bool(support and oppose),
        "status": "unassessed" if not reviews else "insufficient_decisive_reviews",
        "evidence_confidence": None, "pln_used": False, "native": None,
        "input_review_ids": [r["id"] for r in reviews],
        "confirmed_reason_reviews": sum(bool(r.get("analysis_confirmed")) for r in reviews),
        "aspect_assessments": {},
    }
    aspect_reviews = {aspect: [] for aspect in ASPECTS}
    for review in reviews:
        if not review.get("analysis_confirmed") or not review.get("reason_analysis"):
            continue
        for assessment in review["reason_analysis"]["analysis"]["assessments"]:
            aspect_reviews[assessment["aspect"]].append({"id":review["id"], "reviewer_key":review["reviewer_key"],
                "decision":assessment["decision"], "analysis_id":review["analysis_id"]})
    for aspect, items in aspect_reviews.items():
        if not items:
            continue
        result["aspect_assessments"][aspect] = {"strength":None,"confidence":None,
            "support":sum(r["decision"]=="support" for r in items),
            "oppose":sum(r["decision"]=="oppose" for r in items),
            "uncertain":sum(r["decision"]=="uncertain" for r in items),
            "input_review_ids":[r["id"] for r in items], "analysis_ids":[r["analysis_id"] for r in items]}
    numerical_aspects = {k:[r for r in v if r["decision"] in ("support","oppose")] for k,v in aspect_reviews.items()}
    numerical_aspects = {k:v for k,v in numerical_aspects.items() if v}
    if not decisive and not numerical_aspects:
        return result
    manifest = fingerprint()
    executable = shutil.which("swipl")
    if not executable:
        raise NativePLNError("SWI-Prolog is required. On macOS: brew install swi-prolog. No fallback score was saved.")
    # Only controlled identifiers and binary judgments enter executable MeTTa.
    # Clinical free text, reviewer names and JSON fields are never interpolated.
    lines = ["!(import! &self lib_pln)"]

    def premise_group(items, prefix, output):
        for index, review in enumerate(items):
            value = "1.0" if review["decision"] == "support" else "0.0"
            lines.append(f"(= ({prefix}review_{index}) (stv {value} (Truth__w2c 1.0)))")
            if index == 0:
                lines.append(f"(= ({prefix}combined_0) ({prefix}review_0))")
            else:
                lines.append(f"(= ({prefix}combined_{index}) (Truth__Revision ({prefix}combined_{index-1}) ({prefix}review_{index})))")
        lines.append(f"!(let $r ({prefix}combined_{len(items)-1}) ({output} $r))")

    if decisive:
        premise_group(decisive,"","review_consensus")
    for aspect, items in numerical_aspects.items():
        premise_group(items,aspect+"_","aspect_consensus "+aspect)
    program = "\n".join(lines) + "\n"
    env = dict(os.environ)
    # Homebrew Janus can still reference Xcode's Python framework after Xcode is
    # replaced by CommandLineTools. Resolve the existing framework for this child
    # process; do not patch the native interpreter or change system libraries.
    framework = Path("/Library/Developer/CommandLineTools/Library/Frameworks")
    if sys.platform == "darwin" and (framework / "Python3.framework").exists():
        previous = env.get("DYLD_FRAMEWORK_PATH")
        env["DYLD_FRAMEWORK_PATH"] = str(framework) + (":" + previous if previous else "")
    try:
        with tempfile.TemporaryDirectory(prefix="alis-feedback-pln-") as directory:
            tmp = Path(directory)
            shutil.copyfile(ROOT / "vendor/Omega/lib_pln.metta", tmp / "lib_pln.metta")
            request = tmp / "review.metta"
            request.write_text(program)
            process = subprocess.run(
                [executable, "-q", "-s", str(ROOT / "vendor/PeTTa/src/main.pl"),
                 "--", "--silent", str(request)],
                cwd=tmp, env=env, capture_output=True, text=True, timeout=20,
            )
    except subprocess.TimeoutExpired:
        raise NativePLNError("Native PLN timed out. The review was not saved; retry when the runtime is available.") from None
    if process.returncode or re.search(r"\bERROR\b|Exception|failed to process", process.stderr + process.stdout):
        raise NativePLNError("Native PLN failed; no fallback score was saved. " + (process.stderr or process.stdout)[-1000:])
    values = re.findall(r"\(review_consensus\s+\(stv\s+([-+\deE.]+)\s+([-+\deE.]+)\)\)", process.stdout)
    if len(values) != (1 if decisive else 0):
        raise NativePLNError("Native PLN returned no unique assessment; no score was saved.")
    strength = confidence = None

    def verify_pair(pair, items):
        strength, confidence = map(float,pair)
        n=len(items)
        supporting=sum(r["decision"]=="support" for r in items)
        if (not all(math.isfinite(x) and 0<=x<=1 for x in (strength,confidence))
                or not math.isclose(strength,supporting/n,abs_tol=1e-9)
                or not math.isclose(confidence,n/(n+1),abs_tol=1e-9)):
            raise NativePLNError("Native result violated the review-count adapter contract.")
        return strength,confidence

    if decisive:
        strength,confidence=verify_pair(values[0],decisive)
    aspect_values=re.findall(r"\(aspect_consensus\s+([a-z_]+)\s+\(stv\s+([-+\deE.]+)\s+([-+\deE.]+)\)\)",process.stdout)
    if len(aspect_values)!=len(numerical_aspects) or {row[0] for row in aspect_values}!=set(numerical_aspects):
        raise NativePLNError("Native PLN returned invalid claim-specific assessments.")
    for aspect,s,c in aspect_values:
        s,c=verify_pair((s,c),numerical_aspects[aspect])
        result["aspect_assessments"][aspect].update(strength=s,confidence=c)
    result.update(
        strength=strength, confidence=confidence, pln_used=True,
        status="conflicting_reviews" if result["conflict"] else "reviewed" if decisive else "insufficient_decisive_reviews",
        native={
            "engine": "Omega PLN on PeTTa / SWI-Prolog",
            "omega_commit": manifest["omega_commit"], "petta_commit": manifest["petta_commit"],
            "rule_sha256": manifest["files"]["vendor/Omega/lib_pln.metta"],
            "program": program, "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
            "stdout": process.stdout, "stderr": process.stderr, "exit_code": process.returncode,
            "decisive_review_ids": [r["id"] for r in decisive],
            "aspect_input_review_ids": {k:[r["id"] for r in items] for k,items in numerical_aspects.items()},
        },
    )
    return result
