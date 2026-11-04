"""Transparent review-list routing; this is not an extra PLN truth valuation.

Scores describe current decisive clinician judgments. Routing also shows how
many clinicians are uncertain, coverage selected by the user and attributed
record concerns. No paper is rewritten and no intervention is deleted.
"""
from __future__ import annotations

CONCERNS={
    'evidence_mismatch':'Evidence does not match this intervention or condition',
    'entry_error':'Information in the intervention entry needs correction',
    'patient_context':'Patient applicability needs checking',
    'safety_context':'Safety or interaction information needs checking',
    'other':'Other record concern',
}
LABELS={'more_support':'More support','needs_review':'Needs review',
        'lower_support':'More opposition','unreviewed':'Not reviewed'}
ORDER={'more_support':0,'needs_review':1,'unreviewed':2,'lower_support':3}


def review_status(assessment,reviews,min_decisive=0):
    if not isinstance(min_decisive,int) or min_decisive<0:
        raise ValueError('Minimum decisive reviews must be a nonnegative whole number.')
    s=sum(r['decision']=='support' for r in reviews)
    o=sum(r['decision']=='oppose' for r in reviews)
    u=sum(r['decision']=='uncertain' for r in reviews)
    n=s+o;total=n+u
    concerns=[dict(reviewer=r.get('reviewer','Reviewer'),clinician_id=r.get('clinician_id'),
                   kind=r['concern_kind'],label=CONCERNS[r['concern_kind']],
                   text=r.get('concern_text',''),review_id=r.get('id'))
              for r in reviews if r.get('concern_kind') in CONCERNS]
    if total==0:status,reason='unreviewed','No clinician responses yet.'
    elif concerns:status,reason='needs_review',f'{len(concerns)} clinician(s) recorded a concern about this entry.'
    elif n==0:status,reason='needs_review','All current responses are Uncertain; no decisive appraisal is established.'
    elif n<min_decisive:status,reason='needs_review',f'{n} decisive reviews; your selected coverage minimum is {min_decisive}.'
    elif u and u>=max(s,o):status,reason='needs_review','Uncertain is the largest response category (or tied for largest).'
    elif s==o:status,reason='needs_review','Support and opposition are tied.'
    elif s>o:status,reason='more_support','More decisive judgments support retention than oppose it.'
    else:status,reason='lower_support','More decisive judgments oppose retention than support it.'
    return dict(status=status,label=LABELS[status],reason=reason,support=s,oppose=o,uncertain=u,
                decisive=n,total=total,decisive_fraction=n/total if total else None,
                strength=(assessment or {}).get('strength'),confidence=(assessment or {}).get('confidence'),
                concerns=concerns,disagreement=bool(s and o),min_decisive=min_decisive)


def priority_key(row):
    c,a,t=row
    return (ORDER[t['status']],-round((a or {}).get('strength') or 0,10),
            -round((a or {}).get('confidence') or 0,10),c['name'].casefold(),c['id'])
