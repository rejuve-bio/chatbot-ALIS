"""Pure, versioned policy for the post-intervention reviewed atlas.

The native Omega tool invokes this policy. It consumes saved PLN appraisals;
it does not manufacture another truth value or interpret a review note.
"""
from __future__ import annotations

import hashlib
import json

from app.services.feedback_triage import review_status, ORDER

POLICY = 'reviewed-atlas-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                   separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def assemble(context):
    groups = {}
    for entry in context['entries']:
        row = dict(entry)
        row['routing'] = review_status(row['appraisal'], row['reviews'], context['minimum_decisive'])
        groups.setdefault((row['area_index'], row['condition'], row['category']), []).append(row)
    result = []
    for rows in groups.values():
        # Unreviewed items retain the supplied order, not an invented score.
        rows.sort(key=lambda r: (ORDER[r['routing']['status']],
                  -round(r['appraisal']['strength'] or 0, 10),
                  -round(r['appraisal']['confidence'] or 0, 10), r['entry_number']))
        for order, row in enumerate(rows, 1):
            row['research_order'] = order
            result.append(row)
    return {'policy': POLICY, 'dataset_id': context['dataset_id'],
            'minimum_decisive': context['minimum_decisive'], 'entries': result}


def changes(previous, current, context):
    """Explain actual persisted changes, including unchanged scores after note edits."""
    if previous is None:
        return []
    old = {r['id']: r for r in previous['entries']}
    events = {r['id']: r.get('events', []) for r in context['entries']}
    result = []
    for row in current['entries']:
        prior = old.get(row['id'])
        fields = ('version_id', 'source_version', 'research_order', 'statement', 'reviews', 'event_head')
        if prior and all(row[k] == prior[k] for k in fields) and row['routing'] == prior['routing']:
            continue
        keys = ('research_order', 'version_id', 'statement')
        def state(r):
            if r is None:
                return None
            return dict({k: r[k] for k in keys}, status=r['routing']['label'],
                        strength=r['appraisal']['strength'], confidence=r['appraisal']['confidence'],
                        support=r['routing']['support'], oppose=r['routing']['oppose'],
                        uncertain=r['routing']['uncertain'], concerns=len(r['routing']['concerns']))
        causes = [e for e in events[row['id']] if e['id'] > (prior['event_head'] if prior else 0)]
        result.append(dict(id=row['id'], entry_number=row['entry_number'], name=row['name'],
                           condition=row['condition'], before=state(prior), after=state(row),
                           cause_events=causes,
                           reason=('Evidence/statement version changed; only current-version reviews apply.'
                                   if prior and prior['version_id'] != row['version_id'] else
                                   'Review record changed; its current judgment and concerns were reapplied.'
                                   if causes else 'Order or status changed when the current review policy was reapplied.')))
    return result
