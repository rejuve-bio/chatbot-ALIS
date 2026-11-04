"""Read-only feedback tools invoked by the actual native Omega loop."""
import json
import os
from pathlib import Path
from atlas_policy import assemble, digest


def init_memory():
    Path('workflow.json').write_text(json.dumps({'stage': 0, 'history': []}))
    return 'ATLAS-MEMORY-READY'


def history():
    return Path('workflow.json').read_text()


def record_step():
    return 'ATLAS-STEP-RECORDED'


def step(tool, stage, detail):
    value = json.loads(Path('workflow.json').read_text())
    if value['stage'] != stage:
        raise ValueError('Atlas tools were called out of order.')
    value['history'].append(dict(tool=tool, **detail))
    value['stage'] += 1
    Path('workflow.json').write_text(json.dumps(value))


def read_feedback():
    context = json.loads(Path('request.json').read_text())
    step('read-current-feedback', 0, dict(input_digest=digest(context), entries=len(context['entries']),
         current_review_ids=[r['id'] for e in context['entries'] for r in e['reviews']]))
    Path('inspected.json').write_text(json.dumps(context))
    return 'CURRENT-FEEDBACK-INSPECTED'


def apply_policy():
    context = json.loads(Path('inspected.json').read_text())
    atlas = assemble(context)
    Path('atlas.json').write_text(json.dumps(atlas))
    step('apply-review-policy', 1, dict(policy=atlas['policy'], output_digest=digest(atlas)))
    return 'REVIEW-POLICY-APPLIED'


def finish():
    atlas = json.loads(Path('atlas.json').read_text())
    step('finish-reviewed-atlas', 2, dict(output_digest=digest(atlas)))
    manifest = json.loads(Path('/opt/runtime/manifest.json').read_text())
    result = dict(atlas=atlas, input_digest=digest(json.loads(Path('request.json').read_text())),
                  tool_trace=json.loads(Path('workflow.json').read_text())['history'], model_calls=0,
                  runtime=dict(core_commit=manifest['core_commit'], petta_commit=manifest['petta_commit'],
                               profile_sha256=os.environ['ATLAS_PROFILE_HASH'],
                               execution='native-omega-core-tool-loop', planner='local-symbolic-workflow',
                               model=None, feedback_memory='versioned SQLite review snapshots'))
    Path('result.json').write_text(json.dumps(result))
    os._exit(0)
