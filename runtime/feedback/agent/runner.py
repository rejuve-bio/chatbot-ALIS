"""Run a request-isolated profile inside the configured native Omega container."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid


def main():
    package=json.load(sys.stdin)
    files=package['files']
    required={'run.metta','skills.metta','channels.metta','memory.metta','bridge.py','feedback_provider.py','reason_schema.py','source_schema.py'}
    if set(files)!=required or sum(len(s.encode()) for s in files.values())>100000:
        raise ValueError('Invalid feedback profile package.')
    profile_hash=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest=json.loads(Path('/opt/runtime/manifest.json').read_text())
    if manifest['core_commit']!=package['expected_core'] or manifest['petta_commit']!=package['expected_petta']:
        raise ValueError('Native Omega runtime version differs from the feedback contract.')
    for component,root in [('core',Path('/opt/runtime/Omega')),('petta',Path('/opt/runtime/PeTTa'))]:
        for relative,expected in manifest[component+'_files'].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
                raise ValueError('Native Omega source integrity check failed.')
    with tempfile.TemporaryDirectory(prefix='omega-feedback-') as directory:
        work=Path(directory)
        core=work/'repos/Omega'
        shutil.copytree('/opt/runtime/Omega',core)
        (work/'repos/Omega.lock').touch()
        profile=core/'profiles/alis_feedback'
        profile.mkdir(parents=True)
        for name,content in files.items():
            (profile/name).write_text(content)
        program=profile/'run.metta'
        program.write_text(program.read_text().replace('/state/repos',str(work/'repos')))
        (core/'config/plugins.yaml').write_text('- name: feedback_provider\n  loader: python\n  location: "{REPO}/profiles/alis_feedback"\n')
        limit=int(package.get('output_tokens',2048))
        if not 512<=limit<=4096:raise ValueError('Feedback output token limit must be between 512 and 4096.')
        (core/'config/config.yaml').write_text('maxNewInputLoops: 1\nmaxWakeLoops: 0\nsleepInterval: 0\nprovider: Feedback\nmaxOutputToken: '+str(limit)+'\nreasoningMode: none\nwakeupInterval: 86400\nspamShield: false\nsecurityPolicyPath: ""\n')
        (work/'request.json').write_text(json.dumps(package['request']))
        env={**os.environ,'PYTHONPATH':str(profile)+':'+str(core)+':'+str(core/'src')+':/opt/chatbot:/opt/python',
             'PYTHONDONTWRITEBYTECODE':'1','FEEDBACK_PROFILE_HASH':profile_hash,
             'FEEDBACK_REQUEST_TIMEOUT':str(package.get('request_timeout',75)),
             'OMEGA_STAGE':'feedback','OMEGA_LLM_BUDGET_ID':'feedback-'+uuid.uuid4().hex,
             'LLM_MAX_REQUESTS_PER_JOB':'1','LLM_MAX_TOKENS_PER_JOB':'40000','LLM_MAX_INPUT_BYTES':'28000',
             'LLM_STREAM':'false','LLM_MAX_RETRIES':'0'}
        with (work/'core.log').open('w') as log:
            process=subprocess.Popen(['sh','/opt/runtime/PeTTa/run.sh','--silent',str(program)],cwd=work,env=env,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:process.wait(timeout=int(package.get('request_timeout',75))+40)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                raise RuntimeError('Native Omega review deadline reached. No automatic retry was made.')
        if not (work/'result.json').exists():
            error=json.loads((work/'failure.json').read_text())['error'] if (work/'failure.json').exists() else 'Native Omega did not finish the feedback tool call.'
            if os.getenv('FEEDBACK_DEBUG')=='1':
                print((work/'core.log').read_text()[-10000:],file=sys.stderr)
            # Keep a rejected model output reviewable without making another
            # paid call. This contains only the already whitelisted packet.
            rejected={'error':error}
            if (work/'model-proposal.json').exists():rejected['rejected_proposal']=json.loads((work/'model-proposal.json').read_text())
            if (work/'context.json').exists():rejected['inspected_context']=json.loads((work/'context.json').read_text())
            print(json.dumps(rejected));return
        print((work/'result.json').read_text())


try:main()
except Exception as error:
    print(json.dumps({'error':str(error)[:500]}))
    sys.exit(1)
