"""Request-isolated execution of unmodified Omega Core and its three atlas tools."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def main():
    package = json.load(sys.stdin)
    files = package['files']
    required = {'run.metta','skills.metta','channels.metta','memory.metta','bridge.py',
                'atlas_planner.py','atlas_policy.py','triage.py'}
    if set(files) != required or sum(len(s.encode()) for s in files.values()) > 100000:
        raise ValueError('Invalid native atlas profile package.')
    manifest = json.loads(Path('/opt/runtime/manifest.json').read_text())
    if manifest['core_commit'] != package['expected_core'] or manifest['petta_commit'] != package['expected_petta']:
        raise ValueError('Native Omega runtime version differs from the atlas contract.')
    for component, root in [('core', Path('/opt/runtime/Omega')), ('petta', Path('/opt/runtime/PeTTa'))]:
        for relative, expected in manifest[component+'_files'].items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Native Omega source integrity check failed.')
    with tempfile.TemporaryDirectory(prefix='omega-reviewed-atlas-') as directory:
        work = Path(directory); core = work/'repos/Omega'
        shutil.copytree('/opt/runtime/Omega', core)
        (work/'repos/Omega.lock').touch()
        profile = core/'profiles/alis_atlas'; profile.mkdir(parents=True)
        for name, content in files.items():
            (profile/name).write_text(content)
        program = profile/'run.metta'
        program.write_text(program.read_text().replace('/state/repos', str(work/'repos')))
        (core/'config/plugins.yaml').write_text('- name: atlas_planner\n  loader: python\n  location: "{REPO}/profiles/alis_atlas"\n')
        (core/'config/config.yaml').write_text('maxNewInputLoops: 3\nmaxWakeLoops: 0\nsleepInterval: 0\nprovider: AtlasWorkflow\nmaxOutputToken: 128\nreasoningMode: none\nwakeupInterval: 86400\nspamShield: false\nsecurityPolicyPath: ""\n')
        (work/'request.json').write_text(json.dumps(package['context']))
        # Neither a model key nor a network model client is needed by this profile.
        env = {k:v for k,v in os.environ.items() if not any(x in k for x in ('API_KEY','ACCESS_TOKEN','AUTH_TOKEN'))}
        env.update(PYTHONPATH=str(profile)+':'+str(core)+':'+str(core/'src')+':/opt/chatbot:/opt/python',
                   PYTHONDONTWRITEBYTECODE='1', ATLAS_PROFILE_HASH=digest(files))
        with (work/'core.log').open('w') as log:
            p = subprocess.Popen(['sh','/opt/runtime/PeTTa/run.sh','--silent',str(program)],
                                 cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                p.wait(timeout=int(package['timeout'])-3)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
                raise RuntimeError('Native Omega atlas workflow timed out; no model request was made.')
        if not (work/'result.json').exists():
            # Logs contain no provider credentials; keep output bounded and opt-in.
            if os.getenv('FEEDBACK_DEBUG') == '1':
                print((work/'core.log').read_text()[-6000:], file=sys.stderr)
            raise RuntimeError('Native Omega did not complete the atlas tool chain.')
        print((work/'result.json').read_text())


try:
    main()
except Exception as error:
    print(json.dumps({'error': str(error)[:400]}))
    sys.exit(1)
