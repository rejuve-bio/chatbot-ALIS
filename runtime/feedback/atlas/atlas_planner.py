"""Local symbolic planner using Core's provider dispatch interface; no LLM SDK."""
import json
from pathlib import Path
import providers


class AtlasPlanner(providers.LLMProvider):
    def start(self):
        self.turns = 0

    def chat(self, prompt, max_tokens=2048, reasoning_mode='none'):
        self.turns += 1
        stage = json.loads(Path('workflow.json').read_text())['stage']
        if self.turns > 3 or stage not in (0, 1, 2):
            raise RuntimeError('Atlas workflow failed to advance; no retry or model request.')
        return ('read-current-feedback', 'apply-review-policy', 'finish-reviewed-atlas')[stage]


def loadOmegaPlugin():
    providers.registerLLMProvider('AtlasWorkflow', AtlasPlanner())
