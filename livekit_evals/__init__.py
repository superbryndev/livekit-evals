"""
LiveKit Evals: one line, attach(ctx, session), reports each call of a LiveKit agent to SuperBryn.
"""

from .call_report import CallReport, attach
from .prompt_sync import prompt_hash, prompt_ref, push_prompt

__version__ = "0.4.0"
__all__ = ["attach", "CallReport", "prompt_hash", "prompt_ref", "push_prompt"]
