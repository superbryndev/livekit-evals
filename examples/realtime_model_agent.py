"""
LiveKit Agent using Realtime Model with livekit-evals integration

This example shows how to use livekit-evals with OpenAI's Realtime API model.
The integration works identically with both voice pipelines and realtime models.
"""

import logging
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai

# Import livekit-evals
from livekit_evals import attach

logger = logging.getLogger("agent")

load_dotenv(".env")


class Assistant(Agent):
    """A simple voice AI assistant using OpenAI Realtime API."""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You provide clear, concise answers to user questions.
            You are friendly and professional.""",
        )


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent with Realtime Model."""
    
    # Logging setup
    ctx.log_context_fields = {"room": ctx.room.name}

    # Set up session with OpenAI Realtime Model
    # This is simpler than voice pipeline - no separate STT/TTS/LLM configuration
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="alloy",  # OpenAI voice options: alloy, echo, fable, onyx, nova, shimmer
            temperature=0.8,
            instructions="You are a helpful assistant.",
        )
    )

    # ============================================================================
    # LIVEKIT-EVALS: report this call to SuperBryn (before session.start()); works the same with realtime models
    # ============================================================================
    attach(ctx, session)

    # Start the session
    await session.start(agent=Assistant(), room=ctx.room)

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

