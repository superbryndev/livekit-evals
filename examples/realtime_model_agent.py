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
from livekit_evals import create_webhook_handler

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

    # ============================================================================
    # LIVEKIT-EVALS INTEGRATION - STEP 1: Create webhook handler
    # ============================================================================
    webhook_handler = create_webhook_handler(
        room=ctx.room,
        is_deployed_on_lk_cloud=True
    )

    # Set up session with OpenAI Realtime Model
    # This is simpler than voice pipeline - no separate STT/TTS/LLM configuration
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(
            voice="alloy",  # OpenAI voice options: alloy, echo, fable, onyx, nova, shimmer
            temperature=0.8,
            instructions="You are a helpful assistant.",
        )
    )

    # Start the session
    await session.start(agent=Assistant(), room=ctx.room)

    # ============================================================================
    # LIVEKIT-EVALS INTEGRATION - STEP 2 & 3: Attach to session and register callback
    # ============================================================================
    # Works the same way with realtime models!
    if webhook_handler:
        webhook_handler.attach_to_session(session)
        ctx.add_shutdown_callback(webhook_handler.send_webhook)

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

