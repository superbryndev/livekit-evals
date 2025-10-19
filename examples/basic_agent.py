"""
Basic LiveKit Agent with livekit-evals integration

This example shows how to integrate livekit-evals into a simple voice AI agent.
The integration requires just 3 lines of code to track all session metrics.
"""

import logging
from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Import livekit-evals
from livekit_evals import create_webhook_handler

logger = logging.getLogger("agent")

load_dotenv(".env")


class Assistant(Agent):
    """A simple voice AI assistant with weather lookup capability."""
    
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):  # noqa: ARG002
        """Use this tool to look up current weather information in the given location.

        If the location is not supported by the weather service, the tool will indicate this.
        You must tell the user the location's weather is unavailable.

        Args:
            location: The location to look up weather information for (e.g. city name)
        """
        logger.info("Looking up weather for %s", location)
        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    """Prewarm models to reduce cold start latency."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent."""
    
    # Logging setup - add context fields for better debugging
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # ============================================================================
    # LIVEKIT-EVALS INTEGRATION - STEP 1: Create webhook handler
    # ============================================================================
    # This will auto-detect agent_id, version_id, system_prompt, phone_number,
    # SIP trunking, and egress recording from session and room context
    webhook_handler = create_webhook_handler(
        room=ctx.room,
        is_deployed_on_lk_cloud=True  # Set to False if self-hosting
    )

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and LiveKit turn detector
    session = AgentSession(
        # Large Language Model (LLM) - your agent's brain
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        
        # Speech-to-text (STT) - your agent's ears
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        
        # Text-to-speech (TTS) - your agent's voice
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(
            voice='ea8d222e-48d5-465e-aeab-a4ee929d16c6',  # chaithra
            speed="normal",
            language="hi",
        ),
        
        # VAD and turn detection - determines when user is speaking
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        
        # Allow LLM to generate response while waiting for end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Alternative: Use a realtime model instead of voice pipeline
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Handle false positive interruptions (background noise)
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection for pipeline performance monitoring
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    # Start the session, which initializes the voice pipeline and warms up models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # ============================================================================
    # LIVEKIT-EVALS INTEGRATION - STEP 2 & 3: Attach to session and register callback
    # ============================================================================
    # IMPORTANT: This must be AFTER session.start() and BEFORE ctx.connect()
    # This extracts the system prompt from session.agent and starts listening to events
    if webhook_handler:
        webhook_handler.attach_to_session(session)
        ctx.add_shutdown_callback(webhook_handler.send_webhook)

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

