"""
Webhook Handler for LiveKit Agent

Captures events during agent session and sends webhook payload to Supabase edge function.
Designed to work with the webhooks-livekit edge function.
"""

import logging
import os
import re
import json
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

import aiohttp
from .config import S3_CONFIG, is_s3_configured, WEBHOOK_CONFIG, LIVEKIT_CONFIG, AGENT_CONFIG
from livekit.agents import (
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    SpeechCreatedEvent,
    UserInputTranscribedEvent,
    UserStateChangedEvent,
    get_job_context,
    metrics,
)
from livekit.agents.voice.events import AgentState, UserState
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import (
    LLMMetrics,
    STTMetrics,
    TTSMetrics,
    # VADMetrics,
    # EOUMetrics,
    RealtimeModelMetrics,
)
from livekit.rtc import Room

if TYPE_CHECKING:
    from recording_manager import RecordingManager

logger = logging.getLogger("webhook_handler")


def _extract_project_id_from_url(livekit_url: str) -> Optional[str]:
    """
    Extract project ID from LiveKit URL.

    Args:
        livekit_url: LiveKit URL (e.g., "wss://tara-agent-bt2d90rn.livekit.cloud")

    Returns:
        Project ID (e.g., "tara-agent-bt2d90rn") or None if not found
    """
    # Match pattern: wss://PROJECT_ID.livekit.cloud or ws://PROJECT_ID.livekit.cloud
    match = re.match(r"wss?://([^.]+)\.livekit\.cloud", livekit_url)
    if match:
        return match.group(1)
    
    # Also handle custom domains or localhost
    # For custom domains, just use the full hostname as project ID
    match = re.match(r"wss?://([^:/]+)", livekit_url)
    if match:
        hostname = match.group(1)
        # Remove 'livekit.cloud' suffix if present
        if hostname.endswith(".livekit.cloud"):
            return hostname.replace(".livekit.cloud", "")
        return hostname
    
    return None


class WebhookHandler:
    """
    Handles webhook payload construction and delivery for LiveKit agent sessions.
    
    Listens to session events and aggregates data to send to the webhook endpoint
    when the session closes.
    """
    
    def __init__(
        self,
        webhook_url: str,
        api_key: str,
        room: Room,
        is_deployed_on_lk_cloud: bool,
        livekit_project_id: Optional[str] = None,
        call_rate_usd: Optional[float] = None,
        recording_manager: Optional["RecordingManager"] = None,
        disable_recording: bool = False,
    ):
        """
        Initialize webhook handler.
        
        Args:
            webhook_url: URL of the webhook endpoint
            api_key: API key for the webhook endpoint
            room: LiveKit room instance
            is_deployed_on_lk_cloud: Whether agent is deployed on LiveKit Cloud ($0.014/min)
            livekit_project_id: LiveKit project ID for agent uniqueness (optional)
            call_rate_usd: Custom telephony rate per minute ($/min) for cost calculation (optional)
            recording_manager: RecordingManager instance for handling recordings (optional)
            disable_recording: Set to True to disable call recording (default: False, recording enabled)
        """
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.room = room
        self.is_deployed_on_lk_cloud = is_deployed_on_lk_cloud
        self.livekit_project_id = livekit_project_id
        self.call_rate_usd = call_rate_usd
        self.recording_manager = recording_manager
        self.disable_recording = disable_recording
        
        # These will be auto-detected
        self.agent_id: Optional[str] = None
        self.version_id: Optional[str] = None
        self.system_prompt: Optional[str] = None
        self.sip_trunking_enabled: bool = False
        self.egress_enabled: bool = False
        self.phone_number: Optional[str] = None
        
        # Recording URLs (set externally by RecordingManager)
        self.recording_url: Optional[str] = None
        self.stereo_recording_url: Optional[str] = None
        self.egress_id: Optional[str] = None
        
        # Session tracking
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        
        # Transcript tracking
        self.transcript_turns: list[dict[str, Any]] = []
        self.call_start_time_ms: Optional[int] = None
        self.last_user_turn_time_ms: Optional[int] = None
        
        # Usage metrics tracking
        self.usage_metrics = {
            "llm_model": None,
            "llm_provider": None,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_total_tokens": 0,
            "stt_provider": None,
            "stt_model": None,
            "stt_duration_seconds": 0,
            "audio_duration_seconds": 0,
            "tts_provider": None,
            "tts_model": None,
            "tts_voice_id": None,
            "tts_characters": 0,
            "tts_audio_duration_seconds": 0,
        }
        
        # Latency tracking (aggregated)
        self.latency_metrics = {
            "llm_ms": [],
            "stt_ms": [],
            "tts_ms": [],
        }
        
        # Usage collector for metrics (following Whispey's approach)
        self.usage_collector = metrics.UsageCollector()
        
        # Speech events tracking
        self.speech_events: list[dict[str, Any]] = []
        
        logger.info(
            "WebhookHandler initialized: is_deployed_on_lk_cloud=%s, livekit_project_id=%s (agent_id and version_id will be auto-detected)",
            is_deployed_on_lk_cloud,
            livekit_project_id or "not-set",
        )
    
    def set_recording_url(
        self,
        recording_url: Optional[str],
        egress_id: Optional[str] = None,
        stereo_recording_url: Optional[str] = None,
    ) -> None:
        """
        Set the recording URL for this session.
        
        Args:
            recording_url: Primary recording URL
            egress_id: LiveKit egress ID for the recording
            stereo_recording_url: Stereo recording URL (optional)
        """
        self.recording_url = recording_url
        self.egress_id = egress_id
        self.stereo_recording_url = stereo_recording_url
        
        if recording_url:
            self.egress_enabled = True
            logger.info("Recording URL set: %s (egress_id: %s)", recording_url, egress_id)
    
    def _extract_session_config(self, session: AgentSession) -> None:
        """Extract model/provider info from session configuration using Whispey's approach."""
        try:
            # Extract LLM info
            if hasattr(session, 'llm') and session.llm:
                llm_obj = session.llm
                
                # Get model from direct attribute
                if hasattr(llm_obj, 'model'):
                    self.usage_metrics["llm_model"] = llm_obj.model
                
                # Extract provider from module name
                provider_name = llm_obj.__module__.split('.')[-1]
                self.usage_metrics["llm_provider"] = provider_name
                
                # Also try to extract from _opts if available
                if hasattr(llm_obj, '_opts') and llm_obj._opts:
                    opts = llm_obj._opts
                    if hasattr(opts, 'model') and not self.usage_metrics["llm_model"]:
                        self.usage_metrics["llm_model"] = opts.model
            
            # Extract STT info
            if hasattr(session, 'stt') and session.stt:
                stt_obj = session.stt
                
                # Get model from direct attribute
                if hasattr(stt_obj, 'model'):
                    self.usage_metrics["stt_model"] = stt_obj.model
                
                # Extract provider from module name
                provider_name = stt_obj.__module__.split('.')[-1]
                self.usage_metrics["stt_provider"] = provider_name
                
                # Also try to extract from _opts if available
                if hasattr(stt_obj, '_opts') and stt_obj._opts:
                    opts = stt_obj._opts
                    if hasattr(opts, 'model') and not self.usage_metrics["stt_model"]:
                        self.usage_metrics["stt_model"] = opts.model
            
            # Extract TTS info
            if hasattr(session, 'tts') and session.tts:
                tts_obj = session.tts
                
                # Get voice_id from direct attribute
                if hasattr(tts_obj, 'voice_id'):
                    self.usage_metrics["tts_voice_id"] = tts_obj.voice_id
                elif hasattr(tts_obj, 'voice'):
                    self.usage_metrics["tts_voice_id"] = tts_obj.voice
                
                # Get model from direct attribute
                if hasattr(tts_obj, 'model'):
                    self.usage_metrics["tts_model"] = tts_obj.model
                
                # Extract provider from module name
                provider_name = tts_obj.__module__.split('.')[-1]
                self.usage_metrics["tts_provider"] = provider_name
                
                # Also try to extract from _opts if available
                if hasattr(tts_obj, '_opts') and tts_obj._opts:
                    opts = tts_obj._opts
                    if hasattr(opts, 'voice_id') and not self.usage_metrics.get("tts_voice_id"):
                        self.usage_metrics["tts_voice_id"] = opts.voice_id
                    elif hasattr(opts, 'voice') and not self.usage_metrics.get("tts_voice_id"):
                        self.usage_metrics["tts_voice_id"] = opts.voice
                    if hasattr(opts, 'model') and not self.usage_metrics["tts_model"]:
                        self.usage_metrics["tts_model"] = opts.model
                
                # Fallback: if no voice_id found but model exists, use model as voice_id
                # This is common for providers like Sarvam where model name IS the voice
                if not self.usage_metrics["tts_voice_id"] and self.usage_metrics["tts_model"]:
                    self.usage_metrics["tts_voice_id"] = self.usage_metrics["tts_model"]
            
            # Apply provider detection based on model names
            if self.usage_metrics["llm_model"]:
                detected_provider = self._detect_provider_from_model_name(self.usage_metrics["llm_model"])
                if detected_provider != 'unknown':
                    self.usage_metrics["llm_provider"] = detected_provider
            
            if self.usage_metrics["stt_model"]:
                detected_provider = self._detect_provider_from_model_name(self.usage_metrics["stt_model"])
                if detected_provider != 'unknown':
                    self.usage_metrics["stt_provider"] = detected_provider
            
            if self.usage_metrics["tts_model"] or self.usage_metrics.get("tts_voice_id"):
                model_or_voice = self.usage_metrics["tts_model"] or self.usage_metrics.get("tts_voice_id")
                detected_provider = self._detect_provider_from_model_name(model_or_voice)
                if detected_provider != 'unknown':
                    self.usage_metrics["tts_provider"] = detected_provider
            
            logger.info("Extracted session config: %s", self.usage_metrics)
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to extract session config: %s", e)
    
    def _detect_provider_from_model_name(self, model_name: str) -> str:
        """Detect provider from model name with comprehensive provider support."""
        if not model_name:
            return 'unknown'
        
        model_lower = model_name.lower()
        
        # LLM Providers
        if any(x in model_lower for x in ['gpt', 'openai', 'whisper', 'tts-1', 'o1-', 'o3-']):
            return 'openai'
        elif any(x in model_lower for x in ['claude', 'anthropic']):
            return 'anthropic'
        elif any(x in model_lower for x in ['gemini', 'palm', 'bard', 'gemma']):
            return 'google'
        elif any(x in model_lower for x in ['llama', 'meta-llama']):
            return 'meta'
        elif any(x in model_lower for x in ['mistral', 'mixtral']):
            return 'mistral'
        elif any(x in model_lower for x in ['cohere', 'command']):
            return 'cohere'
        elif any(x in model_lower for x in ['perplexity', 'pplx']):
            return 'perplexity'
        elif any(x in model_lower for x in ['groq']):
            return 'groq'
        elif any(x in model_lower for x in ['together', 'togethercomputer']):
            return 'together'
        elif any(x in model_lower for x in ['replicate']):
            return 'replicate'
        elif any(x in model_lower for x in ['huggingface', 'hf-']):
            return 'huggingface'
        
        # TTS Providers
        elif any(x in model_lower for x in ['eleven', 'elevenlabs']):
            return 'elevenlabs'
        elif any(x in model_lower for x in ['cartesia', 'sonic']):
            return 'cartesia'
        elif any(x in model_lower for x in ['playht', 'play.ht', 'play-ht']):
            return 'playht'
        elif any(x in model_lower for x in ['resemble', 'resembleai']):
            return 'resemble'
        elif any(x in model_lower for x in ['murf', 'murf.ai']):
            return 'murf'
        elif any(x in model_lower for x in ['wellsaid', 'wellsaidlabs']):
            return 'wellsaid'
        elif any(x in model_lower for x in ['speechify']):
            return 'speechify'
        elif any(x in model_lower for x in ['saarika', 'sarvam', 'bulbul']):
            return 'sarvam'
        elif any(x in model_lower for x in ['azure', 'microsoft']):
            return 'azure'
        elif any(x in model_lower for x in ['aws', 'polly', 'amazon']):
            return 'aws'
        elif any(x in model_lower for x in ['gcloud', 'google-cloud']):
            return 'google-cloud'
        
        # STT Providers
        elif any(x in model_lower for x in ['deepgram', 'nova', 'aura']):
            return 'deepgram'
        elif any(x in model_lower for x in ['assemblyai', 'assembly']):
            return 'assemblyai'
        elif any(x in model_lower for x in ['rev.ai', 'revai']):
            return 'rev'
        elif any(x in model_lower for x in ['speechmatics']):
            return 'speechmatics'
        elif any(x in model_lower for x in ['gladia']):
            return 'gladia'
        
        # Multi-modal/Realtime Providers
        elif any(x in model_lower for x in ['livekit']):
            return 'livekit'
        elif any(x in model_lower for x in ['twilio']):
            return 'twilio'
        elif any(x in model_lower for x in ['vonage']):
            return 'vonage'
        
        else:
            return 'unknown'
    
    def _detect_sip_trunking(self) -> None:
        """Detect if SIP trunking is enabled by checking for SIP participants."""
        try:
            for participant in self.room.remote_participants.values():
                # Check if participant has SIP-related attributes
                if hasattr(participant, 'attributes'):
                    attributes = participant.attributes
                    # Check for any sip.* attributes
                    if any(key.startswith('sip.') for key in attributes.keys()):
                        self.sip_trunking_enabled = True
                        logger.info("SIP trunking detected from participant attributes")
                        return
                
                # Also check participant identity for SIP patterns
                if hasattr(participant, 'identity'):
                    identity = participant.identity
                    # SIP participants often have phone number-like identities
                    if identity and (identity.startswith('+') or identity.startswith('sip:')):
                        self.sip_trunking_enabled = True
                        logger.info("SIP trunking detected from participant identity: %s", identity)
                        return
            
            logger.info("No SIP participants detected, SIP trunking disabled")
        except Exception as e:
            logger.warning("Failed to detect SIP trunking: %s", e)
    
    async def attach_to_session(self, session: AgentSession) -> None:
        """
        Attach event listeners to the agent session and start recording if configured.
        
        Args:
            session: AgentSession to attach to
        """
        # Mark session start
        self.started_at = datetime.now(timezone.utc)
        self.call_start_time_ms = int(self.started_at.timestamp() * 1000)
        
        # Extract agent_id, version_id, and phone_number from job context if available
        job_ctx = get_job_context()
        if job_ctx and hasattr(job_ctx, 'job') and job_ctx.job and hasattr(job_ctx.job, 'metadata'):
            try:
                metadata = json.loads(job_ctx.job.metadata) if isinstance(job_ctx.job.metadata, str) else job_ctx.job.metadata
                self.agent_id = metadata.get('agent_id') or AGENT_CONFIG['id']
                self.version_id = metadata.get('version_id') or AGENT_CONFIG['version_id']
                self.phone_number = metadata.get('phone_number')
                logger.info("Extracted from job metadata - agent_id: %s, version_id: %s, phone: %s",
                          self.agent_id, self.version_id, self.phone_number)
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to extract from job metadata: %s", e)
        
        # Fallback to config if not extracted from metadata
        if not self.agent_id:
            self.agent_id = AGENT_CONFIG['id']
        if not self.version_id:
            self.version_id = AGENT_CONFIG['version_id']
        
        # Start recording unless disabled
        if self.disable_recording:
            logger.info("Recording disabled by disable_recording flag")
        elif self.recording_manager:
            try:
                logger.info("Starting recording for room %s", self.room.name)
                recording_url, egress_id = await self.recording_manager.start_recording(
                    room_name=self.room.name,
                    phone_number=self.phone_number,
                )
                
                if recording_url:
                    self.set_recording_url(
                        recording_url=recording_url,
                        egress_id=egress_id,
                    )
                    logger.info("Recording started successfully: %s", recording_url)
                else:
                    logger.warning("Failed to start recording")
                    
            except Exception as e:  # noqa: BLE001
                logger.error("Failed to start recording: %s", e, exc_info=True)
        
        # Extract system prompt from session's current agent
        if hasattr(session, 'current_agent') and session.current_agent:
            current_agent = session.current_agent
            if hasattr(current_agent, 'instructions'):
                self.system_prompt = current_agent.instructions
                logger.info("Extracted system prompt from session.current_agent: %s...",
                          self.system_prompt[:100] if self.system_prompt else "None")
        
        # Detect SIP trunking by checking for SIP participants
        self._detect_sip_trunking()
        
        # Extract model/provider info from session configuration
        self._extract_session_config(session)
        
        # Listen to conversation items for transcript
        session.on("conversation_item_added")(self._on_conversation_item_added)
        
        # Listen to metrics for usage and latency
        session.on("metrics_collected")(self._on_metrics_collected)
        
        # Listen to user input for additional transcript metadata
        session.on("user_input_transcribed")(self._on_user_input_transcribed)
        
        # Listen to state changes for precise timing
        session.on("agent_state_changed")(self._on_agent_state_changed)
        session.on("user_state_changed")(self._on_user_state_changed)
        
        # Listen to speech creation for additional tracking
        session.on("speech_created")(self._on_speech_created)
        
        # Listen to session close for cleanup
        session.on("close")(self._on_session_close)
        
        logger.info("Event listeners attached to session")
    
    def _on_conversation_item_added(self, event: ConversationItemAddedEvent) -> None:
        """Handle conversation item added event - fills in text for existing turns."""
        item: ChatMessage = event.item
        
        # Determine speaker role
        speaker = "user" if item.role == "user" else "assistant"
        
        # Get text content
        text = item.text_content or ""
        
        # Find the most recent turn for this speaker WITHOUT text and fill it in
        turn_found = False
        for turn in reversed(self.transcript_turns):
            if turn["speaker"] == speaker and not turn["text"]:
                turn["text"] = text
                turn["interrupted"] = item.interrupted if hasattr(item, "interrupted") else False
                turn["turn_latency"] = self._get_turn_latency(speaker)
                turn["confidence_score"] = self._get_confidence_score(item)
                
                # Calculate response delay for assistant turns
                if speaker == "assistant" and self.last_user_turn_time_ms is not None:
                    turn["response_delay_ms"] = turn["start_time_ms"] - self.last_user_turn_time_ms
                
                # Update last user turn time
                if speaker == "user":
                    self.last_user_turn_time_ms = turn["end_time_ms"] if turn["end_time_ms"] else turn["start_time_ms"]
                
                logger.info("✓ Filled text for %s turn: %s...", speaker, text[:500])
                turn_found = True
                break
        
        if not turn_found:
            logger.warning("No empty %s turn found to fill with text: %s...", speaker, text[:500])
        
        logger.debug("Transcript turn text updated: %s - %s...", speaker, text[:500])
    
    def _on_user_input_transcribed(self, event: UserInputTranscribedEvent) -> None:
        """Handle user input transcribed event for additional metadata."""
        # This gives us language and speaker_id if available
        if self.transcript_turns and event.is_final:
            # Update the last user turn with additional metadata
            for turn in reversed(self.transcript_turns):
                if turn["speaker"] == "user":
                    turn["language"] = event.language
                    if event.speaker_id:
                        turn["speaker_id"] = event.speaker_id
                    break
    
    def _on_agent_state_changed(self, event: AgentStateChangedEvent) -> None:
        """Handle agent state changes for precise timing."""
        try:
            old_state: AgentState = event.old_state
            new_state: AgentState = event.new_state
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            state_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            timestamp = datetime.now(timezone.utc).isoformat()

            # logger.info("Agent state changed: %s -> %s at %dms (turns count: %d)",
            #            old_state, new_state, state_time_ms, len(self.transcript_turns))
            
            # START: non-speaking -> speaking
            if new_state == 'speaking' and old_state != 'speaking':
                logger.info("Agent STARTED speaking at %dms", state_time_ms)
                turn = {
                    "speaker": "assistant",
                    "text": "",  # Will be filled by conversation_item_added
                    "timestamp": timestamp,
                    "start_timestamp": timestamp,
                    "end_timestamp": None,
                    "start_time_ms": state_time_ms,
                    "end_time_ms": None,
                    "response_delay_ms": None,
                    "interrupted": False,
                    "turn_latency": None,
                    "confidence_score": None,
                    "language": None,
                    "speaker_id": None,
                }
                self.transcript_turns.append(turn)
                logger.info("✓ Created assistant turn at start")
            
            # END: speaking -> non-speaking
            elif old_state == 'speaking' and new_state != 'speaking':
                # logger.info("Agent STOPPED speaking at %dms", state_time_ms)
                # Find the last assistant turn without an end time
                for turn in reversed(self.transcript_turns):
                    if turn["speaker"] == "assistant" and turn["end_time_ms"] is None:
                        turn["end_time_ms"] = state_time_ms
                        turn["end_timestamp"] = timestamp
                        # duration_ms = state_time_ms - turn["start_time_ms"]
                        # logger.info("✓ Updated assistant turn end time: duration=%dms", duration_ms)
                        break
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle agent state change: %s", e)
    
    def _on_user_state_changed(self, event: UserStateChangedEvent) -> None:
        """Handle user state changes for precise timing."""
        try:
            old_state: UserState = event.old_state
            new_state: UserState = event.new_state
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            state_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            timestamp = datetime.now(timezone.utc).isoformat()

            # logger.info("User state changed: %s -> %s at %dms (turns count: %d)",
            #            old_state, new_state, state_time_ms, len(self.transcript_turns))
            
            # START: non-speaking -> speaking
            if new_state == 'speaking' and old_state != 'speaking':
                logger.info("User STARTED speaking at %dms", state_time_ms)
                turn = {
                    "speaker": "user",
                    "text": "",  # Will be filled by user_input_transcribed/conversation_item_added
                    "timestamp": timestamp,
                    "start_timestamp": timestamp,
                    "end_timestamp": None,
                    "start_time_ms": state_time_ms,
                    "end_time_ms": None,
                    "response_delay_ms": None,
                    "interrupted": False,
                    "turn_latency": None,
                    "confidence_score": None,
                    "language": None,
                    "speaker_id": None,
                }
                self.transcript_turns.append(turn)
                # logger.info("✓ Created user turn at start")
            
            # END: speaking -> non-speaking
            elif old_state == 'speaking' and new_state != 'speaking':
                # logger.info("User STOPPED speaking at %dms", state_time_ms)
                # Find the last user turn without an end time
                for turn in reversed(self.transcript_turns):
                    if turn["speaker"] == "user" and turn["end_time_ms"] is None:
                        turn["end_time_ms"] = state_time_ms
                        turn["end_timestamp"] = timestamp
                        # duration_ms = state_time_ms - turn["start_time_ms"]
                        # logger.info("✓ Updated user turn end time: duration=%dms", duration_ms)
                        break
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle user state change: %s", e)
    
    def _on_speech_created(self, event: SpeechCreatedEvent) -> None:
        """Handle speech creation event for additional tracking."""
        try:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            speech_time_ms = current_time_ms - self.call_start_time_ms if self.call_start_time_ms else 0
            
            # logger.debug("Speech created: source=%s, user_initiated=%s at %dms",
            #             event.source, event.user_initiated, speech_time_ms)
            
            # Track speech creation for analytics
            if not hasattr(self, 'speech_events'):
                self.speech_events = []
            
            self.speech_events.append({
                'source': event.source,
                'user_initiated': event.user_initiated,
                'timestamp_ms': speech_time_ms,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle speech created event: %s", e)
    
    def _on_session_close(self, event: CloseEvent) -> None:
        """Handle session close event for cleanup."""
        try:
            logger.info("Session closed")
            if event.error:
                logger.error("Session closed with error: %s", event.error)
            
            # Mark session as ended
            self.ended_at = datetime.now(timezone.utc)
            
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to handle session close: %s", e)
    
    def _on_metrics_collected(self, event: MetricsCollectedEvent) -> None:
        """Handle metrics collected event."""
        metrics_obj = event.metrics
        
        # Use Whispey's approach: collect metrics and log them
        self.usage_collector.collect(metrics_obj)
        metrics.log_metrics(metrics_obj)
        
        # Handle different metric types using the discriminator field
        if isinstance(metrics_obj, LLMMetrics):
            # LLM metrics
            self.usage_metrics["llm_input_tokens"] += metrics_obj.prompt_tokens
            self.usage_metrics["llm_output_tokens"] += metrics_obj.completion_tokens
            self.usage_metrics["llm_total_tokens"] += metrics_obj.total_tokens
            
            # Latency (ttft in seconds, convert to ms)
            if metrics_obj.ttft > 0:
                self.latency_metrics["llm_ms"].append(metrics_obj.ttft * 1000)
            elif metrics_obj.duration > 0:
                self.latency_metrics["llm_ms"].append(metrics_obj.duration * 1000)
        
        elif isinstance(metrics_obj, STTMetrics):
            # STT metrics
            self.usage_metrics["stt_duration_seconds"] += metrics_obj.audio_duration
            self.usage_metrics["audio_duration_seconds"] += metrics_obj.audio_duration
            
            # Latency (duration in seconds, convert to ms)
            if metrics_obj.duration > 0:
                self.latency_metrics["stt_ms"].append(metrics_obj.duration * 1000)
        
        elif isinstance(metrics_obj, TTSMetrics):
            # TTS metrics
            self.usage_metrics["tts_characters"] += metrics_obj.characters_count
            self.usage_metrics["tts_audio_duration_seconds"] += metrics_obj.audio_duration
            
            # Latency (ttfb in seconds, convert to ms)
            if metrics_obj.ttfb > 0:
                self.latency_metrics["tts_ms"].append(metrics_obj.ttfb * 1000)
            elif metrics_obj.duration > 0:
                self.latency_metrics["tts_ms"].append(metrics_obj.duration * 1000)
        
        elif isinstance(metrics_obj, RealtimeModelMetrics):
            # Realtime model metrics (e.g., OpenAI Realtime API)
            self.usage_metrics["llm_input_tokens"] += metrics_obj.input_tokens
            self.usage_metrics["llm_output_tokens"] += metrics_obj.output_tokens
            self.usage_metrics["llm_total_tokens"] += metrics_obj.total_tokens

            # Latency (ttft in seconds, convert to ms)
            if metrics_obj.ttft > 0:
                self.latency_metrics["llm_ms"].append(metrics_obj.ttft * 1000)
        
        # VADMetrics and EOUMetrics don't contribute to usage/latency tracking
        # but we log them for debugging
        # logger.debug("Metrics collected: %s", metrics_obj.type)
    
    def _calculate_average_latency(self, latencies: list[float]) -> float:
        """Calculate average latency from a list of measurements."""
        if not latencies:
            return 0.0
        return sum(latencies) / len(latencies)
    
    def _build_webhook_payload(self) -> dict[str, Any]:
        """
        Build the webhook payload in the expected format.
        
        Returns:
            Dictionary matching LivekitWebhookPayload interface
        """
        # Mark end time
        self.ended_at = datetime.now(timezone.utc)
        
        # No post-processing needed - all timestamps come from actual state change events
        # Log any turns with missing end times (shouldn't happen)
        for i, turn in enumerate(self.transcript_turns):
            if turn["end_time_ms"] is None:
                logger.warning("Turn %d (%s) has no end_time_ms - state change event may not have fired",
                              i, turn["speaker"])
        
        # Calculate duration
        duration_seconds = 0
        if self.started_at and self.ended_at:
            duration_seconds = int((self.ended_at - self.started_at).total_seconds())
        
        # Calculate average latencies
        avg_latency = {
            "llm_ms": self._calculate_average_latency(self.latency_metrics["llm_ms"]),
            "stt_ms": self._calculate_average_latency(self.latency_metrics["stt_ms"]),
            "tts_ms": self._calculate_average_latency(self.latency_metrics["tts_ms"]),
        }
        avg_latency["total_ms"] = sum(avg_latency.values())
        
        # Get usage summary from collector (Whispey's approach)
        usage_summary = self.usage_collector.get_summary()
        logger.info("Usage summary from collector: %s", usage_summary)
        
        # Debug: Log final usage metrics
        logger.info("Final usage metrics: %s", self.usage_metrics)
        
        # Filter out turns without text
        turns_with_text = [turn for turn in self.transcript_turns if turn.get("text") and turn["text"].strip()]
        logger.info("Filtered transcript: %d total turns, %d turns with text",
                   len(self.transcript_turns), len(turns_with_text))
        
        # Build payload
        payload = {
            "event": "call.ended",
            "call": {
                "id": self.room.name,  # Use room name as call ID
                "room_name": self.room.name,
                "participant_identity": self._get_participant_identity(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "ended_at": self.ended_at.isoformat() if self.ended_at else None,
                "duration_seconds": duration_seconds,
                "transcript": {
                    "turns": turns_with_text,
                },
                "recording_url": self._get_recording_url(),
                "stereo_recording_url": self._get_stereo_recording_url(),
                "metadata": {
                    "agent_id": self.agent_id,
                    "livekit_project_id": self.livekit_project_id,
                    # Include model/provider info for version config tracking
                    "llm_model": self.usage_metrics["llm_model"],
                    "llm_provider": self.usage_metrics["llm_provider"],
                    "stt_model": self.usage_metrics["stt_model"],
                    "stt_provider": self.usage_metrics["stt_provider"],
                    "tts_model": self.usage_metrics["tts_model"],
                    "tts_provider": self.usage_metrics["tts_provider"],
                    "tts_voice_id": self.usage_metrics["tts_voice_id"],
                    "system_prompt": self.system_prompt,
                    # LiveKit feature flags for cost calculation
                    "sip_trunking_enabled": self.sip_trunking_enabled,
                    "egress_enabled": self.egress_enabled,
                    "lk_agent_enabled": self.is_deployed_on_lk_cloud,
                    # Phone number if available
                    "phone_number": self.phone_number,
                    # Custom telephony rate ($/min) if provided
                    "call_rate_usd": self.call_rate_usd,
                },
                "usage": self.usage_metrics,
                "latency": avg_latency,
            },
        }
        
        return payload
    
    def _get_participant_identity(self) -> str:
        """Get the first non-agent participant identity."""
        for participant in self.room.remote_participants.values():
            return participant.identity
        return "unknown"
    
    def _get_recording_url(self) -> str | None:
        """Get recording URL if available."""
        # Return the recording URL set by RecordingManager
        if self.recording_url:
            return self.recording_url
        
        # Fallback: Check if room has recording info
        if hasattr(self.room, 'recording_url') and self.room.recording_url:
            return self.room.recording_url
        
        # Fallback: Check if room has recording status
        if hasattr(self.room, 'recording_status') and self.room.recording_status:
            # Try to construct URL from room name and LiveKit project
            if self.livekit_project_id and self.livekit_project_id != "not-set":
                return f"https://cloud.livekit.io/projects/{self.livekit_project_id}/recordings/{self.room.name}"
        
        return None
    
    def _get_stereo_recording_url(self) -> str | None:
        """Get stereo recording URL if available."""
        # Return the stereo recording URL set by RecordingManager
        if self.stereo_recording_url:
            return self.stereo_recording_url
        
        # Fallback: Check if room has stereo recording info
        if hasattr(self.room, 'stereo_recording_url') and self.room.stereo_recording_url:
            return self.room.stereo_recording_url
        
        # For now, return None as stereo recording is less common
        return None
    
    def _get_turn_latency(self, speaker: str) -> dict[str, float] | None:
        """Get turn-level latency breakdown if available."""
        if speaker == "assistant":
            # For assistant turns, we can estimate latency from recent metrics
            return {
                "llm_ms": self._calculate_average_latency(self.latency_metrics["llm_ms"]) if self.latency_metrics["llm_ms"] else 0,
                "tts_ms": self._calculate_average_latency(self.latency_metrics["tts_ms"]) if self.latency_metrics["tts_ms"] else 0,
                "total_ms": self._calculate_average_latency(self.latency_metrics["llm_ms"]) + self._calculate_average_latency(self.latency_metrics["tts_ms"]),
            }
        elif speaker == "user":
            # For user turns, STT latency
            return {
                "stt_ms": self._calculate_average_latency(self.latency_metrics["stt_ms"]) if self.latency_metrics["stt_ms"] else 0,
                "total_ms": self._calculate_average_latency(self.latency_metrics["stt_ms"]) if self.latency_metrics["stt_ms"] else 0,
            }
        return None
    
    def _get_confidence_score(self, item: ChatMessage) -> float | None:
        """Get confidence score if available from the message item."""
        # Check if the item has confidence information
        if hasattr(item, 'confidence') and item.confidence is not None:
            return float(item.confidence)
        
        # Check for STT confidence in metadata
        if hasattr(item, 'metadata') and item.metadata:
            if 'confidence' in item.metadata:
                return float(item.metadata['confidence'])
            if 'stt_confidence' in item.metadata:
                return float(item.metadata['stt_confidence'])
        
        return None
    
    async def send_webhook(self) -> None:
        """
        Build and send the webhook payload to the endpoint.
        
        This should be called when the session ends.
        Authenticates using API key from parameter or environment (SUPERBRYN_API_KEY).
        Cleans up recording resources after sending webhook.
        
        Args:
            api_key_override: Optional API key to override environment variable
        """
        try:
            # Clean up recording resources first
            if self.recording_manager:
                try:
                    await self.recording_manager.stop_recording()
                    logger.info("Recording manager cleanup completed")
                except Exception as e:  # noqa: BLE001
                    logger.error("Error during recording cleanup: %s", e, exc_info=True)
            
            # Build webhook payload (egress_enabled is set by set_recording_url if recording is active)
            payload = self._build_webhook_payload()
            
            # Get API key from parameter or environment
            api_key = self.api_key
            if not api_key:
                logger.error("SUPERBRYN_API_KEY not configured and no api_key provided, webhook disabled")
                return
            
            logger.info(
                "Sending webhook to %s for call %s",
                self.webhook_url,
                payload["call"]["id"],
            )
            logger.debug("Webhook payload: %s", payload)
            
            # Send webhook via HTTP POST with API key authentication
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": api_key,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    response_text = await response.text()
                    
                    if response.status == 200:
                        logger.info("SUPERBRYN_WEBHOOK_SENT: %s", response_text)
                    elif response.status == 401:
                        logger.error(
                            "SUPERBRYN_WEBHOOK_UNAUTHORIZED: %s - Check SUPERBRYN_API_KEY",
                            response_text,
                        )
                    elif response.status == 403:
                        logger.error(
                            "SUPERBRYN_WEBHOOK_FORBIDDEN: %s - API key may be expired or disabled",
                            response_text,
                        )
                    else:
                        logger.error(
                            "SUPERBRYN_WEBHOOK_FAILED: status %s: %s",
                            response.status,
                            response_text,
                        )
        
        except Exception as e:  # noqa: BLE001
            logger.error("SUPERBRYN_WEBHOOK_ERROR: %s", e, exc_info=True)


def create_webhook_handler(
    room: Room,
    is_deployed_on_lk_cloud: bool,
    livekit_project_id: Optional[str] = None,
    api_key: Optional[str] = None,
    call_rate_usd: Optional[float] = None,
    disable_recording: bool = False,
) -> Optional[WebhookHandler]:
    """
    Factory function to create a webhook handler from environment variables.
    
    Auto-detects agent_id, version_id, system_prompt, phone_number, SIP trunking,
    and egress recording from session context and room participants.
    
    Recording is ENABLED by default using SuperBryn's S3 credentials.
    S3 credentials in config.py are optional - only needed if you want to use your own bucket.
    
    Requires SUPERBRYN_API_KEY in environment or as parameter for webhook authentication.
    
    Args:
        room: LiveKit room instance
        is_deployed_on_lk_cloud: Whether agent is deployed on LiveKit Cloud ($0.014/min) - REQUIRED
        livekit_project_id: LiveKit project ID (defaults to env var or extracted from LIVEKIT_URL)
        api_key: Override API key (defaults to env var SUPERBRYN_API_KEY)
        call_rate_usd: Custom telephony rate per minute ($/min) for cost calculation (optional)
        disable_recording: Set to True to disable call recording (default: False, recording enabled)
    
    Returns:
        WebhookHandler instance or None if webhook is disabled
    """
    # Get configuration from config.py (with parameter overrides)
    webhook_url = WEBHOOK_CONFIG["url"]
    livekit_project_id = livekit_project_id or LIVEKIT_CONFIG["project_id"]
    
    # Check for API key (parameter override, then config)
    resolved_api_key = api_key or WEBHOOK_CONFIG["api_key"]
    if not resolved_api_key:
        logger.warning("SUPERBRYN_API_KEY not configured in config.py, webhook disabled")
        return None
    
    # If project ID not explicitly set, try to extract from LIVEKIT_URL
    if not livekit_project_id:
        livekit_url = os.getenv("LIVEKIT_URL") or os.getenv("LIVEKIT_WS_URL") or os.getenv("LIVEKIT_WSS_URL")
        if livekit_url:
            livekit_project_id = _extract_project_id_from_url(livekit_url)
            if livekit_project_id:
                logger.info("Extracted project ID from LIVEKIT_URL: %s", livekit_project_id)
    
    # Skip webhook if URL not configured
    if not webhook_url:
        logger.warning("WEBHOOK_URL not configured, webhook disabled")
        return None
    
    # Initialize recording manager (always available with SuperBryn's default S3 credentials)
    recording_manager = None
    
    if disable_recording:
        logger.info("Recording disabled by disable_recording=True flag")
    elif is_s3_configured():
        try:
            from recording_manager import RecordingManager
            recording_manager = RecordingManager(
                s3_bucket=S3_CONFIG["bucket_name"],
                s3_region=S3_CONFIG["region"],
                s3_access_key=S3_CONFIG["access_key"],
                s3_secret_key=S3_CONFIG["secret_key"],
                s3_base_url=S3_CONFIG["base_url"],
            )
            logger.info("Recording manager initialized - using configured S3 bucket: %s", S3_CONFIG["bucket_name"])
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to initialize recording manager: %s", e, exc_info=True)
            logger.warning("Continuing without recording functionality")
    else:
        logger.warning("S3 credentials not configured - this should not happen with default config")
    
    # Create handler - agent_id, version_id, system_prompt, phone_number, sip_trunking, and egress
    # will be auto-detected in attach_to_session and send_webhook
    handler = WebhookHandler(
        webhook_url=webhook_url,
        api_key=resolved_api_key,
        room=room,
        is_deployed_on_lk_cloud=is_deployed_on_lk_cloud,
        livekit_project_id=livekit_project_id,
        call_rate_usd=call_rate_usd,
        recording_manager=recording_manager,
        disable_recording=disable_recording,
    )
    
    logger.info("SUPERBRYN_WEBHOOK_HANDLER_CREATED: is_deployed_on_lk_cloud=%s, recording=%s",
               is_deployed_on_lk_cloud, "disabled" if disable_recording else ("enabled" if recording_manager else "unavailable"))
    return handler
