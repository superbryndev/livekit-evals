"""
Configuration file for LiveKit Evals

Contains default values that can be overridden by environment variables.
Recording credentials are fetched at runtime from a secure endpoint using
the SUPERBRYN_API_KEY -- no S3 keys are stored in this package.
"""

import os

# =============================================================================
# Recording Credentials Configuration
# =============================================================================
# Temporary S3 credentials are fetched per-session from this endpoint.
# The SUPERBRYN_API_KEY is used to authenticate the request.

_DEFAULT_CREDENTIALS_URL = (
    "https://orchestration-service-v2.onrender.com/api/recording-credentials"
)

CREDENTIALS_CONFIG = {
    "url": os.getenv("SUPERBRYN_CREDENTIALS_URL", _DEFAULT_CREDENTIALS_URL),
}

# =============================================================================
# Agent Configuration
# =============================================================================
# These are metadata for tracking/analytics in webhooks (not required for agent to work)
# Customize these to identify different agents/versions in your webhook data

_DEFAULT_AGENT_ID = "livekit-agent"  # Change to identify your agent
_DEFAULT_VERSION_ID = "v1"           # Increment when making updates

AGENT_CONFIG = {
    "id": os.getenv("AGENT_ID", _DEFAULT_AGENT_ID),
    "version_id": os.getenv("VERSION_ID", _DEFAULT_VERSION_ID),
}

# =============================================================================
# LiveKit Configuration
# =============================================================================
_DEFAULT_LIVEKIT_PROJECT_ID = ""  # Auto-extracted from LIVEKIT_URL if not set

LIVEKIT_CONFIG = {
    "project_id": os.getenv("LIVEKIT_PROJECT_ID", _DEFAULT_LIVEKIT_PROJECT_ID),
}

# =============================================================================
# Webhook Configuration
# =============================================================================
_DEFAULT_WEBHOOK_URL = "https://orchestration-service-v2.onrender.com/webhooks/livekit"
_DEFAULT_WEBHOOK_API_KEY = ""

WEBHOOK_CONFIG = {
    "url": os.getenv("WEBHOOK_URL", _DEFAULT_WEBHOOK_URL),
    "api_key": os.getenv("SUPERBRYN_API_KEY", _DEFAULT_WEBHOOK_API_KEY),
}
