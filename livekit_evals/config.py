"""
Configuration file for LiveKit Evals

Contains hardcoded default values that can be overridden by environment variables.

IMPORTANT: Call recording is ENABLED by default using SuperBryn's S3 credentials.
Override S3 settings only if you want to use your own bucket.
"""

import os

# =============================================================================
# S3 Recording Configuration
# =============================================================================
# Recording is ENABLED by default using SuperBryn's S3 credentials below.
# You can override with your own S3 bucket by changing these values or setting env vars.
#
# To use your own S3 bucket, see: S3_SETUP_INSTRUCTIONS.md

_DEFAULT_S3_BUCKET = "superbryn-call-recordings"  # SuperBryn's default bucket
_DEFAULT_S3_REGION = "ap-south-1"
_DEFAULT_S3_ACCESS_KEY = "AKIAUOZE37VAU3LB2RVS"  # SuperBryn's write-only credentials
_DEFAULT_S3_SECRET_KEY = "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"  # SuperBryn's write-only credentials
_DEFAULT_S3_BASE_URL = f"https://{_DEFAULT_S3_BUCKET}.s3.{_DEFAULT_S3_REGION}.amazonaws.com"

S3_CONFIG = {
    "bucket_name": os.getenv("S3_BUCKET_NAME", _DEFAULT_S3_BUCKET),
    "region": os.getenv("S3_REGION", _DEFAULT_S3_REGION),
    "access_key": os.getenv("S3_ACCESS_KEY", _DEFAULT_S3_ACCESS_KEY),
    "secret_key": os.getenv("S3_SECRET_KEY", _DEFAULT_S3_SECRET_KEY),
    "base_url": os.getenv("S3_BASE_URL", _DEFAULT_S3_BASE_URL),
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
_DEFAULT_WEBHOOK_URL = "https://riaahcilmtirmkoulgjy.supabase.co/functions/v1/webhooks-livekit"
_DEFAULT_WEBHOOK_API_KEY = ""  # ← SET YOUR DEFAULT API KEY HERE

WEBHOOK_CONFIG = {
    "url": os.getenv("WEBHOOK_URL", _DEFAULT_WEBHOOK_URL),
    "api_key": os.getenv("SUPERBRYN_API_KEY", _DEFAULT_WEBHOOK_API_KEY),
}


def is_s3_configured() -> bool:
    """Check if S3 credentials are properly configured for recording"""
    return bool(
        S3_CONFIG["bucket_name"]
        and S3_CONFIG["region"]
        and S3_CONFIG["access_key"]
        and S3_CONFIG["secret_key"]
        and S3_CONFIG["base_url"]
    )

