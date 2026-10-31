"""
Recording Manager for LiveKit Agent

Records a call in stereo with LiveKit's Egress API to S3 (the agent on the left channel, everyone else on the
right) and gives the recording's URL, readable after the call ends.
"""

import logging
import urllib.parse
from datetime import datetime
from typing import Optional

import aiohttp
from livekit import api
from livekit.protocol import egress as egress_proto

logger = logging.getLogger("livekit_evals.recording_manager")

RECORDING_FORMAT = egress_proto.EncodedFileType.MP3


class RecordingManager:
    """Manages call recording using LiveKit Egress API: ``DUAL_CHANNEL_AGENT`` audio mixing places the agent
    participant on the left channel and all other participants (e.g. the SIP caller) on the right."""

    def __init__(self, credentials_url: str, api_key: str):
        """Initialize the recording manager.

        Args:
            credentials_url: URL of the endpoint that issues temporary S3 credentials
            api_key: SuperBryn API key used to authenticate the credentials request
        """
        self.credentials_url = credentials_url
        self.api_key = api_key
        self.lkapi = api.LiveKitAPI()
        self.egress_id: Optional[str] = None

        logger.info("RecordingManager initialized — stereo (DUAL_CHANNEL_AGENT), credentials fetched per-session")

    async def _fetch_credentials(self, room_name: str) -> dict:
        """Fetch short-lived S3 credentials from the credentials endpoint.

        Args:
            room_name: Room name included in the request for session scoping

        Returns:
            dict with access_key, secret_key, session_token, bucket, region, base_url
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.credentials_url,
                json={"room_name": room_name},
                headers={"x-api-key": self.api_key},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Failed to fetch recording credentials: {resp.status} {text}"
                    )
                return await resp.json()

    async def start_recording(self, room_name: str) -> Optional[str]:
        """Start recording the room with LiveKit's Egress API; the recording's URL, or None if it failed."""
        try:
            creds = await self._fetch_credentials(room_name)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            folder_path = f"call_recordings/{room_name}/{timestamp}"
            url_folder_path = f"call_recordings/{urllib.parse.quote_plus(room_name)}/{timestamp}"

            file_extension = "mp3" if RECORDING_FORMAT == egress_proto.EncodedFileType.MP3 else "ogg"

            single_file_base = f"{folder_path}/call.{file_extension}"
            url_single_file_actual = f"{url_folder_path}/call.{file_extension}"

            logger.info(
                "Recording paths - API: %s, URL: %s",
                folder_path,
                url_folder_path,
            )

            # Create S3 upload configuration using temporary credentials
            s3_upload = api.S3Upload(  # type: ignore[attr-defined]  # noqa: PGH003
                access_key=creds["access_key"],
                secret=creds["secret_key"],
                session_token=creds["session_token"],
                region=creds["region"],
                bucket=creds["bucket"],
            )

            single_file_req = api.RoomCompositeEgressRequest(  # type: ignore[attr-defined]  # noqa: PGH003
                room_name=room_name,
                audio_only=True,
                audio_mixing=egress_proto.AudioMixing.Value("DUAL_CHANNEL_AGENT"),
                file_outputs=[
                    api.EncodedFileOutput(  # type: ignore[attr-defined]  # noqa: PGH003
                        filepath=single_file_base,
                        file_type=RECORDING_FORMAT,
                        s3=s3_upload,
                    )
                ],
            )

            logger.info("Starting stereo recording for room %s", room_name)
            single_file_res = await self.lkapi.egress.start_room_composite_egress(
                single_file_req
            )

            egress_id = single_file_res.egress_id
            self.egress_id = egress_id
            logger.info("Stereo recording started with egress ID: %s", egress_id)

            recording_url = f"{creds['base_url']}/{url_single_file_actual}"
            logger.info("Recording URL: %s", recording_url)

            return recording_url

        except Exception as e:  # noqa: BLE001
            logger.error("Failed to start recording: %s", e, exc_info=True)
            if hasattr(e, 'details'):
                logger.error("API error details: %s", getattr(e, 'details'))
            return None

    async def stop_egress(self) -> None:
        """Stop the active egress gracefully so the file is finalised on S3.

        Must be called **before** the room is deleted, otherwise the egress
        will be terminated abruptly and the recording may be incomplete.
        """
        if not self.egress_id:
            return

        try:
            await self.lkapi.egress.stop_egress(
                api.StopEgressRequest(egress_id=self.egress_id)  # type: ignore[attr-defined]  # noqa: PGH003
            )
            logger.info("Egress stopped — egress_id=%s (file finalised on S3)", self.egress_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not stop egress %s: %s", self.egress_id, e)
        finally:
            self.egress_id = None

    async def stop_recording(self) -> None:
        """Stop the active egress and clean up resources."""
        await self.stop_egress()
        try:
            await self.lkapi.aclose()
            logger.info("LiveKit API client closed")
        except Exception as e:  # noqa: BLE001
            logger.error("Error closing LiveKit API client: %s", e, exc_info=True)
