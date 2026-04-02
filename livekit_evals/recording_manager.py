"""
Recording Manager for LiveKit Agent

Manages call recording using LiveKit Egress API with S3 storage.
Provides recording URLs that can be accessed after the call ends.
"""

import asyncio
import logging
import urllib.parse
from datetime import datetime
from typing import Optional, Tuple

import aiohttp
from livekit import api
from livekit.protocol import egress as egress_proto

logger = logging.getLogger("recording_manager")

SHOULD_RECORD_SEGMENTS = False

RECORDING_FORMAT = egress_proto.EncodedFileType.MP3


class RecordingManager:
    """Manages call recording using LiveKit Egress API"""

    def __init__(self, credentials_url: str, api_key: str):
        """Initialize the recording manager

        Args:
            credentials_url: URL of the endpoint that issues temporary S3 credentials
            api_key: SuperBryn API key used to authenticate the credentials request
        """
        self.credentials_url = credentials_url
        self.api_key = api_key
        self.lkapi = api.LiveKitAPI()
        self.recording_active = asyncio.Event()
        self.recording_active.set()

        logger.info("RecordingManager initialized (credentials fetched per-session)")

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

    async def start_recording(
        self,
        room_name: str,
        phone_number: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Start recording the call session using LiveKit's Egress API

        Args:
            room_name: Name of the room to record
            phone_number: Phone number to include in the recording path (optional)

        Returns:
            tuple: (recording_url, egress_id) or (None, None) if failed
        """
        try:
            creds = await self._fetch_credentials(room_name)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Build the folder path for the recording
            if phone_number:
                encoded_phone = urllib.parse.quote_plus(phone_number)
                folder_path = f"call_recordings/{phone_number}/{timestamp}"
                url_folder_path = f"call_recordings/{encoded_phone}/{timestamp}"
            else:
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

            # Optional: HLS segmented recording (for reliability)
            if SHOULD_RECORD_SEGMENTS:
                playlist_name = f"{folder_path}/playlist.m3u8"

                hls_req = api.RoomCompositeEgressRequest(  # type: ignore[attr-defined]  # noqa: PGH003
                    room_name=room_name,
                    layout="speaker",
                    audio_only=True,
                    segment_outputs=[
                        api.SegmentedFileOutput(  # type: ignore[attr-defined]  # noqa: PGH003
                            filename_prefix=folder_path,
                            playlist_name=playlist_name,
                            segment_duration=5,
                            s3=s3_upload,
                        )
                    ],
                )

                logger.info("Starting HLS recording for room %s", room_name)
                hls_res = await self.lkapi.egress.start_room_composite_egress(hls_req)
                logger.info("HLS recording started with egress ID: %s", hls_res.egress_id)

            # Start single file recording (primary)
            single_file_req = api.RoomCompositeEgressRequest(  # type: ignore[attr-defined]  # noqa: PGH003
                room_name=room_name,
                layout="speaker",
                audio_only=True,
                file_outputs=[
                    api.EncodedFileOutput(  # type: ignore[attr-defined]  # noqa: PGH003
                        filepath=single_file_base,
                        file_type=RECORDING_FORMAT,
                        s3=s3_upload,
                    )
                ],
            )

            logger.info("Starting single file recording for room %s", room_name)
            single_file_res = await self.lkapi.egress.start_room_composite_egress(
                single_file_req
            )

            egress_id = single_file_res.egress_id
            logger.info(
                "Single file recording started with egress ID: %s",
                egress_id,
            )

            recording_url = f"{creds['base_url']}/{url_single_file_actual}"
            logger.info("Recording URL: %s", recording_url)

            self._start_recording_monitoring(egress_id)

            return recording_url, egress_id

        except Exception as e:  # noqa: BLE001
            logger.error("Failed to start recording: %s", e, exc_info=True)
            if hasattr(e, 'details'):
                logger.error("API error details: %s", getattr(e, 'details'))
            return None, None

    def _start_recording_monitoring(self, egress_id: str) -> None:
        """Start monitoring the recording status in background

        Args:
            egress_id: The egress ID to monitor
        """
        if not egress_id:
            return

        async def check_recording_loop():
            try:
                while self.recording_active.is_set():
                    try:
                        await self.check_recording_status(egress_id)
                    except Exception as e:  # noqa: BLE001
                        logger.error("Error checking recording status: %s", e)

                        err_str = str(e).lower()
                        if "session is closed" in err_str or "connection closed" in err_str:
                            logger.info("LiveKit API session closed, stopping monitoring")
                            self.recording_active.clear()
                            break

                    await asyncio.sleep(30)

                logger.info("Recording monitoring loop finished")

            except asyncio.CancelledError:
                logger.info("Recording monitoring cancelled")
            except Exception as e:  # noqa: BLE001
                logger.error("Unhandled error in recording monitoring: %s", e, exc_info=True)
            finally:
                self.recording_active.clear()

        asyncio.create_task(check_recording_loop())

    async def check_recording_status(self, egress_id: str) -> None:
        """
        Check the status of an egress job

        Args:
            egress_id: The ID of the egress job to check
        """
        try:
            res = await self.lkapi.egress.list_egress(api.ListEgressRequest())  # type: ignore[attr-defined]  # noqa: PGH003

            found = False
            for item in res.items:
                if item.egress_id == egress_id:
                    logger.info("Egress status for %s: %s", egress_id, item.status)
                    found = True
                    break

            if not found:
                logger.warning("No egress item found with ID %s", egress_id)

        except Exception as e:  # noqa: BLE001
            logger.error("Failed to check egress status: %s", e, exc_info=True)
            raise

    async def stop_recording(self) -> None:
        """Stop recording and clean up resources"""
        self.recording_active.clear()
        try:
            await self.lkapi.aclose()
            logger.info("LiveKit API client closed")
        except Exception as e:  # noqa: BLE001
            logger.error("Error closing LiveKit API client: %s", e, exc_info=True)
