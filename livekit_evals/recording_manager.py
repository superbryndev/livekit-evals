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

from livekit import api
from livekit.protocol import egress as egress_proto

# Configure logger
logger = logging.getLogger("recording_manager")

# Set to True if you want HLS segmented recording in addition to single file
SHOULD_RECORD_SEGMENTS = False

# Recording format - options: MP3, OGG, MP4
# MP3: Universal compatibility, familiar format
# OGG: Better compression, open source (default)
RECORDING_FORMAT = egress_proto.EncodedFileType.MP3


class RecordingManager:
    """Manages call recording using LiveKit Egress API"""
    
    def __init__(
        self,
        s3_bucket: str,
        s3_region: str,
        s3_access_key: str,
        s3_secret_key: str,
        s3_base_url: str,
    ):
        """Initialize the recording manager
        
        Args:
            s3_bucket: S3 bucket name
            s3_region: S3 region
            s3_access_key: S3 access key (write-only recommended)
            s3_secret_key: S3 secret key (write-only recommended)
            s3_base_url: S3 base URL for accessing recordings
        """
        self.s3_bucket = s3_bucket
        self.s3_region = s3_region
        self.s3_access_key = s3_access_key
        self.s3_secret_key = s3_secret_key
        self.s3_base_url = s3_base_url
        self.lkapi = api.LiveKitAPI()
        self.recording_active = asyncio.Event()
        self.recording_active.set()  # Initially active
        
        logger.info(
            "RecordingManager initialized - Bucket: %s, Region: %s",
            s3_bucket,
            s3_region,
        )
        
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
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Build the folder path for the recording
            if phone_number:
                # Prepare phone number for use in path
                # The raw phone number gets URL encoded by LiveKit when saving to S3
                encoded_phone = urllib.parse.quote_plus(phone_number)
                folder_path = f"call_recordings/{phone_number}/{timestamp}"
                url_folder_path = f"call_recordings/{encoded_phone}/{timestamp}"
            else:
                # Use room name if phone number not available
                folder_path = f"call_recordings/{room_name}/{timestamp}"
                url_folder_path = f"call_recordings/{urllib.parse.quote_plus(room_name)}/{timestamp}"

            # Determine file extension based on format
            file_extension = "mp3" if RECORDING_FORMAT == egress_proto.EncodedFileType.MP3 else "ogg"
            
            # Base filename for single file recording
            single_file_base = f"{folder_path}/call.{file_extension}"
            url_single_file_actual = f"{url_folder_path}/call.{file_extension}"

            # Log paths for debugging
            logger.info(
                "Recording paths - API: %s, URL: %s",
                folder_path,
                url_folder_path,
            )

            # Create S3 upload configuration
            s3_upload = api.S3Upload(  # type: ignore[attr-defined]  # noqa: PGH003
                access_key=self.s3_access_key,
                secret=self.s3_secret_key,
                region=self.s3_region,
                bucket=self.s3_bucket,
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

            # Construct recording URL
            recording_url = f"{self.s3_base_url}/{url_single_file_actual}"
            logger.info("Recording URL: %s", recording_url)

            # Start monitoring task
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
                        
                        # Check if error indicates session closure
                        err_str = str(e).lower()
                        if "session is closed" in err_str or "connection closed" in err_str:
                            logger.info("LiveKit API session closed, stopping monitoring")
                            self.recording_active.clear()
                            break
                            
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                logger.info("Recording monitoring loop finished")
                
            except asyncio.CancelledError:
                logger.info("Recording monitoring cancelled")
            except Exception as e:  # noqa: BLE001
                logger.error("Unhandled error in recording monitoring: %s", e, exc_info=True)
            finally:
                self.recording_active.clear()

        # Start the monitoring task
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
            raise  # Reraise to handle upstream
    
    async def stop_recording(self) -> None:
        """Stop recording and clean up resources"""
        self.recording_active.clear()
        try:
            await self.lkapi.aclose()
            logger.info("LiveKit API client closed")
        except Exception as e:  # noqa: BLE001
            logger.error("Error closing LiveKit API client: %s", e, exc_info=True)

