"""
YouTube upload and metadata management module.
"""
from pathlib import Path
from typing import Dict, Optional, List
from loguru import logger
import json
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from . import config

class YouTubeUploader:
    """Handles YouTube video uploads and metadata management."""
    
    def __init__(self):
        self.credentials_path = config.BASE_DIR / "credentials" / "youtube_credentials.json"
        self.token_path = config.BASE_DIR / "credentials" / "youtube_token.json"
        self.credentials = None
        self.youtube = None
        
        # Ensure credentials directory exists
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)
        
        # YouTube API scopes needed
        self.SCOPES = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube"
        ]
    
    def authenticate(self) -> None:
        """Authenticate with YouTube API."""
        try:
            if self.token_path.exists():
                with open(self.token_path, 'r') as token:
                    self.credentials = Credentials.from_authorized_user_file(
                        self.token_path,
                        self.SCOPES
                    )
            
            # If credentials don't exist or are invalid, refresh them
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    if not self.credentials_path.exists():
                        raise FileNotFoundError(
                            f"YouTube credentials file not found: {self.credentials_path}"
                        )
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_path),
                        self.SCOPES
                    )
                    self.credentials = flow.run_local_server(port=0)
                
                # Save the credentials for future use
                with open(self.token_path, 'w') as token:
                    token.write(self.credentials.to_json())
            
            # Build YouTube API service
            self.youtube = build('youtube', 'v3', credentials=self.credentials)
            logger.info("Successfully authenticated with YouTube API")
            
        except Exception as e:
            logger.error(f"Error authenticating with YouTube: {str(e)}")
            raise
    
    def upload_video(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = "22",  # Default to "Education"
        privacy_status: str = "private",
        language: str = "en",
        thumbnail_path: Optional[Path] = None,
        playlist_id: Optional[str] = None
    ) -> str:
        """
        Upload a video to YouTube.
        
        Args:
            video_path: Path to the video file
            title: Video title
            description: Video description
            tags: List of video tags
            category_id: YouTube category ID
            privacy_status: Video privacy setting (private/unlisted/public)
            language: Video language code
            thumbnail_path: Optional custom thumbnail
            playlist_id: Optional playlist to add the video to
            
        Returns:
            YouTube video ID
        """
        try:
            if not self.youtube:
                self.authenticate()
            
            # Prepare video metadata
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'categoryId': category_id,
                    'defaultLanguage': language,
                    'defaultAudioLanguage': language
                },
                'status': {
                    'privacyStatus': privacy_status,
                    'selfDeclaredMadeForKids': False
                }
            }
            
            # Create upload request
            insert_request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=MediaFileUpload(
                    str(video_path),
                    chunksize=1024*1024,
                    resumable=True
                )
            )
            
            # Execute upload with progress tracking
            video_id = self._execute_upload_request(insert_request)
            logger.info(f"Successfully uploaded video: {video_id}")
            
            # Set custom thumbnail if provided
            if thumbnail_path and thumbnail_path.exists():
                self._set_thumbnail(video_id, thumbnail_path)
            
            # Add to playlist if specified
            if playlist_id:
                self._add_to_playlist(video_id, playlist_id)
            
            # Save upload metadata
            self._save_upload_metadata(
                video_path,
                {
                    "video_id": video_id,
                    "title": title,
                    "privacy_status": privacy_status,
                    "playlist_id": playlist_id
                }
            )
            
            return video_id
            
        except HttpError as e:
            logger.error(f"HTTP error during upload: {e.resp.status} {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error uploading video: {str(e)}")
            raise
    
    def _execute_upload_request(self, request) -> str:
        """Execute upload request with progress tracking."""
        response = None
        error = None
        retry = 0
        
        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"Upload progress: {progress}%")
            except HttpError as e:
                if e.resp.status in [500, 502, 503, 504]:
                    if retry > 10:
                        raise
                    retry += 1
                    logger.warning(f"Retrying upload (attempt {retry})")
                else:
                    raise
        
        return response['id']
    
    def _set_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        """Set custom thumbnail for a video."""
        try:
            self.youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(str(thumbnail_path))
            ).execute()
            logger.info(f"Set custom thumbnail for video: {video_id}")
        except Exception as e:
            logger.error(f"Error setting thumbnail: {str(e)}")
    
    def _add_to_playlist(self, video_id: str, playlist_id: str) -> None:
        """Add video to a playlist."""
        try:
            self.youtube.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id
                        }
                    }
                }
            ).execute()
            logger.info(f"Added video {video_id} to playlist: {playlist_id}")
        except Exception as e:
            logger.error(f"Error adding to playlist: {str(e)}")
    
    def _save_upload_metadata(self, video_path: Path, metadata: Dict) -> None:
        """Save upload metadata for future reference."""
        metadata_path = video_path.with_suffix(".upload.json")
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved upload metadata to: {metadata_path}")
    
    def get_video_analytics(self, video_id: str) -> Dict:
        """Get analytics data for a video."""
        try:
            if not self.youtube:
                self.authenticate()
            
            # Get video statistics
            stats = self.youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()
            
            if not stats['items']:
                raise ValueError(f"Video not found: {video_id}")
            
            return stats['items'][0]['statistics']
            
        except Exception as e:
            logger.error(f"Error getting video analytics: {str(e)}")
            raise 