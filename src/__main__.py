"""
Main entry point for the video production system.
"""
import argparse
from pathlib import Path
from typing import Optional
from loguru import logger
import asyncio

from . import config
from .script_generator import ScriptGenerator
from .audio_generator import AudioGenerator
from .visual_generator import VisualGenerator
from .video_assembler import VideoAssembler
from .caption_generator import CaptionGenerator
from .background_music import BackgroundMusic
from .video_preferences import VideoPreferences
from .youtube_uploader import YouTubeUploader
from .workflow import create_video_with_retries

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI-Powered Video Production System")
    
    # Basic arguments
    parser.add_argument("topic", help="Topic or title for the video")
    parser.add_argument("--style", default="educational", help="Content style")
    
    # Video format arguments
    parser.add_argument(
        "--format",
        choices=["youtube", "shorts"],
        default="youtube",
        help="Video format (youtube/shorts)"
    )
    parser.add_argument(
        "--niche",
        help="Content niche (e.g., educational_tech, dodstory)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Target duration in seconds"
    )
    
    # Style arguments
    parser.add_argument(
        "--caption-style",
        choices=["basic", "modern", "dynamic"],
        default="modern",
        help="Style for video captions"
    )
    parser.add_argument(
        "--music-style",
        choices=["ambient", "upbeat", "suspenseful", "tech"],
        help="Style for background music"
    )
    
    # Image generator arguments
    parser.add_argument(
        "--image-generator",
        choices=["flux", "stable-diffusion"],
        default="flux",
        help="Image generation model to use"
    )
    parser.add_argument(
        "--image-style",
        choices=["realistic", "cartoon", "artistic", "minimal"],
        default="cartoon",
        help="Visual style for generated images"
    )
    parser.add_argument(
        "--image-quality",
        type=float,
        default=1.0,
        help="Quality emphasis for image generation (0.0-2.0)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for the video file"
    )
    
    # YouTube upload arguments
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the video to YouTube"
    )
    parser.add_argument(
        "--privacy",
        choices=["private", "unlisted", "public"],
        default="private",
        help="YouTube video privacy setting"
    )
    parser.add_argument(
        "--playlist",
        help="YouTube playlist ID to add the video to"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Video language code"
    )
    parser.add_argument(
        "--category",
        default="22",  # Education
        help="YouTube category ID"
    )
    parser.add_argument(
        "--thumbnail",
        type=Path,
        help="Custom thumbnail image path"
    )
    
    args = parser.parse_args()
    
    try:
        # Get format settings and guidelines
        preferences = VideoPreferences()
        format_settings = preferences.get_format_settings(args.format, args.niche)
        
        # Prepare YouTube settings if upload requested
        youtube_settings = None
        if args.upload:
            youtube_settings = {
                "privacy_status": args.privacy,
                "playlist_id": args.playlist,
                "language": args.language,
                "category_id": args.category,
                "thumbnail_path": args.thumbnail
            }
        
        # Prepare additional context with image generation settings
        additional_context = {
            "visual_generation": {
                "generator": args.image_generator,
                "style": args.image_style,
                "quality_emphasis": args.image_quality,
                "format_requirements": {
                    "aspect_ratio": "9:16" if args.format == "shorts" else "16:9",
                    "resolution": (1080, 1920) if args.format == "shorts" else (1920, 1080)
                }
            }
        }
        
        # Create the video
        result = await create_video_with_retries(
            topic=args.topic,
            format=args.format,
            niche=args.niche,
            style=args.style,
            duration=args.duration,
            caption_style=args.caption_style,
            music_style=args.music_style,
            output_path=args.output,
            youtube_settings=youtube_settings,
            additional_context=additional_context
        )
        
        # Print results
        print("\nVideo Production Complete!")
        print(f"Video file: {result['video_path']}")
        
        if args.upload and "youtube_video_id" in result:
            video_id = result["youtube_video_id"]
            print(f"\nUploaded to YouTube:")
            print(f"Video ID: {video_id}")
            print(f"URL: https://youtu.be/{video_id}")
            
            # Get initial analytics
            uploader = YouTubeUploader()
            try:
                stats = uploader.get_video_analytics(video_id)
                print("\nInitial Statistics:")
                print(f"Views: {stats.get('viewCount', 0)}")
                print(f"Likes: {stats.get('likeCount', 0)}")
                print(f"Comments: {stats.get('commentCount', 0)}")
            except:
                pass
        
    except Exception as e:
        logger.error(f"Error during video production: {str(e)}")
        print(f"\nError: {str(e)}")
        print(f"Check the log file for details: {config.OUTPUT_DIR / 'workflow.log'}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 