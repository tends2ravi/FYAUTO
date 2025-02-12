"""
Integration tests for the complete video generation workflow.
"""
import pytest
from pathlib import Path
import asyncio
from typing import Dict, Any

from src.generation import AudioGenerator, VisualGenerator, ScriptGenerator
from src.features import (
    CaptionSystem,
    VideoPreferences,
    WorkflowManager,
    VideoAssembler
)
from src.core.errors import VideoProductionError
from tests.utils import (
    BaseGenerationTest,
    create_test_script,
    create_test_preferences,
    create_test_format_settings
)

class TestVideoWorkflow(BaseGenerationTest):
    """Test complete video generation workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_workflow(self, mock_redis, mock_llm_client, mock_tts_client, mock_image_client):
        """Set up workflow test environment."""
        # Initialize components
        self.script_generator = ScriptGenerator(
            error_handler=self.error_handler,
            output_dir=self.output_dir / "scripts"
        )
        self.audio_generator = AudioGenerator(
            error_handler=self.error_handler,
            output_dir=self.output_dir / "audio"
        )
        self.visual_generator = VisualGenerator(
            error_handler=self.error_handler,
            output_dir=self.output_dir / "visuals"
        )
        self.caption_system = CaptionSystem(
            error_handler=self.error_handler
        )
        self.video_assembler = VideoAssembler(
            error_handler=self.error_handler,
            output_dir=self.output_dir / "videos"
        )
        
        # Set up workflow manager
        self.workflow_manager = WorkflowManager(
            script_generator=self.script_generator,
            audio_generator=self.audio_generator,
            visual_generator=self.visual_generator,
            caption_system=self.caption_system,
            video_assembler=self.video_assembler,
            error_handler=self.error_handler
        )
        
        # Set up test data
        self.test_preferences = create_test_preferences()
        self.test_format = create_test_format_settings()
        
        yield
    
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self):
        """Test successful end-to-end video generation."""
        result = await self.workflow_manager.create_video(
            topic="Test topic about Python programming",
            style="educational",
            duration_minutes=5.0,
            preferences=self.test_preferences,
            format_settings=self.test_format
        )
        
        # Verify script generation
        assert result["script"]["title"]
        assert len(result["script"]["scenes"]) > 0
        
        # Verify audio generation
        assert all(Path(audio["path"]).exists() for audio in result["audio"])
        
        # Verify visual generation
        assert all(Path(visual["path"]).exists() for visual in result["visuals"])
        
        # Verify video assembly
        assert Path(result["video_path"]).exists()
        assert Path(result["video_path"]).stat().st_size > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_retries(self):
        """Test workflow with component failures and retries."""
        # Mock temporary failures
        self.mock_llm_client.generate.side_effect = [
            Exception("Temporary failure"),
            create_test_script()
        ]
        
        result = await self.workflow_manager.create_video(
            topic="Test topic with retries",
            style="educational",
            duration_minutes=5.0,
            preferences=self.test_preferences,
            format_settings=self.test_format
        )
        
        assert result["video_path"]
        assert len(self.error_handler.errors) > 0  # Errors were logged
    
    @pytest.mark.asyncio
    async def test_parallel_processing(self):
        """Test parallel processing of scenes."""
        # Create multiple scenes
        script = create_test_script()
        script["scenes"].extend([
            {
                "id": f"scene_{i}",
                "title": f"Test Scene {i}",
                "content": f"Test content {i}",
                "duration": 60.0,
                "visuals": [{"description": f"Test visual {i}", "duration": 20.0}]
            }
            for i in range(2, 5)
        ])
        
        start_time = asyncio.get_event_loop().time()
        
        result = await self.workflow_manager.process_scenes(
            script=script,
            preferences=self.test_preferences
        )
        
        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time
        
        # Verify parallel processing was faster than sequential would be
        assert processing_time < len(script["scenes"]) * 2.0  # Assuming each scene takes ~2s
        assert len(result["audio"]) == len(script["scenes"])
        assert len(result["visuals"]) == len(script["scenes"])
    
    @pytest.mark.asyncio
    async def test_resource_management(self):
        """Test resource management during video generation."""
        # Monitor resource usage
        initial_temp_files = list(self.temp_dir.glob("**/*"))
        
        result = await self.workflow_manager.create_video(
            topic="Test resource management",
            style="educational",
            duration_minutes=1.0,
            preferences=self.test_preferences,
            format_settings=self.test_format
        )
        
        # Clean up
        await self.workflow_manager.cleanup()
        
        # Verify temporary files were cleaned up
        final_temp_files = list(self.temp_dir.glob("**/*"))
        assert len(final_temp_files) <= len(initial_temp_files)
        
        # Verify only final output remains
        assert Path(result["video_path"]).exists()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error recovery and fallback mechanisms."""
        # Simulate cascading failures
        self.mock_llm_client.generate.side_effect = Exception("Primary LLM failed")
        self.mock_tts_client.synthesize.side_effect = Exception("Primary TTS failed")
        
        with pytest.raises(VideoProductionError) as exc_info:
            await self.workflow_manager.create_video(
                topic="Test error recovery",
                style="educational",
                duration_minutes=1.0,
                preferences=self.test_preferences,
                format_settings=self.test_format
            )
        
        # Verify error handling
        assert "Production failed" in str(exc_info.value)
        assert len(self.error_handler.errors) > 0
        
        # Verify cleanup after failure
        temp_files = list(self.temp_dir.glob("**/*"))
        assert len(temp_files) == len(initial_temp_files)  # No leftover files
    
    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        """Test progress tracking during video generation."""
        progress_updates = []
        
        def progress_callback(progress: float, stage: str):
            progress_updates.append((progress, stage))
        
        result = await self.workflow_manager.create_video(
            topic="Test progress tracking",
            style="educational",
            duration_minutes=1.0,
            preferences=self.test_preferences,
            format_settings=self.test_format,
            progress_callback=progress_callback
        )
        
        # Verify progress tracking
        assert len(progress_updates) > 0
        assert progress_updates[0][0] == 0.0  # Started at 0%
        assert progress_updates[-1][0] == 1.0  # Ended at 100%
        assert all(0.0 <= p[0] <= 1.0 for p in progress_updates)  # All valid percentages 