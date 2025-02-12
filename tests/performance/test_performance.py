"""
Performance tests for the video generation system.
"""
import pytest
import asyncio
import time
from pathlib import Path
import psutil
import numpy as np
from typing import List, Dict, Any

from src.generation import AudioGenerator, VisualGenerator, ScriptGenerator
from src.features import WorkflowManager
from tests.utils import (
    BaseGenerationTest,
    create_test_script,
    create_test_preferences,
    create_test_format_settings
)

class TestSystemPerformance(BaseGenerationTest):
    """Test system performance metrics."""
    
    @pytest.fixture(autouse=True)
    def setup_performance(self, mock_redis, mock_llm_client, mock_tts_client, mock_image_client):
        """Set up performance test environment."""
        self.workflow_manager = WorkflowManager(
            script_generator=ScriptGenerator(error_handler=self.error_handler),
            audio_generator=AudioGenerator(error_handler=self.error_handler),
            visual_generator=VisualGenerator(error_handler=self.error_handler),
            error_handler=self.error_handler
        )
        
        # Set up test data
        self.test_preferences = create_test_preferences()
        self.test_format = create_test_format_settings()
        
        # Performance thresholds
        self.max_memory_mb = 1024  # 1GB
        self.max_processing_time = 300  # 5 minutes
        self.max_temp_files = 100
        
        yield
    
    async def measure_performance(self, coroutine) -> Dict[str, float]:
        """Measure performance metrics for a coroutine."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        result = await coroutine
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        return {
            "execution_time": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "peak_memory": max(start_memory, end_memory),
            "result": result
        }
    
    @pytest.mark.asyncio
    async def test_script_generation_performance(self):
        """Test script generation performance."""
        metrics = await self.measure_performance(
            self.workflow_manager.script_generator.generate_script(
                topic="Performance test topic",
                style="informative",
                duration_minutes=5.0
            )
        )
        
        assert metrics["execution_time"] < 10.0  # Should take less than 10 seconds
        assert metrics["memory_used"] < 100  # Should use less than 100MB
        assert "title" in metrics["result"]
    
    @pytest.mark.asyncio
    async def test_parallel_scene_processing_performance(self):
        """Test performance of parallel scene processing."""
        # Create test script with multiple scenes
        script = create_test_script()
        script["scenes"].extend([
            {
                "id": f"scene_{i}",
                "title": f"Scene {i}",
                "content": f"Content {i}",
                "duration": 30.0,
                "visuals": [{"description": f"Visual {i}", "duration": 10.0}]
            }
            for i in range(2, 10)
        ])
        
        # Test different batch sizes
        batch_sizes = [2, 4, 8]
        processing_times = []
        
        for batch_size in batch_sizes:
            metrics = await self.measure_performance(
                self.workflow_manager.process_scenes_in_batches(
                    script=script,
                    preferences=self.test_preferences,
                    batch_size=batch_size
                )
            )
            processing_times.append(metrics["execution_time"])
        
        # Verify performance improves with larger batch sizes
        assert processing_times[0] > processing_times[1] > processing_times[2]
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage under heavy load."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Generate multiple videos concurrently
        tasks = [
            self.workflow_manager.create_video(
                topic=f"Memory test video {i}",
                style="educational",
                duration_minutes=1.0,
                preferences=self.test_preferences,
                format_settings=self.test_format
            )
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < self.max_memory_mb
        assert all(Path(result["video_path"]).exists() for result in results)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_performance(self):
        """Test performance of resource cleanup."""
        # Create some test files
        test_files = [
            self.temp_dir / f"test_file_{i}.tmp"
            for i in range(100)
        ]
        for file in test_files:
            file.write_text("test content")
        
        start_time = time.time()
        
        # Perform cleanup
        await self.workflow_manager.cleanup()
        
        cleanup_time = time.time() - start_time
        remaining_files = list(self.temp_dir.glob("**/*"))
        
        assert cleanup_time < 5.0  # Cleanup should be quick
        assert len(remaining_files) == 0  # All files should be cleaned up
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test caching system performance."""
        # First generation (no cache)
        no_cache_metrics = await self.measure_performance(
            self.workflow_manager.create_video(
                topic="Cache test video",
                style="educational",
                duration_minutes=1.0,
                preferences=self.test_preferences,
                format_settings=self.test_format
            )
        )
        
        # Second generation (with cache)
        cache_metrics = await self.measure_performance(
            self.workflow_manager.create_video(
                topic="Cache test video",
                style="educational",
                duration_minutes=1.0,
                preferences=self.test_preferences,
                format_settings=self.test_format
            )
        )
        
        # Cache should significantly improve performance
        assert cache_metrics["execution_time"] < no_cache_metrics["execution_time"] * 0.5
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test performance of error handling and recovery."""
        def simulate_error():
            raise Exception("Simulated error")
        
        self.mock_llm_client.generate.side_effect = simulate_error
        
        start_time = time.time()
        
        with pytest.raises(Exception):
            await self.workflow_manager.create_video(
                topic="Error test video",
                style="educational",
                duration_minutes=1.0,
                preferences=self.test_preferences,
                format_settings=self.test_format
            )
        
        error_handling_time = time.time() - start_time
        
        assert error_handling_time < 5.0  # Error handling should be quick
        assert len(self.error_handler.errors) > 0 