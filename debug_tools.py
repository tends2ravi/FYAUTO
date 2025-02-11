"""
Debugging and performance monitoring tools.
"""
import time
import psutil
import os
import sys
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, Any

class PerformanceMonitor:
    """Monitor and analyze performance metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.start_time = 0
        
    def start_monitoring(self):
        """Start monitoring performance metrics."""
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()
    
    def stop_monitoring(self) -> dict:
        """Stop monitoring and return metrics."""
        current_memory = self.process.memory_info().rss
        elapsed_time = time.time() - self.start_time
        
        return {
            "memory_used": (current_memory - self.start_memory) / 1024 / 1024,  # MB
            "elapsed_time": elapsed_time,
            "cpu_percent": self.process.cpu_percent()
        }

class ResourceMonitor:
    """Monitor system resource usage."""
    
    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Get current system resource metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "open_files": len(psutil.Process().open_files()),
            "threads": len(psutil.Process().threads()),
            "python_version": sys.version,
            "platform": sys.platform,
            "cwd": os.getcwd(),
            "env_vars": {
                key: value for key, value in os.environ.items() 
                if key.startswith(("PYTHON", "PATH", "VIRTUAL_ENV"))
            }
        }
    
    @staticmethod
    def log_resource_usage():
        """Log current resource usage."""
        metrics = ResourceMonitor.get_system_metrics()
        logger.info("System Resource Usage:")
        for key, value in metrics.items():
            if key != "env_vars":
                logger.info(f"{key}: {value}")
        logger.debug("Environment Variables:")
        for key, value in metrics["env_vars"].items():
            logger.debug(f"{key}: {value}")

def debug_info(msg: str, data: Any = None):
    """Log debug information with optional data dump."""
    logger.debug(f"DEBUG: {msg}")
    if data is not None:
        logger.debug(f"Data dump: {data}")

async def run_with_monitoring(topic: str, format_type: str = "youtube", style: str = "standard", duration: float = 3.0, output_path: Optional[Path] = None):
    """Run the video creation process with monitoring."""
    from src.workflow import create_video  # Import here to avoid circular imports
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # Run the main process
        await create_video(
            topic=topic,
            format_type=format_type,
            style=style,
            duration=duration,
            output_path=output_path
        )
        
        # Get and log metrics
        metrics = monitor.stop_monitoring()
        logger.info(f"Performance metrics:")
        logger.info(f"Memory used: {metrics['memory_used']:.2f} MB")
        logger.info(f"Elapsed time: {metrics['elapsed_time']:.2f} seconds")
        logger.info(f"CPU usage: {metrics['cpu_percent']}%")
        
        # Log system resource usage
        ResourceMonitor.log_resource_usage()
        
    except Exception as e:
        logger.error(f"Error during monitoring: {str(e)}")
        raise 