"""
Script generation module using Google's Gemini API.
"""
import json
from typing import Dict, Optional
from loguru import logger
import aiohttp
import asyncio
import tempfile
import os

from . import config
from .error_handler import ErrorHandler

class ScriptGenerator:
    """Handles script generation using Google's Gemini API."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        
    async def generate_script(
        self,
        topic: str,
        style: str = "informative",
        duration_minutes: int = 5,
        target_audience: str = "general",
        additional_context: Optional[Dict] = None
    ) -> Dict:
        """
        Generate a video script using Gemini API.
        
        Args:
            topic: Main topic of the video
            style: Writing style (informative, entertaining, educational)
            duration_minutes: Target video duration in minutes
            target_audience: Target audience for the video
            additional_context: Additional context or requirements
            
        Returns:
            Dict containing:
                - script: The generated script text
                - sections: List of script sections
                - metadata: Additional metadata about the script
        """
        prompt = self._create_prompt(
            topic, style, duration_minutes, target_audience, additional_context
        )
        
        try:
            response = await self._call_gemini_api(prompt)
            script_data = self._process_response(response)
            
            # Save the script to a file for inspection
            output_dir = config.OUTPUT_DIR
            script_file = output_dir / "generated_script.json"
            with open(script_file, "w", encoding="utf-8") as f:
                json.dump(script_data, f, indent=2)
            logger.info(f"Saved generated script to: {script_file}")
            
            return script_data
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            raise
    
    def _create_prompt(
        self,
        topic: str,
        style: str,
        duration_minutes: int,
        target_audience: str,
        additional_context: Optional[Dict]
    ) -> str:
        """Create a detailed prompt for the AI."""
        context = additional_context or {}
        
        prompt = f"""You are a professional YouTube script writer. Create a script for a video about {topic}.
        
        Style: {style}
        Target Duration: {duration_minutes} minutes
        Target Audience: {target_audience}
        
        Additional Requirements:
        - Include an engaging hook at the beginning
        - Break down complex topics into simple explanations
        - Include clear section transitions
        - End with a call to action
        
        Additional Context:
        {json.dumps(context, indent=2)}
        
        Format your response EXACTLY as a JSON object with the following structure:
        {{
            "title": "Video title",
            "hook": "Opening hook",
            "sections": [
                {{
                    "title": "Section title",
                    "content": "Section content",
                    "duration_seconds": 60
                }}
            ],
            "call_to_action": "Call to action text",
            "metadata": {{
                "estimated_duration_seconds": 300,
                "key_points": ["point 1", "point 2"],
                "target_keywords": ["keyword1", "keyword2"]
            }}
        }}
        
        Make sure to format the response as valid JSON with all fields present. Each section should have appropriate duration based on the total video length of {duration_minutes} minutes.
        """
        return prompt
    
    async def _call_gemini_api(self, prompt: str) -> str:
        """Make the API call to Gemini using the llm_api.py tool."""
        try:
            # Create a temporary file for the response
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            # Call the llm_api.py tool with venv Python
            cmd = [
                os.path.join("venv", "Scripts", "python"),
                os.path.join("tools", "llm_api.py"),
                "--prompt", prompt,
                "--provider", "gemini",
                "--output", temp_path
            ]
            
            # Use asyncio.create_subprocess_exec for async process execution
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Error calling llm_api.py: {stderr.decode()}")
            
            # Read the response from the temporary file
            with open(temp_path, 'r') as f:
                response = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _process_response(self, response: str) -> Dict:
        """Process and validate the API response."""
        try:
            # Extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            script_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["title", "hook", "sections", "call_to_action", "metadata"]
            missing_fields = [field for field in required_fields if field not in script_data]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in response: {missing_fields}")
            
            return script_data
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error processing Gemini response: {str(e)}")
            raise ValueError("Invalid response format from Gemini API") 