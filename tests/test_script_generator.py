"""
Tests for the script generation system.
"""
import pytest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch, MagicMock
from src.script_generator import ScriptGenerator

@pytest.mark.asyncio
class TestScriptGenerator:
    """Test suite for ScriptGenerator."""
    
    @pytest.fixture
    def script_generator(self):
        """Create script generator instance."""
        return ScriptGenerator()
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return {
            "title": "Test Video Title",
            "hook": "This is an attention-grabbing hook",
            "sections": [
                {
                    "title": "Introduction",
                    "content": "This is the introduction section.",
                    "duration_seconds": 30
                },
                {
                    "title": "Main Content",
                    "content": "This is the main content section.",
                    "duration_seconds": 60
                }
            ],
            "call_to_action": "Like and subscribe!",
            "metadata": {
                "estimated_duration_seconds": 120,
                "key_points": ["point 1", "point 2"],
                "target_keywords": ["test", "video"]
            }
        }
    
    async def test_generate_script(self, script_generator, mock_llm_response, test_dir):
        """Test script generation."""
        # Mock the Gemini API call
        with patch('subprocess.run') as mock_run:
            # Set up the mock response
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = json.dumps(mock_llm_response)
            mock_run.return_value = mock_process
            
            # Generate script
            script_data = script_generator.generate_script(
                topic="Test Topic",
                style="informative",
                duration_minutes=5,
                target_audience="general"
            )
            
            # Verify the result
            assert isinstance(script_data, dict)
            assert "title" in script_data
            assert "hook" in script_data
            assert "sections" in script_data
            assert "call_to_action" in script_data
            assert "metadata" in script_data
            
            # Verify sections
            assert len(script_data["sections"]) == 2
            for section in script_data["sections"]:
                assert "title" in section
                assert "content" in section
                assert "duration_seconds" in section
            
            # Verify metadata
            assert "estimated_duration_seconds" in script_data["metadata"]
            assert "key_points" in script_data["metadata"]
            assert "target_keywords" in script_data["metadata"]
    
    async def test_create_prompt(self, script_generator):
        """Test prompt creation."""
        prompt = script_generator._create_prompt(
            topic="Test Topic",
            style="informative",
            duration_minutes=5,
            target_audience="general",
            additional_context={"key_points": ["point 1", "point 2"]}
        )
        
        # Verify prompt content
        assert "Test Topic" in prompt
        assert "informative" in prompt
        assert "5 minutes" in prompt
        assert "general" in prompt
        assert "point 1" in prompt
        assert "point 2" in prompt
        assert "JSON" in prompt
    
    async def test_process_response_valid(self, script_generator, mock_llm_response):
        """Test processing of valid response."""
        response = json.dumps(mock_llm_response)
        
        processed = script_generator._process_response(response)
        
        assert isinstance(processed, dict)
        assert processed["title"] == mock_llm_response["title"]
        assert processed["hook"] == mock_llm_response["hook"]
        assert len(processed["sections"]) == len(mock_llm_response["sections"])
    
    async def test_process_response_invalid_json(self, script_generator):
        """Test processing of invalid JSON response."""
        invalid_response = "This is not JSON"
        
        with pytest.raises(ValueError) as exc_info:
            script_generator._process_response(invalid_response)
        
        assert "No JSON found in response" in str(exc_info.value)
    
    async def test_process_response_missing_fields(self, script_generator):
        """Test processing of response with missing required fields."""
        incomplete_response = {
            "title": "Test Title",
            # Missing required fields
        }
        
        with pytest.raises(ValueError) as exc_info:
            script_generator._process_response(json.dumps(incomplete_response))
        
        assert "Missing required fields" in str(exc_info.value)
    
    async def test_call_gemini_api_success(self, script_generator, mock_llm_response, test_dir):
        """Test successful Gemini API call."""
        with patch('subprocess.run') as mock_run:
            # Set up the mock response
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = json.dumps(mock_llm_response)
            mock_run.return_value = mock_process
            
            response = script_generator._call_gemini_api("Test prompt")
            
            assert response is not None
            assert mock_llm_response["title"] in response
    
    async def test_call_gemini_api_failure(self, script_generator):
        """Test failed Gemini API call."""
        with patch('subprocess.run') as mock_run:
            # Set up the mock error response
            mock_process = MagicMock()
            mock_process.returncode = 1
            mock_process.stderr = "API Error"
            mock_run.return_value = mock_process
            
            with pytest.raises(Exception) as exc_info:
                script_generator._call_gemini_api("Test prompt")
            
            assert "Error calling llm_api.py" in str(exc_info.value)
    
    async def test_script_output_file(self, script_generator, mock_llm_response, test_dir):
        """Test script output file creation."""
        with patch('subprocess.run') as mock_run:
            # Set up the mock response
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = json.dumps(mock_llm_response)
            mock_run.return_value = mock_process
            
            script_data = script_generator.generate_script(
                topic="Test Topic",
                style="informative",
                duration_minutes=5
            )
            
            # Check if output file exists
            output_file = Path(test_dir) / "output" / "generated_script.json"
            assert output_file.exists()
            
            # Verify file content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data == script_data
    
    async def test_duration_calculation(self, script_generator, mock_llm_response):
        """Test duration calculation in generated script."""
        with patch('subprocess.run') as mock_run:
            # Set up the mock response
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.stdout = json.dumps(mock_llm_response)
            mock_run.return_value = mock_process
            
            script_data = script_generator.generate_script(
                topic="Test Topic",
                duration_minutes=5
            )
            
            # Calculate total duration from sections
            total_duration = sum(
                section["duration_seconds"]
                for section in script_data["sections"]
            )
            
            # Verify duration matches metadata
            assert abs(total_duration - script_data["metadata"]["estimated_duration_seconds"]) <= 30
    
    async def test_style_incorporation(self, script_generator):
        """Test incorporation of style in prompt."""
        styles = ["informative", "entertaining", "educational"]
        
        for style in styles:
            prompt = script_generator._create_prompt(
                topic="Test Topic",
                style=style,
                duration_minutes=5,
                target_audience="general"
            )
            
            assert style in prompt.lower()
    
    async def test_target_audience_incorporation(self, script_generator):
        """Test incorporation of target audience in prompt."""
        audiences = ["beginners", "experts", "general"]
        
        for audience in audiences:
            prompt = script_generator._create_prompt(
                topic="Test Topic",
                style="informative",
                duration_minutes=5,
                target_audience=audience
            )
            
            assert audience in prompt.lower()
    
    async def test_additional_context_handling(self, script_generator):
        """Test handling of additional context in prompt."""
        additional_context = {
            "key_points": ["point 1", "point 2"],
            "tone": "professional",
            "include_examples": True
        }
        
        prompt = script_generator._create_prompt(
            topic="Test Topic",
            style="informative",
            duration_minutes=5,
            target_audience="general",
            additional_context=additional_context
        )
        
        # Verify context incorporation
        assert "point 1" in prompt
        assert "point 2" in prompt
        assert "professional" in prompt
        assert "include_examples" in prompt 