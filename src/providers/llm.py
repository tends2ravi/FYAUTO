"""
LLM providers management module with fallback support.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Any
import asyncio
import json
from pathlib import Path
import tempfile
import os
from loguru import logger

from . import config
from .error_handler import ErrorHandler, APIError

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
    
    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass

class GeminiProvider(LLMProvider):
    """Google's Gemini API provider."""
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text using Gemini API."""
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            cmd = [
                os.path.join("venv", "Scripts", "python"),
                os.path.join("tools", "llm_api.py"),
                "--prompt", prompt,
                "--provider", "gemini",
                "--output", temp_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise APIError(
                    f"Gemini API error: {stderr.decode()}",
                    "GEMINI_API_ERROR",
                    {"stderr": stderr.decode()}
                )
            
            with open(temp_path, 'r') as f:
                response = f.read()
            
            os.unlink(temp_path)
            return response
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise APIError(
                f"Gemini API error: {str(e)}",
                "GEMINI_API_ERROR"
            )
    
    def get_provider_name(self) -> str:
        return "gemini"

class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider."""
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text using DeepSeek API."""
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            cmd = [
                os.path.join("venv", "Scripts", "python"),
                os.path.join("tools", "llm_api.py"),
                "--prompt", prompt,
                "--provider", "deepseek",
                "--output", temp_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise APIError(
                    f"DeepSeek API error: {stderr.decode()}",
                    "DEEPSEEK_API_ERROR",
                    {"stderr": stderr.decode()}
                )
            
            with open(temp_path, 'r') as f:
                response = f.read()
            
            os.unlink(temp_path)
            return response
            
        except Exception as e:
            logger.error(f"Error calling DeepSeek API: {str(e)}")
            raise APIError(
                f"DeepSeek API error: {str(e)}",
                "DEEPSEEK_API_ERROR"
            )
    
    def get_provider_name(self) -> str:
        return "deepseek"

class OpenAIProvider(LLMProvider):
    """OpenAI GPT API provider."""
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text using OpenAI API."""
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            cmd = [
                os.path.join("venv", "Scripts", "python"),
                os.path.join("tools", "llm_api.py"),
                "--prompt", prompt,
                "--provider", "openai",
                "--output", temp_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise APIError(
                    f"OpenAI API error: {stderr.decode()}",
                    "OPENAI_API_ERROR",
                    {"stderr": stderr.decode()}
                )
            
            with open(temp_path, 'r') as f:
                response = f.read()
            
            os.unlink(temp_path)
            return response
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise APIError(
                f"OpenAI API error: {str(e)}",
                "OPENAI_API_ERROR"
            )
    
    def get_provider_name(self) -> str:
        return "openai"

class LocalHuggingFaceProvider(LLMProvider):
    """Local Hugging Face model provider."""
    
    async def generate_text(self, prompt: str) -> str:
        """Generate text using local Hugging Face model."""
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
                temp_path = f.name
            
            cmd = [
                os.path.join("venv", "Scripts", "python"),
                os.path.join("tools", "llm_api.py"),
                "--prompt", prompt,
                "--provider", "local",
                "--output", temp_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise APIError(
                    f"Local model error: {stderr.decode()}",
                    "LOCAL_MODEL_ERROR",
                    {"stderr": stderr.decode()}
                )
            
            with open(temp_path, 'r') as f:
                response = f.read()
            
            os.unlink(temp_path)
            return response
            
        except Exception as e:
            logger.error(f"Error calling local model: {str(e)}")
            raise APIError(
                f"Local model error: {str(e)}",
                "LOCAL_MODEL_ERROR"
            )
    
    def get_provider_name(self) -> str:
        return "local"

class LLMProviderManager:
    """Manages multiple LLM providers with fallback support."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        self.error_handler = error_handler or ErrorHandler()
        
        # Initialize providers in order of preference
        self.providers = [
            GeminiProvider(self.error_handler),
            DeepSeekProvider(self.error_handler),
            OpenAIProvider(self.error_handler),
            LocalHuggingFaceProvider(self.error_handler)
        ]
    
    async def generate_text(self, prompt: str) -> str:
        """
        Generate text using available providers with fallback.
        
        Args:
            prompt: Input prompt for text generation
            
        Returns:
            Generated text from successful provider
            
        Raises:
            APIError: If all providers fail
        """
        errors = []
        
        for provider in self.providers:
            try:
                logger.info(f"Trying provider: {provider.get_provider_name()}")
                response = await provider.generate_text(prompt)
                logger.info(f"Successfully generated text using {provider.get_provider_name()}")
                return response
            except Exception as e:
                logger.warning(f"Provider {provider.get_provider_name()} failed: {str(e)}")
                errors.append({
                    "provider": provider.get_provider_name(),
                    "error": str(e)
                })
        
        # If all providers failed, raise error with details
        raise APIError(
            "All LLM providers failed",
            "ALL_PROVIDERS_FAILED",
            {"errors": errors}
        ) 