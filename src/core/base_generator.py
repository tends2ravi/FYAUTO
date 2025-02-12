"""Base generator class for content generation components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from concurrent.futures import ThreadPoolExecutor
import logging

from src.core.base_provider import BaseProvider
from src.utils.error_handler import handle_errors
from src.utils.logging_config import LoggerMixin
from src.config.settings import settings

class BaseGenerator(ABC, LoggerMixin):
    """Base class for all content generation components."""
    
    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        max_workers: Optional[int] = None
    ):
        """Initialize the generator.
        
        Args:
            provider: Optional service provider instance.
            max_workers: Maximum number of worker threads.
        """
        self.provider = provider or self._get_default_provider()
        self.max_workers = max_workers or settings.MAX_WORKERS
    
    @property
    @abstractmethod
    def provider_class(self) -> Type[BaseProvider]:
        """Get the provider class for this generator.
        
        Returns:
            Provider class type.
        """
        pass
    
    @property
    @abstractmethod
    def default_provider_name(self) -> str:
        """Get the default provider name for this generator.
        
        Returns:
            Default provider name.
        """
        pass
    
    def _get_default_provider(self) -> BaseProvider:
        """Get the default provider instance.
        
        Returns:
            Provider instance.
        """
        return self.provider_class()
    
    @handle_errors(max_retries=3)
    def generate(self, **kwargs) -> Any:
        """Generate content using the provider.
        
        Args:
            **kwargs: Generation parameters.
            
        Returns:
            Generated content.
        """
        self.logger.info(f"Generating content with {self.provider}")
        return self._execute_generation(**kwargs)
    
    @abstractmethod
    def _execute_generation(self, **kwargs) -> Any:
        """Execute the actual content generation.
        
        Args:
            **kwargs: Generation parameters.
            
        Returns:
            Generated content.
        """
        pass
    
    def generate_batch(
        self,
        params_list: List[Dict[str, Any]],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """Generate content in batches using multiple threads.
        
        Args:
            params_list: List of parameter dictionaries for generation.
            batch_size: Optional batch size for processing.
            
        Returns:
            List of generated content.
        """
        batch_size = batch_size or settings.BATCH_SIZE
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for i in range(0, len(params_list), batch_size):
                batch = params_list[i:i + batch_size]
                for params in batch:
                    future = executor.submit(self.generate, **params)
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Batch generation error: {str(e)}")
                        results.append(None)
                
                futures = []
        
        return results
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate generation parameters.
        
        Args:
            parameters: Parameters to validate.
            
        Returns:
            True if parameters are valid, False otherwise.
        """
        return True  # Override in subclasses for specific validation
    
    def __repr__(self) -> str:
        """Get string representation of the generator.
        
        Returns:
            Generator string representation.
        """
        return f"{self.__class__.__name__}(provider={self.provider})" 