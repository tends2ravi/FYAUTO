"""Base provider class for service providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

from src.utils.error_handler import handle_errors
from src.config.settings import settings

class BaseProvider(ABC):
    """Base class for all service providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the provider.
        
        Args:
            api_key: Optional API key for the service.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.api_key = api_key or self._get_default_api_key()
        
        if not self.api_key:
            self.logger.warning(
                f"No API key provided for {self.__class__.__name__}"
            )
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of the provider.
        
        Returns:
            Provider name.
        """
        pass
    
    @abstractmethod
    def validate_credentials(self) -> bool:
        """Validate the provider credentials.
        
        Returns:
            True if credentials are valid, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for the provider.
        
        Returns:
            Dictionary of default parameters.
        """
        pass
    
    def _get_default_api_key(self) -> Optional[str]:
        """Get the default API key for the provider.
        
        Returns:
            API key if available, None otherwise.
        """
        return settings.get_api_key(self.provider_name)
    
    @handle_errors(max_retries=3)
    def make_request(self, **kwargs) -> Any:
        """Make a request to the provider's API.
        
        Args:
            **kwargs: Request parameters.
            
        Returns:
            Response from the API.
            
        Raises:
            RuntimeError: If the request fails after max retries.
        """
        if not self.api_key:
            raise ValueError(
                f"No API key available for {self.provider_name}"
            )
            
        if not self.validate_credentials():
            raise ValueError(
                f"Invalid credentials for {self.provider_name}"
            )
            
        return self._execute_request(**kwargs)
    
    @abstractmethod
    def _execute_request(self, **kwargs) -> Any:
        """Execute the actual API request.
        
        Args:
            **kwargs: Request parameters.
            
        Returns:
            Response from the API.
        """
        pass
    
    def __repr__(self) -> str:
        """Get string representation of the provider.
        
        Returns:
            Provider string representation.
        """
        return f"{self.__class__.__name__}(provider={self.provider_name})" 