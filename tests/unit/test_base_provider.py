"""Unit tests for the base provider class."""

import pytest
from unittest.mock import Mock

from src.core.base_provider import BaseProvider
from src.utils.error_handler import ApplicationError

class TestProvider(BaseProvider):
    """Test implementation of BaseProvider."""
    
    @property
    def provider_name(self) -> str:
        return "test_provider"
    
    def validate_credentials(self) -> bool:
        return bool(self.api_key)
    
    def get_default_parameters(self) -> dict:
        return {"param1": "value1", "param2": "value2"}
    
    def _execute_request(self, **kwargs):
        if not kwargs.get("succeed", True):
            raise ApplicationError("Test error")
        return {"status": "success", "data": kwargs}

def test_provider_initialization():
    """Test provider initialization."""
    # Test with API key
    provider = TestProvider(api_key="test_key")
    assert provider.api_key == "test_key"
    assert provider.provider_name == "test_provider"
    
    # Test without API key
    provider = TestProvider()
    assert provider.api_key is None

def test_provider_validation():
    """Test provider credential validation."""
    # Valid credentials
    provider = TestProvider(api_key="test_key")
    assert provider.validate_credentials() is True
    
    # Invalid credentials
    provider = TestProvider()
    assert provider.validate_credentials() is False

def test_provider_default_parameters():
    """Test provider default parameters."""
    provider = TestProvider()
    params = provider.get_default_parameters()
    assert params == {"param1": "value1", "param2": "value2"}

def test_provider_request_success():
    """Test successful provider request."""
    provider = TestProvider(api_key="test_key")
    result = provider.make_request(test_param="test_value")
    assert result["status"] == "success"
    assert result["data"]["test_param"] == "test_value"

def test_provider_request_failure():
    """Test failed provider request."""
    provider = TestProvider(api_key="test_key")
    with pytest.raises(RuntimeError) as exc_info:
        provider.make_request(succeed=False)
    assert "Operation failed after 3 attempts" in str(exc_info.value)

def test_provider_request_no_api_key():
    """Test provider request without API key."""
    provider = TestProvider()
    with pytest.raises(ValueError) as exc_info:
        provider.make_request()
    assert "No API key available" in str(exc_info.value)

def test_provider_string_representation():
    """Test provider string representation."""
    provider = TestProvider(api_key="test_key")
    assert str(provider) == "TestProvider(provider=test_provider)" 