"""Unit tests for the base generator class."""

import pytest
from typing import Any, Dict, Type
from unittest.mock import Mock, patch

from src.core.base_generator import BaseGenerator
from src.core.base_provider import BaseProvider
from src.utils.error_handler import ApplicationError

class TestGenerator(BaseGenerator):
    """Test implementation of BaseGenerator."""
    
    @property
    def provider_class(self) -> Type[BaseProvider]:
        return Mock(spec=BaseProvider)
    
    @property
    def default_provider_name(self) -> str:
        return "test_provider"
    
    def _execute_generation(self, **kwargs) -> Any:
        if not kwargs.get("succeed", True):
            raise ApplicationError("Test error")
        return {"status": "success", "data": kwargs}

def test_generator_initialization(mock_provider):
    """Test generator initialization."""
    # Test with provider
    generator = TestGenerator(provider=mock_provider)
    assert generator.provider == mock_provider
    
    # Test without provider
    generator = TestGenerator()
    assert isinstance(generator.provider, Mock)

def test_generator_generation_success(mock_provider):
    """Test successful content generation."""
    generator = TestGenerator(provider=mock_provider)
    result = generator.generate(test_param="test_value")
    assert result["status"] == "success"
    assert result["data"]["test_param"] == "test_value"

def test_generator_generation_failure(mock_provider):
    """Test failed content generation."""
    generator = TestGenerator(provider=mock_provider)
    with pytest.raises(RuntimeError) as exc_info:
        generator.generate(succeed=False)
    assert "Operation failed after 3 attempts" in str(exc_info.value)

def test_generator_batch_generation(mock_provider):
    """Test batch content generation."""
    generator = TestGenerator(provider=mock_provider)
    params_list = [
        {"param": f"value{i}"} for i in range(5)
    ]
    
    results = generator.generate_batch(params_list)
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result["status"] == "success"
        assert result["data"]["param"] == f"value{i}"

def test_generator_batch_generation_partial_failure(mock_provider):
    """Test batch generation with some failures."""
    generator = TestGenerator(provider=mock_provider)
    params_list = [
        {"param": f"value{i}", "succeed": i % 2 == 0}
        for i in range(5)
    ]
    
    results = generator.generate_batch(params_list)
    assert len(results) == 5
    assert sum(1 for r in results if r is not None) == 3  # Successful generations
    assert sum(1 for r in results if r is None) == 2  # Failed generations

def test_generator_parameter_validation(mock_provider):
    """Test parameter validation."""
    generator = TestGenerator(provider=mock_provider)
    assert generator.validate_parameters({"test": "value"}) is True

def test_generator_string_representation(mock_provider):
    """Test generator string representation."""
    generator = TestGenerator(provider=mock_provider)
    assert str(generator) == f"TestGenerator(provider={mock_provider})"

def test_generator_max_workers_setting(mock_provider):
    """Test max workers setting."""
    # Test with custom max_workers
    generator = TestGenerator(provider=mock_provider, max_workers=5)
    assert generator.max_workers == 5
    
    # Test with default max_workers from settings
    generator = TestGenerator(provider=mock_provider)
    assert generator.max_workers == 2  # From mock_settings fixture 