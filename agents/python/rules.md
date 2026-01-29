# Python Development Rules

## Code Style & Structure

### PEP 8 Compliance
- Use 4 spaces for indentation (never tabs)
- Maximum line length: 88 characters (Black formatter standard)
- Use snake_case for functions and variables
- Use PascalCase for class names
- Use UPPER_CASE for constants

### Type Hints (Required)
```python
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

def process_data(
    items: List[str],
    config: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Dict[str, int]:
    """Process items according to configuration.
    
    Args:
        items: List of items to process
        config: Configuration dictionary
        output_path: Optional path for output file
        
    Returns:
        Dictionary mapping item names to counts
        
    Raises:
        ValueError: If items list is empty
    """
    if not items:
        raise ValueError("Items list cannot be empty")
    
    results: Dict[str, int] = {}
    for item in items:
        results[item] = len(item)
    
    return results
```

### Docstrings (Google Style)
```python
class DataProcessor:
    """Process and transform data from various sources.
    
    This class provides methods for loading, transforming, and saving
    data in multiple formats.
    
    Attributes:
        source_path: Path to the data source
        cache_enabled: Whether to cache processed results
        
    Example:
        >>> processor = DataProcessor("data.csv")
        >>> results = processor.process()
        >>> processor.save(results, "output.json")
    """
    
    def __init__(self, source_path: str, cache_enabled: bool = True) -> None:
        """Initialize the data processor.
        
        Args:
            source_path: Path to the data source file
            cache_enabled: Enable result caching (default: True)
        """
        self.source_path = Path(source_path)
        self.cache_enabled = cache_enabled
```

## Project Structure

### Standard Layout
```
project/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   └── models.py
│       ├── utils/
│       │   ├── __init__.py
│       │   └── helpers.py
│       └── cli.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core/
│   │   └── test_models.py
│   └── test_utils/
│       └── test_helpers.py
├── docs/
├── pyproject.toml
├── README.md
└── .gitignore
```

### pyproject.toml Configuration
```toml
[project]
name = "mypackage"
version = "0.1.0"
description = "A brief description"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[tool.black]
line-length = 88
target-version = ['py311']

[tool.ruff]
line-length = 88
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --cov=src --cov-report=term-missing"
```

## Best Practices

### Error Handling
```python
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DataError(Exception):
    """Custom exception for data-related errors."""
    pass

def load_data(filepath: str) -> Optional[dict]:
    """Load data from file with proper error handling."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise DataError(f"Data file not found: {filepath}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        raise DataError(f"Invalid JSON format: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error loading {filepath}")
        raise
```

### Context Managers
```python
from contextlib import contextmanager
from typing import Generator

@contextmanager
def database_connection(db_url: str) -> Generator[Connection, None, None]:
    """Context manager for database connections.
    
    Args:
        db_url: Database connection URL
        
    Yields:
        Active database connection
    """
    conn = create_connection(db_url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Usage
with database_connection("postgresql://localhost/db") as conn:
    conn.execute("SELECT * FROM users")
```

### Dataclasses & Pydantic
```python
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
from typing import List

# Simple dataclass
@dataclass
class User:
    """User data container."""
    username: str
    email: str
    age: int
    roles: List[str] = field(default_factory=list)
    
    def __post_init__(self) -> None:
        """Validate data after initialization."""
        if self.age < 0:
            raise ValueError("Age cannot be negative")

# Pydantic model with validation
class UserModel(BaseModel):
    """User model with automatic validation."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)
    roles: List[str] = Field(default_factory=list)
    
    @validator('username')
    def username_alphanumeric(cls, v: str) -> str:
        """Ensure username is alphanumeric."""
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
```

### Async/Await Patterns
```python
import asyncio
from typing import List
import aiohttp

async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch content from URL asynchronously."""
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple(urls: List[str]) -> List[str]:
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Usage
urls = ["https://api.example.com/1", "https://api.example.com/2"]
results = asyncio.run(fetch_multiple(urls))
```

## Testing

### Pytest Structure
```python
import pytest
from pathlib import Path
from mypackage.core import DataProcessor

@pytest.fixture
def sample_data() -> dict:
    """Provide sample data for tests."""
    return {"key": "value", "count": 42}

@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """Create temporary file for testing."""
    file_path = tmp_path / "test.json"
    file_path.write_text('{"test": true}')
    return file_path

class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_initialization(self, temp_file: Path) -> None:
        """Test processor initialization."""
        processor = DataProcessor(str(temp_file))
        assert processor.source_path == temp_file
    
    def test_process_valid_data(self, sample_data: dict) -> None:
        """Test processing with valid data."""
        result = process_data(sample_data)
        assert result is not None
        assert "key" in result
    
    def test_process_invalid_data(self) -> None:
        """Test error handling with invalid data."""
        with pytest.raises(ValueError, match="cannot be empty"):
            process_data([])
    
    @pytest.mark.parametrize("input,expected", [
        ("test", 4),
        ("hello", 5),
        ("", 0),
    ])
    def test_string_length(self, input: str, expected: int) -> None:
        """Test string length calculation."""
        assert len(input) == expected
```

### Test Coverage Requirements
- Minimum 80% code coverage
- 100% coverage for critical paths
- Test edge cases and error conditions
- Use mocking for external dependencies

## Performance

### Profiling
```python
import cProfile
import pstats
from functools import wraps
import time

def profile(func):
    """Decorator to profile function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        return result
    return wrapper

def timeit(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper
```

### Optimization Tips
- Use list comprehensions over loops when appropriate
- Leverage built-in functions (map, filter, reduce)
- Use generators for large datasets
- Cache expensive computations with `@lru_cache`
- Use `__slots__` for memory-intensive classes

## Security

### Input Validation
```python
from pathlib import Path
import re

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove path separators and dangerous characters
    safe_name = re.sub(r'[^\w\s.-]', '', filename)
    # Remove leading dots
    safe_name = safe_name.lstrip('.')
    return safe_name

def validate_path(path: str, base_dir: Path) -> Path:
    """Validate path is within base directory."""
    resolved = (base_dir / path).resolve()
    if not resolved.is_relative_to(base_dir):
        raise ValueError("Path traversal detected")
    return resolved
```

### Secrets Management
```python
import os
from functools import lru_cache

@lru_cache(maxsize=None)
def get_secret(key: str) -> str:
    """Retrieve secret from environment variables.
    
    Raises:
        ValueError: If secret is not found
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Secret {key} not found in environment")
    return value

# Never hardcode secrets
API_KEY = get_secret("API_KEY")
```

## Logging

### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger(__name__)
```

## Code Quality Tools

### Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  
  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Required Checks
- Black formatting
- Ruff linting
- MyPy type checking
- Pytest with coverage
- Security scan with bandit
