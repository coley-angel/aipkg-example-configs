# Python Development Context

## Project Overview

This agent provides comprehensive Python development support with focus on modern Python 3.11+ features, type safety, testing, and best practices.

## Common Patterns

### Application Entry Point

```python
# src/myapp/cli.py
import click
import logging
from pathlib import Path
from typing import Optional

from .core import process_data
from .config import load_config

logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', type=click.Path(exists=True), help='Configuration file')
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config: Optional[str]) -> None:
    """My Application CLI."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    ctx.ensure_object(dict)
    if config:
        ctx.obj['config'] = load_config(config)

@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.pass_context
def process(ctx: click.Context, input_file: str, output: Optional[str]) -> None:
    """Process input file."""
    try:
        result = process_data(input_file, ctx.obj.get('config'))
        if output:
            Path(output).write_text(result)
            click.echo(f"Results written to {output}")
        else:
            click.echo(result)
    except Exception as e:
        logger.exception("Processing failed")
        raise click.ClickException(str(e))

if __name__ == '__main__':
    cli()
```

### Configuration Management

```python
# src/myapp/config.py
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings

class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str
    username: str
    password: str
    
    @property
    def connection_string(self) -> str:
        """Generate connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class AppConfig(BaseModel):
    """Application configuration."""
    app_name: str = Field(..., min_length=1)
    debug: bool = False
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    database: DatabaseConfig
    api_key: str = Field(..., min_length=32)
    
    @validator('api_key')
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if not v.startswith('sk-'):
            raise ValueError("API key must start with 'sk-'")
        return v

class Settings(BaseSettings):
    """Application settings from environment."""
    database_url: str
    api_key: str
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def load_config(config_path: str) -> AppConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
```

### Database Operations

```python
# src/myapp/database.py
from contextlib import contextmanager
from typing import Generator, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel

class User(BaseModel):
    """User model."""
    id: Optional[int] = None
    username: str
    email: str
    created_at: Optional[str] = None

class Database:
    """Database connection manager."""
    
    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
    
    @contextmanager
    def get_connection(self) -> Generator:
        """Get database connection context manager."""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_user(self, user: User) -> User:
        """Create new user."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, email)
                    VALUES (%(username)s, %(email)s)
                    RETURNING id, username, email, created_at
                    """,
                    user.dict(exclude={'id', 'created_at'})
                )
                result = cur.fetchone()
                return User(**result)
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM users WHERE id = %s",
                    (user_id,)
                )
                result = cur.fetchone()
                return User(**result) if result else None
    
    def list_users(self, limit: int = 100) -> List[User]:
        """List all users."""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM users ORDER BY created_at DESC LIMIT %s",
                    (limit,)
                )
                return [User(**row) for row in cur.fetchall()]
```

### API Client

```python
# src/myapp/client.py
from typing import Dict, Any, Optional
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class APIClient:
    """HTTP API client with retry logic."""
    
    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
        )
    
    def __enter__(self) -> 'APIClient':
        return self
    
    def __exit__(self, *args) -> None:
        self.client.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """GET request with retry logic."""
        response = self.client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """POST request with retry logic."""
        response = self.client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

# Usage
with APIClient("https://api.example.com", api_key="sk-xxx") as client:
    result = client.get("/users/123")
    print(result)
```

### Async Operations

```python
# src/myapp/async_processor.py
import asyncio
from typing import List, Dict, Any
import aiohttp
import aiofiles
from pathlib import Path

class AsyncProcessor:
    """Process data asynchronously."""
    
    def __init__(self, max_concurrent: int = 10) -> None:
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_data(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Fetch data from URL with rate limiting."""
        async with self.semaphore:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
    
    async def process_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Process multiple URLs concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_data(session, url) for url in urls]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def read_file_async(self, filepath: Path) -> str:
        """Read file asynchronously."""
        async with aiofiles.open(filepath, 'r') as f:
            return await f.read()
    
    async def write_file_async(self, filepath: Path, content: str) -> None:
        """Write file asynchronously."""
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(content)

# Usage
async def main():
    processor = AsyncProcessor(max_concurrent=5)
    urls = [f"https://api.example.com/item/{i}" for i in range(100)]
    results = await processor.process_urls(urls)
    print(f"Processed {len(results)} URLs")

if __name__ == '__main__':
    asyncio.run(main())
```

### Data Processing Pipeline

```python
# src/myapp/pipeline.py
from typing import Callable, List, Any, TypeVar
from functools import reduce
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class Pipeline:
    """Data processing pipeline."""
    
    def __init__(self) -> None:
        self.steps: List[Callable[[Any], Any]] = []
    
    def add_step(self, func: Callable[[T], T], name: str = "") -> 'Pipeline':
        """Add processing step to pipeline."""
        step_name = name or func.__name__
        
        def wrapped(data: T) -> T:
            logger.debug(f"Executing step: {step_name}")
            try:
                result = func(data)
                logger.debug(f"Step {step_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                raise
        
        self.steps.append(wrapped)
        return self
    
    def execute(self, data: T) -> T:
        """Execute all pipeline steps."""
        logger.info(f"Starting pipeline with {len(self.steps)} steps")
        result = reduce(lambda d, step: step(d), self.steps, data)
        logger.info("Pipeline completed successfully")
        return result

# Usage
def clean_data(data: List[str]) -> List[str]:
    """Remove empty strings."""
    return [item.strip() for item in data if item.strip()]

def uppercase(data: List[str]) -> List[str]:
    """Convert to uppercase."""
    return [item.upper() for item in data]

def deduplicate(data: List[str]) -> List[str]:
    """Remove duplicates."""
    return list(set(data))

pipeline = Pipeline()
pipeline.add_step(clean_data, "clean")
pipeline.add_step(uppercase, "uppercase")
pipeline.add_step(deduplicate, "dedupe")

result = pipeline.execute(["  hello  ", "world", "hello", ""])
print(result)  # ['HELLO', 'WORLD']
```

## Testing Patterns

### Fixtures and Mocking

```python
# tests/conftest.py
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from myapp.database import Database
from myapp.config import AppConfig

@pytest.fixture
def mock_database():
    """Mock database for testing."""
    db = Mock(spec=Database)
    db.get_user.return_value = Mock(id=1, username="testuser", email="test@example.com")
    return db

@pytest.fixture
def sample_config(tmp_path: Path) -> AppConfig:
    """Create sample configuration."""
    return AppConfig(
        app_name="test-app",
        debug=True,
        log_level="DEBUG",
        database={
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "test",
            "password": "test123"
        },
        api_key="sk-test-key-12345678901234567890"
    )

@pytest.fixture
def temp_data_file(tmp_path: Path) -> Path:
    """Create temporary data file."""
    file_path = tmp_path / "data.json"
    file_path.write_text('{"test": true}')
    return file_path
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from click.testing import CliRunner
from myapp.cli import cli

class TestCLIIntegration:
    """Integration tests for CLI."""
    
    @pytest.fixture
    def runner(self):
        return CliRunner()
    
    def test_process_command(self, runner, temp_data_file):
        """Test process command end-to-end."""
        result = runner.invoke(cli, ['process', str(temp_data_file)])
        assert result.exit_code == 0
        assert 'Success' in result.output
    
    def test_process_with_output(self, runner, temp_data_file, tmp_path):
        """Test process command with output file."""
        output_file = tmp_path / "output.txt"
        result = runner.invoke(cli, [
            'process',
            str(temp_data_file),
            '--output', str(output_file)
        ])
        assert result.exit_code == 0
        assert output_file.exists()
```

## Performance Optimization

### Caching

```python
from functools import lru_cache, cache
from typing import List
import time

@lru_cache(maxsize=128)
def expensive_computation(n: int) -> int:
    """Cache results of expensive computation."""
    time.sleep(1)  # Simulate expensive operation
    return n * n

@cache  # Python 3.9+ - unlimited cache
def fibonacci(n: int) -> int:
    """Cached fibonacci calculation."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Custom cache with TTL
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any

class TTLCache:
    """Cache with time-to-live."""
    
    def __init__(self, ttl_seconds: int = 300) -> None:
        self.ttl = timedelta(seconds=ttl_seconds)
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
    
    def get(self, key: str) -> Any:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        self.cache[key] = (value, datetime.now())
```

### Batch Processing

```python
from typing import List, Iterator, TypeVar
from itertools import islice

T = TypeVar('T')

def batch_iterator(items: List[T], batch_size: int) -> Iterator[List[T]]:
    """Yield batches of items."""
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch

# Usage
items = list(range(1000))
for batch in batch_iterator(items, batch_size=100):
    process_batch(batch)
```

## Debugging Tips

### Logging Best Practices

```python
import logging
import sys

def setup_logging(level: str = "INFO") -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

# Use structured logging
logger = logging.getLogger(__name__)
logger.info("Processing started", extra={
    "user_id": 123,
    "action": "process_data",
    "items_count": 50
})
```

### Debugging Decorators

```python
import functools
import inspect

def debug(func):
    """Print function signature and return value."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper

@debug
def add(a: int, b: int) -> int:
    return a + b
```

## Common Utilities

### File Operations

```python
from pathlib import Path
from typing import List
import shutil

def ensure_directory(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def list_files(directory: Path, pattern: str = "*") -> List[Path]:
    """List files matching pattern."""
    return list(directory.glob(pattern))

def copy_tree(src: Path, dst: Path) -> None:
    """Copy directory tree."""
    shutil.copytree(src, dst, dirs_exist_ok=True)

def safe_delete(path: Path) -> bool:
    """Safely delete file or directory."""
    try:
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception as e:
        logger.error(f"Failed to delete {path}: {e}")
        return False
```
