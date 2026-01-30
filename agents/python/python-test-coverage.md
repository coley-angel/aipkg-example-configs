---
description: Run tests with coverage report
---

# Python Test Coverage

Run test suite with coverage analysis and reporting.

## Steps

### 1. Run Tests
```bash
pytest -v --cov=src --cov-report=term-missing
```
Execute test suite with coverage tracking.

### 2. Generate HTML Report
```bash
pytest --cov=src --cov-report=html
```
Generate detailed HTML coverage report.

### 3. Open Report
```bash
open htmlcov/index.html
```
Open coverage report in browser for review.
