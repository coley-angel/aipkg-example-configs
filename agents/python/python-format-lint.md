---
description: Format and lint Python code with Black, isort, Ruff, and mypy
---

# Python Format and Lint

Complete code formatting and linting workflow.

## Steps

### 1. Format with Black
```bash
black .
```
Format code with Black formatter.

### 2. Sort Imports
```bash
isort .
```
Sort and organize imports with isort.

### 3. Lint with Ruff
```bash
ruff check . --fix
```
Lint and auto-fix issues with Ruff.

### 4. Type Check
```bash
mypy src/
```
Run type checking with mypy.
