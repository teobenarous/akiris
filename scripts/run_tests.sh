#!/bin/bash
set -e

project_root=$(pwd)
export PYTHONPATH="$PYTHONPATH:$project_root/app"
export LOG_LEVEL="CRITICAL"

echo "Running Linting (Ruff/Flake8)..."
ruff check . --fix

echo "Running Unit & Integration Tests..."
pytest tests/ \
    --verbose \
    --color=yes \
    --durations=5 \
    --cov=app \
    --cov-report=term-missing

echo "All checks passed"