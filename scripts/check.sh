#!/usr/bin/env bash
set -euo pipefail
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pyright --warnings
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q -m 'not cuda'
