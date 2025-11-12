check:
    uv run ruff check .
    uv run mypy .

fmt:
    uv run ruff format .

audit:
    uv run pip-audit

test:
    uv run pytest
