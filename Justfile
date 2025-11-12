check:
    uv run ruff check .
    uv run basedpyright

fmt:
    uv run ruff format .

audit:
    uv run pip-audit

test:
    uv run pytest
