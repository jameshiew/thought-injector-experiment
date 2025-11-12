lint:
    tombi lint
    uv run ruff check --diff .
    uv run basedpyright

fmt:
    tombi fmt
    uv run ruff format .

dep-check:
    uv run pip-audit

test:
    uv run pytest

check:
    just lint
    just test
