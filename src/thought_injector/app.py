from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import typer
from rich.console import Console

console = Console()
app = typer.Typer(help="Local concept-injection experiments for safetensors-based LMs.")

FuncT = TypeVar("FuncT", bound=Callable[..., Any])


def typed_command(*args: Any, **kwargs: Any) -> Callable[[FuncT], FuncT]:
    """Type-friendly wrapper around Typer's command decorator."""

    decorator = app.command(*args, **kwargs)

    def _wrapper(func: FuncT) -> FuncT:
        return decorator(func)

    return _wrapper
