from __future__ import annotations


class AnchorError(ValueError):
    """Raised when textual anchors cannot be resolved inside a prompt."""


def locate_match_bounds(prompt: str, match: str, occurrence: int, flag: str) -> tuple[int, int]:
    """Locate the raw character span for the Nth occurrence of a substring inside the prompt."""
    if not match:
        raise AnchorError(f"{flag} must be a non-empty string.")
    if occurrence <= 0:
        raise AnchorError(f"{flag}-occurrence must be >= 1.")

    search_index = 0
    match_index = -1
    for _ in range(occurrence):
        match_index = prompt.find(match, search_index)
        if match_index == -1:
            raise AnchorError(
                f"{flag} '{match}' not found inside prompt text (occurrence {occurrence})."
            )
        increment = max(len(match), 1)
        search_index = match_index + increment

    return match_index, match_index + len(match)


def locate_start_anchor(prompt: str, match: str, occurrence: int) -> int:
    """Snap --start-match anchors to the newline before the match (or the match start)."""
    match_index, _ = locate_match_bounds(prompt, match, occurrence, "--start-match")
    newline_index = prompt.rfind("\n", 0, match_index)
    return newline_index if newline_index != -1 else match_index


def locate_end_anchor(prompt: str, match: str, occurrence: int) -> int:
    """Snap --end-match anchors to the newline after the match (or EOF when missing)."""
    _, match_end = locate_match_bounds(prompt, match, occurrence, "--end-match")
    newline_index = prompt.find("\n", match_end)
    if newline_index == -1:
        if not prompt:
            raise AnchorError("--end-match requires a non-empty prompt.")
        return len(prompt)
    return newline_index
