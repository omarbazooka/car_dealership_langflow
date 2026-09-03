from __future__ import annotations

import contextvars

_CURRENT_USER_TEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "CAR_DEALERSHIP_CURRENT_USER_TEXT", default=""
)


def set_current_user_text(text: str) -> contextvars.Token:
    """Bind the current customer utterance to the active request context."""
    return _CURRENT_USER_TEXT.set(str(text or ""))


def get_current_user_text() -> str:
    """Return the customer utterance for the current request context."""
    return _CURRENT_USER_TEXT.get()


def reset_current_user_text(token: contextvars.Token) -> None:
    """Restore the previous request context when an explicit reset is needed."""
    _CURRENT_USER_TEXT.reset(token)
