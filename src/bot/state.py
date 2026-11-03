"""Простое хранилище состояния между командами. Без БД, сбрасывается при рестарте."""
from __future__ import annotations


class BotState:
    def __init__(self) -> None:
        self._last_image: dict[int, str] = {}
        self._last_panel: dict[int, str] = {}
        self._last_caption: dict[int, str] = {}
        self._pending: dict[int, str] = {}

    def set_last_image(self, chat_id: int, path: str) -> None:
        self._last_image[chat_id] = path

    def get_last_image(self, chat_id: int) -> str | None:
        return self._last_image.get(chat_id)

    def set_last_panel(self, chat_id: int, path: str, caption: str = "") -> None:
        self._last_panel[chat_id] = path
        self._last_caption[chat_id] = caption

    def get_last_panel(self, chat_id: int) -> tuple[str | None, str]:
        return self._last_panel.get(chat_id), self._last_caption.get(chat_id, "")

    def set_pending(self, chat_id: int, mode: str) -> None:
        self._pending[chat_id] = mode

    def pop_pending(self, chat_id: int) -> str | None:
        return self._pending.pop(chat_id, None)

    def clear(self, chat_id: int) -> None:
        self._last_image.pop(chat_id, None)
        self._last_panel.pop(chat_id, None)
        self._last_caption.pop(chat_id, None)
        self._pending.pop(chat_id, None)