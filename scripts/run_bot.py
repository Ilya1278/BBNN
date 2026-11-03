"""Entry point Telegram-бота BBNN.

Bootstrap sys.path, чтобы `python scripts/run_bot.py` из корня проекта
видел пакет `src`. Без этого Python кладёт в sys.path[0] папку scripts/,
а не CWD.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.bot.app import main  # noqa: E402

if __name__ == "__main__":
    main()