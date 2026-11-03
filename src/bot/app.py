"""Точка входа Telegram-бота BBNN. Только long-polling."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import yaml
from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from src.bot.handlers import BotRuntime, register_handlers
from src.bot.state import BotState

log = logging.getLogger(__name__)
CONFIG_PATH = Path("configs/bot.yaml")


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Не найден конфиг {path}. Ожидается в корне проекта.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    token = os.getenv("TG_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TG_BOT_TOKEN не задан. Получи токен у @BotFather и положи в .env:\n"
            "  TG_BOT_TOKEN=123456:ABC-DEF..."
        )

    cfg = _load_config(CONFIG_PATH)
    tmp_dir = Path(cfg.get("bot", {}).get("tmp_dir", "runs/bot_tmp"))
    runtime = BotRuntime(cfg, tmp_dir)
    state = BotState()

    bot = Bot(token=token)
    dp = Dispatcher()
    register_handlers(dp, runtime, state, cfg)

    log.info("Бот запущен (polling). tmp_dir=%s", tmp_dir)
    try:
        asyncio.run(dp.start_polling(bot))
    finally:
        runtime.cleanup_all()


if __name__ == "__main__":
    main()