from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from aiogram import Bot, Dispatcher, F  # noqa: E402
from aiogram.filters import Command  # noqa: E402
from aiogram.types import (  # noqa: E402
    CallbackQuery,
    FSInputFile,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder  # noqa: E402

from src.bot.panels import (  # noqa: E402
    build_xai_panel,
    denorm_to_uint8,
    forward_probs,
)
from src.bot.state import BotState  # noqa: E402

log = logging.getLogger(__name__)

HELP_TEXT = (
    "🧠 <b>BBNN Bot</b> — исследование устойчивости к adversarial-атакам.\n\n"
    "Команды:\n"
    "/predict — классификация + Grad-CAM\n"
    "/attack  — FGSM (clean / adv / perturbation)\n"
    "/last    — повторить последнюю панель\n"
    "/metrics — кривые обучения\n"
    "/report  — summary.md\n"
    "/help    — это сообщение"
)
# --------------------------------------------------------------------------- #
# Runtime: ленивая загрузка модели                                          #
# --------------------------------------------------------------------------- #
class BotRuntime:
    def __init__(self, cfg: dict[str, Any], tmp_dir: Path) -> None:
        self.cfg = cfg
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._clf = None

    @property
    def clf(self):
        if self._clf is None:
            # INTERFACES §2: MotorcycleClassifier.__init__(checkpoint_path, cfg, device)
            from src.models.classifier import MotorcycleClassifier
            model_cfg_path = Path(self.cfg["model_config"])
            train_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Загружаю MotorcycleClassifier: ckpt=%s device=%s",
                     self.cfg["checkpoint"], device)
            self._clf = MotorcycleClassifier(
                checkpoint_path=self.cfg["checkpoint"],
                cfg=train_cfg,
                device=device,
            )
        return self._clf

    def tmp_path(self, suffix: str) -> Path:
        return self.tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    def cleanup(self, *paths: Path) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                log.warning("Не удалось удалить %s", p)

    def cleanup_all(self) -> None:
        if not self.tmp_dir.exists():
            return
        for p in self.tmp_dir.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
# --------------------------------------------------------------------------- #
# Утилиты                                                                   #
# --------------------------------------------------------------------------- #

def _photo_to_array(path: Path) -> np.ndarray:
    """Загрузить файл → RGB uint8 HWC. Ровно то, что ест clf.predict()."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


async def _download_photo(
    bot: Bot, message: Message, runtime: BotRuntime, max_mb: int
) -> Path | None:
    if not message.photo:
        return None
    photo = message.photo[-1]  # самый крупный размер
    if photo.file_size and photo.file_size > max_mb * 1024 * 1024:
        await message.answer(f"❌ Файл больше {max_mb} МБ.")
        return None
    dst = runtime.tmp_path(".jpg")
    await bot.download(photo, destination=dst)
    return dst


def _build_attack_panel(
    orig: np.ndarray,
    adv: np.ndarray,
    x_clean: torch.Tensor,
    x_adv: torch.Tensor,
    out_path: Path,
    clean_label: str,
    clean_conf: float,
    adv_label: str,
    adv_conf: float,
) -> None:

    diff = (x_adv - x_clean).detach().cpu().numpy().transpose(1, 2, 0)
    max_abs = float(np.abs(diff).max()) + 1e-8
    diff_disp = np.clip(diff / max_abs * 0.5 + 0.5, 0.0, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    axes[0].imshow(orig)
    axes[0].set_title(f"clean: {clean_label} ({clean_conf:.3f})")
    axes[1].imshow(adv)
    axes[1].set_title(f"adv: {adv_label} ({adv_conf:.3f})")
    axes[2].imshow(diff_disp)
    axes[2].set_title(f"perturbation (norm-space, ±{max_abs:.3f})")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Регистрация                                                               #
# --------------------------------------------------------------------------- #

def register_handlers(
    dp: Dispatcher, runtime: BotRuntime, state: BotState, cfg: dict
) -> None:
    max_mb = int(cfg.get("bot", {}).get("max_photo_size_mb", 10))
    eps = float(cfg.get("attack", {}).get("eps", 0.03))
    eps_space = str(cfg.get("attack", {}).get("eps_space", "normalized"))
    reports_dir = Path(cfg.get("reports_dir", "reports"))
    runs_dir = Path(cfg.get("runs_dir", "runs"))
    latest_run = cfg["latest_run"]

    # ---------------- /start, /help ----------------
    @dp.message(Command("start"))
    @dp.message(Command("help"))
    async def cmd_start(message: Message) -> None:
        await message.answer(HELP_TEXT, parse_mode="HTML")

    # ---------------- /predict ----------------
    @dp.message(Command("predict"))
    async def cmd_predict(message: Message, bot: Bot) -> None:
        if message.photo:
            await _do_predict(message, bot, runtime, state, max_mb)
            return
        state.set_pending(message.chat.id, "predict")
        await message.answer("📷 Пришли фото для классификации.")

    # ---------------- /attack ----------------
    @dp.message(Command("attack"))
    async def cmd_attack(message: Message, bot: Bot) -> None:
        if message.photo:
            await _do_attack(message, bot, runtime, state, max_mb, eps, eps_space)
            return
        last = state.get_last_image(message.chat.id)
        if last and Path(last).exists():
            await _do_attack_from_path(
                message, bot, runtime, state, Path(last), eps, eps_space
            )
            return
        state.set_pending(message.chat.id, "attack")
        await message.answer(
            "📷 Пришли фото для FGSM-атаки (или сначала сделай /predict)."
        )

    # ---------------- голое фото по pending-режиму ----------------
    @dp.message(F.photo)
    async def on_photo(message: Message, bot: Bot) -> None:
        mode = state.pop_pending(message.chat.id) or "predict"
        if mode == "attack":
            await _do_attack(message, bot, runtime, state, max_mb, eps, eps_space)
        else:
            await _do_predict(message, bot, runtime, state, max_mb)

    # ---------------- /last ----------------
    @dp.message(Command("last"))
    async def cmd_last(message: Message, bot: Bot) -> None:
        path, caption = state.get_last_panel(message.chat.id)
        if not path or not Path(path).exists():
            await message.answer("Нет сохранённой панели. Сначала /predict или /attack.")
            return
        await bot.send_photo(message.chat.id, FSInputFile(path), caption=caption)

    # ---------------- /metrics ----------------
    @dp.message(Command("metrics"))
    async def cmd_metrics(message: Message, bot: Bot) -> None:
        fig = reports_dir / latest_run / "figures" / "loss_curves.png"
        if not fig.exists():
            fig = runs_dir / latest_run / "loss_curves.png"
        if not fig.exists():
            await message.answer(f"❌ Не найден loss_curves.png для {latest_run}.")
            return
        await bot.send_photo(
            message.chat.id, FSInputFile(fig),
            caption=f"📈 Кривые обучения — {latest_run}",
        )

    # ---------------- /report ----------------
    @dp.message(Command("report"))
    async def cmd_report(message: Message, bot: Bot) -> None:
        md = reports_dir / latest_run / "summary.md"
        if not md.exists():
            await message.answer(f"❌ Не найден summary.md для {latest_run}.")
            return
        text = md.read_text(encoding="utf-8")
        if len(text) <= 4000:
            await message.answer(text)
        else:
            await message.answer(text[:4000] + "\n…(полностью — файлом)")
            await bot.send_document(message.chat.id, FSInputFile(md))

    # ---------------- inline: 🔍 XAI (adversarial) ----------------
    @dp.callback_query(F.data == "adv_xai")
    async def cb_adv_xai(query: CallbackQuery, bot: Bot) -> None:
        chat_id = query.message.chat.id
        await query.answer("⏳ Считаю XAI для adversarial…")

        last = state.get_last_image(chat_id)
        if not last or not Path(last).exists():
            await bot.send_message(
                chat_id, "❌ Нет картинки — сначала /predict или /attack с фото."
            )
            return

        try:
            await _do_adv_panel(
                bot, chat_id, runtime, state, Path(last), eps, eps_space
            )
        except Exception as exc:
            log.exception("callback adv_xai failed")
            await bot.send_message(
                chat_id, f"❌ Ошибка XAI: {type(exc).__name__}: {exc}"
            )


# --------------------------------------------------------------------------- #
# Сценарии                                                                  #
# --------------------------------------------------------------------------- #

async def _do_predict(
    message: Message, bot: Bot, runtime: BotRuntime,
    state: BotState, max_mb: int,
) -> None:
    src = await _download_photo(bot, message, runtime, max_mb)
    if src is None:
        return
    keep = runtime.tmp_path(".jpg")
    try:
        shutil.copy(src, keep)
        state.set_last_image(message.chat.id, str(keep))

        img = _photo_to_array(src)
        clf = runtime.clf
        x = clf.transform(Image.fromarray(img)).unsqueeze(0).to(clf.device)

        panel_path = runtime.tmp_path(".png")
        label, conf, _ = build_xai_panel(x, clf, panel_path)

        caption = (
            f"🔍 Классификация\n"
            f"Результат: {label}\n"
            f"Уверенность: {conf:.3f}"
        )
        state.set_last_panel(message.chat.id, str(panel_path), caption)
        # Кнопки под clean-панелью нет — XAI для adv вызывается из /attack.
        await bot.send_photo(
            message.chat.id, FSInputFile(panel_path), caption=caption,
        )
    finally:
        runtime.cleanup(src)


async def _do_adv_panel(
    bot: Bot, chat_id: int, runtime: BotRuntime,
    state: BotState, src: Path, eps: float, eps_space: str,
) -> None:
    """Панель XAI 2×3 для adversarial-картинки. Формат идентичен /predict."""
    clf = runtime.clf
    img = _photo_to_array(src)

    clean = clf.predict(img)
    clean_label = clean["label"]
    clean_conf = float(clean["confidence"])

    x = clf.transform(Image.fromarray(img)).unsqueeze(0).to(clf.device)
    y = torch.tensor(
        [clf.CLASSES.index(clean_label)], device=clf.device, dtype=torch.long
    )

    from src.attacks.fgsm import fgsm_attack
    x_adv = fgsm_attack(clf.model, x, y, eps=eps, eps_space=eps_space).detach()

    panel_path = runtime.tmp_path(".png")
    adv_label, adv_conf, _ = build_xai_panel(x_adv, clf, panel_path)

    asr_ok = adv_label != clean_label
    caption = (
        f"🔍 Классификация (adversarial)\n"
        f"Результат: {adv_label}\n"
        f"Уверенность: {adv_conf:.3f}\n\n"
        f"⚔️ FGSM (eps={eps:.2f}, {eps_space})\n"
        f"До атаки:\n  {clean_label}, conf={clean_conf:.3f}\n"
        f"После атаки:\n  {adv_label}, conf={adv_conf:.3f}\n\n"
        f"ASR (по одному примеру): "
        f"{'✅ атака успешна' if asr_ok else '❌ не удалась'}"
    )
    state.set_last_panel(chat_id, str(panel_path), caption)

    kb = InlineKeyboardBuilder()
    await bot.send_photo(
        chat_id, FSInputFile(panel_path),
        caption=caption, reply_markup=kb.as_markup(),
    )


async def _do_attack(
    message: Message, bot: Bot, runtime: BotRuntime,
    state: BotState, max_mb: int, eps: float, eps_space: str,
) -> None:
    src = await _download_photo(bot, message, runtime, max_mb)
    if src is None:
        return
    try:
        await _do_attack_from_path(message, bot, runtime, state, src, eps, eps_space)
    finally:
        runtime.cleanup(src)


async def _do_attack_from_path(
    message: Message, bot: Bot, runtime: BotRuntime,
    state: BotState, src: Path, eps: float, eps_space: str,
) -> None:
    img = _photo_to_array(src)
    clf = runtime.clf

    # 1) Чистое предсказание
    clean = clf.predict(img)
    clean_label = clean["label"]
    clean_conf = float(clean["confidence"])

    # 2) Нормализованный батч — то, что реально видит модель
    x = clf.transform(Image.fromarray(img)).unsqueeze(0).to(clf.device)
    orig_aligned = denorm_to_uint8(x[0])
    y_true = torch.tensor(
        [clf.CLASSES.index(clean_label)], device=clf.device, dtype=torch.long
    )

    # 3) FGSM
    from src.attacks.fgsm import fgsm_attack
    x_adv = fgsm_attack(
        clf.model, x, y_true,
        eps=eps, eps_space=eps_space,
    )

    # 4) Adversarial → пиксели (только для отрисовки панели)
    adv_img = denorm_to_uint8(x_adv[0])

    # 4.1) Предсказание по ТОМУ ЖЕ тензору, что ушёл в XAI — без двойного transform.
    #      Иначе цифры в этой подписи и в панели XAI расходятся (0.676 vs 0.751).
    _, adv_label, adv_conf, _ = forward_probs(clf, x_adv)

    # 5) Панель clean | adv | perturbation
    panel_path = runtime.tmp_path(".png")
    _build_attack_panel(
        orig_aligned, adv_img, x[0], x_adv[0], panel_path,
        clean_label, clean_conf, adv_label, adv_conf,
    )

    asr_ok = adv_label != clean_label
    caption = (
        f"⚔️ FGSM-атака (eps={eps:.2f})\n\n"
        f"До атаки:\n  {clean_label}, conf={clean_conf:.3f}\n\n"
        f"После атаки:\n  {adv_label}, conf={adv_conf:.3f}\n\n"
        f"ASR (по одному примеру): "
        f"{'✅ атака успешна' if asr_ok else '❌ не удалась'}\n"
        f"Картинка визуально идентична — perturbation невидим глазу."
    )
    state.set_last_panel(message.chat.id, str(panel_path), caption)

    kb = InlineKeyboardBuilder()
    kb.button(text="🔍 XAI (adversarial)", callback_data="adv_xai")
    await bot.send_photo(
        message.chat.id, FSInputFile(panel_path),
        caption=caption, reply_markup=kb.as_markup(),
    )