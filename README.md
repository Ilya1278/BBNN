# BBNN — Устойчивость нейросетей к adversarial-атакам

Курсовой проект: бинарная классификация «мотоцикл / не мотоцикл»
с исследованием adversarial-атак (FGSM) и методов защиты.

## Структура
- `src/data` — сбор, очистка, сплит, аугментация
- `src/models` — фабрика моделей (scratch / pretrained)
- `src/train` — цикл обучения и логирование
- `src/xai` — Grad-CAM, saliency
- `src/attacks` — FGSM, оценка успешности
- `src/defense` — adversarial training, JPEG, feature squeezing
- `src/analytics` — графики, детектор переобучения
- `src/bot` — Telegram-интерфейс
- `scripts/` — entry points
