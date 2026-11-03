# INTERFACES.md — контракты между модулями BBNN

> Правило: если твой модуль читает `cfg["x"]["y"]` или принимает параметр —
> ты обязан это здесь задокументировать. Иначе сосед не знает, что положить.

---

## 1. `src/data/` 

### Публичный API

| Функция | Сигнатура | Возвращает |
|---|---|---|
| `get_train_transform` | `(cfg: dict) -> albumentations.Compose` | train-трансформы |
| `get_val_transform` | `(cfg: dict) -> albumentations.Compose` | val/test-трансформы |
| `MotorcycleDataset.__init__` | `(manifest_path: str, split: str, transform, root: str)` | — |
| `MotorcycleDataset.__getitem__` | `(idx: int)` | `(tensor, label)` |

### Ключи YAML, которые читает `src/data/`

Обязательны, дефолтов нет — если отсутствуют, падает KeyError.

```yaml
augment:
  image_size: 224                       # int
  imagenet_mean: [0.485, 0.456, 0.406]  # list[float] × 3
  imagenet_std:  [0.229, 0.224, 0.225]  # list[float] × 3
  jpeg_quality_lower: 50                # int
  jpeg_quality_upper: 95                # int
  jpeg_p: 0.3                           # float, вероятность применения
```

### Формат `manifest.json`

```json
[{"filepath": "train/motorcycle/0001.jpg",
  "label": 1,
  "source": "coco" | "openimages" | "voc" | "imagenet" | "user",
  "sha1": "..."}]
```

- `label`: `1` = motorcycle, `0` = not_motorcycle.
- `filepath`: **относительный**, база — `data/processed/`.
- Склейка пути: **делает `MotorcycleDataset` через параметр `root=`**.
  Вызывающая сторона передаёт `root=<abs path to data/processed>`.

---

## 2. `src/models/`, `src/train/` 
### Публичный API

| Класс / функция | Файл | Сигнатура | Возвращает / примечание |
|---|---|---|---|
| `build_model` | `src/models/factory.py` | `(cfg: dict, num_classes: int = 2) → nn.Module` | переключение по `cfg["model"]["mode"]` |
| `MotorcycleCNN` | `src/models/scratch_cnn.py` | `(num_classes=2, dropout=0.5)` | 5 conv-блоков, ~4.72M параметров |
| `MotorcycleClassifier.__init__` | `src/models/classifier.py` | `(checkpoint_path, cfg, device="cuda")` | грузит state_dict, eval() |
| `MotorcycleClassifier.predict` | `src/models/classifier.py` | `(image: np.ndarray \| str) → dict` | контракт стабилен, не менять |
| `MotorcycleClassifier.predict_batch` | `src/models/classifier.py` | `(images: list) → list[dict]` | контракт стабилен, не менять |
| `MotorcycleClassifier.model` | `src/models/classifier.py` | `@property → nn.Module` | underlying модель, eval(), на device. Для backward (Attack Agent) |
| `MotorcycleClassifier.device` | `src/models/classifier.py` | `@property → torch.device` | куда класть тензоры |
| `MotorcycleClassifier.transform` | `src/models/classifier.py` | `@property → torchvision Compose` | Resize + ToTensor + Normalize(ImageNet) |
| `MotorcycleClassifier.model_mode` | `src/models/classifier.py` | `@property → str` | `"scratch"` \| `"pretrained"` |
| `MotorcycleClassifier.CLASSES` | `src/models/classifier.py` | class attr `list[str]` | `["not_motorcycle", "motorcycle"]` |

### Контракт `predict()` / `predict_batch()` — СТАБИЛЕН

```python
{
  "label": "motorcycle" | "not_motorcycle",
  "confidence": float,          # 0..1
  "probs": {"not_motorcycle": float, "motorcycle": float},
  "model_mode": "scratch" | "pretrained"
}
```

### Публичный доступ к модели (для Attack / XAI)

- `clf.model` — underlying `nn.Module` в `eval()`, на `clf.device`.
  Разрешено делать `loss.backward()`, `x.grad` течёт до входа.
  Проверено mini-тестом: `x.grad.shape = (1, 3, 224, 224)`.
- `clf.device` — `torch.device`, куда класть входные тензоры.
- `clf.transform` — `torchvision.Compose` (Resize → ToTensor → Normalize).
  **Атаку и XAI-карты делать ПОСЛЕ `clf.transform`**, на нормализованных
  тензорах. Иначе возмущение/градиент интерпретируется неверно.

### Ключи YAML (`configs/train_*.yaml`)

```yaml
model:
  mode: "scratch" | "pretrained"
  dropout: 0.5
  num_classes: 2

data:
  manifest: "data/processed/manifest.json"   # читается только для снапшота
  processed_dir: "data/processed"
  image_size: 224
  batch_size: 64
  num_workers: 4

train:
  epochs: 60
  lr: 0.001              # 0.0003 для pretrained
  weight_decay: 0.0001
  label_smoothing: 0.1
  early_stopping_patience: 10
  amp: true
  seed: 42

paths:
  runs_dir: "runs"
  checkpoints_dir: "checkpoints"
  run_name: "scratch_hardaug_v2"   # префикс run_id

# секция augment — принадлежит Data Agent, см. раздел 1
```

### Формат `runs/{run_id}/`

```
runs/{run_id}/
├── config.yaml                # снапшот конфига на старте
├── manifest_snapshot.json     # копия manifest на момент старта (пути НЕ трогаются)
├── metrics.csv                # epoch,train_loss,val_loss,train_acc,val_acc,lr,time_sec
└── model_best.pt              # сохраняется ТОЛЬКО по val_loss
```

- `run_id = f"{run_name}_{YYYYMMDD_HHMMSS}"`, где `run_name` из `paths.run_name`.
- Датасеты читают **`manifest_snapshot.json`**, не живой `data/processed/manifest.json`.
- `model_best.pt` дублируется в `checkpoints/{run_id}_best.pt`.

### Уточнения от 2026-10-08 (прогон v2)

#### augment.py (Data Agent)
- Публичные функции **в единственном числе**: `get_train_transform(cfg)`, `get_val_transform(cfg)`.
- Принимают `cfg` ЦЕЛИКОМ (не `image_size`, не отдельные аргументы).
- Множественное число (`get_train_transforms`) **не существует**.

#### MotorcycleDataset (Data Agent)
- Параметр называется `root=`, **НЕ** `processed_dir=`.
- Молча проглатывает неправильные kwargs при `try/except` — если передать
  `processed_dir=`, он его проигнорирует (через `TypeError`) и будет считать
  `filepath` относительно CWD.

#### Формат `filepath` в manifest
- **ОТНОСИТЕЛЬНЫЙ**: `"train/motorcycle/0001.jpg"`. База — `data/processed/`.
- Склейка — вызывающей стороной через `MotorcycleDataset(root=<abs path>)`.
- Фильтр split внутри датасета: `r["filepath"].startswith(split + "/")`.
  Абсолютные пути **ЛОМАЮТ** фильтр (`train=0`, `ValueError: num_samples=0`).

#### Секция `augment` в YAML — контракт (6 ключей, все обязательны)

| Ключ | Тип | Значение v2 | Дефолт в augment.py |
|---|---|---|---|
| `augment.image_size` | int | 224 | — (обязателен) |
| `augment.imagenet_mean` | list[float]×3 | [0.485, 0.456, 0.406] | — (обязателен) |
| `augment.imagenet_std` | list[float]×3 | [0.229, 0.224, 0.225] | — (обязателен) |
| `augment.jpeg_quality_lower` | int | 50 | — (обязателен) |
| `augment.jpeg_quality_upper` | int | 95 | — (обязателен) |
| `augment.jpeg_p` | float | 0.3 | — (обязателен) |

`.get()` в `augment.py` **не используется** — пропуск любого ключа = `KeyError`.
Владелец ключей — Data Agent. Носитель — `configs/train_*.yaml`.


---

## 3. `src/analytics/` 

**Читает:** `runs/{run_id}/metrics.csv`, `runs/{run_id}/config.yaml`.

**Пишет:** `reports/{run_id}/summary.md`, `reports/{run_id}/figures/*.png`.

**Не трогает:** ничего вне `src/analytics/`, `scripts/analyze.py`, `configs/analytics.yaml`, `reports/`.

**Публичный API:**

| Функция | Сигнатура | Возвращает |
|---|---|---|
| `detect_overfitting` | `(metrics_df, window=5) -> dict` | best_epoch, overfit_start_epoch, underfit_until_epoch, interpretation |
| `generate_report` | `(run_dir, out_dir, y_true=None, y_pred=None)` | пишет `summary.md` + figures |

**Формат вывода:**

```
reports/{run_id}/
├── summary.md
└── figures/
    ├── loss_curves.png
    ├── acc_curves.png
    ├── overfit_zone.png
    ├── lr_schedule.png
    ├── epoch_time.png
    ├── confusion_matrix.png    # при --eval-test
    └── per_class_metrics.png   # при --eval-test
```

**Ключи YAML (`configs/analytics.yaml`):** `window: 5`, `dpi: 150`, `figsize: [10, 6]`, `style: seaborn-whitegrid`.

---

## 4. `src/xai/` — XAI Agent (РЕАЛИЗОВАНО)

**Читает:** `MotorcycleClassifier` (только через публичный API:
`predict()`, `transform`, `model`, `device`, `CLASSES`).

**Пишет:** панели в `reports/xai/` и `reports/xai/sanity/`.

**Не трогает:** ничего вне `src/xai/*`, `scripts/run_xai.py`,
`configs/xai.yaml`, `reports/xai/`.

### Публичный API

| Класс/функция | Файл | Сигнатура | Возвращает |
|---|---|---|---|
| `GradCAM.__init__` | src/xai/gradcam.py | `(model: nn.Module, target_layer: nn.Module)` | — |
| `GradCAM.generate` | src/xai/gradcam.py | `(image: np.ndarray \| Tensor, target_class: int \| None = None)` | heatmap HxW float [0,1] |
| `GradCAM.overlay` | src/xai/gradcam.py | `(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5)` | RGB uint8 HWC |
| `GradCAM.remove_hooks` | src/xai/gradcam.py | `()` | — (обязателен после работы) |
| `get_target_layer` | src/xai/gradcam.py | `(model: nn.Module, target_layer: str = "auto")` | nn.Module (последний Conv2d) |
| `vanilla_saliency` | src/xai/saliency.py | `(model, image, target_class=None)` | HxW float [0,1] |
| `integrated_gradients` | src/xai/saliency.py | `(model, image, target_class=None, steps=32)` | HxW float [0,1] |
| `make_xai_panel` | src/xai/visualize.py | `(image, model, classifier, out_path, ig_steps=32, alpha=0.5, dpi=150, figsize=(12,8))` | Path к PNG |

### Выбор `target_layer`

- `"auto"` → последний `nn.Conv2d` в модели.
- `MotorcycleCNN` (scratch) → последний conv-блок.
- `resnet18` (pretrained) → `layer4[-1].conv2`.
- Явное имя слоя можно передать через `configs/xai.yaml: gradcam.target_layer`.

### Конфиг (`configs/xai.yaml`)

```yaml
gradcam:
  target_layer: auto     # auto | имя слоя из named_modules()
  alpha: 0.5
  colormap: jet
saliency:
  ig_steps: 32
output:
  dpi: 150
  figsize: [12, 8]
```

### CLI (`scripts/run_xai.py`)

```bash
# одна картинка
python scripts/run_xai.py --checkpoint <pt> --config <yaml> \
                          --image <img> --out <png>

# пачка из манифеста
python scripts/run_xai.py --checkpoint <pt> --config <yaml> \
                          --from-manifest data/processed/manifest.json \
                          --split test --n 20 --out-dir reports/xai/

# sanity 5 moto + 5 nonmoto + диагностика концентрации
python scripts/run_xai.py --checkpoint <pt> --config <yaml> \
                          --from-manifest data/processed/manifest.json \
                          --split test --sanity --out-dir reports/xai/sanity/
```

### Диагностика концентрации

`heatmap_focus_score(hm)` → `{top20_mass, center_vs_border}`.
Ориентир «heatmap осмысленный»: `top20_mass ≥ 0.55`.
`center_vs_border` шумит на вытянутых объектах (мотоцикл/велосипед занимает
весь кадр, а не центр) — использовать как вспомогательный сигнал.

---

## 5. `src/attacks/` — Attack Agent (РЕАЛИЗОВАНО)

**Читает:** `MotorcycleClassifier` (только через публичный API: `model`,
`device`, `model_mode`), test-сплит из `manifest.json`, `augment.*` из
`runs/{run_id}/config.yaml`.

**Пишет:** `attack/results/attack_metrics.json`,
`attack/results/attack_metrics_sweep.json`,
`attack/results/attack_examples/*.png`.

**Не трогает:** ничего вне `src/attacks/*`, `scripts/run_attack.py`,
`configs/attack_fgsm.yaml`, `attack/results/`.

### Публичный API

| Функция | Файл | Сигнатура | Возвращает |
|---|---|---|---|
| `fgsm_attack` | src/attacks/fgsm.py | `(model, images, labels, eps, criterion=None, eps_space="normalized", std=None)` | `x_adv: Tensor (N,C,H,W)` detached |
| `evaluate_attack` | src/attacks/fgsm.py | `(classifier, loader, eps, eps_space="normalized", std=None, criterion=None)` | dict метрик (см. ниже) |
| `evaluate_sweep` | src/attacks/fgsm.py | `(classifier, loader, eps_list, eps_space="normalized", std=None, progress=True)` | `list[dict]` |
| `save_success_panels` | src/attacks/fgsm.py | `(classifier, loader, eps, out_dir, cfg_train, eps_space="normalized", max_per_class=1)` | `list[str]` путей к PNG |

### Пространство eps (важно)

Контракт §2 требует атаковать на **нормализованных** тензорах. Пересчёт
в пиксельные единицы: `Δx_pixel = Δx_norm × std`, `std ≈ 0.229`.

- `eps_space="normalized"` (по умолчанию) — `eps` в единицах
  нормализованного тензора, ровно как в §2.
  `eps=0.03` normalized ≈ **1.75/255** в пикселях.
- `eps_space="pixel"` — `eps` в пиксельных единицах [0,1] на канал;
  внутри делается per-channel пересчёт `eps_norm = eps_pixel / std`.

Значение по умолчанию задаётся в `configs/attack_fgsm.yaml: attack.eps_space`.

### Контракт метрик (`attack_metrics.json`)

```json
{
  "attack": "fgsm",
  "eps": 0.03,
  "eps_space": "normalized",
  "asr": 0.7700559701492538,
  "n_samples": 2586,
  "n_clean_correct": 2144,
  "n_adv_success": 1651,
  "clean_acc": 0.8290796597061099,
  "adv_acc": 0.19064191802010827
}
```

- `asr = n_adv_success / n_clean_correct` — доля **изначально верных**
  сэмплов, которые атака сломала. Считается только по `test`-сплиту.
- `clean_acc`, `adv_acc` — по всему сплиту (знаменатель `n_samples`).

`attack_metrics_sweep.json` — массив таких же объектов, по одному на каждый
`eps` из `--eps-sweep`, в порядке перечисления.

### Формат вывода

```
attack/results/
├── attack_metrics.json
├── attack_metrics_sweep.json
└── attack_examples/
    ├── moto_success_01.png
    ├── nonmoto_success_01.png
    └── failure_01.png
```

### CLI (`scripts/run_attack.py`)

```bash
# один eps + панели
python scripts/run_attack.py --config configs/attack_fgsm.yaml --eps 0.03

# sweep по eps (для графика eps vs ASR)
python scripts/run_attack.py --config configs/attack_fgsm.yaml \
        --eps-sweep 0.005 0.01 0.02 0.03 0.05 0.1
```

> **PowerShell:** `\` как продолжение строки не работает. Пиши одной строкой
> или используй backtick `` ` ``.

### Ключи YAML (`configs/attack_fgsm.yaml`)

| Ключ | Тип | Примечание |
|---|---|---|
| `checkpoint` | str | путь к `model_best.pt` |
| `model_config` | str | путь к `runs/{run_id}/config.yaml` (для сборки модели и трансформа) |
| `data.manifest` | str | путь к `manifest.json` (или snapshot) |
| `data.processed_dir` | str | `root=` для `MotorcycleDataset` |
| `data.split` | str | `test` |
| `data.batch_size` | int | 64 |
| `data.num_workers` | int | 4 |
| `attack.method` | str | `"fgsm"` |
| `attack.eps` | float | дефолт для `--eps` |
| `attack.eps_space` | str | `"normalized"` \| `"pixel"` |
| `output_dir` | str | `attack/results/` |
| `seed` | int | 42 |

### Фактический прогон (2026-10-08, `scratch_hardaug_v2_20261008_133340`)

- Чекпоинт: `runs/scratch_hardaug_v2_20261008_133340/model_best.pt`
- Конфиг: `runs/scratch_hardaug_v2_20261008_133340/config.yaml`
- Сплит: `test` из `data/processed/manifest.json`
- `eps_space = "normalized"` (атака на нормализованных тензорах, §2)
- Датасет: `n_samples = 2586`, `n_clean_correct = 2144`
- `clean_acc = 0.82908` — согласуется с `val_acc = 0.838` из трейна
  (разница = val vs test-сплит)

#### `attack/results/attack_metrics.json` (eps=0.03)

```json
{
  "attack": "fgsm",
  "eps": 0.03,
  "eps_space": "normalized",
  "asr": 0.7700559701492538,
  "n_samples": 2586,
  "n_clean_correct": 2144,
  "n_adv_success": 1651,
  "clean_acc": 0.8290796597061099,
  "adv_acc": 0.19064191802010827
}
```

#### `attack/results/attack_metrics_sweep.json` (сводка)

| eps | ASR | adv_acc | n_adv_success |
|---|---|---|---|
| 0.005 | 0.1469 | 0.7073 | 315 |
| 0.010 | 0.3036 | 0.5773 | 651 |
| **0.020** | **0.6035** | 0.3287 | 1294 |
| **0.030** | **0.7701** | 0.1906 | 1651 |
| 0.050 | 0.9132 | 0.0719 | 1958 |
| 0.100 | 0.9767 | 0.0193 | 2094 |

`clean_acc = 0.82908` и `n_samples = 2586` во всех точках — модель и датасет
фиксированы, варьируется только `eps`.

#### Порог «опасно» (ASR ≥ 0.5)

- По сетке sweep'а порог впервые перекрыт на **eps = 0.02** (ASR = 0.6035).
- Линейная интерполяция между (0.01, 0.3036) и (0.02, 0.6035) даёт точку
  перехода **eps ≈ 0.0165**.
- В пиксельных единицах (ImageNet std ≈ 0.229): ≈ **1.17/255**.
- При `eps = 0.03` ASR = 0.7701 (≈ **1.75/255** в пикселях), `adv_acc`
  падает с 0.8291 до 0.1906.

Для сравнения: классический FGSM `8/255` в пикселях эквивалентен
`eps ≈ 0.137` в normalized-пространстве. То есть наша атака при
`eps = 0.03` (≈ 1.75/255) даёт ASR = 0.77 при возмущении в **4.6 раза
меньшем** по пиксельной амплитуде, чем у стандартной 8/255-настройки.

#### Панели успешных атак

`attack/results/attack_examples/`:

| Файл | `true` | `clean pred` | `adv pred` |
|---|---|---|---|
| `moto_success_01.png` | motorcycle | motorcycle | not_motorcycle |
| `nonmoto_success_01.png` | not_motorcycle | not_motorcycle | motorcycle |
| `failure_01.png` | motorcycle | motorcycle | motorcycle (атака не сработала) |

Возмущение визуально высокочастотное и почти невидимое (`|adv − clean| × 8`
почти чёрный) — FGSM ломает модель направленным градиентом, а не видимой
деградацией картинки. Колонка `perturbation (norm-space ×4)` осмысленного
паттерна не показывает — это нормально для одношагового FGSM.

---

## 6. `src/bot/` — Bot Agent (РЕАЛИЗОВАНО)
 
-**Читает:**
-- `MotorcycleClassifier` — команда `/predict`
-- `runs/{run_id}/metrics.csv` — команда `/metrics`
-- `attack/results/attack_metrics.json` — команда `/attack`
-- `reports/{run_id}/summary.md` — команда `/report`
+**Читает (только публичные API, см. §2, §3, §4, §5):**
+- `MotorcycleClassifier.predict()`, `.transform`, `.model`, `.device`, `.CLASSES` — `/predict`, `/attack`
+- `make_xai_panel()` из `src/xai/visualize.py` — `/predict`
+- `fgsm_attack()` из `src/attacks/fgsm.py` — `/attack`
+- `reports/{run_id}/figures/loss_curves.png` (fallback: `runs/{run_id}/loss_curves.png`) — `/metrics`
+- `reports/{run_id}/summary.md` — `/report`
 
-**Пишет:** ничего в проект (только в Telegram).
+**Пишет:** только временные файлы в `runs/bot_tmp/` (удаляются после обработки
+и на shutdown). В `runs/`, `reports/`, `attack/` — ничего не пишет.
 
 **Переменные окружения (`BBNN/.env`):**
 
TG_BOT_TOKEN=<токен от @BotFather>

text

**Команды бота:**

| Команда | Что делает |
|---|---|
-| `/start` | приветствие, справка |
-| `/predict` | принять фото → вернуть label, confidence, heatmap |
-| `/attack fgsm` | принять фото → FGSM → вернуть orig / adv / heatmap |
-| `/metrics` | прислать loss_curves.png из последнего run |
-| `/report` | прислать summary.md |
+| `/start`, `/help` | приветствие, HTML-справка |
+| `/predict` | фото → `clf.predict()` + `make_xai_panel()` → панель + подпись |
+| `/attack` | фото (или последнее из state) → FGSM → clean/adv/perturbation + подпись |
+| `/last` | повторить последнюю панель данного chat_id |
+| `/metrics` | `loss_curves.png` из `reports/{latest_run}/figures/` |
+| `/report` | `summary.md` текстом (файлом при >4000 символов) |

**Публичный API (`src/bot/app.py`):**

| Функция | Сигнатура |
|---|---|
| `main` | `() -> None` (запускает polling) |

+**Публичный API (`src/bot/state.py`):**
+
+| Метод | Сигнатура |
+|---|---|
+| `BotState.set_last_image` | `(chat_id: int, path: str) -> None` |
+| `BotState.get_last_image` | `(chat_id: int) -> str \| None` |
+| `BotState.set_last_panel` | `(chat_id: int, path: str, caption: str = "") -> None` |
+| `BotState.get_last_panel` | `(chat_id: int) -> tuple[str \| None, str]` |
+| `BotState.set_pending` / `pop_pending` | `(chat_id, mode)` / `(chat_id) -> str \| None` |
+
+**Публичный API (`src/bot/handlers.py`):**
+
+| Класс / функция | Сигнатура |
+|---|---|
+| `BotRuntime.__init__` | `(cfg: dict, tmp_dir: Path)` |
+| `BotRuntime.clf` | `@property -> MotorcycleClassifier` (lazy) |
+| `register_handlers` | `(dp: Dispatcher, runtime: BotRuntime, state: BotState, cfg: dict) -> None` |
+
+**Конфиг (`configs/bot.yaml`):**
+
+```yaml
+checkpoint: runs/scratch_hardaug_v2_20261008_133340/model_best.pt
+model_config: runs/scratch_hardaug_v2_20261008_133340/config.yaml
+runs_dir: runs
+reports_dir: reports
+latest_run: scratch_hardaug_v2_20261008_133340
+attack:
+  eps: 0.03
+  eps_space: normalized
+bot:
+  max_photo_size_mb: 10
+  tmp_dir: runs/bot_tmp
+```
+
**Точка входа:** `scripts/run_bot.py`.