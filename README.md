# Лабораторная работа 1: Computer Vision
## Обнаружение дефектов на печатных платах (PCB Defect Detection)
**Задание на оценку 3** | YOLOv11 (Ultralytics) | PyTorch

---

### Датасет

**PCB Defect Detection**
- Ссылка: https://www.kaggle.com/datasets/akhatova/pcb-defects
- Изображений: ~1386
- Классы (6): `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`
- Формат аннотаций: Pascal VOC (XML) → конвертируется в YOLO автоматически

**Обоснование:** Автоматическая инспекция качества PCB — ключевая задача в производстве
кибер-физических систем. Встраивание детектора в производственную линию позволяет
проверять платы в реальном времени и снизить процент брака.

---

### Структура

```
lab1_cv/
├── lab1.py           # Основной скрипт (Jupyter-совместимый, # %% маркеры)
├── dataset.yaml      # Конфигурация датасета для ultralytics
├── requirements.txt  # Зависимости
└── README.md
```

После запуска создаются:
```
runs/
├── pcb_baseline/     # Веса бейзлайна (YOLOv11n)
├── pcb_aug/          # Гипотеза 1: YOLOv11n + аугментации
├── pcb_yolo11s/      # Гипотеза 2: YOLOv11s
└── pcb_improved/     # Улучшенный бейзлайн (YOLOv11s + aug)
custom_baseline.pth   # Веса SimpleYOLO
custom_aug.pth        # Веса SimpleYOLO + aug
results_summary.csv   # Сводная таблица метрик
*.png                 # Графики сравнения
```

---

### Установка и запуск

#### 1. Требования
- Python 3.10+

#### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

#### 3. Настройка Kaggle API
Для автоматической загрузки датасета нужен Kaggle API-ключ:
1. Зайдите на https://www.kaggle.com → Account → Create New Token
2. Сохраните `kaggle.json` в `~/.kaggle/kaggle.json`
3. Установите права: `chmod 600 ~/.kaggle/kaggle.json`

#### 4a. Запуск как Python-скрипт
```bash
cd lab1_cv
python lab1.py
```
