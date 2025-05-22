# **Face ID Tracker: Real-Time Facial Recognition with CUDA Acceleration**

Система распознавания лиц с помощью библиотеки [face_recognition](https://github.com/ageitgey/face_recognition) и аппаратным ускорением через **NVIDIA CUDA**. Проект реализован в изолированном окружении на базе **Conda** для обеспечения воспроизводимости и переносимости.

## **Предварительные требования**

1. **ОС:** Linux x86_64.
2. **Менеджер пакетов python:** Conda (Anaconda, Miniconda, Mamba или Micromamba).
3. **NVIDIA CUDA:** версия CUDA должна быть 11.8.0 или выше, а также должны быть установлены соответствующие драйверы NVIDIA.

## **Структура проекта**

```bash
.
├── README.md   # Документация
├──.env.example # Шаблон конфигурации
├── LICENSE     # Лицензия проекта
├── SBOMs/      # Метаданные зависимостей
├── app/        # Исходный код приложения
└── photos/     # Фотографии для работы
```

## **Инструкция по установке и запуску**

### **I. Клонирование репозитория**

```bash
git clone https://github.com/Sierra-Arn/face-id-tracker-python.git
cd face-id-tracker-python
```

### **II. Создание и активация виртуальной среды**

```bash
conda env create -p ./.venv -f SBOMs/conda-linux-64-lock.yml
conda activate ./.venv
```

### **III. Подготовка данных**

В директорию `photos` Вы можете добавить любые фотографии. Имя файла будет использоваться как имя человека при распознавании.  
В качестве демонстрации, Вы можете скачать фотографии с сайта [Pexels Free Stock Videos](https://www.pexels.com/videos/).  
Например, [Man in brown polo shirt by Simon Robben from Pexels](https://www.pexels.com/photo/man-in-brown-polo-shirt-614810/).

### **IV. Настройка конфигурации через `.env`**

1. Создайте файл `.env` из шаблона:
```bash
cp .env.example .env
```

2. Отредактируйте `.env` в любом текстовом редакторе (например, `nano`, `VSCode`):
```bash
nano .env
```

3. Настройте параметры: файл разделен на секции — измените значения в соответствии с Вашими задачами. Например:

```bash
# === MODEL CONFIGURATION ===
FACE_RECOGNITION_MODEL=hog
FACE_DISTANCE_THRESHOLD=0.6

# === DISPLAY SETTINGS ===
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
RESIZE_FACTOR=2

FONT=FONT_HERSHEY_COMPLEX
COLOR_KNOWN=0,255,0
COLOR_UNKNOWN=0,0,255
COLOR_INFO=255,0,0

# === PATH CONFIGURATION ===
PHOTOS_DIR=photos

# === LOCALIZATION SETTINGS ===
UNKNOWN_PERSON_LABEL=Unknown
TIMEZONE_OFFSET=0
TIMEZONE_LABEL=UTC
```

### **V. Запуск приложения**

```bash
python -m app.main
```

## **Лицензия**

Проект распространяется под лицензией [MIT](LICENSE). 

> **Внимание**  
> Проект включает сторонние компоненты с отдельными лицензиями, которые могут отличаться от MIT.  
> Полный список лицензий зависимостей доступен в файле [THIRD_PARTY_LICENSES.md](SBOMs/THIRD_PARTY_LICENSES.md).

## **Инструменты разработки**
Проект использует следующие инструменты для обеспечения воспроизводимости и управления зависимостями:

- [Micromamba](https://github.com/mamba-org/mamba), ультрабыстрая реализация [Conda](https://github.com/conda/conda) для создания изолированных окружений.
- [conda-lock](https://github.com/conda/conda-lock), утилита для генерации [точного lock-файла](SBOMs/conda-linux-64-lock.yml) для гарантии идентичного воссоздания окружения на любых системах.
- [pip-licenses](https://github.com/raimon49/pip-licenses), утилита для генерации файла [THIRD_PARTY_LICENSES.md](SBOMs/THIRD_PARTY_LICENSES.md) с лицензиями всех conda-зависимостей.