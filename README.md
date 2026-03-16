# **Face ID Tracker: Real-Time Facial Recognition with CUDA Acceleration**

A real-time facial recognition system using the [face_recognition](https://github.com/ageitgey/face_recognition) library with **NVIDIA CUDA** hardware acceleration. The project is implemented in an isolated **Conda** environment for reproducibility and portability.

> [!NOTE]  
**This repository is archived.** This is a personal pet project built purely for learning — it was never intended to be actively maintained or production-ready.

## **Available Documentation / Доступная документация**

- [English Documentation](README.md) (current document / текущий документ)
- [Документация на русском языке](README_RU.md)

## **Prerequisites**

1. **OS:** Linux x86_64.
2. **Python Package Manager:** Conda (Anaconda, Miniconda, Mamba, or Micromamba).
3. **NVIDIA CUDA:** CUDA version 11.8.0 or higher with corresponding NVIDIA drivers installed.

## **Project Structure**

```bash
.
├── .env.example    # Configuration template
├── SBOMs/          # Dependency metadata
├── app/            # Application source code
└── photos/         # Input photo files
```

## **Quick Start**

### **I. Clone the Repository**

```bash
git clone https://github.com/Sierra-Arn/face-id-tracker-python.git
cd face-id-tracker-python
```

### **II. Create and Activate Virtual Environment**

```bash
conda env create -p ./.venv -f SBOMs/conda-linux-64-lock.yml
conda activate ./.venv
```

### **III. Prepare Data**

Add any photos to the `photos/` directory. The filename will be used as the person's name for recognition.  
For demonstration, you can download photos from [Pexels Free Stock Photos](https://www.pexels.com/).  
For example: [Man in brown polo shirt by Simon Robben from Pexels](https://www.pexels.com/photo/man-in-brown-polo-shirt-614810/).

### **IV. Configure via `.env`**

1. Create `.env` file from template:
```bash
cp .env.example .env
```

2. Edit `.env` in any text editor (e.g., `nano`, `VSCode`):
```bash
nano .env
```

3. Configure parameters: the file is divided into sections — modify values according to your needs. For example:

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

### **V. Run the Application**

```bash
python -m app.main
```

## **License**

This project is licensed under the [BSD-3-Clause License](LICENSE).

> **Warning**  
> The project includes third-party components with separate licenses that may differ from BSD-3-Clause.  
> A complete list of dependency licenses is available in [THIRD_PARTY_LICENSES.md](SBOMs/THIRD_PARTY_LICENSES.md).

## **Development Tools**

The project uses the following tools for reproducibility and dependency management:

- [Micromamba](https://github.com/mamba-org/mamba), an ultra-fast implementation of [Conda](https://github.com/conda/conda) for creating isolated environments.
- [conda-lock](https://github.com/conda/conda-lock), a utility for generating [exact lock files](SBOMs/conda-linux-64-lock.yml) to guarantee identical environment recreation across systems.
- [pip-licenses](https://github.com/raimon49/pip-licenses), a utility for generating [THIRD_PARTY_LICENSES.md](SBOMs/THIRD_PARTY_LICENSES.md) with licenses of all conda dependencies.