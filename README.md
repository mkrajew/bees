# WingAI – Automated Bee Wing Landmark Detection

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

WingAI is an open-source software tool for fully automated detection and annotation of morphometric landmarks on bee wing images.  
The system is based on a deep convolutional neural network (U-Net) combined with robust post-processing and Generalized Procrustes Analysis (GPA) to ensure biologically consistent landmark ordering.

WingAI is designed to support high-throughput morphometric analyses, reduce manual annotation effort, and integrate seamlessly with existing classification workflows such as IdentiFly.

In addition to the application itself, this repository contains the **full pipeline for dataset preprocessing, model training, and evaluation**, including all code used to prepare training data, train the neural network models, and reproduce the experimental results.

---

## Key Features

- Fully automated detection of 19 homologous morphometric landmarks
- Robust landmark ordering using Generalized Procrustes Analysis (GPA)
- Batch processing of large image collections
- Interactive web-based user interface (Gradio)
- Manual landmark editing and quality-control flags
- Automatic identification of potentially problematic images
- Export of landmark coordinates in:
  - CSV format compatible with training datasets
  - IdentiFly-compatible metadata format
- GPU acceleration (optional, automatic if available)

---

## Dataset

WingAI was developed and evaluated using the publicly available dataset:

**[Collection of wing images for conservation of honey bees (Apis mellifera) biodiversity in Europe](https://zenodo.org/records/7244070)**.

This dataset contains annotated bee wing images collected across Europe and serves as the primary source for training, validation, and testing of the WingAI model.


---

## Running the Application

> **Note:**  
> This section assumes that a **trained model** and a **precomputed mean wing shape** are already available in the project.  
> If you have not trained the model yet or computed the mean shape, please refer to the  
> **[Model Training](#model-training)** section at the bottom of this README for detailed instructions.

### Requirements

- Python 3.12
- CUDA-capable GPU (optional, recommended)
- Docker (optional, recommended)

### Installation & Environment Setup

This project uses **uv** for dependency and environment management.

#### Sync dependencies

To install the core runtime dependencies:

```bash
uv sync
````

#### Development dependencies (optional)

If you plan to work with notebooks, benchmarks, or development tools, install the development dependencies as well:
```bash
uv sync --dev
```


---

### Run the App (local)

This project includes a `Makefile` that provides a simple command to start the application:

```bash
make app
```

---

## Build & Run with Docker

WingAI can be run using Docker to ensure a consistent and reproducible environment.

### Run without GPU (CPU-only)

```bash
docker compose up --build
```

### Run with GPU support

To run WingAI with GPU acceleration, start the container with access to all available GPUs:

```bash
docker compose up --build --gpus all
```

When a compatible NVIDIA GPU is exposed to the container, WingAI will automatically use it for accelerated inference. Otherwise, the application will fall back to CPU execution.

> **Note:** GPU support requires the NVIDIA Container Toolkit to be installed on the host system.  
> Installation instructions are available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## Performance

The computational performance of WingAI was evaluated using end-to-end benchmarks measuring the time required to process a single bee wing image, from input loading to the generation of ordered landmark coordinates. Benchmarks were executed using `pytest-benchmark` on standard laptop hardware.

**Test platform:**
- CPU: 12th Gen Intel® Core™ i7-12800H (2.40 GHz)
- RAM: 32 GB
- GPU: NVIDIA RTX A3000 Laptop GPU (12 GB)

The results are summarized below:

| Mode | Min (ms) | Max (ms) | Mean (ms) | StdDev (ms) | Median (ms) | Rounds |
|-----|----------|----------|-----------|-------------|-------------|--------|
| GPU | 38.50 | 142.13 | 58.54 | 10.39 | 58.36 | 1000 |
| CPU | 94.59 | 159.58 | 115.11 | 12.39 | 110.46 | 1000 |

With GPU acceleration enabled, WingAI processes a single image in approximately **60 ms**, while CPU-only execution remains well below **120 ms** per image. These results confirm that the software is suitable for high-throughput batch processing on standard desktop or laptop hardware.

---

## Project structure
```
.
├── data                # Datasets
│   ├── raw             # Raw wing images
│   ├── interim         # Intermediate processing results
│   ├── processed       # Final processed datasets
│   └── external        # External data sources
├── models              # Trained neural network weights
├── notebooks           # Research and development notebooks
├── docs                # Documentation
├── references          # Reference materials
├── reports             # Figures and analysis outputs
└── wings               # Core WingAI source code
    ├── app             # Gradio-based user interface
    ├── modeling        # Model definitions and training code
    ├── dataset         # Dataset handling and GPA logic
    ├── gpa.py          # GPA logic
    ├── visualizing     # Visualization utilities
    └── utils           # Shared helper functions

```

---

## Citation

Citation will be added after publication.

---

## Model Training

### Dataset preparation

The model was trained using the dataset described above.

To train the model, unpack the downloaded dataset into: `data/raw`.

Next, preprocess the raw data into training-ready datasets. This can be done using the notebook:
`04_save_datasets.ipynb`.

After generating the datasets, compute the mean wing shape with the notebook:
`09_GPA_impl.ipynb`.

### Model training

Once the training datasets and the mean wing shape have been prepared, configure the training parameters in the following file:

`wings/modeling/training/unet_training.py`

This includes paths to the datasets, training hyperparameters, and output directories.

After setting the desired parameters, start the training by running:

`uv run wings/modeling/training/unet_training.py`

During training, the pipeline automatically monitors validation performance and saves the best-performing model checkpoints to the `models/` directory.

After training is complete, select the appropriate checkpoint file together with the computed mean wing shape and provide them to the application in the designated configuration locations.
These files are required for running inference and for correct landmark ordering during application execution.

---
