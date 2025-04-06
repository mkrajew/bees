"""
Configuration and environment setup for the project.

This module establishes key directory paths, device configuration, logging,
and constants used throughout the project. It ensures a reproducible and
organized development environment by:

- Defining project-relative directories (e.g., data, models, reports).
- Specifying a list of supported countries.
- Setting consistent suffixes for coordinate and image data files.
- Configuring the PyTorch device (CPU or CUDA) and enabling deterministic behavior.
- Setting float32 matrix multiplication precision for improved performance.
- Initializing GPU seeds to ensure reproducibility.
- Integrating loguru logging with tqdm (if available) for clean CLI output.

Attributes:
    PROJ_ROOT (Path): Root directory of the project.
    DATA_PATH (Path): Base data directory.
    RAW_DATA_DIR (Path): Directory for raw input data.
    INTERIM_DATA_DIR (Path): Directory for interim data storage.
    PROCESSED_DATA_DIR (Path): Directory for processed data.
    EXTERNAL_DATA_DIR (Path): Directory for externally sourced data.
    MODELS_DIR (Path): Directory where models are stored.
    MODELLING_DIR (Path): Directory for modeling scripts.
    REPORTS_DIR (Path): Directory for generated reports.
    FIGURES_DIR (Path): Directory for storing figures and plots.
    COUNTRIES (List[str]): List of country codes currently in use.
    COORDS_SUFX (str): Suffix used for raw coordinate CSV files.
    IMG_FOLDER_SUFX (str): Suffix used for wing image folders.
    DEVICE (torch.device): Selected PyTorch device (CPU or CUDA).
"""

from pathlib import Path

import torch
from loguru import logger

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_PATH = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_PATH / 'raw'
INTERIM_DATA_DIR = DATA_PATH / 'interim'
PROCESSED_DATA_DIR = DATA_PATH / 'processed'
EXTERNAL_DATA_DIR = DATA_PATH / 'external'

MODELS_DIR = PROJ_ROOT / 'models'
MODELLING_DIR = PROJ_ROOT / 'wings' / 'modeling'

REPORTS_DIR = PROJ_ROOT / 'reports'
FIGURES_DIR = PROJ_ROOT / 'figures'

# COUNTRIES = ['AT','ES','GR','HR','HU','MD','ME','PL','PT','RO','RS','SI']
COUNTRIES = ['AT', 'GR', 'HR', 'HU', 'MD', 'PL', 'RO', 'SI']

COORDS_SUFX = '-raw-coordinates.csv'
IMG_FOLDER_SUFX = '-wing-images'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"{torch.cuda.get_device_name()=}")
torch.set_float32_matmul_precision('high')

# GPU operations have a separate seed we also want to set
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
