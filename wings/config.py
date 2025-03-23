from pathlib import Path

from loguru import logger
import torch

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
COUNTRIES = ['AT','GR','HR','HU','MD','PL','RO','SI']

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