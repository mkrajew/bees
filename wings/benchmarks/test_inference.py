"""End-to-end inference benchmark with pytest-benchmark: image file -> ordered
landmarks, per checkpoint, device and GPA setting.

Replaces the benchmark that lived in wings/app/test_benchmark.py (removed with
the Gradio prototype). Differences from that one are deliberate:

- it cycles over a random sample of real (YOLO-cropped) images instead of
  repeating one random image: the GPA step gets much slower when the model
  detects extra or missing points, so a single image can be unrepresentative.
  Like the old benchmark, the sample is drawn from the countries the models
  were trained on (wings.config.COUNTRIES): the cropped dataset also holds
  other countries, whose images more often give a wrong number of points and
  then dominate the Max / StdDev columns with slow orderings;
- it times the model pipeline only (decode -> preprocess -> UNet -> mask ->
  coordinates -> GPA ordering); the old one also re-read the image and built
  the Gradio UI's per-landmark overlay masks;
- the GPA settings are explicit. "full" is what the web app runs in production
  (reflection, 8 start angles, PCA pre-alignment: needed for rotated or
  mirrored photos); "basic" is plain handle_coordinates(), which is what the
  old benchmark ran.

Edit CHECKPOINTS to benchmark other models. Run (rounds are pytest-benchmark's
own option; the column/time-unit defaults are set in pyproject.toml):

    uv run pytest --benchmark-min-rounds=1000

Select a subset with -k, e.g. the production GPA settings on GPU only:

    uv run pytest --benchmark-min-rounds=1000 -k "full and cuda"

--benchmark-json=reports/benchmark.json also stores each case's mean time per
pipeline stage (extra_info), which the table doesn't show.
"""

import itertools
import random
import time
from functools import partial

import pytest
import torch

from wings.config import COUNTRIES, MODELS_DIR, PROCESSED_DATA_DIR
from wings.gpa import FULL_ROTATION_MULTISTART_ANGLES, handle_coordinates
from wings.modeling.litnet import LitNet
from wings.modeling.loss import BCEDiceLoss
from wings.modeling.unet import UNet
from wings.utils import load_image
from wings.visualizing.image_preprocess import final_coords, unet_fit_rectangle_preprocess

CHECKPOINTS = [
    MODELS_DIR / "final" / "final-precise.ckpt",
    MODELS_DIR / "final" / "final-rotation-2.ckpt",
]
N_IMAGES = 100  # size of the random sample of real wings (from COUNTRIES) that is cycled over
WARMUP_CALLS = 5  # CUDA init, cuDNN autotune and lazy imports, before pytest-benchmark's own calibration

GPA_SETTINGS = {
    "full": dict(allow_reflection=True, multistart_angles=FULL_ROTATION_MULTISTART_ANGLES, pca_prealign=True),
    "basic": {},
}
STAGES = ("decode + preprocess", "model", "mask -> coords", "GPA ordering")


@pytest.fixture(scope="module")
def files():
    folder = PROCESSED_DATA_DIR / "cropped"
    all_files = sorted(
        p
        for p in folder.rglob("*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and p.name.split("-", 1)[0] in COUNTRIES
    )
    return random.Random(0).sample(all_files, N_IMAGES)


@pytest.fixture(scope="module")
def mean_coords():
    return torch.load(PROCESSED_DATA_DIR / "mask_datasets" / "rectangle" / "mean_shape.pth", weights_only=False)


def load_model(checkpoint, device):
    unet = UNet(in_channels=1, out_channels=1, kernel_size=5, sigmoid=False)
    lit_net = LitNet.load_from_checkpoint(
        checkpoint, model=unet, criterion=BCEDiceLoss(), strict=False, map_location="cpu"
    )
    return lit_net.model.eval().to(device)


def process(path, model, device, mean_coords, preprocess, gpa_kwargs):
    """One image the way production does it. Returns the seconds spent in each
    of STAGES, and the 19 ordered landmarks."""
    t0 = time.perf_counter()
    image, x_size, y_size = load_image(path, preprocess)
    t1 = time.perf_counter()

    with torch.inference_mode():
        probs = torch.sigmoid(model(image.to(device).unsqueeze(0)))
        mask = (probs > 0.5).float().squeeze().cpu().numpy()  # .cpu() also waits for the GPU
    t2 = time.perf_counter()

    coords = torch.tensor(final_coords(mask, x_size, y_size), dtype=torch.float32)
    t3 = time.perf_counter()

    ordered = handle_coordinates(coords, mean_coords, **gpa_kwargs)
    t4 = time.perf_counter()

    return (t1 - t0, t2 - t1, t3 - t2, t4 - t3), ordered


@pytest.mark.parametrize("gpa", list(GPA_SETTINGS))
@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("checkpoint", CHECKPOINTS, ids=lambda p: p.stem)
def test_inference(benchmark, files, mean_coords, checkpoint, device, gpa):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint} not found")

    benchmark.group = f"{'GPU' if device == 'cuda' else 'CPU'}, GPA {gpa}"
    model = load_model(checkpoint, torch.device(device))
    preprocess = partial(unet_fit_rectangle_preprocess, output_size=400)
    images = itertools.cycle(files)
    stage_seconds = []

    def run():
        stages, ordered = process(next(images), model, device, mean_coords, preprocess, GPA_SETTINGS[gpa])
        stage_seconds.append(stages)
        return ordered

    for _ in range(WARMUP_CALLS):
        run()
    stage_seconds.clear()

    ordered = benchmark(run)

    assert ordered.shape == (19, 2)
    means_ms = torch.tensor(stage_seconds).mean(dim=0) * 1000
    benchmark.extra_info.update({f"{name} (mean ms)": round(m, 2) for name, m in zip(STAGES, means_ms.tolist())})
