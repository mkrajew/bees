import os
import random

from wings.app import countries
from wings.app.utils import input_image_time
from wings.config import RAW_DATA_DIR

directory = RAW_DATA_DIR
all_files = []
for subdir, _, files in os.walk(directory):
    if True:
        for file in files:
            if file.lower().endswith('.png') and file.split("-", 1)[0] in countries:
                all_files.append(os.path.join(subdir, file))


def test_benchmark_input_images(benchmark):
    filepath = random.choice(all_files)
    coords = benchmark(input_image_time, filepath)
    assert len(coords) > 0