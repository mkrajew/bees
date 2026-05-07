"""Download Zenodo dataset zip and extract it (including inner zip files)."""

from pathlib import Path
import urllib.request
import zipfile

import typer
from loguru import logger
from tqdm import tqdm

PROJ_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJ_ROOT / "data" / "raw"
DATASET_URL = "https://zenodo.org/api/records/7244070/files-archive"
ARCHIVE_NAME = "zenodo-7244070-files.zip"
USER_AGENT = "WingAI-dataset-downloader/1.0"
CHUNK_SIZE = 8 * 1024 * 1024

app = typer.Typer(add_completion=False)


def _download_zip(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    with urllib.request.urlopen(request) as response, destination.open("wb") as file_out:
        total_size = int(response.headers.get("Content-Length", 0) or 0)
        with tqdm(
            total=total_size or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Download",
        ) as progress:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                file_out.write(chunk)
                progress.update(len(chunk))


def _extract_zip(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = archive.infolist()
        total_uncompressed = sum(member.file_size for member in members)
        with tqdm(
            total=total_uncompressed or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Extract {zip_path.name}",
        ) as progress:
            for member in members:
                archive.extract(member, output_dir)
                progress.update(member.file_size)


def _extract_nested_zips(output_dir: Path, skip: set[Path] | None = None) -> None:
    processed = {path.resolve() for path in (skip or set())}
    while True:
        zip_files = sorted(
            path for path in output_dir.rglob("*.zip") if path.resolve() not in processed
        )
        if not zip_files:
            break
        for zip_path in zip_files:
            logger.info(f"Extracting nested zip: {zip_path}")
            _extract_zip(zip_path, zip_path.parent)
            processed.add(zip_path.resolve())


@app.command()
def main(
    output_dir: Path = typer.Option(
        RAW_DATA_DIR,
        "--output-dir",
        "-o",
        help="Directory where dataset should be downloaded and extracted.",
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME

    logger.info(f"Downloading dataset from {DATASET_URL}")
    _download_zip(DATASET_URL, archive_path)
    logger.info(f"Downloaded archive: {archive_path}")

    logger.info(f"Extracting archive: {archive_path}")
    _extract_zip(archive_path, output_dir)
    _extract_nested_zips(output_dir, skip={archive_path})

    logger.info(f"Done. Dataset extracted to: {output_dir}")


if __name__ == "__main__":
    app()
