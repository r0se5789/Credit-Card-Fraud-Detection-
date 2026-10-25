"""Dataset downloader for Credit Card Fraud Detection."""

import os
import sys
import urllib.request
import time
from pathlib import Path
from typing import Optional
from src.config import RAW_DATA_PATH, DATASET_DOWNLOAD_URLS
from src.utils.logger import logger

def download_dataset(force: bool = False) -> Path:
    """Downloads the creditcard.csv dataset if not already present.
    
    Args:
        force: If True, re-downloads even if the file exists.
        
    Returns:
        Path to the downloaded raw dataset.
    """
    if RAW_DATA_PATH.exists() and not force:
        size_mb = RAW_DATA_PATH.stat().st_size / (1024 * 1024)
        if size_mb > 140:
            logger.info(f"✅ Dataset already exists at {RAW_DATA_PATH} ({size_mb:.2f} MB). Skipping download.")
            return RAW_DATA_PATH
        else:
            logger.warning(f"Existing file {RAW_DATA_PATH} appears incomplete ({size_mb:.2f} MB). Re-downloading...")

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = RAW_DATA_PATH.with_suffix(".tmp")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    success = False
    for url in DATASET_DOWNLOAD_URLS:
        logger.info(f"📥 Attempting download from: {url}")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB chunk
                start_time = time.time()
                last_log_time = start_time

                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        curr_time = time.time()

                        if curr_time - last_log_time > 2.0 or downloaded == total_size:
                            speed_mbps = (downloaded / (1024 * 1024)) / max(1e-3, curr_time - start_time)
                            pct = (downloaded / total_size * 100) if total_size else 0
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024) if total_size else 0
                            logger.info(
                                f"  Progress: {downloaded_mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%) "
                                f"@ {speed_mbps:.1f} MB/s"
                            )
                            last_log_time = curr_time

            # Rename temp to target
            if temp_path.exists():
                if RAW_DATA_PATH.exists():
                    RAW_DATA_PATH.unlink()
                temp_path.rename(RAW_DATA_PATH)

            final_mb = RAW_DATA_PATH.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Successfully downloaded dataset to {RAW_DATA_PATH} ({final_mb:.2f} MB)")
            success = True
            break

        except Exception as e:
            logger.error(f"Failed download from {url}: {e}")
            if temp_path.exists():
                temp_path.unlink()

    if not success:
        raise RuntimeError(
            "Failed to download creditcard.csv from all mirror sources. "
            "Please check network connection or place creditcard.csv manually in data/raw/"
        )

    # Basic integrity check
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().replace('"', '')
        expected_cols = ["Time", "V1", "V28", "Amount", "Class"]
        if not all(col in first_line for col in expected_cols):
            raise ValueError(f"Downloaded CSV header does not match expected schema: {first_line}")

    return RAW_DATA_PATH

if __name__ == "__main__":
    download_dataset()
