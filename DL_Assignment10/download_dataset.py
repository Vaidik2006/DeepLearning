"""Download and extract the public ESC-50 dataset."""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import shutil

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / "ESC-50"
ZIP_PATH = DATA_DIR / "esc50.zip"
URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (TARGET / "meta" / "esc50.csv").exists() and (TARGET / "audio").exists():
        print(f"ESC-50 already exists at: {TARGET}")
        return
    print("Downloading ESC-50 dataset...")
    urlretrieve(URL, ZIP_PATH)
    with ZipFile(ZIP_PATH) as archive:
        archive.extractall(DATA_DIR)
    extracted = DATA_DIR / "ESC-50-master"
    if TARGET.exists():
        shutil.rmtree(TARGET)
    extracted.rename(TARGET)
    ZIP_PATH.unlink(missing_ok=True)
    print(f"Dataset ready: {TARGET}")


if __name__ == "__main__":
    main()
