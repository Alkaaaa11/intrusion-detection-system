"""
download_dataset.py
-------------------
Download the NSL-KDD raw dataset files used by this project.

The script tries multiple public mirrors because large dataset downloads can
occasionally fail or time out depending on the network.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


DATA_DIR = Path("data")

DATASET_FILES = {
    "KDDTrain+.txt": [
        "https://zenodo.org/records/17424143/files/KDDTrain%2B.txt?download=1",
        "https://raw.githubusercontent.com/arjbah/nsl-kdd/master/nsl-kdd/KDDTrain%2B.txt",
    ],
    "KDDTest+.txt": [
        "https://zenodo.org/records/17424143/files/KDDTest%2B.txt?download=1",
        "https://raw.githubusercontent.com/arjbah/nsl-kdd/master/nsl-kdd/KDDTest%2B.txt",
    ],
}


def download_file(url: str, destination: Path, timeout: int = 60) -> None:
    """
    Download a file in chunks. Resume if a partial file already exists.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0

    headers = {"User-Agent": "ids-dataset-downloader/1.0"}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    request = Request(url, headers=headers)
    mode = "ab" if existing_size else "wb"

    print(f"[INFO] Downloading: {url}")
    if existing_size:
        print(f"[INFO] Resuming from {existing_size:,} bytes")

    with urlopen(request, timeout=timeout) as response, destination.open(mode) as output:
        total_header = response.headers.get("Content-Length")
        total_remaining = int(total_header) if total_header else None
        downloaded = existing_size

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)

            if total_remaining:
                expected_total = existing_size + total_remaining
                percent = downloaded / expected_total * 100
                print(f"\r[INFO] {destination.name}: {percent:6.2f}% ({downloaded:,} bytes)", end="")
            else:
                print(f"\r[INFO] {destination.name}: {downloaded:,} bytes", end="")

    print()


def download_with_mirrors(filename: str, urls: list[str]) -> None:
    """
    Try each mirror until one succeeds.
    """
    destination = DATA_DIR / filename

    for url in urls:
        try:
            download_file(url, destination)
            if destination.exists() and destination.stat().st_size > 0:
                print(f"[OK] Saved {filename} to {destination}")
                return
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            print(f"[WARN] Failed from mirror: {error}")

    raise RuntimeError(f"Could not download {filename} from any configured mirror.")


def main() -> None:
    """
    Download all required raw NSL-KDD files.
    """
    for filename, urls in DATASET_FILES.items():
        download_with_mirrors(filename, urls)

    print("[DONE] Dataset download complete.")
    print("[NEXT] Run: python preprocessing.py")


if __name__ == "__main__":
    main()

