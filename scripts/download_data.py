"""
Download NYC Yellow Taxi trip data (24 months: Jan 2024 - Dec 2025).
Spec: Section 7.1 Phase 0, Lines 2818-2825

Downloads approximately 72M records (~8-10 GB compressed) from NYC TLC website.
"""

import requests
from pathlib import Path
from tqdm import tqdm
import sys


def download_file(url: str, output_path: Path) -> bool:
    """Download file with progress bar and verification.

    Args:
        url: Full URL to download
        output_path: Path where file will be saved

    Returns:
        True if download succeeded, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verify file is not empty
        if output_path.stat().st_size == 0:
            raise Exception(f"Empty file: {output_path.name}")

        return True
    except requests.exceptions.RequestException as e:
        print(f"Network error downloading {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False
    except IOError as e:
        print(f"IO error writing {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    """Download all NYC Yellow Taxi data files."""
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    # Download 24 months: Jan 2024 - Dec 2025
    for year in [2024, 2025]:
        for month in range(1, 13):
            filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
            url = f"{base_url}/{filename}"
            output_path = output_dir / filename

            # Skip if file already exists
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / 1e6
                print(f"✓ {filename} exists ({file_size_mb:.1f} MB)")
                success += 1
                continue

            print(f"⬇️  Downloading {filename}...")
            if download_file(url, output_path):
                file_size_mb = output_path.stat().st_size / 1e6
                success += 1
                print(f"✅ {filename} ({file_size_mb:.1f} MB)")
            else:
                failed.append(filename)
                print(f"❌ {filename}")

    # Summary
    total_size_gb = sum(f.stat().st_size for f in output_dir.glob("*.parquet")) / 1e9
    print(f"\n📊 Downloaded: {success}/24 files ({total_size_gb:.2f} GB)")

    if failed:
        print(f"❌ Failed downloads: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("✅ All files downloaded successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
