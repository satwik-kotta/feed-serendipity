"""Download KuaiRec v2 from Zenodo and unpack it into data/KuaiRec/.

Usage
    python data/download.py            # download + unzip + convert matrices to parquet
    python data/download.py --no-parquet

Source: Gao et al., KuaiRec, CIKM 2022. License CC BY-SA 4.0.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
DATA_DIR = Path(__file__).resolve().parent
ZIP_PATH = DATA_DIR / "KuaiRec.zip"
OUT_DIR = DATA_DIR / "KuaiRec"


def _progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done / total_size * 100
    sys.stdout.write(f"\r  {done / 1e6:,.0f} / {total_size / 1e6:,.0f} MB ({pct:4.1f}%)")
    sys.stdout.flush()


def download():
    if ZIP_PATH.exists():
        print(f"Zip already present at {ZIP_PATH}")
        return
    print(f"Downloading {URL}")
    tmp = ZIP_PATH.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp, _progress)
    tmp.rename(ZIP_PATH)
    print("\nDone")


def unzip():
    if (OUT_DIR / "data" / "big_matrix.csv").exists():
        print(f"Already unpacked in {OUT_DIR}")
        return
    print("Unzipping")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)
    # The archive unpacks to KuaiRec 2.0/ in some releases. Normalise the folder name.
    for cand in DATA_DIR.iterdir():
        if cand.is_dir() and cand.name.startswith("KuaiRec") and cand != OUT_DIR:
            shutil.move(str(cand), str(OUT_DIR))
    print(f"Unpacked to {OUT_DIR}")


def to_parquet():
    """Keep only the columns the project uses and store compact dtypes.
    Uses the pyarrow CSV reader, which handles the 1 GB big matrix in well under a minute."""
    import pyarrow as pa
    import pyarrow.csv as pv
    import pyarrow.parquet as pq

    cols = {"user_id": pa.int32(), "video_id": pa.int32(), "play_duration": pa.int64(),
            "video_duration": pa.int64(), "timestamp": pa.float64(), "watch_ratio": pa.float32()}
    src = OUT_DIR / "data"
    for name in ["big_matrix", "small_matrix"]:
        out = src / f"{name}.parquet"
        if out.exists():
            continue
        print(f"Converting {name}.csv to parquet")
        table = pv.read_csv(src / f"{name}.csv",
                            convert_options=pv.ConvertOptions(include_columns=list(cols),
                                                              column_types=cols))
        pq.write_table(table, out)
        print(f"  {table.num_rows:,} rows")
    print("Parquet ready")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-parquet", action="store_true")
    args = ap.parse_args()
    download()
    unzip()
    if not args.no_parquet:
        to_parquet()
