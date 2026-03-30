"""Download a small subset of DataComp-1B for testing UniTok training.

Usage:
    pip install img2dataset datasets pandas pyarrow
    python scripts/download_datacomp_small.py --output_dir data/datacomp_small --num_pairs 20000
"""
import os
import argparse
import subprocess


def download_metadata(num_pairs, output_dir):
    """Stream DataComp-1B metadata from HuggingFace, save first N rows as parquet."""
    import itertools
    import pandas as pd
    from datasets import load_dataset

    print(f"Streaming {num_pairs} metadata rows from DataComp-1B ...")
    ds = load_dataset("mlfoundations/datacomp_1b", streaming=True, split="train")
    rows = list(itertools.islice(ds, num_pairs))
    df = pd.DataFrame(rows)

    # img2dataset needs at minimum: url, caption
    # DataComp columns: url, text, ...
    metadata_path = os.path.join(output_dir, "metadata.parquet")
    df[["url", "text"]].to_parquet(metadata_path, index=False)
    print(f"Saved {len(df)} rows to {metadata_path}")
    return metadata_path


def download_images(metadata_path, output_dir, processes_count=4, thread_count=16):
    """Use img2dataset to download images into webdataset tar format."""
    shards_dir = os.path.join(output_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    print("Downloading images with img2dataset ...")
    cmd = [
        "img2dataset",
        "--url_list", metadata_path,
        "--input_format", "parquet",
        "--url_col", "url",
        "--caption_col", "text",
        "--output_format", "webdataset",
        "--output_folder", shards_dir,
        "--processes_count", str(processes_count),
        "--thread_count", str(thread_count),
        "--image_size", "288",          # slightly larger than 256 for crop
        "--resize_mode", "keep_ratio",
        "--encode_format", "jpg",
        "--encode_quality", "95",
        "--number_sample_per_shard", "1000",
        "--retries", "1",
    ]
    subprocess.run(cmd, check=True)

    # count resulting shards
    tars = sorted(f for f in os.listdir(shards_dir) if f.endswith(".tar"))
    total_size_mb = sum(os.path.getsize(os.path.join(shards_dir, f)) for f in tars) / 1e6
    print(f"\nDone. {len(tars)} shards, {total_size_mb:.0f} MB total in {shards_dir}/")
    print(f"Use --train_data '{shards_dir}/{{00000..{len(tars)-1:05d}}}.tar'")


def main():
    parser = argparse.ArgumentParser(description="Download small DataComp-1B subset")
    parser.add_argument("--output_dir", default="data/datacomp_small")
    parser.add_argument("--num_pairs", type=int, default=20000,
                        help="Number of URL-caption pairs to fetch metadata for. "
                             "Expect ~50%% success rate, so 20k metadata -> ~10k images -> ~1GB.")
    parser.add_argument("--processes_count", type=int, default=4)
    parser.add_argument("--thread_count", type=int, default=16)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    metadata_path = download_metadata(args.num_pairs, args.output_dir)
    download_images(metadata_path, args.output_dir, args.processes_count, args.thread_count)


if __name__ == "__main__":
    main()
