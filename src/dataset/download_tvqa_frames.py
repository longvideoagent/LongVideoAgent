#!/usr/bin/env python3
"""Download selected TVQA frame archives from Hugging Face.

Default behavior downloads only `bbt_frames.tar.gz`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_REPO_ID = "Yuxuan0701/tvqa-frames"

# Dataset archives in the repo (6 files total):
# - bbt_frames.tar.gz
# - castle_frames.tar
# - friends_frames.tar
# - grey_frames.tar
# - house_frames.tar.gz
# - met_frames.tar.gz
SHOW_TO_FILE = {
    "bbt": "bbt_frames.tar.gz",
    "castle": "castle_frames.tar",
    "friends": "friends_frames.tar",
    "grey": "grey_frames.tar",
    "house": "house_frames.tar.gz",
    "met": "met_frames.tar.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download selected TVQA frame subsets from Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Local output directory",
    )
    parser.add_argument(
        "--show",
        type=str,
        default="bbt",
        choices=["bbt", "friends", "grey", "house", "met", "castle"],
        help="Which TV show subset to download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Full list has six archives; keep only bbt enabled for now.
    # files_to_download = [
    #     "bbt_frames.tar.gz",      # enabled below
    #     "castle_frames.tar",
    #     "friends_frames.tar",
    #     "grey_frames.tar",
    #     "house_frames.tar.gz",
    #     "met_frames.tar.gz",
    # ]
    selected_show = args.show
    selected_file = SHOW_TO_FILE[selected_show]

    local_path = hf_hub_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        filename=selected_file,
        local_dir=str(output_dir),
    )

    print(f"Downloaded show: {selected_show}")
    print(f"Downloaded file: {selected_file}")
    print(f"Repository: {args.repo_id}")
    print(f"Saved under: {local_path}")


if __name__ == "__main__":
    main()
