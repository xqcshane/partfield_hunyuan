#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="PartField/model/model_objaverse.ckpt")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id="mikaelaangel/partfield-ckpt",
        filename="model_objaverse.ckpt",
    )
    shutil.copy2(downloaded, output)
    print(output)


if __name__ == "__main__":
    main()
