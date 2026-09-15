"""Download a reproducible local BGE reranker snapshot before starting the API."""

import argparse
from pathlib import Path
import time

from huggingface_hub import snapshot_download
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", default="/models/bge-reranker-v2-m3")
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("--attempts must be a positive integer")
    destination = Path(args.local_dir).expanduser()
    for attempt in range(1, args.attempts + 1):
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                revision=MODEL_REVISION,
                local_dir=str(destination),
                allow_patterns=MODEL_FILES,
                max_workers=2,
            )
            break
        except (ChunkedEncodingError, ConnectionError, Timeout):
            if attempt == args.attempts:
                raise
            print(f"Download interrupted; resuming (attempt {attempt + 1}/{args.attempts})", flush=True)
            time.sleep(min(attempt * 2, 10))
    missing = [name for name in MODEL_FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete reranker download: {missing}")
    print(f"Downloaded {MODEL_ID}@{MODEL_REVISION} to {destination}")


if __name__ == "__main__":
    main()
