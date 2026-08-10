"""Build a train-only LongCLIP text retrieval index for RI-VerbalTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval import LongCLIPTextEmbedder, build_retrieval_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-path", default="./save/Longclip")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    embedder = LongCLIPTextEmbedder(args.model_path, args.device, args.batch_size)
    metadata = build_retrieval_index(
        dataset_folder=args.dataset_folder,
        dataset_name=args.dataset_name,
        output_path=args.output,
        embedder=embedder,
        embedding_model=args.model_path,
        build_params={"batch_size": args.batch_size, "device": args.device},
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved retrieval index to {args.output}")


if __name__ == "__main__":
    main()
