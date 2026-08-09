#!/usr/bin/env python3
"""Create the runtime resource layout without changing downloaded archives."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DATASETS = {
    "BlindWays": "BlindWays",
    "ETTm1": "ETTm1",
    "istanbul_traffic": "istanbul_traffic",
    "synthetic_m": "synth-m",
    "synthetic_u": "synth-u",
    "Weather": "Weather",
}

CHECKPOINTS = {
    "synthetic_m_cttp": "synth-m_cttp",
    "synthetic_m_eval": "synth-m_eval",
    "Weather_cttp": "Weather_cttp",
    "Weather_eval": "Weather_eval",
}

LONGCLIP_FILES = {
    "config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def find_resource(root: Path, name: str, marker: str) -> Path:
    matches = [
        path
        for path in root.rglob(name)
        if path.is_dir() and (path / marker).is_file()
    ]
    if len(matches) != 1:
        found = ", ".join(str(path) for path in matches) or "none"
        raise RuntimeError(
            f"Expected one '{name}' directory with '{marker}' under {root}; found {found}"
        )
    return matches[0]


def place_file(source: Path, target: Path, mode: str, force: bool, mutable: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not force:
            return "existing"
        target.unlink()

    if mode in {"auto", "hardlink"} and not mutable:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise

    shutil.copy2(source, target)
    return "copy"


def place_tree(
    source: Path,
    target: Path,
    mode: str,
    force: bool,
    include: set[str] | None = None,
) -> dict[str, int]:
    counts = {"hardlink": 0, "copy": 0, "existing": 0}
    for path in source.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(source)
        if include is not None and relative.as_posix() not in include:
            continue
        mutable = path.suffix.lower() in {".yaml", ".yml"}
        result = place_file(path, target / relative, mode, force, mutable)
        counts[result] += 1
    return counts


def patch_cttp_config(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.lstrip().startswith("pretrain_model_path:")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one pretrain_model_path entry in {path}")
    index = matches[0]
    indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    lines[index] = f"{indentation}pretrain_model_path: ./save/Longclip"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="VerbalTS repository root",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help="auto tries hard links and falls back to copies",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files already present in datasets/ and save/",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    downloads = root / "_downloads"
    if not downloads.is_dir():
        raise RuntimeError(f"Download directory does not exist: {downloads}")

    summaries: list[tuple[str, dict[str, int]]] = []
    for source_name, target_name in DATASETS.items():
        source = find_resource(downloads / "data", source_name, "meta.json")
        counts = place_tree(source, root / "datasets" / target_name, args.mode, args.force)
        summaries.append((f"datasets/{target_name}", counts))

    longclip_candidates = [
        path
        for path in downloads.glob("LongCLIP-GmP-ViT-L-14")
        if (path / "model.safetensors").is_file()
        and (path / "model.safetensors").stat().st_size > 1_000_000_000
    ]
    if len(longclip_candidates) != 1:
        raise RuntimeError("Could not identify the complete LongCLIP download")
    counts = place_tree(
        longclip_candidates[0],
        root / "save" / "Longclip",
        args.mode,
        args.force,
        LONGCLIP_FILES,
    )
    summaries.append(("save/Longclip", counts))

    checkpoint_root = downloads / "checkpoints"
    for source_name, target_name in CHECKPOINTS.items():
        marker = "clip_model_best.pth" if source_name.endswith("cttp") else "text2ts_msmdiffmv"
        if marker.endswith(".pth"):
            source = find_resource(checkpoint_root, source_name, marker)
        else:
            matches = [
                path
                for path in checkpoint_root.rglob(source_name)
                if path.is_dir() and (path / marker).is_dir()
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Could not uniquely identify checkpoint directory {source_name}")
            source = matches[0]
        counts = place_tree(source, root / "save" / target_name, args.mode, args.force)
        summaries.append((f"save/{target_name}", counts))

    for dataset in ("synth-m", "Weather"):
        patch_cttp_config(root / "save" / f"{dataset}_cttp" / "model_configs.yaml")

    (root / "cache").mkdir(exist_ok=True)

    print("Prepared runtime resources:")
    for name, counts in summaries:
        print(
            f"  {name}: {counts['hardlink']} hard links, "
            f"{counts['copy']} copies, {counts['existing']} existing"
        )
    print("Patched CTTP LongCLIP paths in operational config copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
