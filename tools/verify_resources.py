#!/usr/bin/env python3
"""Validate the prepared VerbalTS datasets, LongCLIP, and released checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np


DATASETS = {
    "BlindWays": ((823, 103, 103), (600, 72), 1),
    "ETTm1": ((13013, 1631, 1631), (120, 1), 1),
    "istanbul_traffic": ((8178, 1023, 1023), (144, 1), 1),
    "synth-m": ((24000, 4000, 4000), (128, 2), 1),
    "synth-u": ((24000, 4000, 4000), (128, 1), 1),
    "Weather": ((10192, 1460, 1448), (36, 21), 3),
}

LONGCLIP_SHA256 = "8e259dc8f7cef4b289d0c4a84667b310b92060addc590e4df2d227ff7029beaf"
LONGCLIP_SIZE = 1_711_063_028
CHECKPOINT_SIZES = {
    "save/synth-m_cttp/clip_model_best.pth": 505_055_758,
    "save/Weather_cttp/clip_model_best.pth": 505_058_318,
    "save/synth-m_eval/text2ts_msmdiffmv/0/ckpts/model_best_loss.pth": 18_829_012,
    "save/synth-m_eval/text2ts_msmdiffmv/1/ckpts/model_best_loss.pth": 18_829_012,
    "save/synth-m_eval/text2ts_msmdiffmv/2/ckpts/model_best_loss.pth": 18_829_012,
    "save/Weather_eval/text2ts_msmdiffmv/0/ckpts/model_best_loss.pth": 16_259_092,
    "save/Weather_eval/text2ts_msmdiffmv/1/ckpts/model_best_loss.pth": 16_259_092,
    "save/Weather_eval/text2ts_msmdiffmv/2/ckpts/model_best_loss.pth": 16_259_092,
}


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def pass_(self, message: str) -> None:
        print(f"PASS  {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"WARN  {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"FAIL  {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def all_finite(array: np.ndarray) -> bool:
    samples_per_chunk = max(1, 5_000_000 // int(np.prod(array.shape[1:])))
    return all(
        np.isfinite(array[start : start + samples_per_chunk]).all()
        for start in range(0, array.shape[0], samples_per_chunk)
    )


def validate_dataset(root: Path, name: str, expected: tuple, deep: bool, report: Report) -> None:
    counts, sample_shape, captions_per_sample = expected
    folder = root / "datasets" / name
    if not folder.is_dir():
        report.fail(f"Missing dataset directory: {folder}")
        return

    meta_path = folder / "meta.json"
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if not {"attr_list", "attr_n_ops"}.issubset(meta):
            raise ValueError("missing attr_list or attr_n_ops")
    except Exception as exc:
        report.fail(f"{name}: invalid meta.json ({exc})")
        return

    for split, expected_count in zip(("train", "valid", "test"), counts):
        paths = {
            "ts": folder / f"{split}_ts.npy",
            "attrs": folder / f"{split}_attrs_idx.npy",
            "caps": folder / f"{split}_text_caps.npy",
        }
        if any(not path.is_file() for path in paths.values()):
            report.fail(f"{name}/{split}: one or more arrays are missing")
            continue
        try:
            ts = np.load(paths["ts"], mmap_mode="r", allow_pickle=False)
            attrs = np.load(paths["attrs"], mmap_mode="r", allow_pickle=False)
            caps = np.load(paths["caps"], mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            report.fail(f"{name}/{split}: cannot load arrays without pickle ({exc})")
            continue

        expected_ts_shape = (expected_count, *sample_shape)
        if ts.shape != expected_ts_shape:
            report.fail(f"{name}/{split}: TS shape {ts.shape}, expected {expected_ts_shape}")
        elif attrs.shape[0] != expected_count or caps.shape[0] != expected_count:
            report.fail(f"{name}/{split}: sample counts do not align")
        elif caps.ndim != 2 or caps.shape[1] != captions_per_sample:
            report.fail(
                f"{name}/{split}: caption shape {caps.shape}, expected "
                f"({expected_count}, {captions_per_sample})"
            )
        elif caps.dtype.kind not in {"U", "S"}:
            report.fail(f"{name}/{split}: captions are not native string arrays")
        else:
            report.pass_(f"{name}/{split}: shapes and text structure")

        if deep:
            if all_finite(ts):
                report.pass_(f"{name}/{split}: all TS values are finite")
            else:
                report.fail(f"{name}/{split}: TS contains NaN or Inf")
            if np.all(np.char.str_len(np.asarray(caps).astype(str)) > 0):
                report.pass_(f"{name}/{split}: captions are nonempty")
            else:
                report.fail(f"{name}/{split}: empty caption found")

        if name == "ETTm1" and split == "train" and np.min(attrs) < 0:
            report.warn("ETTm1/train: negative attribute values exist at released indices 504-507")


def validate_longclip(root: Path, deep: bool, report: Report) -> None:
    folder = root / "save" / "Longclip"
    required = {
        "config.json",
        "merges.txt",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    missing = sorted(name for name in required if not (folder / name).is_file())
    if missing:
        report.fail(f"LongCLIP files missing: {', '.join(missing)}")
        return
    model = folder / "model.safetensors"
    if model.stat().st_size != LONGCLIP_SIZE:
        report.fail(f"LongCLIP size is {model.stat().st_size}, expected {LONGCLIP_SIZE}")
    else:
        report.pass_("LongCLIP file set and model size")
    if deep:
        digest = sha256(model)
        if digest == LONGCLIP_SHA256:
            report.pass_("LongCLIP SHA-256")
        else:
            report.fail(f"LongCLIP SHA-256 mismatch: {digest}")


def validate_checkpoints(root: Path, deep: bool, report: Report) -> None:
    for relative, expected_size in CHECKPOINT_SIZES.items():
        path = root / relative
        if not path.is_file():
            report.fail(f"Missing checkpoint: {relative}")
            continue
        if path.stat().st_size != expected_size:
            report.fail(f"{relative}: size mismatch")
            continue
        if deep:
            if not zipfile.is_zipfile(path):
                report.fail(f"{relative}: not a PyTorch ZIP checkpoint")
                continue
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
            if bad_member is not None:
                report.fail(f"{relative}: CRC failure in {bad_member}")
                continue
        report.pass_(f"{relative}: size" + (" and CRC" if deep else ""))

    for dataset in ("synth-m", "Weather"):
        config_path = root / "save" / f"{dataset}_cttp" / "model_configs.yaml"
        try:
            values = [
                line.split(":", 1)[1].strip().strip("'\"")
                for line in config_path.read_text(encoding="utf-8").splitlines()
                if line.lstrip().startswith("pretrain_model_path:")
            ]
            if len(values) != 1:
                raise ValueError("expected one pretrain_model_path entry")
            configured_path = values[0]
        except Exception as exc:
            report.fail(f"{dataset} CTTP config is invalid ({exc})")
            continue
        if configured_path != "./save/Longclip":
            report.fail(f"{dataset} CTTP LongCLIP path is still {configured_path!r}")
        else:
            report.pass_(f"{dataset} CTTP LongCLIP path")

    report.warn("Synth-M CTTP declares n_var=4 while the released dataset has 2 variables")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="VerbalTS repository root",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also scan all values, captions, LongCLIP hash, and checkpoint CRCs",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    report = Report()
    for name, expected in DATASETS.items():
        validate_dataset(root, name, expected, args.deep, report)
    validate_longclip(root, args.deep, report)
    validate_checkpoints(root, args.deep, report)

    print(f"\nSummary: {report.failures} failures, {report.warnings} advisories")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
