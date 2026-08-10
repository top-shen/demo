"""Relabel an existing train-only strength Teacher without rerunning diffusion."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.strength_teacher import (
    build_teacher_targets,
    estimate_copy_threshold,
    load_teacher_dataset,
    save_teacher_dataset,
    sha256_file,
)


RELABEL_SCHEMA_VERSION = "ri-verbalts-strength-teacher-relabel-v1"


def relabel_payload(payload, epsilon_sem, teacher_temperature, copy_threshold):
    """Return a copied payload with only feasible-set-derived labels replaced."""
    updated = {name: np.array(value, copy=True) for name, value in payload.items()}
    labels = build_teacher_targets(
        updated["candidate_strengths"],
        updated["candidate_semantic_scores"],
        updated["original_semantic_scores"],
        epsilon_sem=float(epsilon_sem),
        teacher_temperature=float(teacher_temperature),
        candidate_copy_distances=updated["candidate_copy_distances"],
        copy_threshold=copy_threshold,
    )
    updated.update(labels)
    return updated


def update_sweep_rows(manifest, payload, copy_threshold, output_path):
    fixed_sweep = copy.deepcopy(manifest.get("fixed_sweep", {}))
    rows = copy.deepcopy(fixed_sweep.get("rows", []))
    distances = np.asarray(payload["candidate_copy_distances"], dtype=np.float32)
    strengths = np.asarray(payload["candidate_strengths"], dtype=np.float32)
    for column, strength in enumerate(strengths):
        if column >= len(rows):
            break
        rows[column]["strength"] = float(strength)
        rows[column]["copy_rate"] = (
            None
            if copy_threshold is None
            else float((distances[:, column] < float(copy_threshold)).mean())
        )
    summary = {"rows": rows, "warning": fixed_sweep.get("warning")}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"path": str(output_path), **summary}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--input-manifest", default="")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-manifest", default="")

    epsilon = parser.add_mutually_exclusive_group(required=True)
    epsilon.add_argument("--epsilon-sem", type=float)
    epsilon.add_argument("--epsilon-relative-original", type=float)
    parser.add_argument("--epsilon-source", default="train-only relabel protocol")
    parser.add_argument("--teacher-temperature", type=float, default=None)

    copy_group = parser.add_mutually_exclusive_group(required=True)
    copy_group.add_argument("--copy-threshold", type=float)
    copy_group.add_argument("--copy-quantile", type=float)
    copy_group.add_argument("--disable-copy-constraint", action="store_true")
    parser.add_argument("--train-ts-path", default="")
    parser.add_argument("--copy-num-pairs", type=int, default=8192)
    parser.add_argument("--copy-seed", type=int, default=2025)
    parser.add_argument("--copy-source", default="train-only random-pair distance quantile")
    return parser


def main():
    args = build_parser().parse_args()
    input_npz = Path(args.input_npz)
    output_npz = Path(args.output_npz)
    if input_npz.resolve() == output_npz.resolve():
        raise ValueError("Relabel output must not overwrite the source Teacher NPZ")

    input_manifest = Path(args.input_manifest) if args.input_manifest else None
    output_manifest = (
        Path(args.output_manifest) if args.output_manifest else output_npz.with_suffix(".json")
    )
    payload, manifest = load_teacher_dataset(input_npz, input_manifest, for_training=True)

    original_mean = float(np.asarray(payload["original_semantic_scores"]).mean())
    if args.epsilon_relative_original is not None:
        if args.epsilon_relative_original < 0:
            raise ValueError("--epsilon-relative-original must be nonnegative")
        epsilon_sem = original_mean * float(args.epsilon_relative_original)
        epsilon_definition = {
            "relative_to_original_mean": float(args.epsilon_relative_original),
            "original_semantic_mean": original_mean,
        }
    else:
        if args.epsilon_sem is None or args.epsilon_sem < 0:
            raise ValueError("--epsilon-sem must be nonnegative")
        epsilon_sem = float(args.epsilon_sem)
        epsilon_definition = {"relative_to_original_mean": None}

    copy_estimation = None
    if args.disable_copy_constraint:
        copy_threshold = None
    elif args.copy_quantile is not None:
        if not args.train_ts_path:
            raise ValueError("--copy-quantile requires --train-ts-path")
        train_ts = np.load(args.train_ts_path, mmap_mode="r")
        copy_estimation = estimate_copy_threshold(
            train_ts,
            quantile=args.copy_quantile,
            num_pairs=args.copy_num_pairs,
            seed=args.copy_seed,
        )
        copy_threshold = float(copy_estimation["threshold"])
    else:
        if args.copy_threshold is None or args.copy_threshold < 0:
            raise ValueError("--copy-threshold must be nonnegative")
        copy_threshold = float(args.copy_threshold)

    teacher_temperature = float(
        args.teacher_temperature
        if args.teacher_temperature is not None
        else manifest.get("teacher_temperature", 0.10)
    )
    updated = relabel_payload(
        payload,
        epsilon_sem=epsilon_sem,
        teacher_temperature=teacher_temperature,
        copy_threshold=copy_threshold,
    )

    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["parent_teacher"] = {
        "data_file": str(input_npz),
        "manifest_file": str(
            input_manifest if input_manifest is not None else input_npz.with_suffix(".json")
        ),
        "data_sha256": sha256_file(input_npz),
    }
    updated_manifest["semantic_threshold"] = {
        "epsilon_sem": epsilon_sem,
        "absolute_value": epsilon_sem,
        "source": args.epsilon_source,
        **epsilon_definition,
    }
    updated_manifest["copy_constraint"] = {
        "enabled": copy_threshold is not None,
        "metric": "dimension_normalized_rmse",
        "threshold": copy_threshold,
        "source": args.copy_source,
        "estimation": copy_estimation,
    }
    updated_manifest["teacher_temperature"] = teacher_temperature
    updated_manifest["relabeling"] = {
        "schema_version": RELABEL_SCHEMA_VERSION,
        "generation_reused": True,
        "diffusion_rerun": False,
    }
    sweep_path = output_npz.with_name(output_npz.stem + "_fixed_sweep.json")
    updated_manifest["fixed_sweep"] = update_sweep_rows(
        updated_manifest, updated, copy_threshold, sweep_path
    )
    save_teacher_dataset(output_npz, output_manifest, updated, updated_manifest)

    gate = np.asarray(updated["gate_targets"], dtype=np.float32)
    split = np.asarray(updated["controller_split_ids"], dtype=np.int8)
    print(f"Relabeled Teacher: {output_npz}")
    print(f"epsilon_sem={epsilon_sem:.8f} copy_threshold={copy_threshold}")
    for split_id, name in ((0, "controller-train"), (1, "controller-validation")):
        mask = split == split_id
        print(
            f"{name}: rows={int(mask.sum())} "
            f"positive={int(gate[mask].sum())} "
            f"negative={int((gate[mask] == 0).sum())}"
        )


if __name__ == "__main__":
    main()
