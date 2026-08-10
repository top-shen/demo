"""Build train-only leave-one-series-out adaptive-strength teacher data.

The default is a 32-query pilot.  A complete build requires the explicit
``--allow-full-build`` flag, so invoking this tool never accidentally starts a
multi-day diffusion sweep.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retrieval.adaptive_strength_controller import (
    SCORE_FEATURE_NAMES,
    compute_pair_features,
    compute_score_features,
    fit_score_normalization,
    fit_similarity_quantiles,
    normalize_score_features,
    strength_to_start_steps,
)
from retrieval.strength_teacher import (
    CTTPTeacherSemanticScorer,
    PrecomputedTeacherSemanticScorer,
    build_teacher_targets,
    generated_to_reference_distance,
    grouped_controller_split,
    save_teacher_dataset,
    sha256_file,
)
from retrieval.text_retriever import LongCLIPTextEmbedder, TextRetriever


DEFAULT_GRID = "0.20,0.35,0.50,0.65,0.80,0.95"


def parse_strength_grid(value: str) -> np.ndarray:
    strengths = np.asarray([float(item) for item in value.split(",")], dtype=np.float32)
    if strengths.size < 2 or np.any((strengths < 0) | (strengths > 1)):
        raise ValueError("strength-grid needs at least two values in [0,1]")
    if np.any(np.diff(strengths) <= 0):
        raise ValueError("strength-grid values must be unique and increasing")
    return strengths


def flatten_train_queries(dataset_folder):
    folder = Path(dataset_folder)
    captions = np.load(folder / "train_text_caps.npy", allow_pickle=True)
    if captions.ndim == 1:
        captions = captions[:, None]
    train_ts = np.asarray(np.load(folder / "train_ts.npy"), dtype=np.float32)
    if captions.ndim != 2 or captions.shape[0] != train_ts.shape[0]:
        raise ValueError("train captions/time series have incompatible shapes")
    sample_ids = np.repeat(np.arange(captions.shape[0], dtype=np.int64), captions.shape[1])
    caption_ids = np.tile(np.arange(captions.shape[1], dtype=np.int64), captions.shape[0])
    return captions.reshape(-1).astype(str), sample_ids, caption_ids, train_ts


def stratified_query_indices(sample_ids, max_queries: int, seed: int):
    """Sample evenly across sample-ID strata, then shuffle deterministically."""
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    if max_queries < 0 or max_queries >= sample_ids.size:
        return np.arange(sample_ids.size, dtype=np.int64)
    if max_queries == 0:
        raise ValueError("max_queries must be positive, or -1 with --allow-full-build")
    rng = np.random.default_rng(int(seed))
    order = np.argsort(sample_ids, kind="stable")
    strata = np.array_split(order, min(max_queries, order.size))
    chosen = np.asarray([rng.choice(group) for group in strata if group.size], dtype=np.int64)
    rng.shuffle(chosen)
    return chosen[:max_queries]


def _part_path(cache_dir: Path, sample_id: int, caption_id: int) -> Path:
    return cache_dir / f"sample_{int(sample_id):08d}_caption_{int(caption_id):03d}.npz"


def _standardize_single_ts(values):
    """Return one time series in canonical [L,V] layout."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected one time series with shape [L] or [L,V], got {values.shape}")
    return values


def _make_generation_batch(torch, caption, sample_id, caption_id, target_ts, device):
    target = _standardize_single_ts(target_ts)[None, ...]
    length = target.shape[1]
    return {
        "ts": torch.as_tensor(target, dtype=torch.float32),
        "tp": torch.arange(length, dtype=torch.float32).reshape(1, length),
        "ts_len": torch.tensor([length], dtype=torch.int32),
        "cap": [str(caption)],
        "sample_id": torch.tensor([int(sample_id)], dtype=torch.long),
        "caption_id": torch.tensor([int(caption_id)], dtype=torch.long),
    }


def _aggregate_generation(torch, output):
    # [S,B,V,L] -> [B,L,V], retaining the repository's pointwise median.
    return output.median(dim=0).values.permute(0, 2, 1).detach().cpu().numpy()


def prepare_generation_configs(diff, cond, args):
    """Apply the same runtime config injections as ``run.py``."""
    diff = copy.deepcopy(diff)
    cond = copy.deepcopy(cond)
    diff["device"] = args.device
    diff.setdefault("generator_pretrain_path", "")
    cond["device"] = args.device
    cond["cond_modal"] = args.cond_modal
    cond["text"]["device"] = args.device
    cond["text"]["output_type"] = args.text_output_type
    cond["text"]["pos_emb"] = args.text_pos_emb
    if args.base_patch is not None:
        diff["diffusion"]["base_patch"] = args.base_patch
    if args.multipatch_num is not None:
        diff["diffusion"]["multipatch_num"] = args.multipatch_num
    if args.patch_length is not None:
        diff["diffusion"]["L_patch_len"] = args.patch_length
    if args.diff_stage_num is not None:
        cond["text"]["num_stages"] = args.diff_stage_num
    return diff, cond


def _load_generation_stack(args):
    import torch
    import yaml

    from models.conditional_generator import ConditionalGenerator

    raw_diff = yaml.safe_load(Path(args.diff_config).read_text(encoding="utf-8"))
    raw_cond = yaml.safe_load(Path(args.cond_config).read_text(encoding="utf-8"))
    diff, cond = prepare_generation_configs(raw_diff, raw_cond, args)
    model = ConditionalGenerator(diff, cond)
    state = torch.load(args.verbalts_checkpoint, map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.to(args.device)
    model.eval()
    model.requires_grad_(False)
    return torch, model


def _load_scorer(args):
    if args.semantic_scorer == "precomputed":
        with np.load(args.precomputed_scores, allow_pickle=False) as archive:
            if "scores" not in archive:
                raise ValueError("precomputed score NPZ must contain 'scores'")
            return PrecomputedTeacherSemanticScorer(archive["scores"])
    import torch
    import yaml
    from models.cttp.cttp_model import CTTP

    config = yaml.safe_load(Path(args.cttp_config).read_text(encoding="utf-8"))
    model = CTTP(config)
    model.load_state_dict(torch.load(args.cttp_checkpoint, map_location=args.device))
    model = model.to(model.device)
    return CTTPTeacherSemanticScorer(model)


def _score_and_embeddings(scorer, captions, generated, lengths):
    if hasattr(scorer, "embed"):
        ts_embedding, text_embedding = scorer.embed(captions, generated, lengths)
        score = (ts_embedding * text_embedding).sum(axis=-1).astype(np.float32)
        return score, ts_embedding, text_embedding
    return scorer.score(captions, generated, lengths), None, None


def generate_query_part(
    args,
    torch,
    model,
    scorer,
    retriever,
    caption,
    sample_id,
    caption_id,
    target_ts,
    strengths,
):
    search = retriever.search([caption], args.top_k, exclude_sample_ids=[[sample_id]])
    selected = retriever.select(
        search,
        [caption],
        [sample_id],
        selection="top1",
        min_similarity=args.min_similarity,
    )
    if selected[0]["fallback"] or selected[0]["reference_ts"] is None:
        raise RuntimeError(
            f"Training query {sample_id}/{caption_id} has no usable leave-one-series-out reference"
        )
    result = selected[0]
    top_rows = np.asarray(search["rows"][0], dtype=np.int64)
    query_embedding = np.asarray(search["query_embeddings"][0], dtype=np.float32)
    reference_embedding = np.asarray(retriever.embeddings[top_rows[0]], dtype=np.float32)
    reference_ts = _standardize_single_ts(result["reference_ts"])
    batch = _make_generation_batch(torch, caption, sample_id, caption_id, target_ts, args.device)
    length = np.asarray([reference_ts.shape[0]], dtype=np.int32)
    base_seed = int(args.seed) * 1000003 + int(sample_id) * 101 + int(caption_id)

    original_config = dict(model.rag_config)
    original_retriever = model.rag_retriever
    model.rag_retriever = retriever
    candidate_scores = []
    candidate_distances = []
    candidate_embeddings = []
    text_embedding = None
    try:
        model.rag_config = {"enabled": False}
        torch.manual_seed(base_seed & 0x7FFFFFFF)
        original = _aggregate_generation(
            torch, model.generate_text(batch, args.samples_per_action, args.sampler)
        )
        original_score, original_embedding, text_embedding = _score_and_embeddings(
            scorer, [caption], original, length
        )
        real_embedding = None
        if hasattr(scorer, "embed"):
            real_embedding, _ = scorer.embed(
                [caption], _standardize_single_ts(target_ts)[None, ...], length
            )

        common = {
            "enabled": True,
            "top_k": args.top_k,
            "selection": "top1",
            "temperature": 1.0,
            "seed": args.seed,
            "min_similarity": args.min_similarity,
            "start_step": -1,
            "mode": "diffusion",
            "diverse_reference": False,
            "adaptive_controller": {"enabled": False},
        }
        context = {"search_result": search, "fixed_results": selected}
        for action_no, strength in enumerate(strengths):
            model.rag_config = dict(common, strength=float(strength))
            # Reuse the same seed across actions so the grid comparison is not
            # dominated by unrelated candidate noise.
            torch.manual_seed(base_seed & 0x7FFFFFFF)
            generated = _aggregate_generation(
                torch,
                model.generate_text(
                    batch,
                    args.samples_per_action,
                    args.sampler,
                    retrieval_context=context,
                ),
            )
            score, embedding, current_text_embedding = _score_and_embeddings(
                scorer, [caption], generated, length
            )
            candidate_scores.append(float(score[0]))
            candidate_distances.append(
                float(generated_to_reference_distance(generated, reference_ts[None, ...])[0])
            )
            if embedding is not None:
                candidate_embeddings.append(embedding[0])
                text_embedding = current_text_embedding

        model.rag_config = dict(common, strength=float(strengths[0]), mode="retrieval_only")
        retrieval_only = reference_ts[None, ...]
        retrieval_only_score = scorer.score([caption], retrieval_only, length)
        retrieval_only_distance = generated_to_reference_distance(
            retrieval_only, reference_ts[None, ...]
        )
    finally:
        model.rag_config = original_config
        model.rag_retriever = original_retriever

    part = {
        "cache_signature": np.asarray(args.cache_signature),
        "query_sample_id": np.asarray(sample_id, dtype=np.int64),
        "query_caption_id": np.asarray(caption_id, dtype=np.int64),
        "reference_sample_id": np.asarray(result["reference_sample_id"], dtype=np.int64),
        "reference_caption_id": np.asarray(result["reference_caption_id"], dtype=np.int64),
        "query_embedding": query_embedding,
        "reference_embedding": reference_embedding,
        "top_k_sample_ids": retriever.sample_ids[top_rows].astype(np.int64),
        "top_k_caption_ids": retriever.caption_ids[top_rows].astype(np.int64),
        "top_k_similarities": np.asarray(search["scores"][0], dtype=np.float32),
        "candidate_semantic_scores": np.asarray(candidate_scores, dtype=np.float32),
        "candidate_copy_distances": np.asarray(candidate_distances, dtype=np.float32),
        "original_semantic_score": np.asarray(original_score[0], dtype=np.float32),
        "retrieval_only_semantic_score": np.asarray(retrieval_only_score[0], dtype=np.float32),
        "retrieval_only_copy_distance": np.asarray(retrieval_only_distance[0], dtype=np.float32),
    }
    if candidate_embeddings:
        part["candidate_ts_embeddings"] = np.stack(candidate_embeddings).astype(np.float32)
        part["original_ts_embedding"] = np.asarray(original_embedding[0], dtype=np.float32)
        part["real_ts_embedding"] = np.asarray(real_embedding[0], dtype=np.float32)
        part["query_text_embedding"] = np.asarray(text_embedding[0], dtype=np.float32)
    return part


def assemble_payload(parts, strengths, num_steps, args):
    def stack(name):
        return np.stack([np.asarray(part[name]) for part in parts])

    top_k_scores = stack("top_k_similarities").astype(np.float32)
    score_features = compute_score_features(top_k_scores, args.score_temperature)
    sample_ids = stack("query_sample_id").astype(np.int64).reshape(-1)
    split_ids = grouped_controller_split(sample_ids, args.validation_fraction, args.split_seed)
    train_mask = split_ids == 0
    normalization = fit_score_normalization(score_features, train_mask)
    quantiles = fit_similarity_quantiles(top_k_scores[:, 0], train_mask)
    target = build_teacher_targets(
        strengths,
        stack("candidate_semantic_scores"),
        stack("original_semantic_score").reshape(-1),
        epsilon_sem=args.epsilon_sem,
        teacher_temperature=args.teacher_temperature,
        candidate_copy_distances=stack("candidate_copy_distances"),
        copy_threshold=None if args.disable_copy_constraint else args.copy_threshold,
    )
    query_embeddings = stack("query_embedding").astype(np.float32)
    reference_embeddings = stack("reference_embedding").astype(np.float32)
    payload = {
        "query_sample_ids": sample_ids,
        "query_caption_ids": stack("query_caption_id").astype(np.int64).reshape(-1),
        "reference_sample_ids": stack("reference_sample_id").astype(np.int64).reshape(-1),
        "reference_caption_ids": stack("reference_caption_id").astype(np.int64).reshape(-1),
        "query_embeddings": query_embeddings,
        "reference_embeddings": reference_embeddings,
        "pair_features": compute_pair_features(query_embeddings, reference_embeddings),
        "top_k_sample_ids": stack("top_k_sample_ids").astype(np.int64),
        "top_k_caption_ids": stack("top_k_caption_ids").astype(np.int64),
        "top_k_similarities": top_k_scores,
        "score_features": score_features,
        "normalized_score_features": normalize_score_features(score_features, normalization),
        "candidate_strengths": strengths.astype(np.float32),
        "candidate_start_steps": strength_to_start_steps(strengths, num_steps),
        "candidate_semantic_scores": stack("candidate_semantic_scores").astype(np.float32),
        "candidate_copy_distances": stack("candidate_copy_distances").astype(np.float32),
        "original_semantic_scores": stack("original_semantic_score").astype(np.float32).reshape(-1),
        "retrieval_only_semantic_scores": stack("retrieval_only_semantic_score").astype(np.float32).reshape(-1),
        "retrieval_only_copy_distances": stack("retrieval_only_copy_distance").astype(np.float32).reshape(-1),
        "controller_split_ids": split_ids,
        **target,
    }
    for optional in (
        "candidate_ts_embeddings",
        "original_ts_embedding",
        "real_ts_embedding",
        "query_text_embedding",
    ):
        if all(optional in part for part in parts):
            payload[optional + ("s" if not optional.endswith("s") else "")] = stack(optional)
    return payload, normalization, quantiles


def _frechet(first, second):
    try:
        from scipy import linalg
    except ImportError:
        return None
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if min(first.shape[0], second.shape[0]) < 2:
        return None
    mu1, mu2 = first.mean(0), second.mean(0)
    cov1, cov2 = np.cov(first, rowvar=False), np.cov(second, rowvar=False)
    covmean = linalg.sqrtm(cov1.dot(cov2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(np.square(mu1 - mu2).sum() + np.trace(cov1 + cov2 - 2 * covmean))
    return value if np.isfinite(value) else None


def write_sweep_summary(payload, output_path, copy_threshold):
    rows = []
    has_embeddings = "candidate_ts_embeddings" in payload
    real_embeddings = payload.get("real_ts_embeddings")
    text_embeddings = payload.get("query_text_embeddings")
    strengths = payload["candidate_strengths"]
    for column, strength in enumerate(strengths):
        distances = payload["candidate_copy_distances"][:, column]
        row = {
            "strength": float(strength),
            "start_step": int(payload["candidate_start_steps"][column]),
            "cttp": float(payload["candidate_semantic_scores"][:, column].mean()),
            "generated_to_reference_distance": float(distances.mean()),
            "copy_rate": None
            if copy_threshold is None
            else float((distances < float(copy_threshold)).mean()),
            "fid": None,
            "j_ftsd": None,
        }
        if has_embeddings and real_embeddings is not None:
            generated_embeddings = payload["candidate_ts_embeddings"][:, column]
            row["fid"] = _frechet(real_embeddings, generated_embeddings)
            if text_embeddings is not None:
                row["j_ftsd"] = _frechet(
                    np.concatenate([real_embeddings, text_embeddings], axis=1),
                    np.concatenate([generated_embeddings, text_embeddings], axis=1),
                )
        rows.append(row)
    valid_fid = [row["fid"] for row in rows if row["fid"] is not None]
    warning = None
    if len(valid_fid) >= 3:
        ranks = np.argsort(np.argsort(np.asarray(valid_fid, dtype=float)))
        trend = float(np.corrcoef(np.arange(len(valid_fid)), ranks)[0, 1])
        if not np.isfinite(trend) or abs(trend) < 0.2:
            warning = (
                "Fixed-strength FID/J-FTSD did not show a clear overall trend; "
                "this weakens the minimum-feasible-strength interpretation."
            )
            warnings.warn(warning)
    elif not has_embeddings:
        warning = "FID/J-FTSD unavailable because the selected scorer did not expose embeddings."
    Path(output_path).write_text(
        json.dumps({"rows": rows, "warning": warning}, indent=2), encoding="utf-8"
    )
    return rows, warning


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-folder", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--longclip-path", default="./save/Longclip")
    parser.add_argument("--diff-config", required=True)
    parser.add_argument("--cond-config", required=True)
    parser.add_argument("--verbalts-checkpoint", required=True)
    parser.add_argument("--cttp-config", default="")
    parser.add_argument("--cttp-checkpoint", default="")
    parser.add_argument("--semantic-scorer", choices=["cttp", "precomputed"], default="cttp")
    parser.add_argument("--precomputed-scores", default="")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-manifest", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-queries", type=int, default=32)
    parser.add_argument("--allow-full-build", action="store_true")
    parser.add_argument("--strength-grid", default=DEFAULT_GRID)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument("--epsilon-sem", type=float, default=0.01)
    parser.add_argument("--epsilon-sem-source", default="controller-validation protocol")
    parser.add_argument("--teacher-temperature", type=float, default=0.10)
    parser.add_argument("--copy-threshold", type=float, default=0.05)
    parser.add_argument("--disable-copy-constraint", action="store_true")
    parser.add_argument("--copy-threshold-source", default="controller-train random-pair pilot")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-action", type=int, default=1)
    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--cond-modal", default="simple_text")
    parser.add_argument("--text-output-type", default="all")
    parser.add_argument("--text-pos-emb", default="none")
    parser.add_argument("--base-patch", type=int)
    parser.add_argument("--multipatch-num", type=int)
    parser.add_argument("--patch-length", type=int)
    parser.add_argument("--diff-stage-num", type=int, default=3)
    return parser


def main():
    args = build_argument_parser().parse_args()
    if args.top_k < 4:
        raise ValueError("Adaptive teacher retrieval requires top_k >= 4")
    if args.max_queries < 0 and not args.allow_full_build:
        raise ValueError("A full teacher build requires --allow-full-build")
    if args.semantic_scorer == "cttp" and (not args.cttp_config or not args.cttp_checkpoint):
        raise ValueError("CTTP scoring requires --cttp-config and --cttp-checkpoint")
    if args.semantic_scorer == "precomputed" and not args.precomputed_scores:
        raise ValueError("Precomputed scoring requires --precomputed-scores")
    strengths = parse_strength_grid(args.strength_grid)
    captions, sample_ids, caption_ids, train_ts = flatten_train_queries(args.dataset_folder)
    indices = stratified_query_indices(sample_ids, args.max_queries, args.seed)
    output_npz = Path(args.output_npz)
    output_manifest = Path(args.output_manifest) if args.output_manifest else output_npz.with_suffix(".json")
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_npz.parent / (output_npz.stem + "_parts")
    cache_dir.mkdir(parents=True, exist_ok=True)

    signature_payload = {
        "dataset": args.dataset_name,
        "index_sha256": sha256_file(args.index_path),
        "verbalts_sha256": sha256_file(args.verbalts_checkpoint),
        "scorer": args.semantic_scorer,
        "scorer_checkpoint": args.cttp_checkpoint or args.precomputed_scores,
        "strengths": strengths.astype(float).tolist(),
        "seed": args.seed,
        "samples_per_action": args.samples_per_action,
        "sampler": args.sampler,
    }
    args.cache_signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    print(
        f"Preparing teacher build: dataset={args.dataset_name} "
        f"queries={len(indices)} samples_per_action={args.samples_per_action}",
        flush=True,
    )
    print("Loading LongCLIP retriever and train index...", flush=True)
    embedder = LongCLIPTextEmbedder(
        args.longclip_path, args.embedding_device, args.embedding_batch_size
    )
    retriever = TextRetriever(args.index_path, embedder=embedder, seed=args.seed)
    if retriever.metadata.get("dataset_name") != args.dataset_name:
        raise ValueError("Retrieval index dataset identity does not match --dataset-name")
    print("Loading frozen VerbalTS generation stack...", flush=True)
    torch, model = _load_generation_stack(args)
    print(f"Loading frozen semantic scorer ({args.semantic_scorer})...", flush=True)
    scorer = _load_scorer(args)
    num_steps = int(model.generator.num_steps)
    print("Teacher generation stack ready.", flush=True)

    parts = []
    progress_path = output_manifest.with_name(output_manifest.stem + ".progress.json")
    for progress, index in enumerate(indices, start=1):
        sample_id = int(sample_ids[index])
        caption_id = int(caption_ids[index])
        part_path = _part_path(cache_dir, sample_id, caption_id)
        if args.resume and part_path.exists():
            with np.load(part_path, allow_pickle=False) as archive:
                part = {name: np.array(archive[name]) for name in archive.files}
            if str(part.get("cache_signature", np.asarray("")).item()) != args.cache_signature:
                raise ValueError(
                    f"Cached teacher query has a different build signature: {part_path}"
                )
        else:
            part = generate_query_part(
                args,
                torch,
                model,
                scorer,
                retriever,
                captions[index],
                sample_id,
                caption_id,
                train_ts[sample_id],
                strengths,
            )
            np.savez_compressed(part_path, **part)
        parts.append(part)
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": "ri-verbalts-strength-teacher-progress-v1",
                    "cache_signature": args.cache_signature,
                    "completed_queries": progress,
                    "total_queries": len(indices),
                    "complete": False,
                    "exclusion_enabled": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"teacher query {progress}/{len(indices)}: "
            f"sample={sample_id} caption={caption_id}",
            flush=True,
        )

    payload, normalization, quantiles = assemble_payload(parts, strengths, num_steps, args)
    sweep_path = output_npz.with_name(output_npz.stem + "_fixed_sweep.json")
    sweep_rows, sweep_warning = write_sweep_summary(
        payload,
        sweep_path,
        None if args.disable_copy_constraint else args.copy_threshold,
    )
    manifest = {
        "dataset": args.dataset_name,
        "split": "train",
        "data_file": str(output_npz),
        "retrieval_index": {
            "path": str(Path(args.index_path)),
            "sha256": sha256_file(args.index_path),
            "identity": retriever.metadata,
        },
        "longclip": {"path": args.longclip_path, "embedding_dim": int(retriever.embeddings.shape[1])},
        "verbalts_checkpoint": {
            "path": args.verbalts_checkpoint,
            "sha256": sha256_file(args.verbalts_checkpoint),
            "num_steps": num_steps,
        },
        "semantic_scorer": {
            "identity": scorer.identity,
            "teacher_semantic_scorer": scorer.identity,
            "metric_optimized": scorer.identity == "cttp",
            "checkpoint": args.cttp_checkpoint if scorer.identity == "cttp" else args.precomputed_scores,
            "checkpoint_sha256": sha256_file(
                args.cttp_checkpoint if scorer.identity == "cttp" else args.precomputed_scores
            ),
        },
        "strength_grid": strengths.astype(float).tolist(),
        "seeds": {"generation": args.seed, "controller_split": args.split_seed},
        "exclusion_policy": {
            "enabled": True,
            "unit": "sample_id",
            "weather_all_captions_excluded": True,
        },
        "retrieval": {"top_k": args.top_k, "selection": "top1"},
        "feature_definitions": {
            "score": list(SCORE_FEATURE_NAMES),
            "pair": ["abs(query-reference)", "query*reference"],
            "score_temperature": args.score_temperature,
        },
        "normalization": normalization,
        **quantiles,
        "semantic_threshold": {
            "epsilon_sem": args.epsilon_sem,
            "absolute_value": args.epsilon_sem,
            "source": args.epsilon_sem_source,
        },
        "copy_constraint": {
            "enabled": not args.disable_copy_constraint,
            "metric": "dimension_normalized_rmse",
            "threshold": None if args.disable_copy_constraint else args.copy_threshold,
            "source": args.copy_threshold_source,
        },
        "teacher_temperature": args.teacher_temperature,
        "pilot": not (args.max_queries < 0),
        "max_queries": args.max_queries,
        "fixed_sweep": {"path": str(sweep_path), "warning": sweep_warning, "rows": sweep_rows},
    }
    save_teacher_dataset(output_npz, output_manifest, payload, manifest)
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": "ri-verbalts-strength-teacher-progress-v1",
                "cache_signature": args.cache_signature,
                "completed_queries": len(indices),
                "total_queries": len(indices),
                "complete": True,
                "exclusion_enabled": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved teacher data: {output_npz}")
    print(f"Saved teacher manifest: {output_manifest}")


if __name__ == "__main__":
    main()
