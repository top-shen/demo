import os
import time
import torch
import numpy as np
from models.cttp.cttp_model import CTTP
import yaml
import tqdm
import numpy as np
from scipy import linalg
import random
import json

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

class BaseEvaluator:
    def __init__(self, configs, dataset, model):
        self._init_cfgs(configs)
        self._init_model(model)
        self._init_data(dataset)
        if "clip_config_path" in configs.keys():
            self._init_clip(configs)
        if self.rag_configs.get("enabled", False):
            self._init_retrieval()

    def _init_retrieval(self):
        from retrieval import LongCLIPTextEmbedder, TextRetriever, load_controller_checkpoint
        from retrieval.strength_teacher import sha256_file

        index_path = self.rag_configs.get("index_path", "")
        if not index_path:
            raise ValueError("rag.index_path is required when RAG is enabled")
        embedder = None
        # Evaluation already loads CTTP, whose text branch contains the same
        # frozen LongCLIP. Reuse it to avoid holding a second ~1.7 GB model.
        if hasattr(self, "clip") and hasattr(self.clip, "text_enc"):
            text_encoder = self.clip.text_enc
            if hasattr(text_encoder, "model") and hasattr(text_encoder, "tokenizer"):
                embedder = LongCLIPTextEmbedder.from_components(
                    text_encoder.model,
                    text_encoder.tokenizer,
                    text_encoder.device,
                    int(self.rag_configs.get("query_batch_size", 64)),
                )
        retriever = TextRetriever(
            index_path=index_path,
            embedding_model_path=self.rag_configs.get("embedding_model_path"),
            embedding_device=self.rag_configs.get("embedding_device", self.model.device),
            query_batch_size=int(self.rag_configs.get("query_batch_size", 64)),
            embedder=embedder,
            seed=int(self.rag_configs.get("seed", 0)),
        )
        adaptive_config = dict(self.rag_configs.get("adaptive_controller", {}))
        controller = None
        controller_manifest = None
        if adaptive_config.get("enabled", False) and self.rag_configs.get("mode", "diffusion") != "retrieval_only":
            if int(self.rag_configs.get("start_step", -1)) >= 0:
                raise ValueError(
                    "rag.start_step and adaptive_controller.enabled cannot be used together"
                )
            checkpoint_path = adaptive_config.get("checkpoint_path", "")
            if not checkpoint_path:
                raise ValueError(
                    "rag.adaptive_controller.checkpoint_path is required when enabled"
                )
            expected = {
                "dataset_identity": retriever.metadata.get("dataset_name"),
                "embedding_dim": int(retriever.embeddings.shape[1]),
                "feature_mode": adaptive_config.get("feature_mode", "score_only"),
                "num_steps": int(self.model.generator.num_steps),
                "retrieval_index_sha256": sha256_file(index_path),
            }
            for key in ("min_strength", "max_strength", "base_gamma", "max_residual"):
                if key in adaptive_config:
                    expected[key] = float(adaptive_config[key])
            controller, controller_manifest, _ = load_controller_checkpoint(
                checkpoint_path, device=self.model.device, expected=expected
            )
        if not hasattr(self.model, "configure_retrieval"):
            raise ValueError("The selected generator does not support RI-VerbalTS retrieval")
        self.model.configure_retrieval(
            retriever,
            self.rag_configs,
            controller=controller,
            controller_manifest=controller_manifest,
        )

    def _init_clip(self, configs):
        model_dict = {
            "clip_patchtst": CTTP,
        }
        clip_configs = yaml.safe_load(open(configs["clip_config_path"]))
        self.clip = model_dict[clip_configs["clip_type"]](clip_configs)
        self.clip.load_state_dict(torch.load(configs["clip_model_path"]))
        self.clip = self.clip.to(self.clip.device)

        fid_mean_cache_path = os.path.join(configs["cache_folder"], "fid_mean.npy")
        fid_cov_cache_path = os.path.join(configs["cache_folder"], "fid_cov.npy")
        jftsd_mean_cache_path = os.path.join(configs["cache_folder"], "jftsd_mean.npy")
        jftsd_cov_cache_path = os.path.join(configs["cache_folder"], "jftsd_cov.npy")
        print("cache_folder: ", configs["cache_folder"])
        if os.path.exists(fid_mean_cache_path) and os.path.exists(fid_cov_cache_path) and os.path.exists(jftsd_mean_cache_path) and os.path.exists(jftsd_cov_cache_path):
            self.ts_mean = np.load(fid_mean_cache_path)
            self.ts_cov = np.load(fid_cov_cache_path)
            self.joint_mean = np.load(jftsd_mean_cache_path)
            self.joint_cov = np.load(jftsd_cov_cache_path)
        else:
            train_loader = self.dataset.get_loader(split="train", batch_size=self.batch_size, shuffle=False, include_self=False)
            all_ts_emb, all_joint_emb = [], []
            with torch.no_grad():
                print("calc the ts mean and cov")
                for batch in tqdm.tqdm(train_loader):
                    ts = batch["ts"].to(self.clip.device).float()
                    ts_len = batch["ts_len"].to(self.clip.device).int()
                    cap = batch["cap"]
                    ts_emb = self.clip.get_ts_coemb(ts, ts_len)
                    cap_emb = self.clip.get_text_coemb(cap, None)
                    all_ts_emb.append(ts_emb)
                    all_joint_emb.append(torch.cat([ts_emb,cap_emb], dim=-1))

            all_ts_emb = torch.cat(all_ts_emb, dim=0)
            all_ts_emb = all_ts_emb.cpu().numpy()
            self.ts_mean = np.mean(all_ts_emb, axis=0)
            self.ts_cov = np.cov(all_ts_emb, rowvar=False)
            all_joint_emb = torch.cat(all_joint_emb, dim=0)
            all_joint_emb = all_joint_emb.cpu().numpy()
            self.joint_mean = np.mean(all_joint_emb, axis=0)
            self.joint_cov = np.cov(all_joint_emb, rowvar=False)

            os.makedirs(configs["cache_folder"], exist_ok=True)
            np.save(fid_mean_cache_path, self.ts_mean)
            np.save(fid_cov_cache_path, self.ts_cov)
            np.save(jftsd_mean_cache_path, self.joint_mean)
            np.save(jftsd_cov_cache_path, self.joint_cov)

    def _init_cfgs(self, configs):
        self.configs = configs
        self.batch_size = self.configs["batch_size"]
        self.n_samples = self.configs["n_samples"]
        self.display_epoch_interval = self.configs["display_interval"]
        self.model_path = self.configs["model_path"]
        self.max_batches = int(self.configs.get("max_batches", -1))
        self.rag_configs = dict(self.configs.get("rag", {}))

    def _init_model(self, model):
        self.model = model
        if self.model_path != "":
            print("Loading pretrained model from {}".format(self.model_path))
            self.model.load_state_dict(torch.load(self.model_path))

    def _init_data(self, dataset):
        self.dataset = dataset
        self.test_loader = dataset.get_loader(split="test", batch_size=self.batch_size, shuffle=False, include_self=False)

    """
    Evaluate.
    """
    def evaluate(self, mode="cond_gen", sampler="ddpm", save_pred=False):
        """
        Args:
            mode: cond_gen or edit.
            sampler: ddpm or ddim.
        """
        print("\n-------------------------------")
        print(f"Evaluating the model with mode={mode} and sampler={sampler}")
        self.model.eval()
        all_tsgen_emb = []
        all_joint_emb = []
        cttp = 0
        sample_num = 0
        retrieval_trace = []
        saved_candidates = []
        saved_predictions = []
        saved_targets = []
        saved_sample_ids = []
        saved_caption_ids = []
        saved_captions = []
        saved_reference_ids = []
        saved_per_sample_cttp = []
        saved_controller_strengths = []
        saved_start_steps = []
        saved_gate_probabilities = []
        saved_selected_reference_ids = []
        saved_copy_distances = []
        saved_controller_actions = []
        saved_fallback_reasons = []
        diverse_reference = bool(
            self.rag_configs.get("enabled", False)
            and self.rag_configs.get("diverse_reference", False)
        )
        save_pred = bool(save_pred or self.configs.get("save_predictions", False) or diverse_reference)

        with torch.no_grad():
            for batch_no, batch in enumerate(self.test_loader):
                if self.max_batches > 0 and batch_no >= self.max_batches:
                    break
                start_time = time.time()
                multi_preds = self.model.generate(batch, self.n_samples, sampler)
                multi_preds = multi_preds.permute(0,1,3,2)
                # References are fixed before generating the default candidates,
                # preserving the paper's elementwise median.  Diverse-reference
                # mode keeps every candidate and uses candidate 0 for aggregate
                # metrics because a cross-reference median can erase peaks.
                if diverse_reference:
                    pred = multi_preds[0]
                else:
                    pred = multi_preds.median(dim=0).values

                ts = batch["ts"].to(self.model.device).float()
                ts_len = batch["ts_len"].to(self.model.device).int()

                if "clip_config_path" in self.configs.keys():
                    cap_tokens = batch["cap"]
                    cap_emb = self.clip.get_text_coemb(cap_tokens, None)
                    ts_gen_emb = self.clip.get_ts_coemb(pred, ts_len)
                    all_tsgen_emb.append(ts_gen_emb)
                    all_joint_emb.append(torch.cat([ts_gen_emb,cap_emb], dim=-1))
                    per_sample_cttp = (ts_gen_emb * cap_emb).sum(dim=-1)
                    cttp += per_sample_cttp.sum().item()
                    sample_num += ts_gen_emb.shape[0]
                    saved_per_sample_cttp.append(
                        per_sample_cttp.detach().cpu().numpy().astype(np.float32)
                    )
                else:
                    saved_per_sample_cttp.append(
                        np.full(pred.shape[0], np.nan, dtype=np.float32)
                    )

                generation_metadata = getattr(self.model, "last_generation_metadata", [])
                if not generation_metadata:
                    diffusion_model = getattr(self.model, "generator", self.model)
                    full_start_step = int(getattr(diffusion_model, "num_steps", 1)) - 1
                    generation_metadata = [
                        {
                            "reference_sample_id": -1,
                            "controller_strength": np.nan,
                            "start_step": full_start_step,
                            "gate_probability": np.nan,
                            "controller_action": "original_gaussian",
                            "fallback_reason": "original_rag_disabled",
                        }
                        for _ in range(pred.shape[0])
                    ]
                saved_controller_strengths.append(
                    np.asarray(
                        [item["controller_strength"] for item in generation_metadata],
                        dtype=np.float32,
                    )
                )
                saved_start_steps.append(
                    np.asarray([item["start_step"] for item in generation_metadata], dtype=np.int64)
                )
                saved_gate_probabilities.append(
                    np.asarray(
                        [item["gate_probability"] for item in generation_metadata],
                        dtype=np.float32,
                    )
                )
                selected_ids = np.asarray(
                    [item["reference_sample_id"] for item in generation_metadata],
                    dtype=np.int64,
                )
                saved_selected_reference_ids.append(selected_ids)
                saved_controller_actions.extend(
                    [str(item["controller_action"]) for item in generation_metadata]
                )
                saved_fallback_reasons.extend(
                    [
                        "" if item["fallback_reason"] is None else str(item["fallback_reason"])
                        for item in generation_metadata
                    ]
                )
                copy_distances = np.full(pred.shape[0], np.nan, dtype=np.float32)
                if self.rag_configs.get("enabled", False) and hasattr(
                    getattr(self.model, "rag_retriever", None), "get_reference_ts"
                ):
                    pred_numpy = pred.detach().cpu().numpy()
                    for row, reference_id in enumerate(selected_ids):
                        if reference_id < 0:
                            continue
                        reference = np.asarray(
                            self.model.rag_retriever.get_reference_ts(int(reference_id)),
                            dtype=np.float32,
                        )
                        if reference.ndim == 1:
                            reference = reference[:, None]
                        if reference.shape == pred_numpy[row].shape:
                            copy_distances[row] = float(
                                np.sqrt(np.mean(np.square(pred_numpy[row] - reference)))
                            )
                saved_copy_distances.append(copy_distances)

                retrieval_trace.extend(getattr(self.model, "last_retrieval_trace", []))
                if save_pred:
                    saved_candidates.append(multi_preds.cpu().numpy())
                    saved_predictions.append(pred.cpu().numpy())
                    saved_targets.append(ts.cpu().numpy())
                    sample_ids = batch.get("sample_id", torch.arange(ts.shape[0]))
                    caption_ids = batch.get("caption_id", torch.full((ts.shape[0],), -1))
                    saved_sample_ids.append(np.asarray(sample_ids))
                    saved_caption_ids.append(np.asarray(caption_ids))
                    saved_captions.extend([str(caption) for caption in batch["cap"]])
                    reference_ids = getattr(self.model, "last_reference_ids", [])
                    if reference_ids:
                        saved_reference_ids.append(np.asarray(reference_ids, dtype=np.int64))

                end_time = time.time()
                if (batch_no+1)%self.display_epoch_interval == 0:
                    print("Batch", batch_no, 
                        "Batch Time {:.2f}s".format(end_time-start_time))
        if sample_num:
            cttp /= sample_num
        self._save_rag_outputs(
            retrieval_trace,
            saved_candidates,
            saved_predictions,
            saved_targets,
            saved_sample_ids,
            saved_caption_ids,
            saved_captions,
            saved_reference_ids,
            saved_per_sample_cttp,
            saved_controller_strengths,
            saved_start_steps,
            saved_gate_probabilities,
            saved_selected_reference_ids,
            saved_copy_distances,
            saved_controller_actions,
            saved_fallback_reasons,
        )
        print("Done!")
        res_dict = {
            "tensorboard":{},
            "df":{},
        }
        if "clip_config_path" in self.configs.keys():
            res_dict["tensorboard"].update({"cttp": cttp})
            res_dict["df"].update({"cttp": cttp})
            fid = None
            jftsd = None
            tsgen_emb = []
            joint_emb = []

            all_tsgen_emb = torch.cat(all_tsgen_emb, dim=0).cpu().numpy()
            tsgen_mean = np.mean(all_tsgen_emb, axis=0)
            tsgen_var = np.cov(all_tsgen_emb, rowvar=False)
            fid = calculate_frechet_distance(self.ts_mean, self.ts_cov, tsgen_mean, tsgen_var)
            all_joint_emb = torch.cat(all_joint_emb, dim=0).cpu().numpy()
            joint_mean = np.mean(all_joint_emb, axis=0)
            joint_var = np.cov(all_joint_emb, rowvar=False)
            jftsd = calculate_frechet_distance(self.joint_mean, self.joint_cov, joint_mean, joint_var)
            
            res_dict["tensorboard"].update({"fid":fid})
            res_dict["df"].update({"fid":fid})
            res_dict["tensorboard"].update({"jftsd":jftsd})
            res_dict["df"].update({"jftsd":jftsd})
            
            print("FID: ", fid)
            print("JFTSD: ", jftsd)
            print("CTTP ", cttp)

        return res_dict

    def _save_rag_outputs(
        self,
        retrieval_trace,
        candidates,
        predictions,
        targets,
        sample_ids,
        caption_ids,
        captions,
        reference_ids,
        per_sample_cttp,
        controller_strengths,
        start_steps,
        gate_probabilities,
        selected_reference_ids,
        copy_distances,
        controller_actions,
        fallback_reasons,
    ):
        trace_path = self.rag_configs.get("trace_path", "")
        if trace_path:
            os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
            with open(trace_path, "w", encoding="utf-8") as trace_file:
                for record in retrieval_trace:
                    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("Retrieval trace: ", trace_path)

        if not candidates:
            return
        prediction_path = self.configs.get("prediction_path", "")
        if not prediction_path:
            return
        os.makedirs(os.path.dirname(prediction_path) or ".", exist_ok=True)
        # Each batch has [S,B,L,V]; concatenate along B to preserve candidates.
        payload = {
            "candidates": np.concatenate(candidates, axis=1),
            "predictions": np.concatenate(predictions, axis=0),
            "targets": np.concatenate(targets, axis=0),
            "test_sample_ids": np.concatenate(sample_ids, axis=0),
            "query_caption_ids": np.concatenate(caption_ids, axis=0),
            "query_captions": np.asarray(captions),
            "per_sample_cttp": np.concatenate(per_sample_cttp, axis=0),
            "controller_strengths": np.concatenate(controller_strengths, axis=0),
            "controller_start_steps": np.concatenate(start_steps, axis=0),
            "controller_gate_probabilities": np.concatenate(gate_probabilities, axis=0),
            "selected_reference_sample_ids": np.concatenate(
                selected_reference_ids, axis=0
            ),
            "generated_to_reference_distances": np.concatenate(copy_distances, axis=0),
            "controller_actions": np.asarray(controller_actions),
            "fallback_reasons": np.asarray(fallback_reasons),
        }
        if reference_ids:
            payload["reference_sample_ids"] = np.concatenate(reference_ids, axis=1)
        np.savez(prediction_path, **payload)
        print("Predictions: ", prediction_path)
