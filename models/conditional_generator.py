import torch
import torch.nn as nn

from models.encoders.attr_encoder import AttributeEncoder
from models.encoders.text_encoder import TextEncoder, CLIPTextEncoder
from models.encoders.cond_projector import TextProjectorMVarMScaleMStep, AttrProjectorAvg
from models.unconditional_generator import UnConditionalGenerator
from models.cttp.cttp_model import CTTP
import time
import random
import yaml
import numpy as np

class ConditionalGenerator(nn.Module):
    def __init__(self, diff_configs, cond_configs):
        super().__init__()
        self.device = diff_configs["device"]
        self.diff_configs = diff_configs
        self.cond_configs = cond_configs
        self.rag_retriever = None
        self.rag_config = {"enabled": False}
        self.last_retrieval_trace = []
        self.last_reference_ids = []
        self._init_condition_encoders(diff_configs, cond_configs)
        self._init_diff(diff_configs)

    def configure_retrieval(self, retriever, configs):
        """Attach an inference-only retriever without changing training state."""
        if "text" not in self.cond_configs["cond_modal"]:
            raise ValueError("RI-VerbalTS retrieval is only supported for text conditioning")
        self.rag_retriever = retriever
        self.rag_config = dict(configs)
        self.rag_config["enabled"] = bool(self.rag_config.get("enabled", False))

    def _init_condition_encoders(self, diff_configs, cond_configs):
        if cond_configs["cond_modal"] == "constraint":
            clip_configs = yaml.safe_load(open(cond_configs["constraint"]["clip_config_path"]))
            self.cond_guide_model = CTTP(clip_configs)
            self.cond_guide_model.load_state_dict(torch.load(cond_configs["constraint"]["clip_model_path"]))
            self.cond_guide_model = self.cond_guide_model.to(self.cond_guide_model.device)
        elif cond_configs["cond_modal"] == "attr":
            cond_configs["attrs"]["device"] = self.device
            self.attr_en = AttributeEncoder(cond_configs["attrs"]).to(self.device)
            self.cond_projector = AttrProjectorAvg(dim_in=cond_configs["attrs"]["attr_emb"], dim_hid=diff_configs["diffusion"]["channels"], dim_out=diff_configs["diffusion"]["channels"])
            self.cond_projector = self.cond_projector.to(self.device)
        elif "text" in cond_configs["cond_modal"]:
            if cond_configs["cond_modal"] == "text":
                cond_configs["text"]["device"] = self.device
                self.attr_en = CLIPTextEncoder(cond_configs["text"]).to(self.device)
            elif cond_configs["cond_modal"] == "simple_text":
                cond_configs["text"]["device"] = self.device
                self.attr_en = TextEncoder(cond_configs["text"]).to(self.device)
            if cond_configs["text"]["text_projector"] == "var_scale_diffstep_multi":
                self.cond_projector = TextProjectorMVarMScaleMStep(n_var=diff_configs["diffusion"]["n_var"],
                                                         n_scale=diff_configs["diffusion"]["multipatch_num"],
                                                         n_steps=diff_configs["diffusion"]["num_steps"],
                                                         n_stages=cond_configs["text"]["num_stages"],
                                                         dim_in=cond_configs["text"]["text_emb"], 
                                                         dim_out=diff_configs["diffusion"]["channels"])
            self.cond_projector = self.cond_projector.to(self.device)

    def _init_diff(self, configs):
        configs["device"] = self.device
        if "text" in self.cond_configs["cond_modal"]:
            configs["diffusion"]["text_projector"] = self.cond_configs["text"]["text_projector"]
        self.generator = UnConditionalGenerator(configs=configs)
        if configs["generator_pretrain_path"] != "":
            self.generator.load_state_dict(torch.load(configs["generator_pretrain_path"]))
            print("Load the pretrain unconditional generator")
        else:
            print("Learn from scratch")
    
    """
    Finetune.
    """
    def forward(self, batch, is_train):
        x, tp, attrs = self._unpack_data_cond_gen(batch)
        attr_emb_raw = self.attr_en(attrs)
        if self.cond_configs["cond_modal"] == "attr" or "diffstep" not in self.cond_configs["text"]["text_projector"]:
            attr_emb = self.cond_projector(attr_emb_raw)

        B = x.shape[0]
        if is_train:
            t = torch.randint(0, self.generator.num_steps, [B], device=self.device)
            if "text" in self.cond_configs["cond_modal"] and "diffstep" in self.cond_configs["text"]["text_projector"]:
                attr_emb = self.cond_projector(attr_emb_raw, t)
            loss = self.generator._noise_estimation_loss(x, tp, attr_emb, t)
            return loss
        
        loss_dict = {}
        for t in range(self.generator.num_steps):
            t = (torch.ones(B, device=self.device) * t).long()
            if "text" in self.cond_configs["cond_modal"] and "diffstep" in self.cond_configs["text"]["text_projector"]:
                attr_emb = self.cond_projector(attr_emb_raw, t)
            tmp_loss_dict = self.generator._noise_estimation_loss(x, tp, attr_emb, t)
            for k in tmp_loss_dict:
                if k in loss_dict.keys():
                    loss_dict[k] += tmp_loss_dict[k]
                else:
                    loss_dict[k] = tmp_loss_dict[k]
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / self.generator.num_steps
        return loss_dict

    def _unpack_data_cond_gen(self, batch):
        ts = batch["ts"].to(self.device).float()
        tp = batch["tp"].to(self.device).float()
        if "text" in self.cond_configs["cond_modal"]:
            attrs = batch["cap"]
        elif "constraint" in self.cond_configs["cond_modal"]:
            attrs = batch["cap"]
        elif self.cond_configs["cond_modal"] == "attr":
            attrs = batch["attrs"].to(self.device).long()
        ts = ts.permute(0, 2, 1)
        return ts, tp, attrs

    def generate(self, batch, n_samples, sampler="ddim"):
        if self.cond_configs["cond_modal"] == "constraint":
            return self.generate_constraint(batch, n_samples, sampler)
        else:
            return self.generate_text(batch, n_samples, sampler)

    """
    Generation.
    """
    def _rag_enabled(self):
        return bool(self.rag_config.get("enabled", False)) and self.rag_retriever is not None

    def _resolve_rag_start_step(self):
        """Map strength [0, 1] linearly onto diffusion indices [0, T-1]."""
        explicit_step = self.rag_config.get("start_step")
        if explicit_step is not None and int(explicit_step) >= 0:
            start_step = int(explicit_step)
            if start_step >= self.generator.num_steps:
                raise ValueError(
                    f"rag_start_step must be in [0, {self.generator.num_steps - 1}]"
                )
            return start_step
        strength = float(self.rag_config.get("strength", 0.5))
        if not 0.0 <= strength <= 1.0:
            raise ValueError("rag_strength must be in [0, 1]")
        return int(round(strength * (self.generator.num_steps - 1)))

    @staticmethod
    def _prepare_reference_tensor(retrieval_results, like):
        """Convert stored [L,V] references to the model's [B,V,L] layout."""
        references = torch.zeros_like(like)
        success = torch.zeros(like.shape[0], dtype=torch.bool, device=like.device)
        expected = (like.shape[2], like.shape[1])
        for row, result in enumerate(retrieval_results):
            reference_ts = result.get("reference_ts")
            if reference_ts is None:
                continue
            reference_ts = np.asarray(reference_ts)
            if reference_ts.ndim == 1 and like.shape[1] == 1:
                reference_ts = reference_ts[:, None]
            if reference_ts.shape != expected:
                raise ValueError(
                    f"Reference shape {reference_ts.shape} does not match expected [L,V] {expected}"
                )
            references[row] = torch.as_tensor(
                reference_ts.T, dtype=like.dtype, device=like.device
            )
            success[row] = True
        return references, success

    def _query_sample_ids(self, batch, batch_size):
        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            return list(range(batch_size))
        if torch.is_tensor(sample_ids):
            sample_ids = sample_ids.detach().cpu().tolist()
        return [int(sample_id) for sample_id in sample_ids]

    def _select_references(self, search_result, attrs, sample_ids, candidate_index):
        return self.rag_retriever.select(
            search_result,
            attrs,
            sample_ids,
            selection=self.rag_config.get("selection", "top1"),
            temperature=float(self.rag_config.get("temperature", 1.0)),
            min_similarity=float(self.rag_config.get("min_similarity", -1.0)),
            candidate_index=candidate_index,
            random_reference=self.rag_config.get("mode", "diffusion") == "random_reference",
        )

    def _initialize_generation_state(self, ts, retrieval_results=None):
        """Return initial x and the last reverse step active for each batch row."""
        base_noise = torch.randn_like(ts)
        full_start = self.generator.num_steps - 1
        start_steps = torch.full(
            (ts.shape[0],), full_start, dtype=torch.long, device=ts.device
        )
        if retrieval_results is None:
            return base_noise, start_steps

        references, success = self._prepare_reference_tensor(retrieval_results, ts)
        mode = self.rag_config.get("mode", "diffusion")
        if mode == "retrieval_only":
            x = torch.where(success[:, None, None], references, base_noise)
            # A successful retrieval is already final and must never enter the
            # reverse loop. Failed rows retain the complete baseline path.
            start_steps = torch.where(
                success,
                torch.full_like(start_steps, -1),
                start_steps,
            )
            return x, start_steps

        start_step = self._resolve_rag_start_step()
        t = torch.full((ts.shape[0],), start_step, dtype=torch.long, device=ts.device)
        noisy_references = self.generator.ddpm.forward(references, t, base_noise)
        x = torch.where(success[:, None, None], noisy_references, base_noise)
        start_steps = torch.where(success, t, start_steps)
        return x, start_steps

    def _reverse_from_start_steps(
        self, x, tp, attr_emb_raw, attr_emb, start_steps, sampler
    ):
        """Reverse only rows whose configured start step includes the current t."""
        if int(start_steps.max().item()) < 0:
            return x
        B = x.shape[0]
        for step in range(int(start_steps.max().item()), -1, -1):
            noise = torch.randn_like(x)
            t = torch.full((B,), step, dtype=torch.long, device=self.device)
            if (
                "text" in self.cond_configs["cond_modal"]
                and "diffstep" in self.cond_configs["text"]["text_projector"]
            ):
                attr_emb = self.cond_projector(attr_emb_raw, t)
            pred_noise, _ = self.generator.predict_noise(x, tp, attr_emb, t)
            if sampler == "ddpm":
                next_x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
            else:
                next_x = self.generator.ddim.reverse(
                    x, pred_noise, t, noise, is_determin=True
                )
            active = start_steps >= step
            if bool(active.all()):
                x = next_x
            else:
                x = torch.where(active[:, None, None], next_x, x)
        return x

    def _record_retrieval_trace(self, results, batch, candidate_index):
        caption_ids = batch.get("caption_id")
        if torch.is_tensor(caption_ids):
            caption_ids = caption_ids.detach().cpu().tolist()
        if caption_ids is None:
            caption_ids = [-1] * len(results)
        configured_start_step = self._resolve_rag_start_step()
        mode = self.rag_config.get("mode", "diffusion")
        for row, result in enumerate(results):
            if result["fallback"]:
                effective_start_step = self.generator.num_steps - 1
            elif mode == "retrieval_only":
                effective_start_step = -1
            else:
                effective_start_step = configured_start_step
            trace = {
                "test_sample_id": int(result["query_sample_id"]),
                "query_caption_id": int(caption_ids[row]),
                "query_caption": result["query_caption"],
                "selected_reference_sample_id": result["reference_sample_id"],
                "selected_reference_caption_id": result["reference_caption_id"],
                "selected_reference_caption": result["reference_caption"],
                "similarity": result["similarity"],
                "top_k_sample_ids": result["top_k_sample_ids"],
                "top_k_caption_ids": result["top_k_caption_ids"],
                "top_k_similarities": result["top_k_similarities"],
                "start_step": effective_start_step,
                "configured_start_step": configured_start_step,
                "strength": float(self.rag_config.get("strength", 0.5)),
                "selection_strategy": self.rag_config.get("selection", "top1"),
                "random_seed": int(self.rag_config.get("seed", 0)),
                "mode": mode,
                "candidate_id": int(candidate_index),
                "fallback": bool(result["fallback"]),
            }
            self.last_retrieval_trace.append(trace)

    @torch.no_grad()
    def generate_text(self, batch, n_samples, sampler="ddim"):
        ts, tp, attrs = self._unpack_data_cond_gen(batch)
        attr_emb_raw = self.attr_en(attrs)
        if self.cond_configs["cond_modal"] == "attr" or "diffstep" not in self.cond_configs["text"]["text_projector"]:
            attr_emb = self.cond_projector(attr_emb_raw)
        else:
            attr_emb = None

        samples = []
        B = ts.shape[0]
        self.last_retrieval_trace = []
        self.last_reference_ids = []
        search_result = None
        fixed_results = None
        sample_ids = self._query_sample_ids(batch, B)
        if self._rag_enabled():
            search_result = self.rag_retriever.search(
                attrs, int(self.rag_config.get("top_k", 1))
            )
            if not bool(self.rag_config.get("diverse_reference", False)):
                fixed_results = self._select_references(
                    search_result, attrs, sample_ids, candidate_index=0
                )
                self._record_retrieval_trace(fixed_results, batch, candidate_index=0)

        for i in range(n_samples):
            retrieval_results = fixed_results
            if self._rag_enabled() and fixed_results is None:
                retrieval_results = self._select_references(
                    search_result, attrs, sample_ids, candidate_index=i
                )
                self._record_retrieval_trace(retrieval_results, batch, candidate_index=i)
            x, start_steps = self._initialize_generation_state(ts, retrieval_results)
            if retrieval_results is None:
                reference_ids = [-1] * B
            else:
                reference_ids = [
                    -1 if result["reference_sample_id"] is None else int(result["reference_sample_id"])
                    for result in retrieval_results
                ]
            self.last_reference_ids.append(reference_ids)
            x = self._reverse_from_start_steps(
                x, tp, attr_emb_raw, attr_emb, start_steps, sampler
            )
            samples.append(x)
        return torch.stack(samples)
    
    def generate_constraint(self, batch, n_samples, sampler="ddim"):
        ts, tp, attrs = self._unpack_data_cond_gen(batch)
        samples = []
        B = ts.shape[0]
        for i in range(n_samples):
            x = torch.randn_like(ts)
            for t in range(self.generator.num_steps-1, -1, -1):
                noise = torch.randn_like(x)
                t = (torch.ones(B, device=self.device) * t).long()
                with torch.no_grad():
                    pred_noise, _ = self.generator.predict_noise(x, tp, None, t)
                if sampler == "ddpm":
                    x = self.generator.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x0 = self.generator.ddim.predict_x0(x, pred_noise, t).permute(0,2,1)
                    with torch.set_grad_enabled(True):
                        x0.requires_grad = True
                        ts_emb = self.cond_guide_model.get_ts_coemb(x0, None)
                        text_emb = self.cond_guide_model.get_text_coemb(attrs, None)
                        negative_cttp = -torch.mm(ts_emb, text_emb.permute(1,0)).trace()
                        negative_cttp.backward()
                    pred_noise -= self.cond_configs["constraint"]["guide_w"] * self.generator.ddim.one_minus_alpha_bar_sqrt[t] * x0.grad.permute(0,2,1)
                    x = self.generator.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)
        return torch.stack(samples)
