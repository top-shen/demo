import os
import yaml
import json
import datetime
import argparse
import pandas as pd
import torch
import numpy as np
import random

from data import GenerationDataset
from models.conditional_generator import ConditionalGenerator
from models.unconditional_generator import UnConditionalGenerator
from evaluation.base_evaluator import BaseEvaluator


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value")

def save_configs(configs, path):
    print(json.dumps(configs, indent=4))
    with open(path, "w") as f:
        yaml.dump(configs, f, yaml.SafeDumper)

def train(training_stage, train_configs, model_diff_configs, model_cond_configs, eval_configs,  output_folder):
    # TensorBoard is a training-only dependency. Importing Trainer at module
    # load time made evaluation fail on otherwise valid inference environments
    # where tensorboard is intentionally not installed.
    from train.trainer import Trainer

    train_configs["train"]["output_folder"] = output_folder

    dataset = GenerationDataset(train_configs["data"])

    if training_stage == "pretrain":
        model = UnConditionalGenerator(model_diff_configs)
    elif training_stage == "finetune":
        if "attrs" in model_cond_configs.keys():
            model_cond_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
        model = ConditionalGenerator(model_diff_configs, model_cond_configs)

    print("\n***** Train Configs *****")
    path = os.path.join(output_folder, "train_configs.yaml")
    save_configs(train_configs, path)

    print("\n***** Model Configs *****")
    path = os.path.join(output_folder, "model_diff_configs.yaml")
    save_configs(model_diff_configs, path)
    if training_stage == "finetune":
        path = os.path.join(output_folder, "model_cond_configs.yaml")
        save_configs(model_cond_configs, path)

    pretrainer = Trainer(train_configs["train"], eval_configs, dataset, model)
    print("Begin training!")
    pretrainer.train()


def evaluate(training_stage, eval_configs, model_diff_configs, model_cond_configs, output_folder, checkpoint_folder=None):
    checkpoint_folder = checkpoint_folder or output_folder
    eval_configs["eval"]["model_path"] = os.path.join(checkpoint_folder, "ckpts/model_best_loss.pth")

    rag_configs = eval_configs["eval"].get("rag", {})
    if rag_configs.get("enabled", False):
        if not rag_configs.get("trace_path"):
            rag_configs["trace_path"] = os.path.join(output_folder, "retrieval_trace.jsonl")
        if not eval_configs["eval"].get("prediction_path"):
            eval_configs["eval"]["prediction_path"] = os.path.join(output_folder, "rag_predictions.npz")

    dataset = GenerationDataset(eval_configs["data"])

    if training_stage == "pretrain":
        model = UnConditionalGenerator(model_diff_configs)
    elif training_stage == "finetune":
        if "attrs" in model_cond_configs.keys():
            model_cond_configs["attrs"]["num_attr_ops"] = dataset.num_attr_ops.tolist()
        model = ConditionalGenerator(model_diff_configs, model_cond_configs)

    print("\n***** Evaluate Configs *****")
    path = os.path.join(output_folder, "eval_configs.yaml")
    save_configs(eval_configs, path=path)

    evaluator = BaseEvaluator(eval_configs["eval"], dataset, model)

    df = _evaluate_cond_gen(
        evaluator,
        sampler=eval_configs["eval"].get("sampler", "ddim"),
        n_sample=eval_configs["eval"].get("n_samples", 10),
    )
    return df


def _evaluate_cond_gen(evaluator, sampler="ddim", n_sample=10):
    evaluator.n_samples = n_sample
    res_dict = evaluator.evaluate(
        mode="cond_gen",
        sampler=sampler,
        save_pred=evaluator.configs.get("save_predictions", False),
    )

    info = {
        "mode": "cond_gen", 
        "sampler": sampler,
        "n_samples": evaluator.n_samples,
        "steps": -1,
    }
    info.update(res_dict["df"])    
    df = pd.DataFrame([info])
    df["steps"].astype(int)
    return df

def run(training_stage, train_configs, eval_configs, model_diff_configs, model_cond_configs, output_folder, data_folder="", only_evaluate=False, checkpoint_folder=None):
    if only_evaluate == False:
        train(training_stage, train_configs, model_diff_configs, model_cond_configs, eval_configs, output_folder)

    eval_configs["data"]["folder"] = data_folder
    df = evaluate(
        training_stage,
        eval_configs,
        model_diff_configs,
        model_cond_configs,
        output_folder,
        checkpoint_folder=checkpoint_folder,
    )
    path = os.path.join(output_folder, "results.csv")
    df.to_csv(path)
    return df

##### Arguments #####
parser = argparse.ArgumentParser(description="TSE")
parser.add_argument("--training_stage", type=str, default="pretrain")
parser.add_argument("--model_diff_config_path", type=str, default="")
parser.add_argument("--model_cond_config_path", type=str, default="")
parser.add_argument("--generator_pretrain_path", type=str, default="")
parser.add_argument("--train_config_path", type=str, default="")
parser.add_argument("--evaluate_config_path", type=str, default="")
parser.add_argument("--data_folder", type=str, default="./datasets")
parser.add_argument("--save_folder", type=str, default="./save")
parser.add_argument("--checkpoint_folder", type=str, default="")
parser.add_argument("--clip_folder", type=str, default="")
parser.add_argument("--start_runid", type=int, default=0)
parser.add_argument("--n_runs", type=int, default=3)
parser.add_argument("--clip_cache_path", type=str, default="cache")

parser.add_argument("--cond_modal", type=str, default="text")
parser.add_argument("--text_output_type", type=str, default="all")
parser.add_argument("--text_pos_emb", type=str, default="none")

parser.add_argument("--base_patch", type=int, default=1)
parser.add_argument("--multipatch_num", type=int, default=3)
parser.add_argument("--L_patch_len", type=int, default=3)
parser.add_argument("--diff_stage_num", type=int, default=3)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--eval_max_batches", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=200)

parser.add_argument("--guide_w", type=float, default=1.0)
parser.add_argument("--only_evaluate", type=bool, default=False)

# Inference-only Retrieval-Initialized VerbalTS.  None defaults let a RAG YAML
# provide values while old configs remain disabled and fully compatible.
parser.add_argument("--rag_enabled", type=str2bool, default=None)
parser.add_argument("--rag_index_path", type=str, default=None)
parser.add_argument("--rag_top_k", type=int, default=None)
parser.add_argument("--rag_selection", choices=["top1", "sample"], default=None)
parser.add_argument("--rag_temperature", type=float, default=None)
parser.add_argument("--rag_seed", type=int, default=None)
parser.add_argument("--rag_min_similarity", type=float, default=None)
parser.add_argument("--rag_start_step", type=int, default=None)
parser.add_argument("--rag_strength", type=float, default=None)
parser.add_argument(
    "--rag_mode", choices=["diffusion", "retrieval_only", "random_reference"], default=None
)
parser.add_argument("--rag_diverse_reference", type=str2bool, default=None)
parser.add_argument("--rag_embedding_model_path", type=str, default=None)
parser.add_argument("--rag_embedding_device", type=str, default=None)
parser.add_argument("--rag_query_batch_size", type=int, default=None)
parser.add_argument("--rag_save_predictions", type=str2bool, default=None)
parser.add_argument("--rag_trace_path", type=str, default=None)
parser.add_argument("--rag_prediction_path", type=str, default=None)
parser.add_argument("--rag_adaptive_enabled", type=str2bool, default=None)
parser.add_argument("--rag_controller_checkpoint_path", type=str, default=None)
parser.add_argument(
    "--rag_controller_feature_mode",
    choices=["score_only", "score_plus_pair"],
    default=None,
)
parser.add_argument("--rag_controller_min_strength", type=float, default=None)
parser.add_argument("--rag_controller_max_strength", type=float, default=None)
parser.add_argument("--rag_controller_base_gamma", type=float, default=None)
parser.add_argument("--rag_controller_max_residual", type=float, default=None)
parser.add_argument("--rag_controller_gate_threshold", type=float, default=None)

args = parser.parse_args()

save_folder = args.save_folder
os.makedirs(save_folder, exist_ok=True)
print("All files will be saved to '{}'".format(save_folder))

train_configs = yaml.safe_load(open(args.train_config_path))
eval_configs = yaml.safe_load(open(args.evaluate_config_path))
model_diff_configs = yaml.safe_load(open(args.model_diff_config_path))
if args.training_stage == "finetune":
    model_cond_configs = yaml.safe_load(open(args.model_cond_config_path))
    model_cond_configs["cond_modal"] = args.cond_modal
else:
    model_cond_configs = None

train_configs["data"]["folder"] = fr"{args.data_folder}"
train_configs["train"]["lr"] = args.lr
train_configs["train"]["epochs"] = args.epochs
train_configs["train"]["batch_size"] = args.batch_size
eval_configs["eval"]["batch_size"] = args.batch_size
if args.eval_max_batches is not None:
    eval_configs["eval"]["max_batches"] = args.eval_max_batches

rag_defaults = {
    "enabled": False,
    "index_path": "",
    "top_k": 1,
    "selection": "top1",
    "temperature": 1.0,
    "seed": 0,
    "min_similarity": -1.0,
    "start_step": -1,
    "strength": 0.5,
    "mode": "diffusion",
    "diverse_reference": False,
    "embedding_model_path": "./save/Longclip",
    "embedding_device": "auto",
    "query_batch_size": 64,
    "trace_path": "",
}
rag_configs = eval_configs["eval"].setdefault("rag", {})
for key, value in rag_defaults.items():
    rag_configs.setdefault(key, value)
rag_overrides = {
    "enabled": args.rag_enabled,
    "index_path": args.rag_index_path,
    "top_k": args.rag_top_k,
    "selection": args.rag_selection,
    "temperature": args.rag_temperature,
    "seed": args.rag_seed,
    "min_similarity": args.rag_min_similarity,
    "start_step": args.rag_start_step,
    "strength": args.rag_strength,
    "mode": args.rag_mode,
    "diverse_reference": args.rag_diverse_reference,
    "embedding_model_path": args.rag_embedding_model_path,
    "embedding_device": args.rag_embedding_device,
    "query_batch_size": args.rag_query_batch_size,
    "trace_path": args.rag_trace_path,
}
for key, value in rag_overrides.items():
    if value is not None:
        rag_configs[key] = value
adaptive_defaults = {
    "enabled": False,
    "checkpoint_path": "",
    "feature_mode": "score_only",
    "min_strength": 0.20,
    "max_strength": 0.95,
    "base_gamma": 1.0,
    "max_residual": 0.15,
    "gate_threshold": 0.50,
}
adaptive_configs = rag_configs.setdefault("adaptive_controller", {})
for key, value in adaptive_defaults.items():
    adaptive_configs.setdefault(key, value)
adaptive_overrides = {
    "enabled": args.rag_adaptive_enabled,
    "checkpoint_path": args.rag_controller_checkpoint_path,
    "feature_mode": args.rag_controller_feature_mode,
    "min_strength": args.rag_controller_min_strength,
    "max_strength": args.rag_controller_max_strength,
    "base_gamma": args.rag_controller_base_gamma,
    "max_residual": args.rag_controller_max_residual,
    "gate_threshold": args.rag_controller_gate_threshold,
}
for key, value in adaptive_overrides.items():
    if value is not None:
        adaptive_configs[key] = value
if args.rag_save_predictions is not None:
    eval_configs["eval"]["save_predictions"] = args.rag_save_predictions
if args.rag_prediction_path is not None:
    eval_configs["eval"]["prediction_path"] = args.rag_prediction_path

model_diff_configs["diffusion"]["multipatch_num"] = args.multipatch_num
model_diff_configs["diffusion"]["L_patch_len"] = args.L_patch_len
model_diff_configs["diffusion"]["base_patch"] = args.base_patch
if "text" in args.model_cond_config_path and args.training_stage == "finetune":
    model_cond_configs["text"]["output_type"] = args.text_output_type
    model_cond_configs["text"]["num_stages"] = args.diff_stage_num
    model_cond_configs["text"]["pos_emb"] = args.text_pos_emb

if args.clip_folder != "":
    eval_configs["eval"]["cache_folder"] = args.clip_cache_path
    eval_configs["eval"]["clip_model_path"] = fr"{args.clip_folder}/clip_model_best.pth"
    eval_configs["eval"]["clip_config_path"] = fr"{args.clip_folder}/model_configs.yaml"
    
    if model_cond_configs["cond_modal"] == "constraint":
        model_cond_configs["constraint"]["clip_config_path"] = fr"{args.clip_folder}/clip_model_best.pth"
        model_cond_configs["constraint"]["clip_model_path"] = fr"{args.clip_folder}/model_configs.yaml"
        model_cond_configs["constraint"]["guide_w"] = args.guide_w

seed_list = [1, 7, 42]
df_list = []
eval_record_folder = eval_configs["data"]["folder"]
configured_trace_path = rag_configs.get("trace_path", "")
configured_prediction_path = eval_configs["eval"].get("prediction_path", "")
for n in range(args.start_runid, args.n_runs):
    fix_seed = seed_list[n]
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    print(f"\nRun: {n}")
    output_folder = os.path.join(save_folder, str(n))
    os.makedirs(output_folder, exist_ok=True)
    eval_configs["eval"]["model_path"] = ""
    eval_configs["data"]["folder"] = eval_record_folder
    rag_configs["trace_path"] = configured_trace_path
    eval_configs["eval"]["prediction_path"] = configured_prediction_path
    if args.generator_pretrain_path != "":
        model_diff_configs["generator_pretrain_path"] = f"{args.generator_pretrain_path}/{n}/ckpts/model_best_loss.pth"
    else:
        model_diff_configs["generator_pretrain_path"] = ""
    checkpoint_folder = (
        os.path.join(args.checkpoint_folder, str(n))
        if args.checkpoint_folder
        else output_folder
    )
    df = run(
        args.training_stage,
        train_configs,
        eval_configs,
        model_diff_configs,
        model_cond_configs,
        output_folder,
        data_folder=args.data_folder,
        only_evaluate=args.only_evaluate,
        checkpoint_folder=checkpoint_folder,
    )

    n_records = df.shape[0]
    df.insert(0, column="run", value=[n]*n_records)
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
path = os.path.join(save_folder, "results.csv")
df.to_csv(path)

df_stat = df.groupby(["mode", "sampler", "steps", "n_samples"], as_index=False).agg(["mean", "std"])
df_stat.to_csv(os.path.join(save_folder, "results_stat_condgen.csv"))
