# 可学习的检索自适应 diffusion strength 控制器

本模块在 phase-1 [Retrieval-Initialized VerbalTS](README_RAG.md) 上增加一个仅训练小型控制器的阶段。VerbalTS、LongCLIP 与 CTTP/其他 teacher scorer 默认全部冻结；推理输入仍然只有文本检索特征，不把 CTTP 时间序列 embedding 送入控制器。

## 1. 方法

对 query `q` 检索训练集 Top-K caption（`K>=4`，最终 reference 始终选 Top-1）。控制器输出使用检索的概率和逐样本连续 strength：

```text
C_theta(q,r) -> (p_use, s),  s in [s_min,s_max]
t = round(s * (T-1))
```

检索分数特征为：

```text
z_score = [sim1, sim1-sim2, mean(sim1:K), std(sim1:K),
           entropy(softmax(sim1:K / score_temperature))]
```

`score_only` 只使用这五维特征。`score_plus_pair` 还使用冻结 LongCLIP 文本 embedding：

```text
z_pair = [abs(e_q-e_r), e_q*e_r]
```

pair 特征先经 `LayerNorm -> Linear -> GELU -> Dropout` 投影，再与 score 特征拼接。默认 768 维 LongCLIP、128 维 projection/hidden 时参数量低于 500K。所有五个 scalar 的均值和标准差只在 controller-train 行上拟合，并同时写入 teacher manifest 和 controller checkpoint。

### Similarity prior 与 residual

controller-train Top-1 similarity 的 5%/95% 分位数定义置信度：

```text
c = clip((sim1-q05)/(q95-q05), 0, 1)
s_base = s_min + (s_max-s_min) * (1-c)^gamma
delta = delta_max * tanh(f_theta(z))
s = clip(s_base + delta, s_min, s_max)
```

默认 `s_min=0.20`、`s_max=0.95`、`gamma=1.0`、`delta_max=0.15`。这里使用加号形式，因为它满足 `c=0 -> s_max`、`c=1 -> s_min`，也与“可靠检索保留更多 reference”的定义一致；`s_min-(s_max-s_min)(1-c)^gamma` 会在最终 clip 后使绝大多数输入退化到 `s_min`。

`--direct-strength-head` 是“不使用 similarity residual prior”的消融：网络用 sigmoid 直接预测 `[s_min,s_max]` 内的 strength。第一版不把单网络 heteroscedastic variance 当作 epistemic uncertainty，也不启用 uncertainty fallback；未来可在 checkpoint schema 之上增加 controller ensemble。

### Retrieval-use gate

```text
p_use = sigmoid(g_theta(z))
```

当 `p_use < gate_threshold` 时，该样本从独立纯高斯噪声开始，并走 `T-1,...,0` 完整 reverse trajectory。它不是把 reference 加噪到 `strength=1.0`。检索 similarity threshold 的 fallback 先于 controller gate；缺失 reference 也直接走 Original 路径。

运行优先级为：

1. `rag.enabled=false`：Original VerbalTS，不加载 index、LongCLIP 或 controller；
2. `rag.mode=retrieval_only`：直接使用 reference，不加载 controller；
3. adaptive 与显式非负 `rag.start_step` 同时配置：立即报错；
4. `adaptive_controller.enabled=true`：逐样本 controller；
5. 否则保持 phase-1 fixed-strength RAG；
6. retrieval 自身 fallback 永远先于 controller。

同一测试 query 的默认 10 个候选共享 reference、strength、start step 和 gate decision，只改变 diffusion noise；pointwise median 逻辑不变。adaptive 官方配置拒绝 `diverse_reference=true`。

## 2. Teacher 构建

Teacher builder 只读取 `train_text_caps.npy`、`train_ts.npy` 和 train-only retrieval index。每个训练 caption 作为 query 时，检索会排除相同 time-series sample ID：Synth-M 排除该样本，Weather 会一次排除该 sample 的全部三条 caption，而不是只排除当前 caption row。若排除后不足 Top-K，会报告 query、requested K 和 eligible 行数。

默认 action grid 为 `0.20,0.35,0.50,0.65,0.80,0.95`，另生成纯高斯 Original 和仅用于诊断的 Retrieval-only。默认调用只跑 32 个分层抽样 query；全量构建必须显式传 `--max-queries -1 --allow-full-build`。每个 query 单独缓存，`--resume` 校验 build signature 后跳过已完成项，并持续写 progress JSON。

每个候选计算逐样本 semantic score `C_ij` 与 dimension-normalized RMSE copy distance。scorer 接口是：

```python
class TeacherSemanticScorer:
    def score(self, query_captions, generated_ts, ts_lengths):
        ...
```

已提供 `CTTPTeacherSemanticScorer`、顺序读取外部分数的 `PrecomputedTeacherSemanticScorer`，测试可直接注入 fake scorer。copy constraint 可关闭；阈值必须由 controller-train/独立训练序列随机 pair 的低分位数估计，`estimate_copy_threshold` 只采样不同序列对，不构造 `N^2` 矩阵。禁止在 test 上校准 `epsilon_sem`、copy threshold 或 gate threshold。

可行集与 soft teacher 为：

```text
F_i = {s_j: C_ij >= C_original_i - epsilon_sem
             and distance(y_ij,x_ref_i) >= copy_threshold}
w_ij = exp(-s_j/tau_teacher) / sum_{k in F_i} exp(-s_k/tau_teacher)
s*_i = sum_j w_ij s_j
v*_i = sum_j w_ij (s_j-s*_i)^2
```

若 `F_i` 为空，`gate_target=0`，strength target 为 `NaN` 并在 loss 中 mask；否则 `gate_target=1`。`v*` 仅保存为 ambiguity 诊断，不参与第一版 inference fallback。

使用 CTTP teacher 时 manifest 明确记录：

```json
{"teacher_semantic_scorer":"cttp","metric_optimized":true}
```

若论文仍用同一 CTTP checkpoint 作为主要评价指标，这会有 metric-targeting 风险。正式实验应至少补充独立 semantic scorer、属性级指标，或使用与最终评价不同的 CTTP checkpoint 验证。

builder 还输出 fixed-strength sweep JSON，包括 FID、J-FTSD、CTTP、copy rate 和 generated-to-reference distance。CTTP scorer 可提供 embedding 时，FID/J-FTSD 以本次 train-query subset 的真实序列为 reference；pilot 数字只是管线检查，不能当论文结果。如果固定 strength 与 FID/J-FTSD 没有整体趋势，工具只发警告；这会削弱 minimum-feasible-strength teacher 的解释，而不会被改写成正面结论。

## 3. 数据格式与泄漏防护

Teacher 输出一个压缩 NPZ 和一个 JSON manifest。NPZ 包含：

- query/reference sample IDs、caption IDs；
- query/reference embeddings 与 pair features；
- Top-K sample/caption IDs、similarities；
- raw/normalized score features；
- candidate strengths/start steps；
- per-candidate semantic scores/copy distances；
- Original 与 Retrieval-only 诊断分数；
- feasible masks、teacher weights、soft targets、ambiguity variance、gate targets；
- 以 query sample ID 分组的 controller train/validation split IDs；
- CTTP 可用时的 sweep embeddings。

Manifest 记录 schema、dataset/split、index SHA-256 和 metadata、LongCLIP、VerbalTS checkpoint SHA-256、scorer、grid、seed、exclusion policy、feature 定义、只用 controller-train 拟合的 normalization/q05/q95、semantic/copy threshold 绝对值和来源、pilot/progress 与 sweep。loader 拒绝非 `split=train`、字段缺失或 schema 不匹配的数据；同一 sample 的多个 caption 不会跨 controller train/validation。

## 4. Pilot、训练与评估命令

以下命令均从项目根目录运行。先跑单 run、8 个 query 的 Synth-M pilot：

```bash
python tools/build_strength_teacher.py \
  --dataset-folder ./datasets/synth-m --dataset-name synth-m \
  --index-path ./cache/rag/synth-m/train_longclip.npz \
  --longclip-path ./save/Longclip \
  --diff-config configs/synth-m/diff/model_text2ts_dep.yaml \
  --cond-config configs/synth-m/cond/text_msmdiffmv.yaml \
  --verbalts-checkpoint ./save/synth-m_eval/text2ts_msmdiffmv/0/ckpts/model_best_loss.pth \
  --cttp-config ./save/synth-m_cttp/model_configs.yaml \
  --cttp-checkpoint ./save/synth-m_cttp/clip_model_best.pth \
  --base-patch 4 --multipatch-num 3 --patch-length 3 \
  --max-queries 8 --samples-per-action 1 --seed 42 \
  --output-npz ./cache/adaptive/synth-m/run0_teacher_pilot.npz \
  --resume
```

Weather pilot 将 dataset/config/checkpoint 路径替换为 `Weather`，并使用 `--base-patch 1`。正式 teacher 模板必须在 validation protocol 锁定阈值后才运行：

```bash
python tools/build_strength_teacher.py \
  --dataset-folder ./datasets/synth-m --dataset-name synth-m \
  --index-path ./cache/rag/synth-m/train_longclip.npz \
  --longclip-path ./save/Longclip \
  --diff-config configs/synth-m/diff/model_text2ts_dep.yaml \
  --cond-config configs/synth-m/cond/text_msmdiffmv.yaml \
  --verbalts-checkpoint ./save/synth-m_eval/text2ts_msmdiffmv/0/ckpts/model_best_loss.pth \
  --cttp-config ./save/synth-m_cttp/model_configs.yaml \
  --cttp-checkpoint ./save/synth-m_cttp/clip_model_best.pth \
  --base-patch 4 --multipatch-num 3 --patch-length 3 \
  --max-queries -1 --allow-full-build --resume \
  --strength-grid 0.20,0.35,0.50,0.65,0.80,0.95 \
  --epsilon-sem 0.01 --epsilon-sem-source controller-validation-v1 \
  --copy-threshold 0.05 --copy-threshold-source controller-train-random-pairs-q05 \
  --samples-per-action 10 --seed 42 \
  --output-npz ./cache/adaptive/synth-m/run0_teacher.npz
```

训练 score-plus-pair controller：

```bash
python tools/train_strength_controller.py \
  --teacher-npz ./cache/adaptive/synth-m/run0_teacher.npz \
  --output-dir ./save/adaptive_controller/synth-m/score_plus_pair \
  --feature-mode score_plus_pair --epochs 100 --patience 12 \
  --lambda-gate 1 --lambda-strength 1 \
  --lambda-monotonic 0.2 --lambda-residual 0.01 \
  --device cuda:0 --seed 42
```

score-only 将 `--feature-mode` 改为 `score_only`。Synth-M/Weather adaptive 评估：

```bash
bash scripts/synth-m/eval_adaptive.sh
bash scripts/Weather/eval_adaptive.sh
```

先做 checkpoint smoke：

```bash
bash scripts/synth-m/eval_adaptive.sh \
  --start_runid 0 --n_runs 1 --batch_size 4 --eval_max_batches 1
```

可用 CLI 覆盖：`--rag_adaptive_enabled`、`--rag_controller_checkpoint_path`、`--rag_controller_feature_mode`、`--rag_controller_min_strength`、`--rag_controller_max_strength`、`--rag_controller_base_gamma`、`--rag_controller_max_residual`、`--rag_controller_gate_threshold`。checkpoint 会校验 dataset identity、index SHA-256、embedding dimension、feature mode、normalization schema、diffusion steps 和 controller version。

## 5. Loss 与训练输出

最小化的总 loss 使用正权重加和（原草案中的负号会鼓励增大误差，因此未采用）：

```text
L_gate = BCEWithLogits(gate_logit, gate_target)
L_step = Huber((T-1)*s, round((T-1)*s*))  # 只在 gate_target=1
L_mono = max(0, s_a-s_b), c_a > c_b + eta
L_residual = delta^2
L = lambda_g L_gate + lambda_s L_step + lambda_m L_mono + lambda_r L_residual
```

正类权重通过 `--gate-pos-weight` 显式设置，默认 1；训练启动时打印并保存类别数量。early stopping 只看 controller-validation loss。输出包括 `best.pt`、`last.pt`、training/calibration JSON、epoch CSV、class balance、strength/start-step MAE、AUROC/F1/precision/recall、monotonic violation、residual 分布、validation predictions 和独立 checkpoint manifest。

## 6. Trace、预测与分析

retrieval trace 新增 Top-1/Top-2/margin/mean/std/entropy、base strength、residual、predicted strength/start step、gate probability/threshold、action、checkpoint、feature mode 和 fallback reason。fallback reason 区分 `retrieval_below_threshold`、`controller_gate_reject`、`missing_reference`；RAG 关闭时 prediction metadata 使用 `original_rag_disabled`，不会加载检索资源；若显式配置 `rag.trace_path`，也可以写出 Original 决策 trace。

`rag_predictions.npz` 保留原 candidates/median/target/reference IDs，并新增 per-sample CTTP、controller strength/start step/gate probability、selected reference ID、copy distance、action 和 fallback reason。

```bash
python tools/analyze_strength_controller.py \
  --teacher-npz ./cache/adaptive/synth-m/run0_teacher.npz \
  --controller-predictions ./save/adaptive_controller/synth-m/score_plus_pair/validation_predictions.npz \
  --evaluation-npz ./save/synth-m_adaptive_eval/text2ts_msmdiffmv/0/rag_predictions.npz \
  --retrieval-trace ./save/synth-m_adaptive_eval/text2ts_msmdiffmv/0/retrieval_trace.jsonl \
  --fixed-sweep ./cache/adaptive/synth-m/run0_teacher_fixed_sweep.json \
  --copy-threshold 0.05 --output-dir ./save/adaptive_analysis/synth-m/run0
```

核心依赖只输出 CSV/JSON；matplotlib 不是必需依赖。输出覆盖 similarity-strength 关系、teacher/predicted step error、similarity decile CTTP、gate/fallback/copy rate、strength histogram、monotonic violation、teacher target error 和 fixed sweep。

## 7. 实验组

所有 test 设置在 validation 结束后锁定。每组至少运行 seeds/runs 0、1、2，并保存命令与 checkpoint hash。

| 组 | 可复现设置/命令模板 |
|---|---|
| 1. Original | `bash scripts/synth-m/eval.sh`，或 adaptive 脚本加 `--rag_enabled false` |
| 2. Retrieval-only | `bash scripts/synth-m/eval_rag.sh --rag_mode retrieval_only --rag_top_k 4 --rag_selection top1` |
| 3. Fixed 0.4 | `bash scripts/synth-m/eval_rag.sh --rag_strength 0.4 --rag_top_k 4 --rag_selection top1` |
| 4. validation-best global fixed | 只在 validation sweep 选 `S_BEST`，test 用 `--rag_strength S_BEST`，不得重选 |
| 5. Handcrafted continuous | 训练时 `--max-residual 0 --disable-gate-loss`，评估匹配 `--rag_controller_max_residual 0 --rag_controller_gate_threshold 0` |
| 6. Learned score-only | train `--feature-mode score_only`，评估匹配 checkpoint/config |
| 7. Learned score-plus-pair | 默认完整配置 |
| 8. Without gate | 重训加 `--disable-gate-loss`，评估加 `--rag_controller_gate_threshold 0` |
| 9. Without monotonic | 重训加 `--lambda-monotonic 0` |
| 10. Without residual prior | 重训加 `--direct-strength-head --lambda-residual 0` |
| 11. Shuffled features | 重训加 `--shuffle-retrieval-features`，独立输出目录 |
| 12. Random reference + diffusion | fixed RAG：`--rag_mode random_reference --rag_adaptive_enabled false --rag_strength S_BEST` |
| 13. Validation-only oracle | 直接从 validation teacher candidate grid 按可行集/oracle 选 action；只报告 upper bound，不在 test 建 teacher |

关键检验不是只比较 0.4，而是 `LearnedController > validation-best fixed strength` 和 `LearnedController > handcrafted continuous mapping`。

建议成功标准：CTTP 相对本地 Original 非劣（预注册约 1% margin）；FID/J-FTSD 仍优于 Original；learned 优于 validation-best fixed；full 优于 score-only；shuffle 后下降；copy rate 明显低于 Retrieval-only；三个 run 方向一致；test 参数在 validation 后完全锁定。

## 8. 已知限制与正式实验顺序

- 当前代码提供完整接口、pilot/resume 路径和轻量测试，但仓库提交本身不包含正式 teacher、controller 或 test 结果。
- 完整 teacher 成本约为 `query_count * (grid+Original) * samples_per_action` 次生成，必须先用 8/32 query pilot 验证显存、耗时和 scorer 数值。
- pilot subset 的 FID/J-FTSD 方差很大，只用于单调性前置检查。
- 单控制器不提供可信 epistemic uncertainty；ensemble fallback 只预留 schema 扩展方向。
- CTTP-targeting 风险必须用独立 scorer/属性指标/独立 CTTP checkpoint 复核。
- 正式顺序应为：三 run pilot → controller-train 阈值估计 → validation fixed sweep/超参选择 → 锁定 config/hash → 三 run full teacher/controller → validation sanity check → 一次性 test evaluation。

轻量测试：

```bash
python tests/run_numpy_retrieval_tests.py
python tests/run_adaptive_numpy_tests.py
```

完整 CPU smoke（需要 PyTorch，但不加载真实 LongCLIP/checkpoint/GPU）：

```bash
python tests/run_rag_smoke_tests.py
# 或 python -m pytest tests/test_retrieval.py tests/test_adaptive_controller.py -q
```
