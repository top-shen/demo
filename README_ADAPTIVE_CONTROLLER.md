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

已有 Synth-M pilot teacher 后，可用一条命令依次训练两种控制器、完成各自的单 batch adaptive evaluation，并生成分析输出：

```bash
bash scripts/synth-m/run_adaptive_pilot_end_to_end.sh
```

该总控脚本不会启动 full teacher build。可通过环境变量覆盖 pilot 设置，例如 `DEVICE=cuda:1 EPOCHS=30 bash scripts/synth-m/run_adaptive_pilot_end_to_end.sh`。

8-query smoke 全部通过后，使用下面的一条命令运行默认 256-query 的
train-only 中型开发 pilot：

```bash
bash scripts/synth-m/run_adaptive_medium_pilot.sh
```

该脚本可断点续建 Teacher，并训练、分析 `score_only` 与
`score_plus_pair`。它只使用 Teacher 内按 sample ID 分组的
controller-train/controller-validation，不运行 dataset test evaluation。默认参数可通过环境变量覆盖，例如
`MAX_QUERIES=512 SAMPLES_PER_ACTION=3 DEVICE=cuda:1 bash scripts/synth-m/run_adaptive_medium_pilot.sh`。
`epsilon_sem=0.01` 与 copy threshold `0.05` 在此阶段仍标记为 train-only
provisional 参数，不应当作已经锁定的论文设置。

若中型 pilot 显示 copy constraint 未激活、gate 恒为 use，或 learned strength
未超过常数/prior 基线，可在不重新运行 diffusion 的情况下执行修复诊断：

```bash
bash scripts/synth-m/run_adaptive_repair_pilot.sh
```

默认修复设置仅由训练数据得到：`epsilon_sem` 取中型 pilot Original CTTP
均值的 1%，copy threshold 取 8192 个随机不同训练序列对 RMSE 的 q05。脚本会把
重标后的 Teacher 写入新文件，不覆盖原 Teacher；随后显式记录
`negative/positive` gate class weight，并训练 `score_only`/`score_plus_pair` 的
prior+monotonic、prior without monotonic、direct head without prior/monotonic
六组诊断模型。最后输出 learned、similarity prior、controller-train constant
三类 strength MAE，以及仅在 controller-validation 上选择的 balanced-accuracy
gate threshold。该流程不生成新时间序列，也不读取 dataset valid/test split。

若共享 trunk 出现 gate/strength 多任务干扰，可继续运行独立轻量 task tower
诊断：

```bash
DEVICE=cpu bash scripts/synth-m/run_adaptive_decoupled_pilot.sh
```

`--separate-task-towers` 为可选训练参数，默认关闭，因此旧 checkpoint 的网络结构和
加载行为保持不变。启用后 gate 与 strength 使用独立的小型 projection/trunk，完整
`score_plus_pair` 模型仍受 500K 参数上限约束。诊断脚本显式提高 gate loss 权重，
并比较原始 `max_residual=0.15`、较宽 residual 以及 direct strength head；这些都是
controller-validation 消融，不会自动替代正式方法定义。

若独立塔出现同时超过 constant strength/start-step 基线且 gate 有区分力的候选，
先用三初始化 seed 与 shuffled-feature 负对照检查稳定性：

```bash
DEVICE=cpu bash scripts/synth-m/run_adaptive_robustness_pilot.sh
```

默认对同一 train-only Teacher 使用 seeds 42/43/44，并为每个 seed 训练真实特征和
完全打乱 retrieval features 的成对负对照。只有真实特征多数 seed 稳定优于常数、
且 gate 指标整体优于 shuffled control，才继续扩大 Teacher；该检查不读取 dataset
valid/test。

上述鲁棒性 pilot 完成后，可执行最终的 1024-query × 3 samples/action 稳定性检查：

```bash
bash scripts/synth-m/run_adaptive_stable_teacher_go_no_go.sh
```

该命令只使用 Synth-M 训练集，构建 1024 个 query 的可恢复 Teacher。默认候选包含
6 个 strength 与 Original，因此总计生成 21,504 条 diffusion trajectory；中断后
重新运行同一命令会按 query 续建。原始候选和使用训练集随机不同序列对 q05 copy
阈值、Original CTTP 均值 1% semantic margin 重标后的 Teacher 分开保存，不覆盖
已有 256-query 结果。

脚本随后使用 seeds 42/43/44 训练 `score_plus_pair`、`score_only` 和 shuffled-pair
三组 direct-head 独立 task tower，并输出
`save/adaptive_pilot_analysis/synth-m/stable_q1024_spa3_seed42_q05_eps1pct/go_no_go.json`。
只有 pair 模型的 strength/start-step 误差在多数 seed 和跨 seed 均值上超过
controller-train constant，且 strength 相对 shuffled control 的优势也同时满足多数
seed 与均值条件时，才报告 `GO`；否则报告 `NO_GO`。pair 相对 score-only 的增益和
gate AUROC 作为独立诊断字段报告，不与 adaptive-strength 的可学习性判定混为一谈。
这是 train-only 的方法可学习性检查，不是 dataset validation/test 结论，也不会自动
启动正式评估。

脚本默认直接开始构建，不要求额外确认变量。若 GPU 不在 0 号卡，可设置
`GENERATION_DEVICE=cuda:1`。原始 Teacher 和训练 checkpoint 默认复用；需要重新进入
Teacher 的 resume/recombine 流程或重训时，分别设置 `FORCE_TEACHER_BUILD=1` 或
`FORCE_RETRAIN=1`。若有意覆盖 q05/1% 阈值，必须同步设置新的 `LABEL_TAG`，避免不同
研究定义共用输出目录。

若稳定 Teacher 输出 `decision=GO` 但 `gate_signal=NOT_SUPPORTED`，下一阶段只验证
关闭 gate 的 adaptive strength，不允许 gate 拒绝样本影响主结果。运行完整的
Synth-M dataset-validation benchmark：

```bash
bash scripts/synth-m/run_adaptive_validation.sh
```

该脚本显式传递 `--eval_split valid`，并在汇总前逐个检查保存的 `eval_configs.yaml`
和 prediction split；任何 `test` artifact 或 `max_batches>0` 的 partial pilot 都会被
拒绝。现有配置仍默认 `split: test`，所以旧的正式评估行为保持兼容，但不得用旧
`eval_adaptive.sh` 默认值做参数选择。

validation 条件包括 Original、Retrieval-only、fixed
`0.20/0.35/0.40/0.50/0.65/0.80/0.95`、handcrafted similarity prior、learned
score-plus-pair、learned score-only 和 shuffled-pair；每组默认评估 VerbalTS runs
0/1/2。稳定 Teacher 的 controller seed 42 在读取 validation 结果前固定为主 checkpoint，
所有 adaptive 条件使用 `gate_threshold=0`。已完成条件通过 `results.csv` 自动复用。

validation-best fixed strength 的规则在读取结果前锁定为：先要求平均 CTTP 不低于
Original 的 99%，再选择平均 J-FTSD 最低者；若相同，依次使用更低 FID 和更高 CTTP
作为 tie-breaker。最终输出：

```text
save/adaptive_validation/synth-m/stable_q1024_spa3/validation_decision.json
save/adaptive_validation/synth-m/stable_q1024_spa3/validation_summary.csv
```

只有 `validation_decision.json` 报告 `GO_TO_TEST` 后才锁定 best-fixed strength、controller
checkpoint/hash 和 gate-disabled 配置，并运行一次 test。`STOP_OR_REVISE` 不得通过查看
test 结果补救。

完整 CPU smoke（需要 PyTorch，但不加载真实 LongCLIP/checkpoint/GPU）：

```bash
python tests/run_rag_smoke_tests.py
# 或 python -m pytest tests/test_retrieval.py tests/test_adaptive_controller.py -q
```

## 9. Validation-only Oracle ceiling 诊断

dataset validation 表明 learned controller 没有超过 validation-best fixed strength 后，可运行一个严格的、仅用于诊断的 policy-specific empirical ceiling。它回答的是：在已经保存的 fixed-strength action grid 内，如果事后知道每个 validation 样本该选哪个 strength，当前 action space 是否仍有可用 headroom。它不是数学意义上的全局上界，也不是可部署模型；Oracle 使用了 validation label/metric，不能替代 controller，更不能据此自动查看 test。

Oracle 只读取每个 fixed condition 已保存的最终 `predictions`，即 10 个随机候选的逐点中位数。`candidates` 字段被明确忽略，禁止在 10 个随机候选内部再次选择，否则会额外利用采样结果并严重夸大 ceiling。每个 run 只在同一 run 的 fixed actions 内选择，run 0/1/2 绝不交叉混合。

运行前会 fail-fast 审计：

- split 必须为 `valid`，`max_batches <= 0`，sample ID 完整且无重复；
- 不同 strength 按 sample ID 对齐，并核对 caption ID、caption、target 和 reference ID；
- 目录 strength 与保存配置一致，CTTP checkpoint/config/training statistics 身份一致；
- 保存的逐样本 CTTP 均值能够复现 results.csv，离线重算路径能够复现每个 fixed action 的 CTTP/FID/J-FTSD；
- 所有输入路径、配置摘要和 SHA-256 写入 manifest，结束时再次核对输入未被修改。

默认 action grid 从真实目录和 `eval_configs.yaml` 联合发现，不硬编码。四类 policy 为：

1. `max_cttp`：逐样本选择 CTTP 最大的 fixed action；CTTP 容差内依次偏好非 near-reference、更大 reference distance、更高 strength、固定 action index。
2. `non_reference_max_cttp_qXX`：只在 `distance >= threshold` 的 action 中最大化 CTTP；无可行 action 时，若已有 Original prediction 则退回 Original，否则选择 reference distance 最大的 fixed action，并保留 `constraint_unmet=true`。
3. `pareto_*_lambda_*`：逐样本归一化 CTTP 和 strength，最大化 `normalized_cttp - lambda * normalized_strength`。同时报告 unconstrained 与 q05 non-reference constrained family，lambda 固定为 `0,0.02,0.05,0.10,0.20,0.50,1,2,5`。报告三 run 共用 lambda，以及每个 run 独立挑 lambda 的更乐观 ceiling；主要判断只参考 shared-lambda。
4. `original_hybrid`：仅在 Original 的逐样本 prediction 已经保存时可用。若某个 fixed action 同时满足 `CTTP >= 0.99 * Original per-sample CTTP` 和 non-reference constraint，则选择最低 strength，否则回退 Original。若 Original predictions 缺失，标记 `hybrid_oracle_available=false`，绝不补跑生成。

near-reference 使用 generated-to-reference 的维度归一化 RMSE，`distance < threshold` 即标记 near-reference。默认 q05 必须来自 train-only Teacher manifest/calibration；当前 Synth-M 估计约为 `0.6874167`。若 train-only 文件还保存 q01/q10，则同时输出敏感性分析；缺少时明确标记 unavailable。该指标只是 reference-retention 风险启发式，不等同于已证明的抄袭或 exact copy。

Oracle 挑选完成后会形成完整的新 prediction embedding 集合，并在这个重组后的集合上重算 CTTP、FID 和 J-FTSD。FID/J-FTSD 是集合级指标，绝不能对各 fixed action 已有的 aggregate 数字做加权平均。max-CTTP 和 Pareto 都直接使用 CTTP 做事后选择，因此结果存在明确的 metric-targeting/乐观偏差。

运行命令：

```bash
bash scripts/synth-m/run_oracle_ceiling.sh
```

这条命令不会调用 `run.py`、diffusion generation、Teacher/controller 训练或 test。可覆盖离线编码设备，例如 `DEVICE=cpu bash scripts/synth-m/run_oracle_ceiling.sh`。只有在 reference 本身随 strength 改变、且明确希望诊断 reference+strength 联合上限时，才设置 `ALLOW_JOINT_REFERENCE_ORACLE=1`；报告会据此标为 joint oracle，不能冒充 strength-only 结果。

历史 validation evaluator 没有显式调用 `CTTP.eval()`，而当前 CTTP 的时间序列分支配置含 `dropout=0.1`。因此历史 aggregate metrics 是 frozen-parameter、train-mode dropout scorer 的一次随机实现，不能用 deterministic eval-mode 重编码后要求 `1e-3` 绝对一致。Oracle 脚本默认使用 `CTTP_RUNTIME_MODE=legacy_train`，保持参数冻结并用固定 seed 做 3 次 Monte Carlo 重编码；每个 fixed action 与历史结果按预先声明的 5% 对称相对误差做 stochastic compatibility audit，同时在每个 Oracle metric 中保存 scorer-repeat 标准差。该 5% 只用于判断是否仍属于同一个历史 scorer 分布，不会把历史 aggregate FID/J-FTSD 混入重组集合计算。可以显式设置 `CTTP_RUNTIME_MODE=eval` 做 deterministic sensitivity analysis，但它与历史 Original aggregate 不再是同一评价口径，不能替换 primary legacy-compatible 诊断。

输出位于 `save/adaptive_validation/synth-m/stable_q1024_spa3/oracle_ceiling/`：manifest、integrity report、每 run/汇总 metrics、decision、strength usage、Pareto frontier、near-reference sensitivity，以及每个 run 的 CSV/NPZ assignments。研究分类只能是：

- `USABLE_HEADROOM_PRESENT`：non-reference Pareto 或 hybrid 同时满足 CTTP 为 Original 的至少 99%、FID/J-FTSD 优于 Original、J-FTSD 优于 best-fixed、near-reference 不超过 cap，且至少 2/3 run 方向一致；说明瓶颈更可能在 Teacher/features/controller。
- `METRIC_TARGETED_OR_COPY_HEADROOM_ONLY`：只有 max-CTTP/unconstrained 等乐观策略改善，安全 non-reference/hybrid 不成立；不能据此支持部署。
- `NO_USABLE_HEADROOM_IN_EVALUATED_GRID`：在当前 fixed grid、reference 和 validation Oracle policies 下未观察到可用 headroom；不能外推为任何 adaptive strength 都不可能有效。
- `INCONCLUSIVE_ARTIFACTS_MISSING`：必要 prediction/config/threshold/alignment 或离线复现审计不成立。

无论分类为何，test 都保持未读取、未运行。依赖无关的 Oracle 单元测试命令为：

```bash
python tests/run_oracle_ceiling_tests.py
```
