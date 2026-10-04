# MemReason-RL：ADNI 异步纵向神经认知记忆建模实施规范

> **用途**：本文件用于直接交给代码大模型 / research agent 执行。目标不是让模型“自由发挥”，而是按照本文定义的数据假设、模块接口、训练阶段、损失函数、验证门槛和实验矩阵逐步实现并汇报结果。
>
> **暂定全名**：**MemReason-RL: Question-Grounded Asynchronous Neuro-Cognitive Memory Dynamics with Verifiable Reinforcement Learning**
>
> **核心任务**：从 ADNI 中**不规则、异步、部分缺失**的纵向认知量表题目级记录与稀疏 T1 MRI 中，持续估计患者的潜在记忆状态（latent memory belief state），预测未来认知轨迹，并使用真实未来随访构造可验证奖励（RLVR）对未来轨迹与推理报告进行 GRPO 后训练。

---

## 0. 执行代理必须遵守的原则

1. **不要假设每个访视同时存在 MRI、MMSE、MoCA、ADAS、CDR、FAQ。**
2. **不要把数据先强行整理成固定长度的 `T × D` 完整矩阵再统一插值。** 患者历史应建模为 asynchronous event stream。
3. **真实缺失值不能填 0 后当真实观测。** 缺失用于 attention/loss/reward mask；必要时使用显式 missingness metadata，但必须做 protocol leakage 检查。
4. **MRI 是 sparse structural anchor，不是每个认知访视存在 latent state 的必要条件。**
5. **认知量表的 item/question semantics 必须保留。** 不允许只用总分替代 item-level evidence。
6. **Diagnosis、CN/MCI/AD、未来 FAQ 不参与 latent memory state 的定义。** 它们主要作为 downstream validity / outcome。
7. **GRPO 不从零训练 MRI encoder 或时序主干。** 表征和 dynamics 先用监督/自监督目标学稳定，再做 SFT，最后做 RLVR/GRPO。
8. **第一版不生成完整 future 3D MRI。** 只预测 future MRI latent state / structural representation，降低计算量与任务发散。
9. **不要为了“有 Diffusion / Agent / GRPO”而堆模块。** 只有在明确解决 stochastic future / reasoning alignment / adaptive decision 的问题时才引入。
10. 每完成一个阶段必须输出：数据量、训练/验证曲线、核心指标、失败 case、下一阶段是否满足 gate；**不满足 gate 不进入下一阶段。**

---

# 1. 科研问题与顶会/顶刊叙事

## 1.1 现有方法的问题

真实 AD 纵向队列并不是同步完整的多模态序列。患者之间存在：

- 随访长度不同；
- 随访间隔不同；
- 不同访视使用的认知量表不一致；
- 某些量表整个 instrument 缺失；
- 同一量表内部也可能存在 item 缺失；
- T1 MRI 比认知评估更稀疏；
- MRI 与量表日期可能不完全同步。

因此传统写法：

\[
X_t=[MRI_t,MMSE_t,MoCA_t,ADAS_t,CDR_t]
\]

并不符合真实数据生成过程。

## 1.2 核心研究问题

> **How can we identify and forecast an individual's latent memory state from asynchronously observed, semantically heterogeneous, and partially missing neuro-cognitive evidence?**

进一步：

> **How can real future follow-ups be converted from ordinary supervision labels into verifiable rewards that optimize an entire future neuro-cognitive trajectory rather than a single future score?**

## 1.3 核心假设

将患者记忆状态视为持续存在但不可直接观测的 latent process：

\[
z^{mem}(t)
\]

量表题目是行为观测：

\[
y^{cog}_{i,t}\sim p(y\mid z^{mem}(t), q_i)
\]

T1 MRI 是更低频的结构观测/锚点：

\[
x^{MRI}_{t_m}\sim p(x\mid z^{mem}(t_m))
\]

因此模型学习的是：

\[
p(z^{mem}(t)\mid \mathcal{O}_{\le t})
\]

以及：

\[
p(z^{mem}(t+\Delta),Y^{cog}_{t+\Delta}\mid\mathcal{O}_{\le t})
\]

其中 `O` 可以是任意不完整的认知/MRI观测子集。

## 1.4 论文贡献建议压缩为三点

### Contribution 1 — Question-Grounded Partial-Observation Modeling

把不同量表的 item 从无语义 tabular variable 重新表示为：

\[
\text{Question/Task Semantics} + \text{Response} + \text{Score} + \text{Domain}
\]

从而在 MMSE / MoCA / ADAS / CDR 等 instrument 间学习共享 cognitive construct。

### Contribution 2 — Asynchronous Neuro-Cognitive Memory Belief Dynamics

将高频但不完整的认知事件与低频 T1 MRI structural anchors 融合进连续时间 probabilistic belief state：

\[
q(z^{mem}(t))=\mathcal N(\mu_t,\sigma_t^2)
\]

显式处理：variable-length follow-up、irregular intervals、scale-level missingness、MRI missingness。

### Contribution 3 — Availability-Aware Trajectory RLVR

把真实 future cognition / future MRI / evidence / temporal trend 转成 machine-verifiable rewards，用 GRPO 优化：

\[
\text{entire future trajectory + grounded reasoning}
\]

而不只是 point-wise MAE。

---

# 2. 当前已知数据

## 2.1 认知纵向 CSV

当前文件：

```text
adni_all_longitudinal_scales_visit(1).csv
```

已核对基本统计：

- `15447` visits
- `4316` subjects
- `149` columns
- 平均 visits/subject ≈ `3.579`
- 中位数 visits/subject = `2`
- 最大 visits/subject = `21`
- ≥2 visits：`2568` subjects
- ≥3 visits：`2097` subjects
- ≥4 visits：`1703` subjects
- ≥5 visits：`1279` subjects
- ≥6 visits：`953` subjects

量表 visit-level availability：

| Scale | Present visits | Approx. proportion |
|---|---:|---:|
| ADAS | 13,170 | 85.3% |
| CDR | 14,635 | 94.7% |
| FAQ | 13,642 | 88.3% |
| MMSE | 14,467 | 93.7% |
| MoCA | 9,270 | 60.0% |
| DXSUM | 14,366 | 93.0% |

**结论**：主要 missingness 是 scale-level / visit-level，而不是简单随机 item NaN。

## 2.2 T1 MRI

用户另有 ADNI longitudinal T1 MRI，经验上平均每位患者约 3 次 T1 scan。

**MRI 不在当前 CSV 内，需要建立单独 metadata 表。**

执行代理必须要求/生成：

```text
mri_metadata.csv
```

最少字段：

```text
subject_id / rid
scan_id
scan_date
visit_code
visit_month (若可映射)
image_path
sequence_name
field_strength
site/scanner（若可得）
qc_pass
```

如有多个 T1 series 同一时间点，先定义 selection rule，不能随意重复使用。

推荐规则：优先 ADNI 推荐/已校正 T1；同 visit 多扫描时按 QC、官方 preferred scan、最近日期等固定规则选 1 个主扫描，并保留 alternate scan metadata。

---

# 3. CSV 字段分组与角色

## 3.1 Metadata

不作为 cognition observation：

```text
subject_id
rid
visit_month
visit_code
*_visit_date
*_userdate
*_present
scale_source_count
```

其中 `visit_month` / 日期只用于 time encoding 和匹配。

## 3.2 ADAS（19 columns）

```text
adas_visit_date
adas_userdate
adas_q1score ... adas_q13score
adas_q5score_cue
adas_totscore
adas_total13
adas_present
```

### 建议核心 memory evidence

```text
adas_q1score   # Word recall / immediate recall related
adas_q4score   # Delayed word recall
adas_q8score   # Word recognition
adas_q9score   # weak/secondary memory-related evidence
```

### 非 memory 主域

```text
q2  -> command / language-executive
q3  -> construction / visuospatial-executive
q5  -> naming/language
q6  -> praxis/executive
q7  -> orientation
q10 -> spoken language
q11 -> word finding
q12 -> comprehension
```

### 必须先核验

`adas_q13score` 的具体语义与不同 ADNI phase 的 harmonization 必须查对应数据字典后再加入模型。

**不能凭字段编号推断。**

### 总分

```text
adas_totscore
adas_total13
```

主要用于 prediction / downstream validation，不用于直接定义 latent memory state。

## 3.3 CDR（11 columns）

Memory：

```text
cdr_cdmemory
```

Auxiliary cognition / function：

```text
cdr_cdorient   -> Orientation
cdr_cdjudge    -> Executive/Judgment
cdr_cdcommun   -> Function/Communication
cdr_cdhome     -> Function
cdr_cdcare     -> Function
```

Global severity / validation：

```text
cdr_cdglobal
cdr_cdrsb
```

CDR memory 是 clinician/informant ordinal rating，不等同于 performance-based recall test；应保留独立 scale/instrument embedding 或 decoder。

## 3.4 FAQ（14 columns）

FAQ 全部作为 **Function / downstream outcome**，不直接构造 `z_mem`。

```text
faq_faqfinan
faq_faqform
faq_faqshop
faq_faqgame
faq_faqbevg
faq_faqmeal
faq_faqevent
faq_faqtv
faq_faqrem
faq_faqtravl
faq_faqtotal
```

即便 `faq_faqrem` 名称含 remember，也应视为“日常功能后果”，而不是纯 memory performance。

## 3.5 MMSE（42 columns）

### Orientation

```text
mmse_mmyear
mmse_mmmonth
mmse_mmday
mmse_mmseason
mmse_mmhospit
mmse_mmfloor
mmse_mmcity
mmse_mmarea
mmse_mmstate
```

### Memory — registration/acquisition related

```text
mmse_word1
mmse_word2
mmse_word3
mmse_mmtrials   # learning efficiency / trials
```

### Attention / working memory

```text
mmse_mmd
mmse_mml
mmse_mmr
mmse_mmo
mmse_mmw
mmse_worldscore
```

### Memory — delayed recall core

```text
mmse_word1dl
mmse_word2dl
mmse_word3dl
```

### Language / praxis / executive

```text
mmse_mmwatch
mmse_mmpencil
mmse_mmrepeat
mmse_mmhand
mmse_mmfold
mmse_mmonflr
mmse_mmread
mmse_mmwrite
mmse_mmdraw
```

### 数据极少，默认 drop

```text
mmse_mmltr1 ... mmse_mmltr7
```

这些字段在当前表中有效值极少，默认不用于 v0；执行时再次统计非空数并保存 audit report。

### Global

```text
mmse_mmscore
```

用于 forecast / validation，不直接定义 memory latent。

## 3.6 MoCA（36 columns）

### Executive / visuospatial

```text
moca_trails
moca_cube
moca_clockcon
moca_clockno
moca_clockhan
moca_abstran
moca_absmeas
```

### Language / naming

```text
moca_lion
moca_rhino
moca_camel
moca_repeat1
moca_repeat2
moca_ffluency
```

### Attention / working memory

```text
moca_digfor
moca_digback
moca_letters
moca_serial1 ... moca_serial5
```

### Memory core

```text
moca_delw1 ... moca_delw5
```

**重要**：当前数据中这些字段取值可达 `0~3`，不能简单当作 binary correct/incorrect。必须依据 ADNI MoCA 数据字典确认其编码是否包含 free recall / category cue / multiple choice cue 等层级信息，再设计 response text 与 ordinal loss。

### Orientation

```text
moca_date
moca_month
moca_year
moca_day
moca_place
moca_city
```

### Global

```text
moca_moca
```

注意：当前数据中 `moca_present` 的 visit 数明显高于 `moca_moca` 总分非空数，因此不能依赖总分作为唯一 MoCA 标记。

## 3.7 DXSUM（22 columns）

全部归入 `Clinical outcome / downstream label`：

```text
dxsum_diagnosis
...
dxsum_diagnosis_label
```

**绝对不允许作为 latent memory encoder 的输入。**

用于：

- CN/MCI/AD downstream probing
- conversion prediction
- external clinical validity
- stratified analysis

---

# 4. Cognitive Item Dictionary：必须首先建立

创建：

```text
configs/cognitive_item_dictionary.yaml
```

每个 item 至少包含：

```yaml
moca_delw1:
  instrument: MoCA
  domain: Memory
  construct: DelayedRecall
  task_description: "Delayed recall of a previously learned word"
  value_type: ordinal
  valid_values: [0, 1, 2, 3]   # 仅在数据字典核验后保留
  direction: verify             # higher=better / higher=worse
  core_memory: true
  use_for_state: true
  use_for_aux: false
  use_for_outcome: false
  verified_from_dictionary: false
```

必须有字段：

```text
instrument
domain
construct
task_description
value_type
range/categories
direction
core_memory
use_for_state
use_for_aux
use_for_outcome
verified_from_dictionary
```

### 强制规则

- `verified_from_dictionary=false` 的 item **不得进入最终主实验**。
- 可以进入 exploratory run，但必须明确标记。
- ADAS q13、MoCA delayed word 编码、任何 range 异常字段必须优先核对。

---

# 5. MRI 与认知访视匹配

## 5.1 不要求严格同步

定义 cognitive event 时间 `t_c`，MRI 时间 `t_m`。

允许：

\[
|t_c-t_m|\le \delta
\]

其中 `δ` 需要做 sensitivity：

```text
30 days
60 days
90 days
180 days (仅 exploratory)
```

使用时间衰减权重：

\[
w_{cm}=\exp(-|t_c-t_m|/\tau)
\]

### 主实验建议

优先：

- 同一 visit code；或
- 日期差 ≤ 90 天；
- 多个 MRI 满足时选时间最近、QC 通过者。

必须输出：

```text
reports/mri_cognition_matching_stats.csv
```

包含：

- matched pairs 数量
- date difference 分布
- 每人 matched MRI 数
- unmatched cognition visits
- unmatched MRI scans

---

# 6. 患者数据表示：Asynchronous Event Stream

不要先构造固定：

```text
[visit, MRI, MMSE, MoCA, ADAS, CDR]
```

而是构造：

\[
\mathcal E_p=\{e_1,e_2,...,e_{K_p}\}
\]

其中：

\[
e_k=(t_k,type_k,payload_k,mask_k)
\]

`type_k` 可为：

```text
COG_MMSE
COG_MOCA
COG_ADAS
COG_CDR
FUNCTION_FAQ
MRI_T1
```

或者更细化为 item event。

推荐实践：

- 同一 instrument 同一 visit 的 item 先形成一个 `instrument event`；
- instrument 内保留 item tokens；
- 同一日期的多个 instrument events 作为同一 time bucket，但不要求全部存在；
- MRI 是独立 event。

示例：

```text
M0  MRI
M0  MMSE(items)
M0  CDR(items)
M6  ADAS(items)
M12 MRI
M12 MoCA(items)
M12 CDR(items)
M18 MoCA(items)
M24 MMSE(items)
M24 FAQ(items)
M36 MRI
```

不存在的模态/量表：**不生成伪 token。**

---

# 7. Module 1 — Question-Grounded Cognitive Event Encoder

## 7.1 输入

每个 item：

\[
c_{i,t}=(q_i,a_{i,t},s_{i,t},d_i,r_i)
\]

- `q_i`: task/question semantics
- `a_i`: response category / correctness / cue level
- `s_i`: numeric score
- `d_i`: cognitive domain
- `r_i`: instrument identity

构造文本模板：

```text
[Instrument] MoCA
[Domain] Memory
[Task] Delayed recall of a previously learned word
[Response] <decoded response category>
[Score] <raw score>
[Time] Month 12
```

注意：不要把 diagnosis / future label 写进 prompt。

## 7.2 Semantic branch

\[
e^{sem}_{i,t}=E_{text}(q_i,d_i,r_i,a_i)
\]

v0 推荐：

- 冻结或 LoRA 的轻量 LLM/text encoder；
- 不要求大模型端到端吃 3D MRI；
- semantic encoder 主要负责 construct grounding。

## 7.3 Numeric branch

\[
e^{num}_{i,t}=E_{num}(s_{i,t}, value\_type_i)
\]

对不同 value type：

- binary → embedding
- ordinal → ordinal embedding / scalar projection
- continuous → normalized scalar + MLP

## 7.4 Fusion

\[
g_{i,t}=\sigma(W_g[e^{sem}_{i,t};e^{num}_{i,t}])
\]

\[
e^{cog}_{i,t}=g_{i,t}\odot e^{sem}_{i,t}+(1-g_{i,t})\odot e^{num}_{i,t}
\]

输出：`item cognitive token`。

---

# 8. Module 2 — Domain-Aware Cognitive Aggregation

域：

```text
Memory
Executive
Attention
Language
Orientation
```

Function 不参与 memory state 主定义。

对每个 domain 使用 learnable query：

\[
h_t^d=Attn(q_d,\{e^{cog}_{i,t}:domain_i=d\})
\]

如果某个 domain 在该访视没有任何观测：

- 不创建假观测；
- 由 temporal prior / history 提供 belief；
- domain availability mask 进入 aggregation。

输出：

\[
H_t^{cog}=\{h_t^{mem},h_t^{exec},h_t^{att},h_t^{lang},h_t^{ori}\}
\]

---

# 9. Module 3 — Sparse MRI Structural Encoder

输入：3D T1 MRI。

v0 backbone 可选：

- 3D Swin Transformer
- 3D ViT
- 已有可信 MRI encoder

要求输出 spatial tokens：

\[
V_t=\{v_{1,t},...,v_{P,t}\},\quad v_j\in\mathbb R^d
\]

**禁止在 grounding 前直接 global average pooling。**

可同时保留 global token：

\[
g_t^{MRI}=Pool(V_t)
\]

但 spatial tokens 必须保留用于 question-to-anatomy grounding。

MRI preprocessing 至少统一：

- orientation
- voxel spacing / resampling
- skull stripping（若 pipeline 要求）
- bias field correction（按既有预处理保持一致）
- intensity normalization
- crop/pad
- QC

必须保存 preprocessing manifest。

---

# 10. Module 4 — Question-to-Anatomy Grounding

仅在 MRI 与 cognition 可匹配的窗口内训练。

认知 item 作为 Query：

\[
Q_i=W_Qe^{cog}_{i,t_c}
\]

MRI spatial token：

\[
K=W_KV_{t_m},\quad V'=W_VV_{t_m}
\]

\[
A_{ij}=softmax(\frac{Q_iK_j^T}{\sqrt d})
\]

\[
r^{MRI}_{i,t}=\sum_j A_{ij}V'_j
\]

融合：

\[
u_{i,t}=Fuse(e^{cog}_{i,t},r^{MRI}_{i,t})
\]

定义 `u_i,t` 为 **Neuro-Cognitive Evidence Token**。

加入时间匹配权：

\[
L_{ground}=w_{cm}\cdot L_{ground}^{pair}
\]

### v0 Grounding 不要求手工 ROI mask

第一版可使用：

- contrastive correspondence
- matched vs mismatched subject/time negatives
- same-visit / near-visit positive

不要一开始就声称 attention = causal anatomy。

如果未来加入 atlas / FastSurfer ROI，可作为额外 anatomical prior / evaluation，不是 v0 前置条件。

---

# 11. Module 5 — Probabilistic Latent Memory Belief State

## 11.1 为什么是 belief state

不同时间点观测量不同，因此 latent state 的不确定性应变化。

定义：

\[
q(z^{mem}(t))=\mathcal N(\mu_t,diag(\sigma_t^2))
\]

建议 `d_mem = 16~64`，默认 32。

## 11.2 Memory-centered contextualization

Memory 主域：

\[
h_t^{mem}
\]

辅助 context：

\[
c_t=F(h_t^{exec},h_t^{att},h_t^{lang},h_t^{ori})
\]

\[
g_t=\sigma(W[h_t^{mem};c_t])
\]

\[
\tilde h_t^{mem}=h_t^{mem}+g_t\odot W_cc_t
\]

这保证 latent memory state 以 memory evidence 为主，其他认知域只是条件修正。

## 11.3 Initial posterior

首个可用 cognitive event：

\[
(\mu_0,\log\sigma_0)=PosteriorNet(\tilde h_0^{mem})
\]

若同时间有 MRI，可同时 structural correction。

---

# 12. Module 6 — Continuous-Time Predict–Update State Model

核心思想：**没有新观测时向前预测；有认知/MRI观测时修正 belief。**

## 12.1 Predict step

事件 `k-1 → k` 时间差：

\[
\Delta t_k=t_k-t_{k-1}
\]

\[
(\mu_k^-,\sigma_k^-)=F_\theta(\mu_{k-1}^+,\sigma_{k-1}^+,\Delta t_k)
\]

v0 backbone 优先：

1. time-aware Transformer / event Transformer
2. Neural CDE
3. GRU-ODE / latent ODE

先选一种主干，不并行堆三种。

## 12.2 Cognitive update

若新事件为 cognition：

\[
g_k^{Cog}=\sigma(W[\mu_k^-;h_k^{Cog}])
\]

\[
\mu_k^+=\mu_k^-+g_k^{Cog}\odot U_{Cog}(h_k^{Cog})
\]

同时更新 uncertainty：

\[
\sigma_k^+=UpdateUncertainty(\sigma_k^-,h_k^{Cog})
\]

## 12.3 MRI structural correction

若新事件为 MRI：

\[
g_k^{MRI}=\sigma(W_m[\mu_k^-;g_k^{MRI}])
\]

\[
\mu_k^+=\mu_k^-+g_k^{MRI}\odot U_{MRI}(g_k^{MRI})
\]

对应 uncertainty 应在获得结构证据后适当降低，但不要硬编码固定下降量；通过 variational/calibration objective 学习。

## 12.4 没有 MRI 的 cognition visit

完全合法：

\[
z_t^{mem}=F(\mathcal H_{cog,\le t},\mathcal H_{MRI,\le t})
\]

无需构造伪 MRI。

---

# 13. Module 7 — Masked Observation & Cross-Scale Learning

真实缺失不先填补，但训练时主动 mask 已存在的观测，提高 partial-observation robustness。

## 13.1 Masked item modeling

随机遮掉已观测 item：

\[
L_{item}=\ell(\hat y_{i,t},y_{i,t})
\]

## 13.2 Masked instrument modeling

例如：

```text
ADAS + CDR -> reconstruct MMSE
MMSE + CDR -> reconstruct MoCA
MoCA + MMSE -> reconstruct ADAS
```

使用 scale-specific decoders：

\[
\hat Y_t^k=D_k(z_t^{mem},h_t^{aux})
\]

## 13.3 Cross-scale latent consistency

由不同 instrument subset 得到的 memory state 应接近：

\[
L_{cross}=D(z_t^{A},z_t^{B})
\]

其中 A/B 是不同的量表子集。

必须避免把不同 instrument 的 score 原值直接比较。

---

# 14. Module 8 — Irregular Interpolation & Extrapolation

## 14.1 Interpolation

至少有 3 个 longitudinal observations 的 subject：随机 mask 中间 event / state。

\[
\mathcal H_{<t_i},\mathcal H_{>t_i}\rightarrow \hat z(t_i)
\]

\[
L_{int}=D(\hat z(t_i),z(t_i))
\]

更稳健的 v0 可以用观测重建代替 latent-to-latent GT：

\[
L_{int-obs}=\ell(D(\hat z(t_i)),Y_{t_i}^{obs})
\]

因为 latent 本身没有真实 GT。

## 14.2 Extrapolation

随机 cutoff `t_c`：

\[
\mathcal H_{\le t_c}\rightarrow \hat Y_{t_c+\Delta}
\]

未来 target 只对实际观测字段计算 loss。

支持任意 horizon：

```text
6 / 12 / 24 / 36 months
```

允许实际时间落入 tolerance window，例如 12±3 months，具体在 config 中定义。

---

# 15. Variable-History Training Sample Construction

对每个 subject：

1. 将所有 cognition / MRI events 按真实时间排序；
2. 随机选 cutoff `t_c`；
3. history = 所有 `t <= t_c` events；
4. target = `t > t_c` 的真实 events；
5. 随机选择 forecast horizon；
6. 找 horizon tolerance 内真实 future observations；
7. 没有 target 的 sample 跳过 forecast loss，但仍可参与 representation / masking tasks。

伪代码：

```python
for subject in subjects:
    events = sort_by_time(load_events(subject))
    cutoff = sample_cutoff(events)
    history = [e for e in events if e.time <= cutoff]

    horizon = sample([6, 12, 24, 36])
    target_events = select_future_events(
        events,
        center=cutoff + horizon,
        tolerance=HORIZON_TOLERANCE[horizon]
    )

    sample = {
        "history": history,
        "target_events": target_events,
        "forecast_horizon": horizon,
    }
```

一个 subject 每个 epoch 可以生成不同 sample。

---

# 16. Availability-Aware Loss

定义：

\[
M_{i,t}=1
\]

表示 target item 实际存在，否则 0。

\[
L_{future}=\frac{\sum_{i,t}M_{i,t}\ell(\hat y_{i,t},y_{i,t})}{\sum_{i,t}M_{i,t}+\epsilon}
\]

### 各 value type loss

- binary：BCE
- ordinal：ordinal CE / cumulative link loss
- continuous score：Huber / normalized MAE
- scale total：MAE + correlation metric（loss 可只用 Huber）

不同 scale 必须按训练集统计量 normalization，不能让 ADAS 数值范围主导。

---

# 17. Module 9 — Multi-Hypothesis Future Trajectory Generator

从当前 belief：

\[
q(z_t)=\mathcal N(\mu_t,\sigma_t^2)
\]

生成 `G` 个未来候选：

\[
\tau^{(1)},...,\tau^{(G)}
\]

每个候选至少包含：

```text
future latent memory states
future cognitive item/scale predictions
uncertainty
(optional) future MRI latent representation
```

第一版不强制 Diffusion；可通过 stochastic latent sampling + temporal transition 得到多候选。

若 v0 已证明 multimodal future 明显多峰，再考虑 latent diffusion 作为升级实验。

---

# 18. Module 10 — Structured Longitudinal Reasoning SFT

在 GRPO 前，先让 LLM 学会固定结构输出。

输出 schema：

```text
[CURRENT_STATE]
Memory: ...
Executive: ...
Attention: ...
Language: ...
Orientation: ...

[OBSERVED_EVIDENCE]
- <instrument/item/time/value>
- ...

[TEMPORAL_INTERPRETATION]
...

[FUTURE_FORECAST]
Horizon: 12 months
Predicted memory state: ...
Predicted cognitive items/scales: ...

[UNCERTAINTY]
...
```

推荐机器可解析 JSON 作为训练底层格式，最终再渲染自然语言：

```json
{
  "current_state": {},
  "evidence": [],
  "trend": {},
  "forecast": {},
  "confidence": {}
}
```

SFT 目标：

\[
L_{SFT}=-\sum_s\log p(y_s\mid y_{<s},H)
\]

不得把自由 CoT 当成 ground truth；优先生成可验证的结构化 evidence reasoning。

---

# 19. Module 11 — Availability-Aware Trajectory RLVR / GRPO

## 19.1 RL 的科研作用

RL 不用于“让 MAE 再低一点”，而用于优化同时包含多个约束的未来轨迹：

```text
future correctness
+ evidence faithfulness
+ temporal coherence
+ cognitive-domain correctness
+ MRI-cognition consistency (when MRI exists)
+ uncertainty calibration
+ output format
```

## 19.2 每个 history 采样 G 个输出

\[
o_1,...,o_G\sim\pi_\theta(\cdot\mid H_t)
\]

每个输出包含：

```text
trajectory choice
future predictions
structured evidence
reasoning
confidence
```

## 19.3 总 reward

\[
R=\lambda_OR_{outcome}+\lambda_ER_{evidence}+\lambda_TR_{temporal}+\lambda_DR_{domain}+m_{MRI}\lambda_MR_{MRI}+\lambda_CR_{crossmodal}+\lambda_UR_{uncertainty}+\lambda_FR_{format}
\]

其中所有 reward 必须 availability aware。

---

# 20. Reward 设计

## 20.1 Outcome Reward

未来真实观测集合：

\[
\Omega_{future}
\]

\[
R_{outcome}^{(g)}=\frac{\sum_{j\in\Omega}w_jr(\hat y_j^{(g)},y_j)}{\sum_{j\in\Omega}w_j}
\]

continuous 可用：

\[
r=\exp(-|\hat y-y|/(\sigma_y+\epsilon))
\]

binary/ordinal 使用相应 log-prob / accuracy-aware score。

**必须除以 available target 数量，避免 reward 因目标较多而尺度不同。**

## 20.2 Evidence Faithfulness Reward

对报告引用：

```text
scale
item
visit/time
value
```

逐条查原始 event store。

若引用不存在的数据：强惩罚。

定义：

```text
precision of cited evidence
value correctness
visit correctness
```

例如：

\[
R_{evidence}=0.5P_{citation}+0.3P_{value}+0.2P_{time}
\]

权重可调。

## 20.3 Temporal Reward

不要强制所有 cognition 单调下降。

基于真实不规则时间：

\[
\beta_{true}=Slope(\{(t_i,y_i)\})
\]

\[
\beta_{pred}=Slope(\{(t_i,\hat y_i)\})
\]

\[
R_T=\exp(-|\beta_{pred}-\beta_{true}|/s_\beta)
\]

文本趋势（stable / decline / improve）必须与预测 trajectory 一致。

## 20.4 Domain Reward

来自 Cognitive Item Dictionary。

如模型说：

```text
MoCA delayed recall -> language decline
```

则错误。

正确：

```text
MoCA delayed recall -> memory / delayed recall construct
```

reward 可完全机器验证。

## 20.5 MRI Reward

仅在 future MRI 存在时：

\[
R_{MRI}=sim(\hat z^{MRI}_{future},z^{MRI}_{future})
\]

若 future MRI 不存在：

```text
m_MRI = 0
```

**不是把 reward 设为 0，而是从分子和归一化中跳过。**

## 20.6 Cross-modal Consistency Reward

训练 critic：

\[
C_\phi(z^{MRI},z^{Cog})
\]

正样本：same subject + matched time。

负样本：

- different subject
- distant time
- shuffled cognition

未来候选：

\[
R_{cross}=C_\phi(\hat z^{MRI}_{future},\hat z^{Cog}_{future})
\]

不允许手写“海马萎缩一定对应某个分数”这类过硬因果规则。

## 20.7 Uncertainty Reward

模型输出：

\[
(\mu,\sigma)
\]

使用 proper scoring rule：

\[
R_U=-NLL(y\mid\mu,\sigma)
\]

或 interval coverage + sharpness。

## 20.8 Format Reward

机器检查 JSON/schema 完整性。

只占较小权重，避免模型靠格式刷 reward。

---

# 21. GRPO Optimization

同一 history 的 G 个候选得到 reward：

\[
R_1,...,R_G
\]

组内 advantage：

\[
A_i=\frac{R_i-\bar R}{std(R)+\epsilon}
\]

采用 clipped group-relative objective + reference KL：

\[
L_{GRPO}=-\frac1G\sum_i\min(r_iA_i,clip(r_i,1-\epsilon,1+\epsilon)A_i)+\beta KL(\pi_\theta\|\pi_{SFT})
\]

### Trainable parameters

v0 推荐只训练：

- LLM LoRA
- trajectory selector / reasoner head
- 少量 projection/adapters

冻结：

- 主 MRI encoder
- 已稳定的 cognitive encoder 主干
- 大部分 temporal dynamics backbone

后续 ablation 再开放部分层。

---

# 22. Supervised Total Objective

GRPO 之前：

\[
L_{sup}=\lambda_1L_{item}+\lambda_2L_{cross}+\lambda_3L_{ground}+\lambda_4L_{int}+\lambda_5L_{future}+\lambda_6L_{unc}
\]

每一项由 availability mask 激活。

mini-batch 内允许：

- A 患者只贡献 item loss
- B 患者贡献 temporal loss
- C 患者贡献 MRI grounding
- D 患者贡献 future MRI

即 **partial-supervision multi-task learning**。

---

# 23. 四阶段训练计划

## Stage I — Cognitive Event Representation

数据：所有可用认知 visits，包括单次访视患者。

训练：

```text
Question semantics
Numeric response
Masked item
Masked instrument
Cross-scale consistency
```

目标：证明 question/response representation 优于纯 numeric baseline。

### Gate 1

必须至少满足：

- masked item reconstruction 优于 numeric-only；
- cross-scale prediction 至少在 2 个 instrument pair 上稳定提升；
- 无明显 diagnosis leakage。

未满足：不进入 MRI grounding。

## Stage II — Sparse Structural Anchoring

数据：MRI–cognition matched pairs。

训练：

```text
MRI tokens
question-to-anatomy cross attention
matched/mismatched contrastive objective
```

### Gate 2

- same-subject/time alignment > shuffled baseline；
- MRI+Q/A 在 future cognition 或 cross-scale 上有可复现增益；
- grounding attention 不出现全部 collapse 到同一 token。

## Stage III — Asynchronous Memory Dynamics

数据：≥2 visits 做 forecast；≥3 visits 做 interpolation。

训练：

```text
predict-update belief state
irregular time gaps
interpolation
extrapolation
uncertainty
```

### Gate 3

至少：

- 在 12/24 month forecast 上优于 last observation carried forward / LME / GRU-D 等强基线；
- uncertainty 与缺失程度/预测误差有合理 calibration；
- variable-length 不导致严重 performance collapse。

## Stage IV — Reasoning SFT + Trajectory GRPO

先 SFT，再 GRPO。

### Gate 4

GRPO 必须在不显著损伤数值 forecast 的前提下提升至少：

- evidence correctness
- temporal consistency
- domain correctness
- calibration / entailment

如果只提升格式，不能声称 RLVR 成功。

---

# 24. 数据划分

必须 **subject-wise split**，不能 visit-wise split。

建议：

```text
70% train
10% validation
20% test
```

或固定 5-fold CV；根据论文定位选择一种主设置。

同一 subject 所有 visits、MRI 必须在同一 split。

### 外部验证（若后续可得）

AIBL / OASIS3 作为 external generalization。

如果量表 schema 不完全相同，重点测试：

- shared task semantics
- unseen instrument generalization

---

# 25. 主要 Evaluation Tasks

## Task A — Masked Item Reconstruction

指标：

- binary ACC/AUROC
- ordinal MAE / weighted kappa
- continuous MAE

## Task B — Cross-Scale Prediction

例：

```text
ADAS + CDR -> MoCA memory items
MMSE + CDR -> ADAS memory items
```

核心验证：semantic representation 是否实现 scale-invariant construct。

## Task C — Future Cognitive Forecasting

Horizon：

```text
6m / 12m / 24m / 36m
```

指标：

- MAE / RMSE
- Pearson/Spearman
- ordinal metrics
- calibration

## Task D — Future Functional Decline

\[
z_t^{mem}\rightarrow FAQ_{future}
\]

FAQ 只作为 outcome。

## Task E — Clinical Progression Validity

冻结 representation：

```text
z_mem -> CN/MCI/AD
z_mem -> conversion
```

只用于 external validity，不作为 representation construction input。

## Task F — Grounded Report

指标：

```text
Evidence precision/recall
Numeric citation accuracy
Temporal consistency
Domain correctness
Hallucinated evidence rate
Calibration
```

BLEU/ROUGE 可报告，但不能作为主指标。

---

# 26. Baselines

## Classical longitudinal / psychometric

至少：

```text
Last Observation Carried Forward
Linear Mixed Effects
Latent Growth / simple state-space model
Gaussian Process (可选)
```

## Deep irregular time series

选 4~6 个强基线：

```text
GRU / LSTM
GRU-D
T-LSTM
Transformer + time embedding
Neural CDE
Latent ODE / ODE-RNN
```

## Multimodal baseline

```text
MRI global feature + scale totals concat
MRI global feature + item numeric concat
MRI + cognitive Transformer without question semantics
```

## LLM-specific

```text
Numeric-only prompt
Variable-name + numeric prompt
Question semantics + response
Question semantics + response + MRI anchor
```

这组是证明“为什么需要语言语义”的关键实验。

---

# 27. 关键 Ablation

至少做：

```text
w/o Question Semantics
w/o Numeric Branch
w/o Domain Aggregation
w/o MRI Structural Anchoring
w/o Asynchronous Predict-Update
w/o Belief Uncertainty
w/o Masked Instrument Training
w/o Interpolation
w/o Multi-Hypothesis Sampling
w/o GRPO
w/o Evidence Reward
w/o Temporal Reward
w/o Domain Reward
w/o Cross-modal Reward
```

还必须做 missingness robustness：

```text
Randomly drop 20/40/60% cognition instruments
Randomly drop MRI anchors
Short history vs long history
1 MRI vs 2 MRI vs >=3 MRI
```

---

# 28. Protocol Leakage 检查

ADNI 某些量表缺失可能与 study phase / protocol 有关，而非患者状态。

因此：

1. missingness mask 主要用于 masking/loss/uncertainty；
2. 不要直接让 `scale_missing=1` 成为高权重 disease feature；
3. 做 `with vs without missingness embedding` ablation；
4. 按 ADNI phase / cohort stratify；
5. 检查模型是否能仅凭“哪些量表没测”预测 diagnosis；若能，说明 leakage 风险大。

---

# 29. 推荐代码目录

```text
memreason_rl/
├── README.md
├── configs/
│   ├── data.yaml
│   ├── cognitive_item_dictionary.yaml
│   ├── model.yaml
│   ├── train_stage1.yaml
│   ├── train_stage2.yaml
│   ├── train_stage3.yaml
│   └── train_stage4_grpo.yaml
├── data/
│   ├── build_cognitive_events.py
│   ├── build_mri_metadata.py
│   ├── match_mri_cognition.py
│   ├── split_subjects.py
│   ├── dataset.py
│   └── collate.py
├── models/
│   ├── cognitive_encoder.py
│   ├── numeric_encoder.py
│   ├── domain_aggregator.py
│   ├── mri_encoder.py
│   ├── cognitive_anatomy_grounder.py
│   ├── belief_state.py
│   ├── temporal_dynamics.py
│   ├── decoders.py
│   ├── trajectory_generator.py
│   └── reasoning_llm.py
├── losses/
│   ├── masked_losses.py
│   ├── cross_scale.py
│   ├── grounding.py
│   ├── temporal.py
│   └── calibration.py
├── rl/
│   ├── rewards.py
│   ├── evidence_verifier.py
│   ├── temporal_verifier.py
│   ├── domain_verifier.py
│   ├── crossmodal_critic.py
│   └── grpo_trainer.py
├── train/
│   ├── stage1_cognition.py
│   ├── stage2_grounding.py
│   ├── stage3_dynamics.py
│   └── stage4_grpo.py
├── eval/
│   ├── forecast.py
│   ├── cross_scale.py
│   ├── calibration.py
│   ├── grounding.py
│   └── report_eval.py
└── reports/
    ├── data_audit/
    ├── matching/
    ├── experiments/
    └── figures/
```

---

# 30. 推荐配置（v0）

```yaml
model:
  cognitive_dim: 256
  mri_token_dim: 256
  memory_state_dim: 32
  num_domains: 5
  temporal_backbone: "time_aware_transformer"
  probabilistic_state: true

mri_matching:
  primary_window_days: 90
  sensitivity_days: [30, 60, 90, 180]

forecast:
  horizons_months: [6, 12, 24, 36]
  tolerance_months:
    6: 3
    12: 3
    24: 6
    36: 6

masking:
  item_mask_prob: 0.15
  instrument_mask_prob: 0.20

trajectory:
  num_candidates: 4

rl:
  use_grpo: false   # Stage IV 才打开
  group_size: 4
  kl_beta: 0.1
```

以上超参只作为起点，所有最终值必须通过 validation 决定。

---

# 31. Stage-wise 执行清单

## Phase 0 — Data Audit（必须先做）

执行：

```text
[ ] CSV shape / unique subject / visit distribution
[ ] each column dtype / non-null / unique values / min-max
[ ] scale-level co-occurrence matrix
[ ] item-level availability
[ ] value direction check
[ ] ADNI phase / protocol metadata if available
[ ] T1 metadata table
[ ] MRI per subject distribution
[ ] MRI-cognition time gap distribution
[ ] subject-wise split
```

生成：

```text
reports/data_audit/summary.md
reports/data_audit/item_stats.csv
reports/data_audit/scale_cooccurrence.csv
reports/data_audit/followup_distribution.csv
reports/matching/mri_cognition_matching.csv
```

## Phase 1 — Numeric baseline

先做：

```text
LME
GRU-D
time-aware Transformer
```

只用 numeric item + time + mask。

**目的**：建立必须被后续 semantic/MRI/RL 超越的可靠 baseline。

## Phase 2 — Question semantics

加入 item dictionary / text encoder。

比较：

```text
numeric-only
variable-name + numeric
question/task semantics + numeric
```

## Phase 3 — MRI sparse anchors

先只做 paired-window grounding + global MRI anchor。

证明 MRI 有增益后，再做更复杂 spatial cross-attention。

## Phase 4 — Belief dynamics

从 deterministic state 先跑通，再升级 probabilistic state。

必须报告：

```text
forecast error
uncertainty calibration
performance vs history length
performance vs missingness
```

## Phase 5 — SFT report

结构化 JSON reasoning。

先保证 evidence parser 100% 可运行。

## Phase 6 — GRPO

只在 SFT 稳定后开启。

每轮必须记录 reward decomposition：

```text
R_outcome
R_evidence
R_temporal
R_domain
R_MRI
R_crossmodal
R_uncertainty
R_format
```

避免只看 total reward。

---

# 32. 输出结果表设计

## Table 1 — Dataset & Missingness

```text
Subjects
Visits
Visits/subject
Scale availability
MRI availability
MRI-cognition matched pairs
Follow-up length
```

## Table 2 — Main Forecasting

```text
Method | 6m | 12m | 24m | 36m | Avg
```

按：MMSE / MoCA / ADAS / CDR 分层。

## Table 3 — Cross-Scale Generalization

```text
Input Instruments | Held-out Instrument | Baseline | MemReason
```

## Table 4 — Missingness Robustness

```text
Drop Ratio | Numeric | Semantic | +MRI Anchor | Full
```

## Table 5 — RLVR

```text
Variant | Forecast | Evidence | Temporal | Domain | Calibration
SFT
+ outcome reward
+ evidence
+ temporal
+ domain
Full GRPO
```

---

# 33. Figure 设计

## Figure 1 — Main Method

视觉中心必须是：

```text
Asynchronous cognitive event stream
Sparse T1 MRI structural anchors
        ↓
Question-grounded cognitive evidence
        ↓
Probabilistic latent memory belief state
        ↓
Continuous predict-update dynamics
        ↓
Multiple future trajectories
        ↓
Availability-aware RLVR / GRPO
        ↓
Grounded future cognitive report
```

不要画成每个 M0/M6/M12 都有 MRI。

## Figure 2 — Missingness Pattern

一个真实 patient example：

```text
Time: M0 M6 M12 M18 M24 M36
MMSE  ✓  ✓   ✓       ✓
MoCA  ✓      ✓   ✓
ADAS  ✓  ✓       ✓
MRI   ✓      ✓           ✓
```

展示模型如何持续更新 belief。

## Figure 3 — Latent Memory Trajectories

按不同临床组事后染色，但 diagnosis 不作为 encoder input。

## Figure 4 — Calibration / uncertainty

history sparse → uncertainty broad；新 cognition/MRI anchor → posterior uncertainty shrink。

## Figure 5 — Success / Failure Cases

Grounded report + evidence pointers + future trajectory。

---

# 34. 不允许的实现方式

以下做法默认视为不符合本项目设计：

```text
× 只选 complete-case subjects
× 把所有缺失值统一填 0
× 每个 visit 强行拼接 MRI+MMSE+MoCA+ADAS
× 把 diagnosis 当 input
× 只用 total score，不保留 item semantics
× 让 LLM 直接无约束预测 future score
× GRPO 从零训练整个模型
× MRI 不存在时给 MRI reward=0
× 用自由 CoT 当医学真实 reasoning label
× attention map 直接宣称 causal anatomical explanation
× 没核对 ADNI item coding 就写 question text
× 用 visit-wise split 导致同一患者泄漏到 train/test
```

---

# 35. MVP（最小可行版本）

如果资源有限，先实现：

```text
1. Item dictionary
2. Numeric + semantic cognitive encoder
3. Domain-aware memory state
4. Time-aware event Transformer
5. Masked instrument + future forecast
6. Subject-wise split
7. Numeric vs semantic ablation
```

只有 v0 证明：

\[
Question\ Semantics > Numeric\ Only
\]

以及：

\[
Asynchronous\ model > fixed-grid\ baseline
\]

再加入 MRI。

MRI 证明有效后再加入 probabilistic belief / report / GRPO。

---

# 36. 推荐开发顺序（避免一次性工程过大）

```text
Milestone 0  Data audit
Milestone 1  Numeric irregular-time baseline
Milestone 2  Question-grounded cognition
Milestone 3  Cross-scale reconstruction
Milestone 4  Sparse MRI anchors
Milestone 5  Probabilistic memory belief dynamics
Milestone 6  Multi-hypothesis future generation
Milestone 7  Structured reasoning SFT
Milestone 8  Availability-aware GRPO
Milestone 9  External/robustness experiments
```

每个 Milestone 单独 commit / checkpoint / report。

---

# 37. 给代码大模型的首轮任务 Prompt

可直接复制以下内容：

```text
你现在是本项目的 research engineer。请严格按照 MemReason-RL Implementation Spec 实现，不要自行改变科研假设。

当前第一阶段只执行 Phase 0 Data Audit，不训练复杂模型。

输入：
1. adni_all_longitudinal_scales_visit(1).csv
2. ADNI T1 MRI 根目录或 mri_metadata.csv（如果尚未给出，则只为 MRI metadata 定义接口，不虚构路径）

你需要：
1. 读取 CSV，确认 subject、visit、column 统计；
2. 输出每列 dtype、非空数、unique 数、min/max（适用时）；
3. 按 ADAS/CDR/FAQ/MMSE/MoCA/DXSUM 分组；
4. 计算每个量表的 visit-level availability；
5. 计算每个 subject 的随访次数与随访跨度；
6. 计算 scale co-occurrence matrix；
7. 统计核心 memory items 的有效数和取值集合；
8. 标记编码需要 ADNI data dictionary 核验的变量；
9. 建立 subject-wise split 文件；
10. 生成 reports/data_audit/summary.md 和相关 CSV。

约束：
- 不进行缺失值填补；
- 不使用 diagnosis 作为 feature；
- 不假设 MRI 与认知同步；
- 不修改原始文件；
- 所有随机操作固定 seed；
- 运行结束后汇报发现的数据异常和下一步实现风险。
```

---

# 38. 最终项目成功标准

该项目的成功不是“CN/MCI/AD AUC 更高”这一项。

需要同时证明：

1. **Semantic value**：question/task semantics 相比 numeric-only 提升跨量表与 future cognition 建模；
2. **Missingness robustness**：对 variable-length、scale missing、MRI missing 稳健；
3. **Neural value**：sparse MRI anchor 提供额外 future-memory 信息；
4. **State validity**：latent memory state 能解释/预测未来 cognition 与 function；
5. **Temporal value**：irregular dynamics 比固定 grid / static model 更好；
6. **RL value**：GRPO 提升 evidence/temporal/domain/calibration consistency，而不牺牲 forecast；
7. **Auditability**：最终报告中的关键结论可回溯到真实 item、时间点以及可用 MRI evidence。

最终论文要回答的不是：

> “我们能否预测 AD？”

而是：

> **“在真实不完整、异步的神经认知随访中，我们能否持续识别一个人的潜在记忆状态，并生成可验证的未来认知轨迹？”**

---

# 39. 当前明确待补充给执行代理的信息

在真正进入 MRI 阶段前，需要用户提供：

```text
[ ] ADNI T1 MRI 根目录
[ ] 文件命名规则 / RID 映射规则
[ ] scan date / visit code metadata 来源
[ ] 已完成的 MRI preprocessing 情况
[ ] 是否已有 FreeSurfer / ROI / atlas 结果
[ ] GPU 节点、显存、并行策略
[ ] 希望采用的 LLM backbone
```

这些缺失时，执行代理只能预留接口，不得虚构。

---

# 40. 与已有方法的关系（用于论文 Related Work / Method Motivation）

- **EMAD (CVPR 2026)**：借鉴“生成结论必须回指证据”的 evidence-centric 思想，以及在 SFT 后通过 executable rewards + GRPO 提升临床一致性的训练范式；本项目把这一思想从 current AD diagnosis 推进到 longitudinal future trajectory verification。
- **MCI-Diff (CVPR 2026)**：借鉴 irregular longitudinal sequence 中 interpolation / extrapolation 以及 multiple future candidates + language-guided plausibility 的思想；本项目不以 future MRI generation 为主，而是学习 asynchronous memory belief dynamics，并用真实 future cognition / MRI 作为 verifiable rewards。

注意：本项目的创新性不能表述为“把两篇论文拼起来”。必须围绕以下独立问题组织：

```text
asynchronous partial observations
question semantics
sparse structural anchoring
latent memory belief state
trajectory-level verifiable RL
```

---

**END OF IMPLEMENTATION SPEC**
