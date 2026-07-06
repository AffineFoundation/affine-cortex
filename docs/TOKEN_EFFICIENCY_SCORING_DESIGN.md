# Token Efficiency 派生评分项系统设计

## 1. 目标

新增一个用于奖励低 token 消耗的评分项，命名为 `TOKEN-EFFICIENCY`。它在配置、battle scoring、rank 展示上表现为一个 env，但不是普通 Docker/affinetes env。

核心目标：

- 统计 champion 与 challenger 在同一场 battle 中、相同 runtime env、相同 task overlap 上的平均 token 消耗。
- token 消耗越低越好，但需要转换成现有 comparator 可处理的“越大越好”分数。
- 支持 `system_config.environments` 中的 `enabled_for_sampling` 和 `enabled_for_scoring`。
- rank 能稳定看到 token 平均值、相对 champion 的节省比例、coverage 和最终 verdict。
- 不新增 DynamoDB 表，不新增 `sample_results` 顶层字段。
- 不在 scheduler、monitor、API、CLI 中重复实现 token 提取、均值、阈值和 verdict 逻辑。

非目标：

- 不把 `TOKEN-EFFICIENCY` 做成单独容器镜像。
- 不生成独立 task pool。
- 不启动 executor worker。
- 不把缺失 usage 当作 0 token。
- 不让 CLI 重新计算评分，只负责展示后端给出的 payload。

## 2. 当前事实

当前项目中与该设计相关的事实：

- executor 每个 task 的结果写入 `sample_results`。
- `sample_results` 中已有 `extra_compressed`，保存 env 返回的 `extra`。
- 当前每个 task 的 `extra` 中已经有 usage 信息。
- battle 决策只需要 champion/challenger 当前 refresh 的 overlap tasks。
- rank live 数据来自 `LiveScoresMonitor` 写入的 `miner_stats.scores_by_env`。
- terminated miner 的 frozen 数据也保存在 `miner_stats.scores_by_env`。
- `/rank/current` 当前会把 live `scores_by_env` 拆成 `sample_counts`、`sample_averages`、`champion_overlap_avgs`、`terminal_scores`。

关键约束：

- 不新增 `sample_results.token_count` 等顶层字段。
- token usage 从 `extra_compressed` 解压后的 `extra.usage` 或等价白名单字段读取。
- 缺失 usage 的 sample pair 不参与 token 计算。

## 3. 核心模型

### 3.1 Runtime Env

普通评测环境，例如 `SWE-INFINITE`、`NAVWORLD`、`MEMORY`、`TERMINAL`。

特点：

- 有 Docker/affinetes evaluator。
- 有 task pool。
- executor 会采样。
- sample row 有 `score` 和 `extra_compressed`。

### 3.2 Derived Env

`TOKEN-EFFICIENCY` 是 derived env。

特点：

- 没有 evaluator。
- 没有 task pool。
- 不进入 executor。
- 输入来自 runtime env 的 sample rows。
- 输出是一个可进入 comparator 和 rank 的 env payload。

### 3.3 单一职责模块

新增一个模块：

```text
affine/src/scorer/token_efficiency.py
```

它是 token efficiency 的唯一业务逻辑入口，负责：

- 解析 config。
- 从 sample extra 中提取 usage。
- 根据 overlap sample metrics 计算 token stats。
- 生成 comparator input。
- 生成 rank/live/frozen payload。
- 生成 snapshot audit metadata。

其他模块禁止重复实现：

- usage 提取路径。
- token 均值计算。
- token ratio 计算。
- dominant/not_worse/worse 判定。
- coverage 计算与 availability 判定。

CLI 只展示 payload，不计算 verdict。

### 3.4 模块 API 契约

`token_efficiency.py` 对外只暴露少量稳定函数，其他模块只允许调用这些函数，不允许复制内部公式：

```python
load_token_efficiency_config(environments: dict) -> TokenEfficiencyConfig | None

extract_token_usage(extra: dict) -> TokenUsage | None

compute_token_efficiency(
    *,
    env: str,
    config: TokenEfficiencyConfig,
    basis_metrics_by_runtime_env: dict[str, dict[int, SampleMetric]],
    subject_metrics_by_runtime_env: dict[str, dict[int, SampleMetric]],
    overlap_ids_by_runtime_env: dict[str, set[int]],
    subject_is_reference: bool = False,
) -> TokenEfficiencyComputation
```

语义：

- `basis` 是比较基准，battle 中通常是 champion。
- `subject` 是被展示或被评分的对象，battle 中通常是 challenger。
- `overlap_ids_by_runtime_env` 只能由 scheduler/monitor 按 runtime score overlap 传入，token module 不自行选 task。
- `subject_is_reference=true` 用于 champion 自身展示，此时 subject 与 basis 相同，但仍按同一套 missing usage 和 coverage 规则计算。

第一版只支持一个 `TOKEN-EFFICIENCY` derived env，不设计多个 token derived env 或多套 token 评分实例。

## 4. System Config

建议配置：

```json
{
  "TOKEN-EFFICIENCY": {
    "display_name": "TOKENS",
    "kind": "derived",
    "derived_metric": "token_efficiency",
    "enabled_for_sampling": true,
    "enabled_for_scoring": false,
    "scoring": {
      "min_pairs": 100,
      "savings_margin": 0.05,
      "extra_token_tolerance": 0.02,
      "max_score_ratio": 2.0
    }
  }
}
```

字段语义：

- `kind=derived`
  - 表示该 env 不进入 sampler/executor。
- `derived_metric=token_efficiency`
  - 选择 token efficiency 计算器。
- `enabled_for_sampling`
  - 对 runtime env 表示采样。
  - 对 derived env 表示计算并在 rank 展示。
- `enabled_for_scoring`
  - 是否把该 derived env 注入 comparator。
- `min_pairs`
  - scoring availability 的最小有效 usage pair 数。
  - 低于该值时，token env 本轮 unavailable，不注入 comparator。
  - rank 仍可展示 partial token 数据和 coverage。
- `savings_margin`
  - challenger 至少比 champion 少多少 token 才算 dominant。
- `extra_token_tolerance`
  - challenger 最多比 champion 多多少 token 仍算 not_worse。
- `max_score_ratio`
  - synthetic score 上限，避免极端小 token 值把展示分数放大得过分。
  - 实现时必须保证不低于 `dominant_ratio_threshold`，避免 ratio 被 clamp 后 comparator verdict 与 token payload verdict 不一致。

第一版固定策略：

- 只统计当前 runtime scoring env 的 overlap tasks。
- token 口径固定为 `total_tokens`，缺失时 fallback 到 `prompt_tokens + completion_tokens`。
- 聚合方式固定为 sample-weighted mean。
- usage 不足时固定 skip，不注入 comparator，不判 worse。
- token env available 时作为正式新 env 参与 dominant/not_worse/worse。
- token payload 固定不进入 `average_score`。

配置组合规则：

| enabled_for_sampling | enabled_for_scoring | 结果 |
|---|---|---|
| false | false | 完全关闭 token efficiency |
| true | false | 只在 rank 展示，不影响 battle |
| true | true | rank 展示，且参与 battle scoring |
| false | true | 非法配置，启动时应拒绝或自动关闭 scoring |

不允许 `enabled_for_scoring=true` 但 `enabled_for_sampling=false`，原因是这会产生“参与胜负但 rank 不显示”的隐藏评分项，破坏可解释性。

## 5. Env 读取接口

必须拆分 env 读取语义，避免 derived env 污染 runtime env 链路。

建议 `StateStore` 提供：

```python
get_runtime_environments()
get_runtime_scoring_environments()
get_derived_environments()
get_scoring_environments()
get_rank_display_environments()
```

调用关系：

| 调用方 | 接口 |
|---|---|
| task pool refresh | `get_runtime_environments()` |
| executor manager | `get_runtime_environments()` |
| executor worker | `get_runtime_environments()` |
| champion completion gate | `get_runtime_scoring_environments()` |
| battle overlap gate | `get_runtime_scoring_environments()` |
| `_decide()` runtime score | `get_runtime_scoring_environments()` |
| `_decide()` token derived env | `get_derived_environments()` |
| `LiveScoresMonitor` runtime score | `get_runtime_environments()` |
| `LiveScoresMonitor` token derived metric | `get_derived_environments()` |
| `/rank/current.enabled_envs` | `get_rank_display_environments()` |
| `/scores/latest` env filter | `get_scoring_environments()` |

兼容规则：

- 旧 env config 没有 `kind` 时，默认 `kind="runtime"`。
- `get_environments()` 如保留，应保持原有 runtime 语义，不能返回 derived env。

## 6. Sample 读取与 Usage 提取

### 6.1 不新增字段

token usage 从现有数据读取：

```text
sample_results.extra_compressed
-> decompress
-> extra
-> extract_token_usage(extra)
```

不新增：

- `sample_results.token_count`
- `sample_results.prompt_tokens`
- `sample_results.completion_tokens`
- 独立 token metrics 表

原因：

- battle 决策低频。
- monitor 定时运行，可控。
- usage 已经在 extra 中。
- 新字段会扩大写入路径和兼容成本。

### 6.2 统一读取方法

为避免 score 查询和 usage 查询重复扫描同一 partition，建议新增一个统一 reader：

```python
read_sample_metrics_for_tasks(
    hotkey: str,
    revision: str,
    env: str,
    task_ids: list[int],
    refresh_block: int,
    include_extra_usage: bool = False,
) -> dict[int, SampleMetric]
```

`SampleMetric`：

```python
@dataclass
class SampleMetric:
    score: float
    usage: TokenUsage | None = None
```

读取规则：

- Query 当前 `(hotkey, revision, env)` partition。
- 只保留请求的 `task_ids`。
- 只保留 `refresh_block` 匹配的 row。
- `include_extra_usage=false` 时只读 score 字段，用于普通路径。
- `include_extra_usage=true` 时 projection 增加 `extra_compressed`，解压并提取 usage。

这样：

- scheduler 普通 score 与 token usage 可共用同一个 reader。
- monitor 普通 score 与 token usage 也可共用同一个 reader。
- 不需要 `read_scores_for_tasks()` 和 `read_usage_for_tasks()` 两套重复逻辑长期并存。

如果为了渐进改造保留 `read_scores_for_tasks()`，也应让它内部调用 `read_sample_metrics_for_tasks(..., include_extra_usage=False)`。

### 6.3 Usage 提取

唯一提取函数：

```python
extract_token_usage(extra: dict) -> TokenUsage | None
```

`TokenUsage`：

```python
@dataclass
class TokenUsage:
    total_tokens: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    source: str
```

白名单路径：

```text
extra["usage"]
extra["openai_usage"]
extra["inference_usage"]
extra["inference_calls"][*]["usage"]
extra["calls"][*]["usage"]
```

提取优先级：

```text
extra["usage"]
-> extra["openai_usage"]
-> extra["inference_usage"]
-> extra["inference_calls"][*]["usage"]
-> extra["calls"][*]["usage"]
```

只使用第一个命中的有效来源，不把多个来源相加。这样可以避免同一次推理同时写了顶层 usage 和 calls usage 时被重复计数。

单次 usage：

```text
total_tokens 优先
否则 prompt_tokens + completion_tokens
```

多次 calls：

```text
逐 call 提取 usage 后求和
```

禁止：

- 从模型输出文本解析 token。
- 从非白名单字段猜 token。
- 把缺失 usage 当作 0。

缺失 usage：

```text
usage = None
该 sample pair 不参与 token 均值
```

## 7. Battle 评分算法

### 7.1 输入

runtime scoring env 先按现有逻辑得到 overlap task ids：

```text
overlap_ids(env) = champion scored task ids ∩ challenger scored task ids
```

expected pairs：

```text
expected_pairs = sum(len(overlap_ids(env)) for env in runtime_scoring_envs)
```

有效 token pairs：

```text
valid_pairs = [
  (env, task_id, champion_usage, challenger_usage)
  for env in runtime_scoring_envs
  for task_id in overlap_ids(env)
  if champion_usage is not None
  and challenger_usage is not None
]
```

缺失 usage 的 pair 不参与计算。

### 7.2 Coverage

```text
token_pairs = len(valid_pairs)
coverage_ratio = token_pairs / expected_pairs
```

如果：

```text
token_pairs < min_pairs
```

则：

```text
TOKEN-EFFICIENCY unavailable
不注入 comparator
rank/snapshot 记录 reason、coverage；如果 token_pairs > 0，也可以记录 partial avg_tokens
```

这里的 `available` 精确定义为“是否可参与 scoring”。它不等于“是否有任何 token 数据可展示”。scoring availability 只由 `min_pairs` 控制。coverage ratio 不参与本轮判罚，只用于 rank 展示和 snapshot 审计。建议默认 `min_pairs=100`，避免只有少量有效 usage pair 时 token env 产生噪声 verdict。

### 7.3 平均 token

默认 sample-weighted：

```text
champ_avg_tokens = sum(champion total_tokens over valid_pairs) / token_pairs
chal_avg_tokens  = sum(challenger total_tokens over valid_pairs) / token_pairs
```

### 7.4 Ratio

转换为越大越好的 synthetic score：

```text
ratio_to_champion = champ_avg_tokens / chal_avg_tokens
champion_synthetic_score = 1.0
challenger_synthetic_score = clamp(ratio_to_champion, 0, max_score_ratio)
saving_rate = 1 - chal_avg_tokens / champ_avg_tokens
```

健壮性规则：

- 如果 `champ_avg_tokens <= 0` 或 `chal_avg_tokens <= 0`，token env 本轮 unavailable，reason=`invalid_token_average`。
- 必须先通过上述合法性检查，才能计算 ratio；不要用 `max(chal_avg_tokens, 1)` 掩盖坏数据。
- 正常 OpenAI-compatible usage 应为正数；该规则只用于防御坏数据。

含义：

- `ratio_to_champion > 1`：challenger 更省 token。
- `ratio_to_champion = 1`：平均 token 相同。
- `ratio_to_champion < 1`：challenger 更耗 token。

### 7.5 Verdict

阈值：

```text
dominant_ratio_threshold = 1 / (1 - savings_margin)
not_worse_ratio_threshold = 1 / (1 + extra_token_tolerance)
```

判定：

```text
ratio_to_champion > dominant_ratio_threshold
=> dominant

ratio_to_champion >= not_worse_ratio_threshold
=> not_worse

otherwise
=> worse
```

示例：

```text
savings_margin = 0.05
dominant 需要 challenger token < champion token * 0.95

extra_token_tolerance = 0.02
not_worse 允许 challenger token <= champion token * 1.02
```

### 7.6 注入 comparator

当 token env available 且 `enabled_for_scoring=true`：

```text
champ_scores["TOKEN-EFFICIENCY"] = {0: 1.0}
chal_scores["TOKEN-EFFICIENCY"] = {0: ratio_to_champion}
```

`EnvComparisonConfig`：

```text
margin = dominant_ratio_threshold - 1.0
not_worse_tolerance = 1.0 - not_worse_ratio_threshold
min_tasks_per_env = 1
```

`TOKEN-EFFICIENCY` 是正式新 env：available 且 `enabled_for_scoring=true` 时，它和普通 env 一样参与 dominant/not_worse/worse 计数。不提供“不计 dominant”的第一版分支。

### 7.7 Metadata 传递

`WindowComparator` 只能处理数字 score，无法保存 `avg_tokens`、`saving_rate`、`coverage_ratio` 等 token 原始指标。因此 token module 必须返回一个 sidecar result，而不是只返回 synthetic score。

建议结构：

```python
@dataclass
class TokenEfficiencyComputation:
    env: str
    available: bool
    reason: str
    champion_score: float | None
    challenger_score: float | None
    comparison_config: EnvComparisonConfig | None
    champion_payload: dict
    challenger_payload: dict
    snapshot_metric: dict
```

使用方式：

- `available=true` 且 `enabled_for_scoring=true` 时，把 `champion_score`、`challenger_score`、`comparison_config` 注入 comparator。
- frozen scores 不从 `EnvComparison` 反推 token payload，而是直接使用 `champion_payload` / `challenger_payload`。
- snapshot outcome 中 token env 的 raw `metric` 直接来自 `snapshot_metric`。
- CLI 和 API 只消费 payload，不反算 ratio/verdict。

这样可以避免三处重复逻辑：

- `_comparison_scores_by_env()` 重建 token payload。
- `_final_scores_from_result()` 重建 token payload。
- `_outcome_for_snapshot()` 从 synthetic score 反推 raw token stats。

## 8. Scheduler 集成

### 8.1 `_refresh_task_ids`

只处理 runtime env。

`TOKEN-EFFICIENCY` 不生成 task ids。

### 8.2 `_samples_complete`

只检查 runtime scoring env。

`TOKEN-EFFICIENCY` 不增加 champion completion gate。

### 8.3 `_battle_overlap_ready`

只检查 runtime scoring env 的 score overlap。

`TOKEN-EFFICIENCY` 不增加 overlap gate。

原因：

- token scoring 依赖 runtime overlap。
- 缺失 usage 不应阻塞 battle。

### 8.4 `_decide`

决策流程：

1. 读取 runtime scoring env 的 champion/challenger sample metrics。
2. 求每个 runtime env 的 overlap ids。
3. 构造普通 env 的 `champ_scores`、`chal_scores`、`env_configs`。
4. 如果存在 enabled token derived env：
   - 从同一批 sample metrics 中提取 usage；如果普通 score 路径还没有读取这些 row，则直接用 `read_sample_metrics_for_tasks(..., include_extra_usage=True)` 一次读出 score 和 usage，避免同一 subject/env 二次查询。
   - 调 `token_efficiency.compute(...)`。
   - 如果 available 且 `enabled_for_scoring=true`，注入 comparator input。
   - 保存 `TokenEfficiencyComputation` sidecar。
5. 调 `WindowComparator.compare(...)`。
6. 把普通 env payload 和 sidecar 中的 token payload 一起冻结到 loser 的 `miner_stats.scores_by_env`。
7. 若 challenger 赢，写 `scores` 和 `score_snapshots`：
   - new champion 的 `scores_by_env` 合并 challenger token payload。
   - previous champion 的 `scores_by_env` 合并 champion token payload。
   - snapshot outcome 合并 sidecar 中的 token audit metadata。

注意：

- token module 只能使用 `_decide()` 已经确定的 runtime overlap ids。
- token module 不自行选择 task。
- unavailable token env 不参与 comparator，但仍可写 rank payload。
- `_final_scores_from_result()` 和 `_comparison_scores_by_env()` 不应包含 token 专用重建逻辑；它们只合并 token sidecar payload。
- `_write_weights()` 构造 champion/previous champion `WeightSubject.scores_by_env` 时也必须合并 token sidecar payload，避免 `/scores/latest` 与 `/rank/current` 对最终结果展示不一致。

### 8.5 Early Regression

第一版不让 token env 参与 early regression。

原因：

- token usage 缺失不应造成 early lost。
- early path 需要与 full decide bit-consistent，token unavailable 会增加复杂度。
- token scoring 在 final decide 生效即可。

后续如果要支持，必须复用同一个 `token_efficiency.compute(...)`，不能另写一套逻辑。

## 9. LiveScoresMonitor 集成

### 9.1 目标

rank 需要在 battle 进行中看到 token 指标。

`LiveScoresMonitor` 定期计算并写入：

```text
miner_stats.scores_by_env["TOKEN-EFFICIENCY"]
```

### 9.2 数据范围

token 指标不应无条件对所有 valid miners 解压 `extra_compressed`。第一版只计算 rank 中有当前 live 意义的主体：

- 当前 champion。
- active battle challenger。
- predeployed challengers。

这些主体来自 `system_config` state，而不是扫描 sample_results 推断。queue 中尚未部署、没有当前 refresh samples 的 miner 不计算 token payload，rank 显示 `-`。

对这些主体：

- champion row：展示 champion 当前可解析 usage 的平均 token，`is_reference=true`。
- challenger/predeployed row：展示该 miner 与 champion 在 runtime env overlap 上的 token ratio。

对于非 champion miner：

```text
valid_pairs = champion usage ∩ miner usage
```

缺失 usage 的 pair 不参与计算。

### 9.3 Monitor 不重复算法

`LiveScoresMonitor` 不直接计算 verdict、ratio、coverage。

它只负责：

1. 读取 sample metrics。
2. 构造 overlap inputs。
3. 调 `token_efficiency.compute(...)`。
4. 把返回 payload 写入 `miner_stats.scores_by_env`。

这样 scheduler final decide 和 live monitor 使用同一套 token 逻辑。

### 9.4 Monitor 成本控制

为了避免 rank 展示引入过高的 DynamoDB/CPU 成本：

- 只有存在 `enabled_for_sampling=true` 的 token derived env 时，才开启 token monitor 分支。
- token monitor 只处理 champion、active battle challenger、predeployed challengers。
- 对同一 `(hotkey, revision, env, refresh_block)`，runtime score 与 usage 尽量通过 `read_sample_metrics_for_tasks(..., include_extra_usage=True)` 一次读取完成。
- 并发沿用 `LiveScoresMonitor` 的全局并发限制。
- 如果某 subject/env 的 runtime sample metrics 已经被本轮 monitor 读取，token 分支必须复用本轮缓存结果，不再调用第二套 usage DAO。具体实现是：active token roster 在普通 live score pass 中直接读取 `score+usage`，token pass 只消费缓存；只有缓存缺失时才补读。
- unavailable 或无 overlap 时写轻量 payload，不继续额外查询。

### 9.5 写入策略

当 `TOKEN-EFFICIENCY.enabled_for_sampling=true`：

- monitor 计算 token payload。
- 写入 `miner_stats.scores_by_env["TOKEN-EFFICIENCY"]`。

当 `enabled_for_sampling=false`：

- monitor 不计算 token payload。
- rank 不显示该列。

当 `enabled_for_scoring=false`：

- monitor 仍可计算和展示。
- scheduler 不注入 comparator。

## 10. Rank/API 展示闭环

### 10.1 问题

当前 `/rank/current` 对 live scores 的拆分会丢字段：

```text
scores_by_env -> sample_counts/sample_averages/champion_overlap_avgs
```

普通 env 只需要 `count` 和 `avg`，所以没问题。

token env 需要：

- `unit`
- `avg_tokens`
- `champion_overlap_avg_tokens`
- `ratio_to_champion`
- `saving_rate`
- `coverage_ratio`
- `available`
- `reason`

如果 API 只返回 `sample_averages`，CLI 只能看到 synthetic ratio，不能正常展示平均 token。

### 10.2 API 契约

`/rank/current.window` 新增：

```json
"sample_details": {
  "<uid>": {
    "TOKEN-EFFICIENCY": {
      "count": 1432,
      "avg": 1.084,
      "unit": "tokens",
      "lower_is_better": true,
      "include_in_average_score": false,
      "available": true,
      "avg_tokens": 1287.4,
      "champion_overlap_avg_tokens": 1395.6,
      "ratio_to_champion": 1.084,
      "saving_rate": 0.0775,
      "coverage_ratio": 0.94,
      "expected_pairs": 1520,
      "token_pairs": 1432,
      "verdict": "dominant",
      "reason": "challenger_used_fewer_tokens"
    }
  }
}
```

兼容规则：

- 保留 `sample_counts`、`sample_averages`、`champion_overlap_avgs`。
- 新 CLI 优先读 `sample_details`。
- 老 CLI 仍能看到普通 env。
- `terminal_scores` 已经保留 full payload，用于 terminated rows。

### 10.2.1 `rank_state` 实现契约

当前后端实现中，`affine/api/rank_state.py::_split_display_scores(display_map)` 会把 live `scores_by_env` 投影成四个字段：

```text
sample_counts
sample_averages
champion_overlap_avgs
terminal_scores
```

这个函数需要改成返回第五个字段：

```text
sample_details
```

建议返回结构：

```python
return counts, averages, overlap_avgs, terminal, details
```

其中：

- live token row：`details[uid]["TOKEN-EFFICIENCY"] = item`，保留完整 payload。
- frozen row：仍进入 `terminal_scores[uid][env] = item`，不需要复制进 `sample_details`。
- 普通 env：继续使用 `sample_counts/sample_averages/champion_overlap_avgs`，不需要复制进 `sample_details`。
- token env：必须保留完整 payload，包括 `avg_tokens`、`saving_rate`、`coverage_ratio`、`available`、`reason`。

`get_rank_window()` 返回 payload 时必须包含：

```json
"sample_details": sample_details
```

如果不改这一层，即使 `LiveScoresMonitor` 已经把 token payload 写入 `miner_stats.scores_by_env`，API 也会在拆分时丢掉 token 专用字段，rank CLI 只能看到 synthetic ratio，无法展示平均 token。这是 rank 闭环的必要修改，不是展示优化。

### 10.3 `enabled_envs`

`window.enabled_envs` 改为：

```text
runtime sampling envs + derived envs with enabled_for_sampling=true
```

这样 `TOKEN-EFFICIENCY` 能作为 rank 列出现。

### 10.4 CLI 渲染

cell 数据优先级：

```text
1. live_detail = sample_details[uid][env]
2. terminal_entry = terminal_scores[uid][env]
3. legacy live_count/live_avg fallback
4. "-"
```

如果 payload:

```text
unit == "tokens" or lower_is_better == true
```

显示：

```text
1.29k tok(-7.8%)/1432
```

字段含义：

- `1.29k tok`：该 row 的平均 token，按人类可读单位展示。
- `-7.8%`：相对 champion overlap 的 token 节省率。
- `1432`：参与 token 计算的有效 usage pair 数。

如果 unavailable：

```text
1.29k tok(partial 42%)/64
```

含义：

- token env 本轮未参与 scoring。
- coverage 42%。
- 有 64 个有效 usage pair。

如果完全没有有效 usage pair：

```text
tok n/a(0%)/0
```

CLI 不计算 threshold 和 verdict；只展示 payload 中已有字段。

token 数字展示规则：

- 计算和 payload 存储始终使用原始 token 数字，不做 k/M 缩放。
- CLI/API 展示层才做格式化。
- `< 1000` 显示整数，例如 `842 tok`。
- `>= 1000` 且 `< 1_000_000` 显示 k，例如 `1.29k tok`、`48.7k tok`。
- `>= 1_000_000` 显示 M，例如 `1.42M tok`。
- pair 数仍显示原始整数，例如 `/1432`，不做 k 缩写，方便判断 coverage 和样本量。

### 10.5 Rank 数据完整性检查

rank 闭环必须满足以下不变量：

- `enabled_for_sampling=true` 决定 rank 是否出现 `TOKEN-EFFICIENCY` 列。
- live token row 通过 `sample_details` 承载完整 payload；frozen token row 通过 `terminal_scores` 承载完整 payload。
- `sample_counts/sample_averages/champion_overlap_avgs` 只作为兼容字段，不作为 token 展示的数据源。
- CLI 不根据 `sample_averages["TOKEN-EFFICIENCY"]` 推断 token 文案。
- token unavailable 时也可以有 `avg_tokens`，但必须同时有 `available=false` 和明确 `reason`。
- token payload 缺失时显示 `-`，不要把缺失 payload 解释成 0 token 或 0 分。

## 11. scores/latest 与 average_score

`/scores/latest` 仍按 scoring env 过滤。

当 `TOKEN-EFFICIENCY.enabled_for_scoring=false`：

- 不进入 `/scores/latest.scores_by_env`。
- 仍可通过 `/rank/current.window.sample_details` 展示 live 指标。

当 `enabled_for_scoring=true`：

- 可进入 `/scores/latest.scores_by_env`。
- payload 必须包含 `include_in_average_score=false`。

`WeightWriter._average_of_env_scores()` 必须跳过：

```text
include_in_average_score == false
unit == "tokens"
lower_is_better == true
```

原因：

- token ratio 和普通 task score 不是同一量纲。
- token env 可参与 comparator，但不应污染公开 `average_score`。

链上权重不受影响，仍由 battle winner 决定。

当前 `_average_of_env_scores()` 会从每个 env payload 的 `score/mean/avg/average` 中取第一个数字。实现 token payload 后，如果不显式跳过，`TOKEN-EFFICIENCY.avg=ratio_to_champion` 会被错误混入 `average_score`。因此这里不是软建议，而是必须修改：

```text
if env_payload.include_in_average_score is false: skip
elif env_payload.unit == "tokens": skip
elif env_payload.lower_is_better is true: skip
else: include existing numeric score logic
```

## 12. Snapshot 审计

`score_snapshots.config.outcome.per_env` 中 token env 需要记录 raw metadata：

```json
{
  "env": "TOKEN-EFFICIENCY",
  "champion_avg": 1.0,
  "challenger_avg": 1.084,
  "champion_n": 1432,
  "challenger_n": 1432,
  "delta": 0.084,
  "margin": 0.052631,
  "not_worse_tolerance": 0.019608,
  "verdict": "dominant",
  "reason": "challenger_used_fewer_tokens",
  "metric": {
    "unit": "tokens",
    "token_metric": "total_tokens",
    "available": true,
    "champion_avg_tokens": 1395.6,
    "challenger_avg_tokens": 1287.4,
    "ratio_to_champion": 1.084,
    "saving_rate": 0.0775,
    "coverage_ratio": 0.94,
    "expected_pairs": 1520,
    "token_pairs": 1432,
    "scoring_config": {
      "min_pairs": 100,
      "savings_margin": 0.05,
      "extra_token_tolerance": 0.02,
      "max_score_ratio": 2.0,
      "dominant_ratio_threshold": 1.052631,
      "not_worse_ratio_threshold": 0.980392
    }
  }
}
```

如果 unavailable：

```json
{
  "env": "TOKEN-EFFICIENCY",
  "verdict": "unavailable",
  "reason": "insufficient_token_pairs",
  "metric": {
    "available": false,
    "coverage_ratio": 0.0,
    "expected_pairs": 1520,
    "token_pairs": 0,
    "scoring_config": {
      "min_pairs": 100,
      "savings_margin": 0.05,
      "extra_token_tolerance": 0.02,
      "max_score_ratio": 2.0
    }
  }
}
```

Unavailable token env 不进入 comparator winner 计算，但 snapshot 可记录它为何未生效。

## 13. Payload 单一格式

token module 输出统一 payload。该 payload 描述的是“subject 相对 basis”的 token 指标：

- champion live row：subject 是 champion，basis 也是 champion，`is_reference=true`。
- challenger live row：subject 是 challenger，basis 是 champion。
- challenger lost frozen row：subject 是 challenger，basis 是当时 champion。
- champion dethroned frozen row：subject 是旧 champion，basis 是新 champion。

这样 monitor、scheduler、rank 不需要分别定义 champion/challenger/terminated 的 token 字段。

字段最小化原则：

- `avg` 是为了兼容现有 rank/terminal score payload 的 numeric average 字段，值等于 `ratio_to_champion`。
- `ratio_to_champion` 是 token 语义字段，用于展示和审计。
- 不额外写 `score` 字段，避免 `score` 与 `avg` 表达同一个 synthetic ratio。
- `avg_tokens` 才是用户在 rank 中主要看到的 token 平均消耗。

```json
{
  "count": 1432,
  "avg": 1.084,
  "unit": "tokens",
  "lower_is_better": true,
  "include_in_average_score": false,
  "is_reference": false,
  "available": true,
  "avg_tokens": 1287.4,
  "champion_overlap_avg_tokens": 1395.6,
  "ratio_to_champion": 1.084,
  "saving_rate": 0.0775,
  "coverage_ratio": 0.94,
  "expected_pairs": 1520,
  "token_pairs": 1432,
  "verdict": "dominant",
  "reason": "challenger_used_fewer_tokens"
}
```

同一个 payload 用于：

- monitor live 写入。
- scheduler final freeze。
- rank API `sample_details`。
- rank API `terminal_scores`。
- scores snapshot audit。

不得为 live、frozen、snapshot 分别定义不同 token 字段名。

reference row 规则：

```json
{
  "is_reference": true,
  "available": true,
  "count": 1432,
  "avg": 1.0,
  "unit": "tokens",
  "avg_tokens": 1395.6,
  "champion_overlap_avg_tokens": 1395.6,
  "ratio_to_champion": 1.0,
  "saving_rate": 0.0,
  "coverage_ratio": 0.94,
  "expected_pairs": 1520,
  "token_pairs": 1432
}
```

CLI 对 reference row 显示：

```text
1.40k tok/1432
```

不显示 `(+0.0%)`，避免误导。

## 14. 缺失 Usage 规则

缺失 usage 的 pair：

- 不参与均值。
- 不计入 `token_pairs`。
- 不按 0 处理。
- 不直接判 challenger worse。

coverage 只用于：

- rank 展示。
- snapshot 审计。

coverage 本身不直接判断 token env 是否 available。available 的第一版条件是：

```text
token_pairs >= min_pairs
champ_avg_tokens > 0
subject_avg_tokens > 0
```

因此 `coverage_ratio=0.4` 但 `token_pairs >= min_pairs` 时，token env 仍可参与 scoring；`coverage_ratio=0.9` 但 `token_pairs < min_pairs` 时，token env 仍 unavailable。这样避免同时存在“比例阈值”和“pair 数阈值”两套判定口径。

当 token env unavailable：

- 不注入 comparator。
- 不影响 winner。
- 如果 `token_pairs > 0`，rank 可以显示 partial avg token、coverage 和 pair 数。
- 如果 `token_pairs == 0`，rank 显示 n/a 和 coverage。

当 token env available：

- 正常计算 ratio。
- 可产生 dominant/not_worse/worse。
- 如果 worse，能阻止 challenger dethrone。
- 如果 dominant，能贡献 dominant env。

## 15. 测试计划

### 15.1 Token Module

- 单次 `usage.total_tokens`。
- `prompt_tokens + completion_tokens` fallback。
- 多次 `inference_calls[*].usage` 累加。
- 缺失 usage 返回 `None`。
- 缺失 usage pair 不参与均值。
- `min_pairs` 不足时 unavailable。
- coverage 本身不触发 unavailable，只作为展示和审计字段。
- challenger 省 5% 以上 -> dominant。
- challenger 多 2% 以内 -> not_worse。
- challenger 多超过 2% -> worse。

### 15.2 DAO

- `read_sample_metrics_for_tasks(..., include_extra_usage=False)` 不解压 extra。
- `include_extra_usage=True` 解压 `extra_compressed`。
- refresh_block 不匹配的 row 被过滤。
- task_id 不在请求列表中被过滤。
- 缺失 usage 返回 `usage=None`。
- `read_scores_for_tasks()` 如保留，应复用统一 reader。

### 15.3 Scheduler

- derived env 不进入 task sampler。
- derived env 不启动 executor worker。
- token env disabled for scoring 时不影响 winner。
- token env enabled 且 available 时注入 comparator。
- token env unavailable 时不注入 comparator。
- token env dominant 可贡献 dominant count。
- token env worse 可阻止 dethrone。
- final freeze 写入统一 token payload。

### 15.4 LiveScoresMonitor

- monitor 只在 `enabled_for_sampling=true` 时计算 token payload。
- monitor 调用 token module，不重复实现计算。
- live payload 写入 `miner_stats.scores_by_env["TOKEN-EFFICIENCY"]`。
- usage 缺失时显示 coverage，不报错。

### 15.5 Rank/API/CLI

- `/rank/current.window.enabled_envs` 包含 `TOKEN-EFFICIENCY`。
- `/rank/current.window.sample_details` 保留完整 token payload。
- 老字段 `sample_counts/sample_averages` 仍存在。
- `rank_state._split_display_scores()` 不丢弃 live token payload 专用字段。
- CLI 优先使用 `sample_details`。
- CLI 对 token payload 显示平均 token，不显示 synthetic ratio 百分比。
- terminal row 从 `terminal_scores` 展示 frozen token payload。
- token payload 缺失时 cell 显示 `-`，不是 `0tok`。

### 15.6 average_score

- `WeightWriter._average_of_env_scores()` 跳过 `unit=tokens`。
- `include_in_average_score=false` 生效。

## 16. 实施顺序

1. 增加 `token_efficiency.py`
   - config parser。
   - usage extractor。
   - compute function。
   - payload builder。

2. 增加统一 sample metrics reader
   - score-only 模式。
   - score + usage 模式。
   - 旧 score reader 复用它。

3. 增加 env 分类接口
   - runtime env。
   - runtime scoring env。
   - derived env。
   - rank display env。

4. 接入 LiveScoresMonitor
   - shadow 展示优先。
   - 写入统一 payload。

5. 接入 `/rank/current`
   - 新增 `sample_details`。
   - `enabled_envs` 使用 rank display env。

6. 接入 CLI
   - token cell renderer。
   - unavailable renderer。

7. 接入 scheduler `_decide`
   - 使用 runtime overlap。
   - 调 token module。
   - available 时注入 comparator。
   - freeze/snapshot 写入统一 payload。

8. 调整 `WeightWriter._average_of_env_scores`
   - 跳过 token payload。

9. shadow 运行
   - `enabled_for_sampling=true`
   - `enabled_for_scoring=false`
   - 观察 coverage、avg_tokens、rank 展示。

10. 正式 scoring
   - `enabled_for_scoring=true`
   - usage 不足仍按 unavailable/skip 处理

## 17. 最终结论

`TOKEN-EFFICIENCY` 是一个 derived env，不是 runtime env。

系统闭环如下：

```text
sample_results.extra_compressed
-> unified sample metrics reader
-> token_efficiency.extract_token_usage
-> token_efficiency.compute
-> monitor live payload / scheduler frozen payload / snapshot audit payload
-> /rank/current.sample_details + terminal_scores
-> CLI token renderer
```

这条链路满足：

- 不新增字段。
- 不新增表。
- 缺失 usage 不参与计算。
- battle 和 monitor 复用同一 token module。
- rank 能看到完整 token 信息。
- derived env 不污染 sampler/executor。
- token ratio 不混入 `average_score`。
- `enabled_for_sampling` 控制展示计算，`enabled_for_scoring` 控制 battle 记分。
