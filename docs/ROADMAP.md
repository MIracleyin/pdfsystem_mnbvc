# pdfsys-mnbvc · Roadmap

> 优化方案与实施计划 · v0.1 · 2026-04-17
>
> 本文档把 [`PRD.md`](./PRD.md) 描述的目标转化为**带优先级、带工作量、带验收标准**的可执行任务池。PRD 回答"我们要做什么"，ROADMAP 回答"按什么顺序做、怎么做、做完怎么验证"。

---

## 0 · 摘要

**一句话**：设计文档与架构框架一流，工程基础设施缺失严重，6 个 stage 只落地了 1.5 个。

**冲刺计划**：以 2 周"可协作化"冲刺（P0）作为一切后续工作的前提，再用 4 周打磨性能与可靠性（P1），最后 10–16 周补齐 6-stage 闭环（P2）。P3 是 PB 级规模化与生态，作为长期背景项。

---

## 1 · 现状评分卡

| 维度 | 状态 | 评分 |
|---|---|---|
| 设计文档（PRD） | 441 行，取舍清晰 | 9/10 |
| 架构分包 | 7 个 workspace 包，边界合理 | 8/10 |
| 核心契约（`pdfsys-core`） | frozen dataclass + 零依赖 + 原子写 | 9/10 |
| MVP 闭环（Router→MuPDF→Scorer） | 跑通 OmniDocBench-100 | 7/10 |
| **测试** | **零测试文件，零 CI** | **0/10** |
| **依赖管理** | 无 lock 文件，依赖无上界 | 2/10 |
| **Observability** | 无 logging，无 metrics | 2/10 |
| 实现完成度 | 2180 行，4/7 包是 stub | 3/10 |
| Demo & 贡献者体验 | Gradio + Cursor rules 完善 | 8/10 |

**关键风险**：当前状态下 1 人可 hack 前进；**任何超过 3 人的协作会立刻失控**——没有测试保护 parity、没有 CI、没有 lock 文件，第一次依赖升级就会毒化路由器。

---

## 2 · 优化全景

```
┌──────────────────────────────────────────────────────────────────┐
│  P0  工程基础（2 周，阻塞一切后续）                                │
│  ├─ 1.1 测试框架 pytest + 关键单测                                 │
│  ├─ 1.2 代码质量 ruff + mypy + pre-commit                         │
│  ├─ 1.3 GitHub Actions CI                                         │
│  ├─ 1.4 uv.lock 入库 + 依赖上界                                    │
│  └─ 1.5 Parity harness（router 回归守门）                          │
├──────────────────────────────────────────────────────────────────┤
│  P1  性能与可靠性（4 周）                                          │
│  ├─ 2.1 Router 热路径优化（49 ms → 10 ms）                         │
│  ├─ 2.2 Quality scorer 批量推理                                    │
│  ├─ 2.3 structlog 日志系统                                         │
│  ├─ 2.4 Prometheus metrics 导出                                    │
│  └─ 2.5 错误分类 + quarantine 桶                                   │
├──────────────────────────────────────────────────────────────────┤
│  P2  功能补全（8-12 周，按 PRD roadmap）                           │
│  ├─ 3.1 Layout analyser（PP-DocLayoutV3 ONNX INT8）                │
│  ├─ 3.2 Pipeline parser（RapidOCR 简单版式）                       │
│  ├─ 3.3 Stage-B router（layout-cache 驱动）                        │
│  ├─ 3.4 VLM parser（MinerU 2.5 + LMDeploy）                        │
│  ├─ 3.5 Stage-3 后处理                                             │
│  ├─ 3.6 Stage-4 质量 / PII / MinHash 去重                          │
│  └─ 3.7 Stage-5 Parquet 打包                                       │
├──────────────────────────────────────────────────────────────────┤
│  P3  规模化与生态（3-6 个月）                                      │
│  ├─ 4.1 datatrove 编排集成                                         │
│  ├─ 4.2 Slurm / K8s runner                                         │
│  ├─ 4.3 对象存储后端（S3 / OSS / MinIO）                           │
│  ├─ 4.4 中文 EduScore 训练                                         │
│  └─ 4.5 竖排古籍 LoRA                                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3 · P0 工程基础（Week 1-2）

### 3.1 测试框架 · pytest

**目标**：2 周内 `pdfsys-core` ≥ 90% / `pdfsys-router` ≥ 60% / `pdfsys-parser-mupdf` ≥ 60% 覆盖率。

**为什么优先**：`.cursor/rules/01-architecture-invariants.mdc` 里 7 条不变式（BBox 归一化、frozen dataclass、原子写、schema 同构等）**全部可单测验证**。没有测试，"不要违反不变式"只是一句空话。

**交付物结构**：

```
tests/
├── conftest.py                         # 共享 fixtures
├── fixtures/pdfs/                      # 5-10 个跨类型 PDF（< 100 KB/file，入库）
├── unit/
│   ├── core/
│   │   ├── test_bbox.py               # BBox 边界、转换、非法值
│   │   ├── test_serde.py              # to_dict/from_dict roundtrip
│   │   ├── test_cache.py              # LayoutCache 原子写 + 崩溃恢复
│   │   └── test_types.py              # Backend / RegionType 枚举稳定性
│   ├── router/
│   │   ├── test_classifier_smoke.py   # classify() 不 raise 任何畸形输入
│   │   ├── test_feature_shape.py      # 输出必须 124 列，列名锁定
│   │   └── test_error_taxonomy.py     # encrypted/corrupt/empty 错误分类
│   ├── parser_mupdf/
│   │   ├── test_extract_basic.py      # 正常 PDF 段落抽取
│   │   ├── test_bbox_normalized.py    # 所有 bbox ∈ [0, 1]
│   │   └── test_corrupted_pdf.py      # 坏 PDF 不 crash
│   └── bench/
│       └── test_loop_never_raises.py  # 坏 PDF 进去，JSONL 行出来
├── contract/
│   ├── test_extracted_doc_schema.py   # 所有 parser 输出同构
│   └── test_cursor_rules_valid.py     # .mdc frontmatter 合法
└── integration/
    └── test_bench_smoke.py            # python -m pdfsys_bench --limit 3
```

**关键样例**：

```python
# tests/unit/core/test_bbox.py
import pytest
from pdfsys_core import BBox

class TestBBoxInvariants:
    @pytest.mark.parametrize("x0,y0,x1,y1", [
        (-0.1, 0, 0.5, 0.5),   # 负坐标
        (0, 0, 1.1, 0.5),      # 超过 1
        (0.5, 0, 0.3, 0.5),    # x1 < x0
        (0, 0, 0, 0),          # 零面积
    ])
    def test_rejects_invalid(self, x0, y0, x1, y1):
        with pytest.raises(ValueError):
            BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def test_to_pixels_roundtrip(self):
        box = BBox(0.1, 0.2, 0.9, 0.8)
        assert box.to_pixels(1000, 500) == (100, 100, 900, 400)
```

```python
# tests/unit/router/test_feature_shape.py
EXPECTED_COLUMNS = 124

def test_feature_vector_has_124_columns(sample_pdf):
    router = Router()
    decision = router.classify(sample_pdf)
    assert not decision.error
    assert len(decision.features) == EXPECTED_COLUMNS, (
        f"Feature vector drifted from 124 to {len(decision.features)}. "
        "If intentional, retrain XGBoost weights."
    )
```

**实施步骤**：

1. `uv add --group dev pytest pytest-cov pytest-xdist hypothesis`
2. 根 `pyproject.toml` 加 `[tool.pytest.ini_options]` 和 `[tool.coverage.run]`
3. `conftest.py` 提供 `sample_pdf` / `encrypted_pdf` / `corrupted_pdf` fixture
4. 按上表顺序写测试（每天 1 个子目录）
5. 加 `Makefile` 或 `scripts/test.sh`：`uv run pytest -n auto tests/`

**验收**：CI 跑通全部测试 < 2 分钟；三包覆盖率达标。

**工作量**：1 人 · 10 天

---

### 3.2 代码质量 · ruff + mypy + pre-commit

**目标**：零 ruff 错误、`pdfsys-core` 零 mypy 错误、commit 前自动拦截。

**根 `pyproject.toml` 新增**：

```toml
[tool.ruff]
target-version = "py311"
line-length = 100
src = ["packages/pdfsys-core/src", "packages/pdfsys-router/src",
       "packages/pdfsys-parser-mupdf/src", "packages/pdfsys-bench/src",
       "demo"]

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "PLC0415", "BLE001", "RET", "ARG"]
ignore = ["E501"]
per-file-ignores = { "packages/pdfsys-bench/**" = ["BLE001"] }

[tool.mypy]
python_version = "3.11"
strict = true
exclude = ["^packages/pdfsys-parser-(pipeline|vlm)/", "^packages/pdfsys-layout-analyser/"]

[[tool.mypy.overrides]]
module = ["pymupdf.*", "xgboost.*", "gradio.*"]
ignore_missing_imports = true
```

**`.pre-commit-config.yaml`**：

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        files: ^packages/pdfsys-core/
  - repo: local
    hooks:
      - id: no-committed-weights
        name: Reject committed model weights
        entry: bash -c '! git diff --cached --name-only | grep -E "\.(ubj|safetensors|pt|bin)$"'
        language: system
        pass_filenames: false
      - id: validate-cursor-rules
        name: Validate .cursor/rules YAML frontmatter
        entry: python scripts/validate_rules.py
        language: system
        files: ^\.cursor/rules/.*\.mdc$
```

**实施步骤**：

1. `uv add --group dev ruff mypy pre-commit`
2. 写上面两个配置
3. `uv run ruff check --fix .` + `uv run ruff format .` 修现存问题
4. `uv run mypy packages/pdfsys-core` 直到零错
5. `pre-commit install` 追加到 `scripts/setup_cursor.sh`
6. 把 `03-doc-sync.mdc` 里提到的 `scripts/validate_rules.py` 落地

**验收**：`pre-commit run --all-files` 全绿。

**工作量**：1 人 · 3 天

---

### 3.3 GitHub Actions CI

**`.github/workflows/ci.yml`**：

```yaml
name: CI
on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { version: "0.4.x", enable-cache: true }
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy packages/pdfsys-core

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with: { python-version: "${{ matrix.python }}" }
      - run: uv sync --frozen
      - run: uv run python -m pdfsys_router.download_weights
      - run: uv run pytest -n auto --cov --cov-report=xml tests/
      - uses: codecov/codecov-action@v4
        if: matrix.python == '3.11'

  parity:
    runs-on: ubuntu-latest
    if: contains(github.event.pull_request.changed_files, 'feature_extractor.py')
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 2 }
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --frozen
      - run: uv run python -m pdfsys_router.download_weights
      - run: bash scripts/check_parity.sh origin/main HEAD
```

**实施步骤**：

1. 写上面 workflow
2. 可选：`.github/workflows/preview-hf-space.yml` PR 自动部署预览 Space
3. GitHub Settings → Branches 把 `main` 设为 protected、必须通过 CI

**验收**：PR 打开 3 分钟内看到 ✅ × 3。

**工作量**：1 人 · 1 天

---

### 3.4 uv.lock 入库 + 依赖上界

**当前痛点**：
- `.gitignore:14` 把 `uv.lock` 排除了（反模式，lock 文件必须入库）
- 所有依赖只有下界：`pymupdf>=1.24` 明天升级到 2.0 会被自动拉进来

**修复**：

1. 从 `.gitignore` 移除 `uv.lock`
2. 给所有依赖加上界（保守策略 major+1）：

```toml
# packages/pdfsys-router/pyproject.toml
dependencies = [
    "pdfsys-core",
    "pymupdf>=1.24,<2.0",
    "xgboost>=2.0,<3.0",
    "scikit-learn>=1.3,<2.0",
    "pandas>=2.0,<3.0",
    "numpy>=1.26,<3.0",
]
```

3. `uv lock && git add uv.lock`
4. CI 用 `uv sync --frozen`（见 §3.3）

**工作量**：0.5 天

---

### 3.5 Parity Harness

**背景**：`.cursor/rules/21-router-parity.mdc` 已描述 parity 验证流程，但**缺可执行脚本**。

**`scripts/check_parity.sh`**：

```bash
#!/usr/bin/env bash
# Verify router ocr_prob drift between two refs.
# Usage: bash scripts/check_parity.sh <baseline_ref> <candidate_ref>
set -euo pipefail

BASELINE="${1:-origin/main}"
CANDIDATE="${2:-HEAD}"
SAMPLE_DIR="${PARITY_SAMPLE_DIR:-tests/fixtures/pdfs}"
EPSILON="${PARITY_EPSILON:-1e-6}"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

run_bench() {
    local ref="$1" out="$2"
    git worktree add "$WORK_DIR/$ref" "$ref"
    (cd "$WORK_DIR/$ref" && uv sync --frozen --quiet \
       && uv run python -m pdfsys_router.download_weights >/dev/null \
       && uv run python -m pdfsys_bench --pdf-dir "$SAMPLE_DIR" --out "$out" --no-quality)
    git worktree remove --force "$WORK_DIR/$ref"
}

run_bench "$BASELINE"  "$WORK_DIR/baseline.jsonl"
run_bench "$CANDIDATE" "$WORK_DIR/candidate.jsonl"

uv run python scripts/parity_diff.py \
    "$WORK_DIR/baseline.jsonl" "$WORK_DIR/candidate.jsonl" \
    --epsilon "$EPSILON"
```

**`scripts/parity_diff.py`**：接收两个 JSONL、逐 PDF 对比 `ocr_prob`、漂移超阈值 exit 非零。

**工作量**：1 天

---

## 4 · P1 性能与可靠性（Week 3-6）

### 4.1 Router 热路径优化

**现状**：49 ms/PDF（PRD 目标 ≤10 ms）。跑 1 PB 语料 ≈ 浪费 10+ 小时 CPU。

**优化点**（先 profile 后改，要求 P0 测试先到位）：

#### (a) 去掉 pandas DataFrame 构造

```python
# ❌ 现状 (packages/pdfsys-router/src/pdfsys_router/xgb_model.py)
df = pd.DataFrame([features])
names = getattr(self.model, "feature_names_in_", None)
if names is not None:
    df = df.reindex(columns=list(names), fill_value=0)
probs = self.model.predict_proba(df)

# ✅ 优化：缓存列序 + numpy array
class XgbRouterModel:
    def __init__(self, path):
        self._feature_order: list[str] | None = None

    def predict_proba(self, features: dict[str, float]) -> float:
        if self._feature_order is None:
            self._feature_order = list(self.model.feature_names_in_)
        arr = np.fromiter(
            (features.get(k, 0.0) for k in self._feature_order),
            dtype=np.float32, count=len(self._feature_order),
        ).reshape(1, -1)
        return float(self.model.predict_proba(arr)[0, 1])
```

预估：~15 ms → ~2 ms。

#### (b) PyMuPDF 文本读取去重

`_get_garbled_text_per_page` 对每页 `get_text()`，后续 `compute_features_per_chunk` 对采样页再读一次——同一页读两次。
优化：读所有采样页文本时就缓存 `page → text` 字典，复用。预估 ~25 ms → ~12 ms。

#### (c) 早 return

`is_encrypted` / `needs_pass` / `len(doc) == 0` 这类硬错误应在特征提取前 short-circuit。

**验收**：Parity harness 验证 `|diff(ocr_prob)| < 1e-6`；OmniDocBench-100 上 p50 ≤ 10 ms。

**工作量**：2-3 天

---

### 4.2 Quality scorer 批量推理

**现状**：单条 3.6 s；10 万文档 ≈ 100 小时。

**改动**：`OcrQualityScorer.score_many` 从循环改成真正 batch：

```python
def score_many(self, texts: list[str], batch_size: int = 8) -> list[QualityScore]:
    self._ensure_loaded()
    torch = self._torch
    results: list[QualityScore] = []
    for i in range(0, len(texts), batch_size):
        batch = [t[:self.max_chars] or " " for t in texts[i:i + batch_size]]
        enc = self._tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=self.max_tokens, padding=True,
        ).to(self._device)
        with torch.inference_mode():
            logits = self._model(**enc).logits.squeeze(-1)
        for j, text in enumerate(batch):
            score = max(0.0, min(3.0, float(logits[j].item())))
            results.append(QualityScore(
                score=score,
                num_chars=len(text),
                num_tokens=int(enc["attention_mask"][j].sum()),
                model=self.model_name,
            ))
    return results
```

**配套**：`pdfsys_bench.loop.run_loop` 改成"先全部 extract → 批量 score → 展回 JSONL"，保持输出顺序。

**验收**：batch=8 相比 batch=1 吞吐 ≥ 3×；单样本数值差 `< 1e-3`。

**工作量**：3 天

---

### 4.3 structlog 日志系统

**现状**：全仓 `print(...)` × 12 处；无级别、无结构。

**方案**：`pdfsys-core` 之外的包引入 `structlog`（core 保持零依赖）：

```python
# packages/pdfsys-router/src/pdfsys_router/_log.py
import structlog
log = structlog.get_logger("pdfsys.router")

# 使用：
log.info("classified", backend=decision.backend.value,
         ocr_prob=decision.ocr_prob, pdf=str(path),
         num_pages=decision.num_pages)
```

生产用 `JSONRenderer()`（便于 Grafana/ELK 摄入），dev 用 `ConsoleRenderer()`。

**工作量**：2 天

---

### 4.4 Prometheus metrics

**最小实现**：

```python
# packages/pdfsys-bench/src/pdfsys_bench/_metrics.py
from prometheus_client import Counter, Histogram, start_http_server

router_decisions = Counter("pdfsys_router_decisions_total",
                           "Router decisions by backend", ["backend"])
router_latency = Histogram("pdfsys_router_duration_seconds",
                           "Router classification latency",
                           buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0])
extract_failures = Counter("pdfsys_extract_failures_total",
                           "Extraction failures", ["backend", "error_class"])

def enable_metrics_endpoint(port: int = 9000) -> None:
    start_http_server(port)
```

`pdfsys-bench` CLI 新增 `--metrics-port` flag。

**工作量**：2 天

---

### 4.5 错误分类 + quarantine 桶

**现状**：失败写 `extract_error: "classify_failed: X"` 自由字符串，无法聚合。

**方案**：`pdfsys-core` 新增 `errors.py`：

```python
from enum import Enum

class ErrorClass(str, Enum):
    OPEN_FAILED = "open_failed"
    ENCRYPTED = "encrypted"
    EMPTY = "empty"
    CORRUPTED_STREAM = "corrupted_stream"
    FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"
    MODEL_INFERENCE_FAILED = "model_inference_failed"
    OOM = "oom"
    UNKNOWN = "unknown"
```

`RouterDecision.error_class: ErrorClass` 替代自由字符串。Bench 按 class 聚合计数。

Quarantine 桶：`out/quarantine/<error_class>/<sha256>.json` 保留失败记录（路径 + error + 完整特征向量，**不保留 PDF**），离线分析用。

**工作量**：3 天

---

## 5 · P2 功能补全（Week 7-16）

### 依赖 DAG

```
Layout Analyser (3.1) ──┬──► Pipeline Parser (3.2) ──┐
                        │                             │
                        └──► VLM Parser     (3.4) ────┼──► Stage-3 (3.5) ──► Stage-4 (3.6) ──► Stage-5 (3.7)
                                                       │
                        ┌──► Stage-B Router (3.3) ─────┘
                        │
                  (reads LayoutCache)
```

### 5.1 Layout Analyser · P2-1

**选型**：PP-DocLayoutV3 ONNX INT8（CPU ~50 ms/页），未来可接 docling-layout-heron。

**交付物**：

```
packages/pdfsys-layout-analyser/src/pdfsys_layout_analyser/
├── __init__.py
├── analyser.py              # LayoutAnalyser 主类
├── runners/
│   ├── pp_doclayoutv3.py    # ONNX runtime 驱动
│   └── heuristic.py         # bbox 列数聚类 fallback
├── render.py                # PDF 页 → PNG（DPI 可调）
└── postprocess.py           # 阅读顺序 + 跨栏合并
```

**API**：

```python
class LayoutAnalyser:
    def __init__(self, config: LayoutConfig = LayoutConfig()): ...
    def analyse(self, pdf_path: str | Path) -> LayoutDocument: ...
    def analyse_with_cache(
        self, pdf_path: str | Path, cache: LayoutCache
    ) -> LayoutDocument: ...   # idempotent
```

**验收**：
- OmniDocBench-100 上 mAP ≥ 0.85
- CPU INT8 吞吐 ≥ 20 页/s/core
- `LayoutDocument` 能被 `LayoutCache.save/load` 完整 roundtrip
- 空 / 加密 / 损坏 PDF 全部不 crash

**工作量**：1 人 · 10 天

---

### 5.2 Pipeline Parser · P2-2

**选型**：RapidOCR（PaddleOCR ONNX 前向，无 Paddle 依赖）。

**交付物**：

```
packages/pdfsys-parser-pipeline/src/pdfsys_parser_pipeline/
├── extract.py              # extract_doc / extract_doc_bytes
├── ocr_engine.py           # RapidOCR wrapper (lazy load)
├── region_processor.py     # 按 RegionType 派发
├── image_cropper.py        # bbox → image crop
└── markdown_emitter.py     # region + OCR → Segment
```

**核心逻辑**：

```python
def extract_doc(pdf_path, *, layout_cache: LayoutCache) -> ExtractedDoc:
    layout = layout_cache.load_or_compute(pdf_path, analyser)
    segments = []
    for page in layout.pages:
        for region in page.regions:
            img = crop_region_from_pdf(pdf_path, page.index, region.bbox)
            text = ocr_engine.recognise(img, region.type)
            segments.append(Segment(
                index=len(segments),
                backend=Backend.PIPELINE,
                page_index=page.index,
                type=region.type,
                content=text,
                bbox=region.bbox,
                source_region_id=region.region_id,
            ))
    return ExtractedDoc(
        sha256=sha256_of_file(pdf_path),
        backend=Backend.PIPELINE,
        segments=tuple(segments),
        markdown=merge_segments_to_markdown(tuple(segments)),
        stats={"page_count": len(layout.pages)},
    )
```

**验收**：
- OmniDocBench 扫描件子集中文字符 F1 ≥ 0.90
- 输出 schema 与 `parser-mupdf` 同构（`tests/contract/test_extracted_doc_schema.py` 保护）
- CPU 吞吐 ≥ 5 页/s/core

**工作量**：1 人 · 12 天

---

### 5.3 Stage-B Router · P2-3

把当前 4 行 stub `decider.py` 做实：

```python
def decide_complex_vs_simple(
    layout: LayoutDocument, config: RouterConfig
) -> Backend:
    if not config.vlm_enabled:
        return Backend.PIPELINE
    if layout.has_complex_content:
        return Backend.VLM
    return Backend.PIPELINE
```

`Router._route()`：`ocr_prob ≥ threshold` 时先查 `LayoutCache`，命中 → 调 `decide_complex_vs_simple`；未命中 → 返回 `DEFERRED`。

**工作量**：2 天

---

### 5.4 VLM Parser · P2-4

**选型**（PRD §4.4）：生产用 LMDeploy 驱动 MinerU 2.5-Pro 1.2B。

**交付物**：

```
packages/pdfsys-parser-vlm/src/pdfsys_parser_vlm/
├── extract.py
├── engines/
│   ├── mineru.py           # LMDeploy wrapper
│   └── paddleocr_vl.py     # 备选
├── batching.py             # dynamic batching
├── rendering.py            # 高 DPI 页面渲染
└── fallback.py             # OOM 降 batch 重试
```

**关键约束**：
- Worker 常驻模型（单例懒加载）
- `max_batch_size=16, max_seq=8192`（PRD §4.4）
- 超长页：单页 > 8192 tokens 按 bbox 聚类切两块
- 单页 OOM 自动降 batch 重试 ≤ 2 次后写 quarantine（见 §4.5）

**工作量**：1 人 · 15 天（含 LMDeploy 调通）

---

### 5.5 Stage-3 后处理

独立成新包 `packages/pdfsys-postproc/`：

```
├── reading_order.py       # 跨页合并、脚注挂回正文、双栏交错修正
├── paragraph_merge.py     # 折行还原 + 中文断句
├── formula_norm.py        # KaTeX 语法校验，失败转 image placeholder
├── table_norm.py          # HTML↔Markdown 双格式，行列校验
└── unicode_norm.py        # NFC + 全半角统一 + 零宽字符清理
```

**工作量**：1 人 · 10 天

---

### 5.6 Stage-4 质量 / PII / MinHash 去重

独立成 `packages/pdfsys-quality/`，复用 `datatrove` 的 MinHash block（PRD §4.6.5）：

```
├── lang_id.py         # GlotLID 段落级语种识别
├── heuristic.py       # 重复 n-gram、非 CJK 比例、行长方差
├── edu_score.py       # 中文 EduScore (fastText → DeBERTa-v3-tiny)
├── pii.py             # 正则 + NER 兜底
└── dedup/
    ├── exact.py       # md5 内容精确去重
    └── minhash.py     # datatrove MinHash LSH wrapper
```

**工作量**：2 人 · 3 周（MinHash 跨 shard 需全局 shuffle，最复杂）

---

### 5.7 Stage-5 Parquet 打包

独立成 `packages/pdfsys-output/`：
- Parquet 分片 ~1 GB/shard，zstd 压缩
- 分桶路径：`v1/lang=zh/source=arxiv/qb=high/shard-NNNNN.parquet`
- JSONL 镜像 + Markdown 抽样存档（每 shard 0.1%）

**工作量**：1 人 · 5 天

---

## 6 · P3 规模化与生态（3-6 个月）

| 项 | 说明 | 工作量 |
|---|---|---|
| **datatrove 集成** | 把现有 stage 包成 `datatrove.Block`，原生 Slurm 后端 | 2-3 周 |
| **Slurm / K8s runner** | 新包 `pdfsys-runner`，支持 shard checkpoint + 反压 | 3-4 周 |
| **对象存储后端** | `pdfsys-core` 抽象 `FSBackend` 协议，支持 `file://` / `s3://` / `oss://` / `minio://` | 1-2 周 |
| **中文 EduScore 训练** | fastText → DeBERTa-v3-tiny 分类器 + 数据标注 | 4-6 周（含标注） |
| **竖排古籍 LoRA** | MinerU 2.5 针对性 LoRA 微调 | 4-6 周（GPU 密集） |

---

## 7 · 里程碑时间线

| 里程碑 | 周 | 标志 |
|---|---|---|
| **M1 · 可协作化** | 2 | CI 绿灯；覆盖率达标；lock 文件入库；parity harness 守门 |
| **M2 · 生产级核心** | 6 | Router p50 ≤ 10 ms；scorer 3× 吞吐；统一 log+metrics；错误可聚合 |
| **M3 · 6-stage 打通** | 16 | 10 GB 数据集端到端跑完；三种 backend 同构 schema |
| **M4 · PB 就绪** | 24 | datatrove + Slurm runner；对象存储后端；TCO 估算入库 |
| **M5 · v0.1 数据集** | 32 | 首个 1 TB 级对外可发布数据集 + 评测报告 |

---

## 8 · Quick Wins · 两周内可立即启动

如果只能挑最高 ROI 的 5 件事立刻做：

1. **写 15 个 core / router / parser-mupdf 单测** — 2 天 · 把不变式变成机器可验证
2. **配 ruff + pre-commit** — 0.5 天 · 新 PR 质量底线立起来
3. **写 `.github/workflows/ci.yml`** — 0.5 天 · 反馈从"review 时"提前到"push 时"
4. **`uv.lock` 入库 + 依赖加上界** — 0.5 天 · 依赖不会突然不一样
5. **`scripts/check_parity.sh` + 10 个样本 PDF 入 fixtures** — 2 天 · router 改动自动守门

合计 **5-6 个工作日**，换来"可协作化"的全部前提。强烈建议以这作为第一冲刺。

---

## 9 · 风险与"不做的事"

### 必须克制的诱惑

- ❌ **不要在 P0 之前碰 stub 实现**——没有测试和 parity harness 保护，任何功能添加都是技术债的利息
- ❌ **不要替换 PyMuPDF**——它在中文场景的工程成熟度是第一梯队，换 pdfminer/PyPDF2 会立刻倒退
- ❌ **不要引入 LangChain / LlamaIndex**——这是数据处理 pipeline，不是 RAG 应用
- ❌ **不要在 `pdfsys-core` 引入 pydantic**——现有 `dataclass(frozen=True, slots=True)` + `serde.py` 够用，换 pydantic 破坏零依赖不变式

### 长期风险对应策略

| 风险 | 对应 |
|---|---|
| MinerU 2.5 新版许可变化 | PaddleOCR-VL 保持热备，`pdfsys-parser-vlm` 做成 engine 抽象 |
| PyMuPDF AGPL 限制 | 评估 pikepdf / pdfplumber 作为退路（低优先级） |
| PB 级对象存储成本失控 | P0 阶段写 `scripts/tco.py` 估算 |
| 中文 PII 召回不足 | NER 模型兜底，保留审计表便于事后补救 |

---

## 10 · 如何跟踪进度

- **短期（P0-P1）**：GitHub Projects / Milestones。每个子项一 issue，带验收标准。
- **中期（P2）**：每个 stage 落地时开一个"tracking issue"聚合子 PR，`CHANGELOG.md` 按 SemVer 更新。
- **长期（P3）**：PRD §10 的 P0/P1/P2/P3 roadmap 每月复盘一次，本文档 v0.N 同步迭代。

进度状态在根 `README.md` §What's implemented 表里维护——按 `.cursor/rules/03-doc-sync.mdc` 的映射表，任何 Stage 状态从 ❌→✅ 都必须同步该表。

---

## 附录 · 总量一览

| 阶段 | 周期 | 核心交付 | 人力 |
|---|---|---|---|
| **P0 工程基础** | 2 周 | pytest + ruff + CI + lock + parity | 1 人 |
| **P1 性能/可靠性** | 4 周 | router 5×、scorer 3×、log/metrics | 1-2 人 |
| **P2 功能补全** | 10-12 周 | 6 stage 闭环 | 2-3 人 |
| **P3 规模化** | 3-6 月 | datatrove + Slurm + PB 级运行 | 3-4 人 |

从 0 到"PB 级准备"约 24 周，累计约 20-30 人周。与 PRD §6 的资源预算 "100 × A100 + 32 节点 CPU 墙钟 ~2 个月"相匹配——**先把工具链造好，再把大算力接上**。
