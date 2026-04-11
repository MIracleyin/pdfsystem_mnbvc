# PDFSystem-MNBVC · PRD

> PB 级 PDF → 预训练数据处理系统
> 对标 HuggingFace FinePDFs · 面向 MNBVC 中文语料
> v0.1 · 2026-04-11

---

## 0. TL;DR

构建一套可在 10–100 GPU 小集群上长期稳定运行的 PDF → 预训练数据处理系统，将 PB 级原始 PDF 高质量、高吞吐、低成本地转换为结构化中文预训练数据。

关键取舍：

- **双路径 + 前置分流**：用 CPU 上 ≤10 ms 的 XGBoost 路由器把 ~90% 的页面送入 CPU 文本路径（PyMuPDF），只让 ~10% 的页面看到 GPU（借鉴 FinePDFs）。
- **小模型够用**：GPU 路径主选 **MinerU 2.5-Pro 1.2B**（中文字符 F1 0.965，A100 2.12 fps），备选 **PaddleOCR-VL 0.9B**（吞吐快 15.8%）。不调用任何商业 API。
- **编排复用**：直接使用 `datatrove`（FinePDFs 同款），省去自研编排器 80% 的工作量。
- **资源反推**：1 PB 原始 PDF ≈ 5 亿文档 ≈ 100 亿页，在 100 × A100 + 32 节点 CPU 集群下全量处理墙钟约 2 个月。

---

## 1. 背景与目标

MNBVC 是一个长期收集中文语料的开源项目，累积的原始 PDF 已逼近 PB 量级且仍在增长。这些 PDF 覆盖学术论文、政府公文、行业报告、电子书、扫描古籍、报纸期刊等极其异质的来源，是中文大模型预训练数据中一座尚未被充分开采的金矿。

本项目目标是把这些 PDF 转成可直接用于大模型预训练的结构化文本，并保持完整的可复现性与可追溯性。设计上重度借鉴 HuggingFace 2025 年开源的 FinePDFs（475M 文档、1733 语种、3T tokens）的工程经验，同时针对中文语料、有限算力与本地部署做关键取舍。

### 1.1 关键非功能目标

| 维度 | 目标 |
| --- | --- |
| 吞吐 | 10–100 GPU 集群下，单月可消化 50–500 TB 原始 PDF |
| 成本 | GPU 成本占比 ≤ 35%，其余由 CPU 路径承担 |
| 质量 | 中文 OCR 字符 F1 ≥ 0.95；阅读顺序还原正确率 ≥ 0.90 |
| 可复现 | 任意 shard 可独立重跑且产物字节级一致 |
| 断点续跑 | 允许任意节点失败，整体进度不丢 |
| 合规 | PII 自动脱敏、来源可追溯、license 元数据完整保留 |

### 1.2 非目标

- 不做多模态数据抽取（图像、图表作为语义单元进入训练数据），这是未来版本的事。
- 不做实时处理，整体是 batch pipeline。
- 不做训练侧的数据混合策略，那是下游训练框架的职责。

---

## 2. 关键设计洞察

在画架构图之前，先把 FinePDFs 等先行项目踩过的坑提炼为六条贯穿全局的设计原则。后续每一个模块的取舍都要回到这六条做合理性检验。

### 2.1 分流先行：90% 的页面不该看到 GPU

FinePDFs 团队最重要的工程发现是：一份普通的 PDF 语料里，绝大多数页面其实是 born-digital 的，只有约 5–10% 是扫描件或文本流损坏的页面。如果用统一的 GPU OCR 流水线处理全部页面，整个项目的 GPU 成本会比理论下限高 10–20 倍。本系统沿用 FinePDFs 的 XGBoost OCR 路由器思路，这是整个成本模型的命门。

### 2.2 小模型已经够用：拒绝商业 API 与百亿 VLM

2025 H2–2026 Q1 的开源社区里，MinerU 2.5-Pro 1.2B（OpenDataLab）和 PaddleOCR-VL 0.9B（百度）这两个亚 1.5B 参数的解耦式 VLM 已经在 OmniDocBench 上全面超越 Gemini 2.5 Pro 等闭源大模型，尤其是 MinerU 2.5 在中文文档上拿下 0.965 的字符 F1。一张 A100 80G 可以并行驻留 16 个以上的 1.2B 模型副本。

### 2.3 CPU 与 GPU 必须分车道

PyMuPDF 在单核上的吞吐是 10–30 PDF/秒，MinerU 2.5 在单卡 A100 上是 2.12 页/秒。两者吞吐差三个数量级。如果共享同一个调度器和队列，慢的一端会立刻变成快的一端的瓶颈（典型 head-of-line blocking）。系统为 CPU 与 GPU 各开独立通道，中间用对象存储 staging 解耦，反压机制让快通道根据慢通道水位自动节流。

### 2.4 以页为最小调度单位，文档只是聚合视图

一份 PDF 内部的页面同质性远低于直觉。一篇论文常常前 10 页 born-digital、最后附录的扫描表格是图像。如果以"文档"为最小单位，会强迫整篇文档走最重的那条路径。本系统以"页"为最小处理单元、以"文档"为最终聚合单元。

### 2.5 廉价过滤前置：不要对垃圾页跑 OCR

语种识别、长度过滤、模板化页面（页眉页脚、版权页、空白页）这些动作都能在 CPU 路径里以毫秒级成本完成。把它们前置在 OCR 之前，可以再砍掉 20–40% 的 GPU 工作量。

### 2.6 idempotent + checkpoint：失败是常态

在 100 GPU × 数月的时间窗口里，节点失败、网络抖动、显存 OOM、模型 NaN、对象存储限流都是必然事件。每个 stage 都设计为幂等（同一输入 → 同一输出）+ shard 级 manifest checkpoint，任意时刻杀掉所有 worker 重启都能从断点继续。

---

## 3. 总体架构

6 个串行 stage + 3 层数据存储。串行只是逻辑视图——实际上每个 stage 内部有大量并行 worker，相邻 stage 之间通过对象存储解耦，可以异步推进。

### 3.1 数据流

```text
原始 PDF 对象存储
        │
        ▼
[Stage 0] Ingestion & Sharding   ── manifest.parquet (sha256, size, src)
        │
        ▼
[Stage 1] Triage Classifier (CPU, XGBoost)
        │
        ├── TEXT_OK ──▶ [Stage 2A] CPU 文本路径
        │                  PyMuPDF + 轻量 layout
        │                       │
        │                       ▼
        │                  Markdown + meta
        │
        ├── NEEDS_OCR ─▶ [Stage 2B] GPU 视觉路径
        │                  PP-DocLayoutV3 + MinerU 2.5 / PaddleOCR-VL
        │                       │
        │                       ▼
        │                  Markdown + meta
        │
        └── REJECT ───▶ Quarantine bucket（人工/重训）
                                │
                                ▼
[Stage 3] Postprocess: 阅读顺序、跨页合并、公式表格归一
                                │
                                ▼
[Stage 4] Quality / Lang / PII / Dedup（精确 + MinHash）
                                │
                                ▼
[Stage 5] 输出打包：Parquet shards + JSONL + Markdown 抽样
```

### 3.2 三层存储

- **L0 原始层（cold）**：S3/OSS/MinIO，PB 级，原始 PDF 不可变，按 sha256 前缀分目录。
- **L1 中间层（warm）**：对象存储 + Parquet/JSONL 分片，存放每个 stage 的中间产物。L1 设计为可丢弃——任意时刻清空都能用上游重建。
- **L2 输出层（hot）**：最终 Parquet 数据集，按语种 / 来源 / 质量分桶，供训练框架和 HuggingFace datasets 直接消费。

---

## 4. 模块详解

### 4.1 Stage 0：数据接入与切片

入库时流式扫描计算 sha256、大小、来源 URL、首个 PDF Producer 字段，写入 `manifest.parquet`。这个 manifest 是后续所有 stage 的唯一 source of truth。

- **Sharding**：按 sha256 前两位 hex 切成 256 个 shard（再细可到 1024），每 shard 200 万–500 万个 PDF，单 worker 处理粒度可控。
- **前置精确去重**：同 sha256 只保留一条。
- **PDF 健康检查**：用 PyMuPDF 尝试 open + pages，捕获损坏文件并打 tag，避免后续 worker 反复崩。

### 4.2 Stage 1：PDF 分流分类器

整个系统省钱最重要的模块，直接复刻 FinePDFs 的 OCR Predictor 思路。XGBoost 是有意为之的选择——纯 CPU 推理 ≤10 ms/PDF，模型几 MB，部署零负担。

**特征**（CPU 提取，全部来自 PyMuPDF）：

- 内嵌文本字节数 / PDF 总字节数（关键比值，扫描件接近 0）
- 页面数、平均页面像素面积、是否含字体子集
- 图像对象总面积 / 页面总面积
- ToUnicode CMap 缺失率（中文古籍 / 老 PDF 的关键信号）
- 第一页、中间页、最后一页可提取文本长度三元组
- PDF Producer / Creator 字段（Word / LaTeX / 扫描软件 / Office）
- XObject 数量与 Form XObject 占比

**输出 3 类标签**：

- `TEXT_OK` → 走 Stage 2A
- `NEEDS_OCR` → 走 Stage 2B
- `REJECT` → 损坏 / 加密 / 0 页 / 全空白，进 quarantine

训练数据：5–10 万份人工标注 + 启发式弱标注。目标精确率 ≥ 95%、召回率 ≥ 90%。误判 TEXT_OK 但实际抽不出文字的样本会经 Stage 2A 的失败回退送回 Stage 2B。

### 4.3 Stage 2A：CPU 文本路径

目标是用最少的 CPU 时间把可提取文本流的 PDF 转成结构良好的 Markdown。FinePDFs 在这一路径上用的是 Docling + Layout Heron int8 量化版；本系统做两点中文本土化调整：

- **解析后端**：PyMuPDF 1.27+（基于 MuPDF 1.27.x）。中文 cmap 处理和文本流还原比 Docling 默认后端更稳，每核 10–30 PDF/秒。
- **轻量布局**：PP-DocLayoutV3 的 ONNX int8 量化版，CPU 推理 ~50 ms/页；triage 非常干净时可 fallback 到纯启发式（bbox 列数聚类 + 字号聚类）。
- **阅读顺序**：双栏检测 + 段落合并 + 跨页折行还原。
- **失败回退**：若提取出的字符数小于阈值（如 < 0.3 × 期望字符数），自动改写 manifest 把该 PDF 丢回 Stage 2B。这是 triage 误差的安全网。

### 4.4 Stage 2B：GPU 视觉路径

处理真正难啃的部分：扫描件、图像 PDF、文本流损坏、版式极端的页面。

**主选：MinerU 2.5-Pro 1.2B**

- 中文字符 F1 0.965（OmniDocBench 中文 SOTA）
- 解耦式 VLM：先布局再 patch-level OCR，对长页友好
- 吞吐：A100 80G vLLM async 2.12 页/s；H200 4.47 页/s
- 端到端覆盖文本 / 表格 / 公式 / 图像区，输出结构化 JSON → Markdown
- 显存占用 fp16 ≈ 3 GB，A100 80G 可并行 16+ 副本

**备选 / 混部：PaddleOCR-VL 0.9B**

- 吞吐比 MinerU 2.5 高 15.8%，显存比 dots.ocr 省 40%
- OmniDocBench v1.5 总分 92.56（高于 MinerU 2.5 的 90.67）
- 中文略弱于 MinerU 2.5，但在多语种与吞吐敏感场景上更好

**调度策略**：路由器在 Stage 1 输出之外再附一个二级 hint——主语种为中文且含较多公式表格 → MinerU 2.5；其它 → PaddleOCR-VL。两套模型共享同一 worker pool，只是加载不同权重。

**推理引擎与批处理**：

- 生产环境用 **LMDeploy**（FinePDFs 同款，比 vLLM 省显存、首 token 延迟更低）
- 动态 batching：max batch 16、max seq 8192、超长页强制切块
- 常驻模型：worker 一次加载、长生命周期
- 失败兜底：单页 OOM 自动降 batch 重试 ≤ 2 次后写 quarantine

### 4.5 Stage 3：后处理

无论来自 CPU 还是 GPU 路径，都进入统一的后处理流水线：

- 阅读顺序最终重排（跨页表格合并、脚注挂回正文、双栏交错修正）
- 段落合并（基于行尾标点与中文断句规则修复折行）
- 公式归一（LaTeX 用 KaTeX / MathJax 解析校验，失败的退化为图像占位）
- 表格归一（HTML / Markdown 双格式存储，行列校验失败的转 image-placeholder）
- Unicode 归一（NFC + 全半角统一 + 控制字符剔除 + 零宽字符清理）
- 元数据补全（每段记录来源页码、bbox、置信度）

### 4.6 Stage 4：质量过滤、语种、PII、去重

#### 4.6.1 语种识别

GlotLID（FinePDFs 同款），**段落级**而非文档级——一篇中英混排的论文里，参考文献段打 en，正文打 zh，下游可以分别处理。

#### 4.6.2 启发式质量过滤

- 重复 n-gram 比例（去除 OCR 串行错位产物）
- 非 CJK / 非 ASCII 符号比例（去除符号噪声）
- 行长方差与平均行长（去除 OCR 抖动）
- URL / email 占比、纯数字占比（去除目录页 / 广告页）
- 最短文档长度阈值（按语种自适应：zh ≥ 200 chars，en ≥ 500 chars）

#### 4.6.3 模型质量分类器

训练中文版 EduScore：fastText 起步 → DeBERTa-v3-tiny 升级。训练数据用高质量中文教育/百科语料 vs 论坛灌水/SEO 文本做对比。每段打 0–5 分，下游训练时按分桶 mix。

#### 4.6.4 PII 脱敏

- 正则：身份证 18 位、手机号、银行卡（Luhn 校验）、邮箱、IPv4/IPv6
- 中国特化：车牌号、统一社会信用代码、护照号
- 命名实体兜底：MiniLM / BERT-tiny NER，仅在正则未命中时启用
- 策略：替换为 `⟨PII:phone⟩` 等占位符，原值哈希存入审计表（不入训练数据）

#### 4.6.5 去重

- 第一遍精确去重：sha256（Stage 0 完成）
- 第二遍内容精确去重：normalize 后文本的 md5
- 第三遍模糊去重：MinHash LSH（5-gram、num_hashes=128、threshold 0.85），`datatrove` 的 minhash block 可直接复用
- **跨 shard 是全局 shuffle 唯一点**：需要一个独立 pass，是整个 pipeline 里最昂贵的单个 stage

### 4.7 Stage 5：输出打包

最终对外的数据集采用 **Parquet 主格式 + JSONL 副格式 + Markdown 抽样存档**三件套。

- Parquet 分片：~1 GB / shard，按 `lang / source / quality_bucket` 分桶
- schema：`id, lang, source, text_md, text_plain, meta(json), quality_score, dedup_cluster_id, pii_redacted(bool)`
- 命名约定：`pdfsystem_mnbvc/v1/lang=zh/source=arxiv/qb=high/shard-00001.parquet`
- JSONL：与 Parquet 1:1 镜像，便于 grep / 抽样审计
- Markdown 抽样存档：每 shard 随机抽 0.1% 文档落盘原始 Markdown，长期保留作为人工审核基线

---

## 5. 编排与资源调度

编排框架直接选用 **datatrove**（FinePDFs 同款），它原生提供 Slurm 后端、shard 级 manifest checkpoint、minhash block 等关键能力。集群层面支持 Slurm 与 Kubernetes 双后端。

### 5.1 队列拓扑

- **Lane A（CPU）**：节点级数据并行，每节点 64–128 worker；节点本地 NVMe 做热数据缓存；产出写入 L1。
- **Lane B（GPU）**：每张 GPU 一个 worker 进程，模型常驻；输入来自 Stage 1 直接喂入或 Lane A 的失败回退。
- **Lane C（global ops）**：MinHash dedup、跨 shard 合并这类 shuffle 操作单独排队，避开 A/B 的高吞吐节奏。

### 5.2 反压与水位

Lane A 吞吐远高于 Lane B。如果不加控制，Lane B 的 staging 队列会无限增长。系统在 staging 桶上设置 **high / low watermark**：超过 high watermark 时 Lane A worker 主动 sleep，掉到 low watermark 再恢复。简单可靠，避免引入复杂的消息中间件。

### 5.3 Checkpoint 与断点续跑

- 每个 shard 完成后写一条 manifest record：`{shard_id, stage, status, output_path, n_docs, n_tokens, sha256}`
- worker 启动时先扫 manifest，跳过已完成的 shard
- 中间层产物本身就是 checkpoint，允许某个 stage 部分失败后只重跑该 stage

---

## 6. 资源预算（PB 级反推）

以 1 PB 原始 PDF 为单位，参数取自 FinePDFs 公开数据与 MinerU 2.5 / PaddleOCR-VL 公开 benchmark。数量级估算，用于反推集群规模而非签 SLA。

### 6.1 数据量估算

| 指标 | 假设值 | 推导 |
| --- | --- | --- |
| 原始数据 | 1 PB | 目标输入 |
| 平均 PDF 大小 | 2 MB | MNBVC 抽样观测 |
| PDF 总数 | ≈ 5 × 10⁸ | 1 PB ÷ 2 MB |
| 平均页数 / PDF | 20 | 学术 + 报告混合 |
| 页面总数 | ≈ 1.0 × 10¹⁰ | 5e8 × 20 |
| 分流比例 | 90% CPU / 10% GPU | FinePDFs 经验，中文略偏 GPU |
| CPU 路径页数 | 9 × 10⁹ | |
| GPU 路径页数 | 1 × 10⁹ | |

### 6.2 CPU 路径预算

| 项 | 数值 | 说明 |
| --- | --- | --- |
| PyMuPDF 吞吐 | ~30 页/s/core | 现代数字 PDF，单核 |
| 所需 core·秒 | 3.0 × 10⁸ | 9e9 ÷ 30 |
| 所需 core·小时 | ≈ 8.3 × 10⁴ | |
| 32 节点 × 64 core | 2048 core 并行 | |
| CPU 路径墙钟 | ≈ 40 小时纯计算 | 不含 IO |
| 现实墙钟 | 1–2 周 | 含对象存储与 manifest 开销 |

### 6.3 GPU 路径预算

| 项 | 数值 | 说明 |
| --- | --- | --- |
| MinerU 2.5 吞吐 | ~2 页/s/A100 | 公开 2.12 fps |
| 所需 GPU·秒 | 5 × 10⁸ | 1e9 ÷ 2 |
| 所需 GPU·小时 | ≈ 1.39 × 10⁵ | |
| 50 × A100 满载 | ~115 天 | 理想吞吐 |
| 100 × A100 满载 | ~58 天 | 推荐配置 |
| 100 × H200 满载 | ~26 天 | 高端配置 |

**结论**：100 A100 + 32 节点 CPU 规模下，1 PB 原始 PDF 全量处理墙钟约 2 个月；100 H200 可压到 1 个月以内。与 FinePDFs 团队公开的处理周期数量级一致。

### 6.4 存储预算

- L0 原始：1 PB（不可压缩）
- L1 中间：~30 TB（每 PDF ~60 KB Markdown + meta）
- L2 输出 Parquet：~15 TB（zstd 压缩）
- Manifest + 索引：~50 GB

---

## 7. 存储与数据布局

```text
s3://pdfsystem-mnbvc/
├── L0_raw/                       # 原始 PDF，不可变
│   └── ab/cd/abcd1234....pdf     # 按 sha256 前 4 位分目录
├── L1_intermediate/
│   ├── stage1_triage/            # XGBoost 路由结果
│   │   └── shard-00001.parquet
│   ├── stage2a_text/             # CPU 路径产物
│   ├── stage2b_vision/           # GPU 路径产物
│   ├── stage3_postproc/
│   └── stage4_quality/
├── L2_output/
│   └── v1/lang=zh/source=arxiv/qb=high/shard-00001.parquet
├── manifest/
│   ├── ingest.parquet            # Stage 0 写入
│   └── stage_status.parquet      # 全 stage 状态
└── audit/
    ├── pii_hash_table.parquet    # 不可逆审计
    └── md_samples/               # 0.1% 抽样 Markdown
```

---

## 8. 可观测、容错、质量保证

### 8.1 指标

- 吞吐：每 stage 的 docs/s、pages/s、bytes/s（shard × worker 维度）
- 路由分布：`TEXT_OK / NEEDS_OCR / REJECT` 三类比例随时间变化
- 回退率：Stage 2A → 2B 回退占比（拐点立即告警）
- GPU 利用率：SM busy %、显存占用、batch 平均长度
- 失败率：按 stage 与失败原因分类计数
- 成本：估算的 GPU·小时 / TB 原始数据

### 8.2 质量回归

维护一份 **500 份手工对齐的中文 PDF 基准集**（学术 / 报告 / 扫描古籍 / 报纸 / 双栏论文各 100 份），每天对当天 pipeline 输出做一次自动 diff，超阈值触发人工复核。这是防止"悄悄变烂"的最重要防线。

### 8.3 故障策略

| 粒度 | 策略 |
| --- | --- |
| worker | 单页失败 retry 3 次，仍失败写 quarantine |
| 节点 | 心跳超时由调度器自动重排 |
| shard | manifest 标记 failed，人工审视后决定重跑或丢弃 |
| 全局 | 每周回看 quarantine 桶，按错误聚类决定是否升级 triage 分类器 |

---

## 9. 参考方案对比

| 方案 | 类型 | 中文 | 吞吐 | License | 本系统位置 |
| --- | --- | --- | --- | --- | --- |
| PyMuPDF / MuPDF | CPU 文本 | 好 | 10–30 PDF/s/core | AGPL | Stage 2A 主力 |
| Docling Heron int8 | CPU 布局 | 中 | 依赖 OpenVINO | MIT | 可选 |
| PP-DocLayoutV3 | 布局检测 | 好 | CPU/GPU 均可 | Apache 2.0 | Stage 2B 布局头 |
| **MinerU 2.5-Pro 1.2B** | VLM 端到端 | **极佳** | 2.12 fps@A100 | Apache 2.0 | **Stage 2B 主选** |
| PaddleOCR-VL 0.9B | VLM 端到端 | 好 | 比 MinerU 快 15.8% | Apache 2.0 | Stage 2B 备选 |
| RolmOCR | OCR | 中 | 需 vLLM / LMDeploy | Apache 2.0 | FinePDFs 原选，不用 |
| olmOCR | OCR | 中 | — | Apache 2.0 | baseline |
| Gemini 2.5 Pro | 闭源 API | 好 | API 限速 | 商业 | 不采用（成本 / 合规） |

---

## 10. 实施路线图

### P0 · 单机 PoC（2 周）

- 1 万份多样化 PDF 走通端到端 6 个 stage
- MinerU 2.5 + PyMuPDF 双路径独立验证
- 产出第一批 Parquet shard，人工对比抽样
- **交付物**：可运行的 docker compose + Jupyter 验证 notebook

### P1 · 分类器与 datatrove 集成（4 周）

- 标注 5–10 万份 PDF 训练 XGBoost triage 分类器
- 接入 `datatrove`，跑通单 shard 的 6-stage Slurm 作业
- 实现 manifest checkpoint 与回退闭环
- **交付物**：100 万 PDF 试运行报告 + 成本估算

### P2 · 10 GPU 集群试点（6 周）

- 1 TB 原始 PDF 全链路压测
- 调通反压、重跑、quarantine 流程
- 中文质量回归基准集上线
- **交付物**：可对外发布的 v0.1 数据集 + 评测报告

### P3 · PB 全量与持续迭代（持续）

- 100 GPU 满载推进到 PB 量级
- 按月发布 v0.2、v0.3，每版带 CHANGELOG 与质量 diff
- 长尾专项：竖排古籍、繁体、表格密集型行业报告
- 社区贡献：开源 triage 分类器与中文 EduScore

---

## 11. 风险与开放问题

- **中文古籍与竖排**：MinerU 2.5 在横排中文表现极好，但竖排古籍未经专门评测。P3 阶段计划微调一个古籍专用 LoRA。
- **公式与表格忠实度**：LaTeX 还原失败的样本比例需要持续监控；P2 质量回归基准集要专门覆盖公式表格密集的论文。
- **数据来源合规**：PB 级语料的来源 license 必须随数据流转保留，输出 Parquet 独立列存储。
- **MinerU 2.5 商用授权**：Apache 2.0，但需关注 OpenDataLab 后续版本条款，保留 PaddleOCR-VL 作为热备。
- **PII 召回率**：中文 PII 模式比英文复杂（地址、姓名），正则不够，可能需要小模型 NER 兜底。
- **对象存储成本**：PB 级数据 + 中间层 + 输出层每月存储与流量费用需在 P0 阶段完成 TCO 估算。

---

## 附录 A · 参考资料

- [FinePDFs 数据集](https://huggingface.co/datasets/HuggingFaceFW/finepdfs)
- [FinePDFs 博客](https://huggingface.co/spaces/HuggingFaceFW/FinePDFsBlog)
- [FinePDFs 代码库](https://github.com/huggingface/finepdfs)
- [MinerU 2.5 Pro 1.2B](https://modelscope.cn/models/OpenDataLab/MinerU2.5-Pro-2604-1.2B)
- [MinerU 2.5 论文 (arXiv:2509.22186)](https://arxiv.org/abs/2509.22186)
- [PaddleOCR-VL 论文 (arXiv:2510.14528)](https://arxiv.org/abs/2510.14528)
- [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- [MuPDF 1.27 文档](https://mupdf.readthedocs.io/en/1.27.2/)
- [datatrove](https://github.com/huggingface/datatrove)
- [MinerU 项目主页](https://github.com/opendatalab/MinerU)
- [OmniDocBench](https://github.com/opendatalab/OmniDocBench)
