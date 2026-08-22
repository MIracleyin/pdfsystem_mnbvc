# 图文交错文档数据集格式设计 — `pdfsys.doc/v1`

**Date:** 2026-08-22
**Scope:** 定义 L2 发布层的 Parquet 数据格式，同时承载纯文本、图文交错、图文对三种消费形态；给出与既有 `ExtractedDoc` / MinerU 产物的映射，以及可运行的写入器与视图投影。
**Status:** 已实现并在 `output/` 的 19 份真实 MinerU 产物上验证通过。
**Out of scope:** 分片调度、MinHash 去重、PII 脱敏、语种识别接线——这些是 Stage-4/5 的事，本格式只负责给它们留好列。

---

## 1 · 一句话设计

> **一行一文档，一个有阅读序的 `blocks` 列表——顺序本身就是交错关系；图像字节按内容寻址存到独立侧表。**

交错视图、图文对视图都是对 `blocks` 的投影，不是第二份存储。

---

## 2 · 现状盘点：手上已经有什么

### 2.1 解析产物

| 产物 | 位置 | 形态 |
|---|---|---|
| `ExtractedDoc` / `Segment` | `pdfsys_types.extract` | `segments` 有序块列表 + 合并后的 `markdown` + 开放的 `stats` |
| MinerU `content_list.json` | 解析 sidecar | **已经是阅读序的交错列表** |
| MinerU `middle.json` | 解析 sidecar | 每页 `page_size`、preproc/para blocks |
| MinerU `images/*.jpg` | 解析 sidecar | 图片/图表/表格裁剪图 |
| `dataset.parquet` | `pdfsys_cli.parquet_writer` | 一行一 PDF 的**扁平运行记录**，23 列，只有 markdown 字符串，无图 |

关键观察有两条：

1. **`content_list.json` 就是我们要的交错结构**，不需要重新解析。在 19 份真实样本上做字段普查（629 个条目）：

   ```
   type:  text 488 · image 67 · header 15 · table 15 · page_number 15 ·
          footer 14 · aside_text 8 · chart 4 · list 3
   keys:  type/bbox/page_idx (629) · text (540) · img_path (86) ·
          text_level (70) · image_caption (67) · image_footnote (67) ·
          content (28) · table_caption/footnote/body (15) · sub_type (8) ·
          chart_caption/footnote (4) · list_items (3)
   ```

   也就是说 caption、表格 HTML、VLM 生成的图像描述（`content`）**MinerU 都已经给了**，只是现在全部被丢在 sidecar 里没人消费——`ExtractedDoc.segments` 对 pipeline/vlm 两条链路目前恒为空元组。

2. **`content_list` 的 bbox 在 0–1000 网格上，与页面尺寸无关** —— 每个轴独立映射到 0–1000。这一点容易踩坑：`middle.json` 里同时有 `page_size`，看上去像是 bbox 的坐标空间，但不是。19 份样本的实测反证：

   | doc | `page_size` | bbox maxX | bbox maxY |
   |---|---|---|---|
   | b8b2757a | `[558, 773]` | 874 | 940 |
   | a24e8b3f | `[480, 350]` | 943 | 997 |
   | 6ffc0b0a | `[4000, 2853]` | 942 | 999 |
   | 444b59c2 | `[1500, 2121]` | 944 | 998 |

   `page_size` 跨度 8 倍，bbox 上界却恒定贴着 1000。按 `page_size` 归一会把大量框 clamp 成 1.0（`[558,773]` 那份的正文块直接塌成零宽），MinerU [官方文档](https://opendatalab.github.io/MinerU/reference/output_files/)也写明 bbox "mapped to a range of 0-1000"。所以除以 1000，`middle.json` 只用来取页数。

### 2.2 一个必须先解决的前置条件

`external/parsers` 里 pipeline 与 vlm 两个 parser 调用 `mineru-api` 时都写死了：

```python
"return_images": "false",   # extract.py:138 (pipeline) / :155 (vlm)
```

**生产 run 目前拿不到任何图片字节。** 本格式的图像部分要落地，必须先把这个开关翻成 `true`（并给 `output_dir` 落盘路径）。这是 submodule 里的改动，不在本次范围内，但它是唯一的硬阻塞项——本文档所有图像相关验证跑的是 `output/` 下更早一批带 `images/` 目录的产物。

---

## 3 · 相关工作调研

调研目标很具体：**别人怎么把"图和文的相对位置"存进列式格式**。

| 数据集 / 格式 | 记录单位 | 图文对齐方式 | 图像存储 | 值得抄的 | 不抄的 |
|---|---|---|---|---|---|
| **MMC4** (AI2, 2023) | 网页文档 | `text_list` + `image_info[].matched_text_index` + 全量 `similarity_matrix` | 外链 URL | 显式记录"这张图配哪句话"及其置信度 | CLIP 相似度矩阵是 `n_img × n_text`，对 PDF 是纯浪费——PDF 的图文关系是版面给的，不用猜 |
| **OBELICS** (HF, 2023) | 网页文档 | `images[]` / `texts[]` 两条**等长平行数组**，同一位置恰好一侧非空 | 外链 URL | 平行数组的"交错"语义极简单，生态里代码最多 | 一半数组是 null 填充；无法承载页码、bbox、标题层级、表格结构 |
| **MINT-1T-PDF** (mlfoundations, 2024) | PDF 文档 | 同 OBELICS 平行数组 | `images[]` 字符串引用 + `image_metadata[{page, xref, sha256, width, height}]` | 第一个把 OBELICS 形态搬到 PDF 上的；`image_metadata` 里带 `page` 是对的 | 官方 dataset viewer 至今是坏的——**PyArrow schema mismatch，字段顺序在不同记录间不一致**。这是"用松散结构装元数据"的直接代价 |
| **OmniCorpus** (ICLR 2025) | 多源文档 | 流式格式，声称可退化为纯文本 / 图文对 / 交错三种形态 | CAS + CLIP 分数元数据 | **"一份存储、三种视图"这个目标本身**就是我们要的 | 具体 schema 未公开到可复用的程度 |
| **PMC-InterCPT** (2026) | 生物医学文章 | `images[]` / `texts[]` + `metadata` JSON 串；排布规则是"图后紧跟 caption，再跟引用该图的正文段落" | base64 内嵌 | ① **caption 之外必须带"引用该图的正文"**——他们的核心发现是 caption 单独用信息量不够；② 图后紧跟 caption 的排布；③ LLM 打标 + 小模型分类器的二级质量过滤 | `metadata` 塞成一个 JSON 字符串——查不了、push 不下去谓词 |
| **FinePDFs** (HF, 2025) | PDF 文档（纯文本） | 不涉及 | 不涉及 | ① `text` 单列 + **`page_ends` 字符偏移数组**恢复页粒度，不存第二份文本；② `extractor` 列记录抽取器；③ per-page 分数数组（`ocr_quality_scores`、`fw_edu_scores`）| 纯文本，没有图的位置 |
| **DoclingDocument** (IBM, 2025) | 任意文档 | `texts`/`tables`/`pictures` 分表 + `body`/`furniture` 两棵树，靠 JSON pointer 串联 | 引用 | **`furniture` 概念**——页眉页脚页码是版面装饰，不是正文 | 树 + JSON pointer 对"喂给 tokenizer"这个用途过重，列式存储里也没法做谓词下推 |
| **Qwen3-VL** (2025) | 书籍级交错 | 微调模型做多模态解析后对齐图文，跨页合并到 256K token 序列 | — | 跨页合并成长序列是训练侧的事，格式侧只要保证阅读序连续 | — |
| **HF `datasets.Image`** | — | — | `struct<bytes: binary, path: string>` | **这是事实标准**，`cast_column("image", Image())` 直接解码 | — |

### 调研结论

- 平行数组（OBELICS/MINT-1T/PMC）是**网页语料的形态**。PDF 白给的页码、bbox、标题层级、表格结构、caption 归属，在平行数组里全部无处安放，只能塞进一个 JSON 字符串元数据列——MINT-1T-PDF 的 viewer 到今天还是坏的，就是这条路的账单。
- 反过来，**从"有序 typed block 列表"投影出平行数组是两行代码**，反向则是有损的。所以存 block 列表，把 OBELICS 形态作为视图提供。
- PMC-InterCPT 的"图文对不能只有 caption"和 FinePDFs 的 `page_ends`，是两个可以直接拿来的具体技巧。
- Docling 的 `furniture` 区分值得抄：页码"- 1 -"进预训练语料是纯污染。

---

## 4 · 设计决策

每条都是"决策 / 理由 / 代价"。

**D1 · 一行一文档，`blocks: list<struct>` 就是交错本身**
不设独立的对齐结构。块在数组里的位置即阅读序，图块和文本块混排。
*代价*：消费者要会读嵌套列。缓解手段是 D4 的冗余 `text` 列。

**D2 · 图像字节走内容寻址侧表，不进文档表**
实测这批数据：`documents/` 84 KiB、`images/` 771 KiB——**图像占 90% 的字节**。文本扫描和元数据过滤不该被 JPEG 拖着走 row group。内容寻址（SHA-256）顺带把扫描件里反复出现的抬头、公章、logo 去了重。
*代价*：要 join。给了 `image_ids` 冗余列，让 join 不必先读嵌套列。

**D3 · `image` 列就是 HF `datasets.Image` 的 wire struct**
`struct<bytes: large_binary, path: string>`。已验证 `load_dataset(...).cast_column("image", Image())` 直接出 PIL 对象，尺寸与我们探测的一致。
*代价*：无。

**D4 · 保留冗余的 `text` 列 + `page_ends`**
90% 的消费者只要纯文本；从 blocks 重渲染要读整个嵌套列。`page_ends` 抄 FinePDFs，用字符偏移恢复页粒度，不存第二份文本。
*代价*：实测 `text` 占 documents 文件的 39.4%，`blocks[].text` 占 33.8%——重复大约 1.9×。摊到整个 shard（含图）是 ~4% 的额外字节。写入器给了 `include_text=false` 开关，纯文本语料（没有图来摊薄）时可以关掉，实测 84 KiB → 52 KiB。

**D5 · caption / footnote / alt 挂在图块上，不拆成独立 caption 块**
图文对是本格式的一等公民，挂在块上意味着取图文对不用自连接。`alt`（MinerU VLM 生成的图像描述）与 `caption`（人写的）分列，消费者可以 `WHERE source != 'alt'` 把合成文本滤掉。
*代价*：块结构多 3 个可空列。Parquet 的 null 近乎零成本。渲染 markdown 与交错视图时按 PMC-InterCPT 的排布，caption 紧跟在图后输出，不会漏进正文。

**D6 · `mentions`：图号引用回链**
从 caption 里解析 `图 3` / `Figure 3` / `表 2`，再在正文里找同号引用，把引用段落的块下标记在图块上。这是 PMC-InterCPT 那条发现的廉价版本。实测在样本里正确命中（`表 6-3-2` 的 caption 与正文引用配上了）。
*代价*：正则匹配，会有假阳性；所以它只是 `iter_pairs` 的第 4 优先级，caption 永远优先。

**D7 · furniture 分类，默认不进 `text`**
`page_header` / `page_footer` / `page_number` / `aside` 四类保留在 blocks 里，但渲染时跳过。
*代价*：`page_ends` 需要处理"整页只有页眉页脚"的情况——已覆盖测试。

**D8 · bbox 要么归一化到 [0,1]，要么为空；越界是拒绝而不是 clamp**
源坐标除以 `bbox_scale`（MinerU 默认 1000）；任何一个分量落在 [0,1] 之外就整个写 null。
*理由*：缺失的 bbox 是诚实的，错误的 bbox 会静默毁掉下游每一次裁剪。**不 clamp** 是关键——框超出声明的 scale 意味着 scale 判断错了，clamp 会把这个错误藏起来（第一版就是这么踩的，见 §2.1）。

**D8b · 图块渲染进 `text` 时 alt 留空，题注单独成段**
`![](img://<id>)` + 题注段落，而不是 `![题注](img://<id>)` + 题注段落。
*理由*：后者每条题注计两遍 token；前者用 `![]\(img://…\)` 一条正则剥掉图引用后，题注仍留在正文里。模型生成的 `alt` **永不进 `text`** —— `text` 只承载人写 / OCR 出来的内容，合成描述留在 `blocks[].alt` 里由消费者显式选用。

**D9 · 固定强类型列，不用 JSON 字符串装元数据**
MINT-1T-PDF 和 PMC-InterCPT 都把元数据塞进 JSON 字符串，前者的后果是 schema 漂移 + viewer 永久损坏。唯一的 JSON 逃生舱是 `provenance`（上游 license / 批次），且明确声明 pdfsys 不解析它。

**D10 · `type` / `backend` / `lang` 用 dictionary encoding**
十几个取值跑在十亿行上，压缩到接近于零，且保住 `blocks.type == 'image'` 这类谓词下推。

**D11 · schema 版本写进 Parquet file-level key-value metadata**
`pdfsys.schema = "pdfsys.doc/1"`。读端不需要旁路信息就能分派。

---

## 5 · Schema

三张表。`documents` 是唯一的事实来源；`images` 是它的字节侧表；`pairs` 是可选的物化视图。

规范定义：[`docs/schema/doc_dataset.v1.json`](../../schema/doc_dataset.v1.json)（有测试守着它和 Arrow schema 不漂移）。

```
dataset/v1/lang=zho_Hans/source=arxiv/qb=high/
├── documents/shard-00000.parquet
├── images/shard-00000.parquet
├── pairs/shard-00000.parquet        # 可选
└── shard-00000.meta.json
```

### 5.1 `documents`

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | string | 源 PDF 的 SHA-256，与 `ExtractedDoc.sha256` 同一身份 |
| `source_uri` | string | 来源 |
| `provenance` | string | 上游不透明 JSON（license / 批次 / 策略层级） |
| `text` | large_string | blocks 的 Markdown 渲染；furniture 已剔除，图为 `![](img://<image_id>)`，题注紧跟其后单独成段 |
| `page_ends` | list\<int32\> | 每页在 `text` 中结束的字符偏移，长度 = `n_pages` |
| `blocks` | list\<struct\> | **权威记录**，见 5.2 |
| `image_ids` | list\<string\> | 本文档引用的去重 image_id，首次出现序 |
| `n_pages` `n_blocks` `n_chars` `n_images` `n_tables` `n_formulas` | int32 | 便宜的过滤列，全部可从 blocks 导出 |
| `backend` | dict\<string\> | `mupdf` \| `pipeline` \| `vlm` |
| `router_ocr_prob` | float32 | Stage-A 路由器 P(needs OCR) |
| `quality_score` `quality_model` | float32 / string | ModernBERT 质量分 [0,3] |
| `lang` `lang_score` | dict\<string\> / float32 | 语种 |

### 5.2 `blocks[]` struct

| 字段 | 类型 | 说明 |
|---|---|---|
| `idx` | int32 | 文档内阅读序下标，稠密 |
| `page` | int32 | 0 基页码 |
| `type` | dict\<string\> | `text` `title` `list` `code` `table` `formula` `image` `chart` + 四类 furniture：`page_header` `page_footer` `page_number` `aside` |
| `text` | large_string | 沿用 `Segment` 的编码约定：text/title/list = Markdown，table = HTML，formula = LaTeX；image/chart 为 null |
| `level` | int8 | title 的标题层级 |
| `caption` | string | 人写的图/表/图表题注 |
| `footnote` | string | 图/表脚注 |
| `alt` | string | 模型生成的图像描述（MinerU VLM `content`） |
| `bbox` | struct\<x0,y0,x1,y1: float32\> | 归一化 [0,1]，左上原点；不可归一化时为 null |
| `image_id` | string | 图像字节的 SHA-256，join 键 |
| `mentions` | list\<int32\> | 正文中按图号引用本图的块下标 |

### 5.3 `images`

| 列 | 类型 | 说明 |
|---|---|---|
| `image_id` | string | `image.bytes` 的 SHA-256，主键 |
| `image` | struct\<bytes: large_binary, path: string\> | HF `datasets.Image` wire format |
| `format` | dict\<string\> | jpeg / png / webp / gif / unknown |
| `width` `height` `n_bytes` | int32 | 由容器头解析（不引入 Pillow，`pdfsys-core` 保持零依赖） |

### 5.4 `pairs`（可选物化视图）

`doc_id` · `image_id` · `block_idx` · `page` · `text` · `source`。

`source` 记录文本来自哪一档，由好到差：

| `source` | 含义 |
|---|---|
| `content` | caption + 该块自己的转写（表格裁剪图 + 它的 HTML，即表格识别的标准配对形态） |
| `caption` | 人写题注（+ 脚注），ground truth |
| `alt` | 模型生成描述，合成文本 |
| `mention` | 正文中按图号引用该图的段落（PMC-InterCPT 那条） |
| `context` | 相邻文本块，最弱兜底 |

每张图最多产出一行，消费者按 `source` 过滤即可控制合成文本比例。

---

## 6 · 三种视图

### 纯文本

```python
pq.read_table("documents", columns=["id", "text", "quality_score", "lang"])
```

关掉 `text` 列写入时，用 `pdfsys_core.render_markdown(blocks)` 现渲染。

### 图文交错（OBELICS / MINT-1T 形态）

```python
from pdfsys_core import to_interleaved
images, texts = to_interleaved(doc.blocks)
# len(images) == len(texts)，同一位置恰好一侧非空——与 OBELICS 逐字段兼容
```

写给那两个数据集的代码可以原样跑。

### 图文对

```python
from pdfsys_core import iter_pairs
for p in iter_pairs(doc):
    ...  # p.image_id / p.text / p.source
```

或直接读物化的 `pairs/`，join `images/` 拿字节：

```sql
SELECT p.text, i.image
FROM   'pairs/*.parquet'  p
JOIN   'images/*.parquet' i USING (image_id)
WHERE  p.source IN ('caption', 'content')   -- 只要人写的，不要合成
```

---

## 7 · 规模

19 份真实文档、84 张图，zstd：

| 表 | 行数 | 文件 | 备注 |
|---|---|---|---|
| `documents` | 19 | 84.0 KiB | ≈ 4.4 KiB/doc |
| `images` | 84 | 770.9 KiB | ≈ 9.2 KiB/img，**占 shard 90%** |
| `pairs` | 77 | 10.4 KiB | |

`documents` 内部各列压缩后占比：`text` 39.4%、`blocks[].text` 33.8%、`image_id` 4.6%、`image_ids` 4.3%、bbox 四列合计 10.0%。

分片仍按 PRD §4.7：~1 GB/shard，路径 `v1/lang=/source=/qb=/shard-NNNNN.parquet`。

---

## 8 · 与现有 `dataset.parquet` 的关系

**不替换。** 两者是不同层的东西：

- `pdfsys_cli.parquet_writer.ParquetSink` → **L1 运行遥测**：一行一 PDF，记录路由概率、各阶段耗时、错误分类、`kept` 标志。它是排障和调阈值用的，扁平结构正合适。
- `pdfsys_cli.dataset_writer.DatasetWriter` → **L2 发布数据集**：对外分发给训练框架的东西。

`pdfsys dataset --meta results.jsonl` 按 sha256 把前者的 `quality_score` / `ocr_prob` / `backend` join 进后者，两条链路各司其职。

---

## 9 · 实现

| 位置 | 内容 |
|---|---|
| `pdfsys_core/dataset.py` | 格式定义 + 全部纯函数变换（零依赖，stdlib only） |
| `pdfsys_cli/dataset_writer.py` | Arrow schema + `DatasetWriter` |
| `pdfsys_cli/dataset_build.py` | MinerU 目录 / `ExtractedDoc` → `DocRecord` |
| `pdfsys dataset` | CLI 子命令 |
| `docs/schema/doc_dataset.v1.json` | 规范定义（有测试守同步） |
| `tests/schema/test_dataset_v1.py` · `test_dataset_parquet.py` | 70 个测试 |

```sh
pdfsys dataset --from-mineru ./out --to ./dataset/v1 \
               --meta ./out/results.jsonl --pairs
```

---

## 10 · 待办

1. **翻开 `return_images`**（见 §2.2）——图像部分落地的唯一硬阻塞。
2. **mupdf 快车道目前不产图块**：`parser-mupdf` 按设计丢弃 image block。文本类 PDF 里的插图要进语料的话，这条链路要补。
3. **同 sha256 多 backend**：一份 PDF 被 pipeline 和 vlm 各跑一遍会产生两行同 `id`。发布前需要按 `(id, backend)` 择优去重——建议按 `quality_score` 取高。
4. **v1.1 候选**：`sub_type`（MinerU 的 `natural_image` / `chemical` / …，实测出现 8 次）、整页渲染图（同 `images` 表，用于版面/VLM 训练）、图像感知哈希做近重去重、per-page 质量分数组（FinePDFs 那种）。
5. **跨页合并**：Qwen3-VL 那种把连续页并成长序列的做法属于训练侧，本格式靠 `page` + 连续阅读序已经能支持，不需要改 schema。

---

## 参考

- [MMC4 (Multimodal C4)](https://github.com/allenai/mmc4) — `text_list` / `image_info.matched_text_index` / `similarity_matrix`
- [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) — 平行 `images[]` / `texts[]`
- [MINT-1T](https://arxiv.org/abs/2406.11271) · [MINT-1T-PDF 数据卡](https://huggingface.co/datasets/mlfoundations/MINT-1T-PDF-CC-2024-18) — PDF 版平行数组 + `image_metadata`，及其 schema 漂移问题
- [OmniCorpus](https://arxiv.org/abs/2406.08418) (ICLR 2025) — 一份存储、三种形态
- [PMC-InterCPT](https://arxiv.org/abs/2606.01049) (2026) — figure-referencing 正文、图后跟 caption 的排布、二级质量过滤
- [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) (2025) — `page_ends` / `extractor` / per-page 分数数组
- [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/) · [Docling 论文](https://arxiv.org/abs/2501.17887) (2025) — `furniture` 概念
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) (2025) — book-scale 交错数据构造
- [HF Image Dataset 文档](https://huggingface.co/docs/hub/en/datasets-image) — `struct<bytes, path>` 约定
