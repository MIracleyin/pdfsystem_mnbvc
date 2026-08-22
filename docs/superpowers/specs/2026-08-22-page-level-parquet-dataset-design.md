# 页级图文交错数据集格式设计 — `pdfsys.page/v2`

**Date:** 2026-08-22
**Scope:** 定义 L2 发布层的 Parquet 数据格式，同时承载纯文本、图文交错、图文对三种消费形态；给出与既有 `ExtractedDoc` / MinerU 产物的映射，以及可运行的写入器与视图投影。
**Status:** 已实现。在 `output/` 的 19 份真实 MinerU 产物 + 一份人工拼接的 4 页 PDF 上端到端验证通过。
**Out of scope:** 分片调度、MinHash 去重、PII 脱敏、语种识别接线——这些是 Stage-4/5 的事，本格式只负责给它们留好列。

---

## 1 · 一句话设计

> **一行一页，主键 `(doc_id, page_index)` ——这个身份来自 PDF 本身，不是模型造出来的；页文本里内联图标记来承载图文交错；模型派生的结构是旁边一列可丢弃的增强；图像像素要么是裁剪图、要么是整页光栅，二选一，绝不两存。**

交错视图、图文对视图都是投影，不是第二份存储。

---

## 2 · 现状盘点：手上已经有什么

### 2.1 解析产物

| 产物 | 位置 | 形态 |
|---|---|---|
| `ExtractedDoc` / `Segment` | `pdfsys_types.extract` | `segments` 有序块列表 + 合并后的 `markdown` + 开放的 `stats` |
| MinerU `content_list.json` | 解析 sidecar | **已经是阅读序的交错列表**，每条带 `page_idx` |
| MinerU `middle.json` | 解析 sidecar | 每页 `page_size`（PDF 点）、preproc/para blocks |
| MinerU `images/*.jpg` | 解析 sidecar | 图片/图表/表格裁剪图 |
| `dataset.parquet` | `pdfsys_cli.parquet_writer` | 一行一 PDF 的**扁平运行记录**，23 列，只有 markdown 字符串，无图 |

关键观察有三条：

1. **`content_list.json` 就是我们要的交错结构**，不需要重新解析。在 19 份真实样本上做字段普查（629 个条目）：

   ```
   type:  text 488 · image 67 · header 15 · table 15 · page_number 15 ·
          footer 14 · aside_text 8 · chart 4 · list 3
   keys:  type/bbox/page_idx (629) · text (540) · img_path (86) ·
          text_level (70) · image_caption (67) · image_footnote (67) ·
          content (28) · table_caption/footnote/body (15) · sub_type (8) ·
          chart_caption/footnote (4) · list_items (3)
   ```

   caption、表格 HTML、VLM 生成的图像描述（`content`）**MinerU 都已经给了**，只是现在全部被丢在 sidecar 里没人消费——`ExtractedDoc.segments` 对 pipeline/vlm 两条链路目前恒为空元组。

2. **`content_list` 的 bbox 在 0–1000 网格上，与页面尺寸无关**。这一点容易踩坑：`middle.json` 里同时有 `page_size`，看上去像是 bbox 的坐标空间，但它是页面几何（PDF 点）——`[612, 792]` 是 US Letter，`[595, 841]` 是 A4。19 份样本的实测反证：

   | doc | `page_size` | bbox maxX | bbox maxY |
   |---|---|---|---|
   | b8b2757a | `[558, 773]` | 874 | 940 |
   | a24e8b3f | `[480, 350]` | 943 | 997 |
   | 6ffc0b0a | `[4000, 2853]` | 942 | 999 |
   | 444b59c2 | `[1500, 2121]` | 944 | 998 |

   `page_size` 跨度 8 倍，bbox 上界却恒定贴着 1000。按 `page_size` 归一会把大量框 clamp 成 1.0（`[558,773]` 那份的正文块直接塌成零宽），MinerU [官方文档](https://opendatalab.github.io/MinerU/reference/output_files/)也写明 bbox "mapped to a range of 0-1000"。所以除以 1000；`page_size` 拿去填页面几何列。

3. **MinerU 本来就是按页工作的**——`content_list_v2.json` 字面上就是 list-of-pages，路由设计里的 DEFERRED 也是页级概念。

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
| **FinePDFs** (HF, 2025) | PDF 文档（纯文本） | 不涉及 | 不涉及 | ① `extractor` 列记录抽取器；② **per-page 分数数组**（`ocr_quality_scores`、`fw_edu_scores`、`per_page_languages`）——他们需要页粒度信号；③ `page_ends` 字符偏移恢复页边界 | 行是文档，所以页粒度只能塞成数组（见 §4 D1） |
| **DoclingDocument** (IBM, 2025) | 任意文档 | `texts`/`tables`/`pictures` 分表 + `body`/`furniture` 两棵树，靠 JSON pointer 串联 | 引用 | **`furniture` 概念**——页眉页脚页码是版面装饰，不是正文 | 树 + JSON pointer 对"喂给 tokenizer"这个用途过重，列式存储里也没法做谓词下推 |
| **olmOCR-mix / DocLayNet** | 页 | — | 整页渲染图 + 该页文本 | **页是天然单位**，(页图, 页文本) 是 OCR/VLM 训练的标准配对 | — |
| **Qwen3-VL** (2025) | 书籍级交错 | 微调模型做多模态解析后对齐图文，跨页合并到 256K token 序列 | — | 跨页合并成长序列是训练侧的事，格式侧只要保证阅读序连续 | — |
| **HF `datasets.Image`** | — | — | `struct<bytes: binary, path: string>` | **这是事实标准**，`cast_column("image", Image())` 直接解码 | — |

### 调研结论

- 平行数组（OBELICS/MINT-1T/PMC）是**网页语料的形态**。PDF 白给的页码、bbox、标题层级、表格结构、caption 归属，在平行数组里全部无处安放，只能塞进一个 JSON 字符串元数据列——MINT-1T-PDF 的 viewer 到今天还是坏的，就是这条路的账单。
- 反过来，**投影出平行数组是几行代码**，反向则是有损的。所以 OBELICS 形态作为视图提供，不作为存储。
- **文档级行的代价在 FinePDFs 里看得最清楚**：他们需要页粒度的语种和质量分，但行是文档，只能存成 `list<float>` 数组。行改成页，这些全部变回标量。
- PMC-InterCPT 的"图文对不能只有 caption"和 Docling 的 `furniture` 区分，是两个可以直接拿来的具体技巧。

---

## 4 · 设计决策

每条都是"决策 / 理由 / 代价"。

**D1 · 一行一页，主键 `(doc_id, page_index)`**
v1 是一行一文档 + 嵌套 `blocks`。改成页的三个理由：

- **行的身份从模型手里拿回来。** v1 唯一的细粒度句柄是 `blocks[].idx`，完全由跑了哪个版面模型决定。`(doc_id, page_index)` 来自 PDF。
- **行大小变均匀。** 一本 500 页的书在 v1 里是一行、约两万个 block、几 MB 文本；Parquet row group 被撑歪，谓词只能全有全无。
- **`page_ends` 整个消失。** 那一列和渲染器里那套页边界跟踪逻辑，存在的唯一理由就是从文档级文本块里把页找回来。**能删掉一整套机制，是页更自然的强信号。** FinePDFs 的 `ocr_quality_scores[]` / `per_page_languages[]` 是同一个问题的另一种妥协。

*代价*：跨页上下文要 `GROUP BY doc_id ORDER BY page_index`（文件按此序写就是顺序扫描）；文档级字段反规范化到每页。

**D2 · 空页也必须出现在表里**
`split_pages(n_pages=...)` 用 PDF 的真实页数补齐，抽取器一无所获的页会是一条 `text=""`、`blocks=[]` 的行。
*理由*：**悄悄消失的页，是你事后无法发现丢了的页。** 实测里那条空行照样拿到了整页渲染图——正好能看出抽取器漏了什么。

**D3 · 图文交错编码进 `text`，而不是 `blocks`**
页文本里内联图标记（两种形态见 D6c）。`IMAGE_REF_RE` 是格式契约的一部分。
*理由*：这是"降低对解析模型依赖"的**实际着力点**。改成页级行本身并不能减轻依赖——如果图的位置只存在于 `blocks` 里，那只是把问题挪了个地方。交错关系放进字符串后，`blocks` 才真正变成可以整列丢掉的东西。
*代价*：纯文本消费者要么容忍这一行标记，要么用 `strip_image_refs()` 剥掉（题注不会跟着丢，因为它单独成段）。

**D3b · 图块渲染时 alt 留空，题注单独成段**
`![](img://<id>)` + 题注段落，而不是 `![题注](img://<id>)` + 题注段落。
*理由*：后者每条题注计两遍 token；前者剥掉标记后题注仍在正文里。模型生成的 `alt` **永不进 `text`**——那一列只承载人写 / OCR 出来的内容，合成描述留在 `blocks[].alt` 由消费者显式选用。

**D4 · 稳定性分三级，消费者可以停在任意一级**
扫描件里凡是涉及阅读顺序的东西，没有一样是真正模型无关的——图插在哪两段之间是版面模型判的。所以老实分层：

| 级 | 内容 | 依赖 |
|---|---|---|
| ① | `doc_id` / `page_index` / `width_pt` / `height_pt` / 整页渲染图 | PDF 本身，无模型 |
| ② | `text` + 它引用的图像 blob | 抽取器，工具级，`extractor` 列标注 |
| ③ | `blocks`：类型、阅读序、bbox、题注 | 版面模型，`layout_model` 列标注 |

丢掉 ③ 损失的是 bbox 和题注，**不包括交错关系**。

**D5 · 图像字节走内容寻址侧表**
文本扫描不该被 JPEG 拖着走 row group。内容寻址（SHA-256）顺带把扫描件里反复出现的抬头、公章、logo 去了重（实测：一张图被三页引用，只存一份）。

**D6 · 裁剪图与整页图互斥，不能都存**
这是全套设计里最省字节的一条，值得先摆证据。

**MinerU 的裁剪图，本身就是 200 dpi 整页光栅的子矩形。** 反推 19 份文档、172 张裁剪图的隐含 DPI：

```
中位 201.0 dpi · p90 204.2 · 落在 200±10 的占 97.1%
```

拿真实数字对一次：`page_size=[558, 773] pt` 的页，某表格 bbox `[0.134, 0.222, 0.874, 0.369]`，MinerU 存的裁剪图是 `1148×318`；把同一页按 200 dpi 渲成 `1550×2147` 再按 bbox 裁，得到 `1147×315`。差的 1–3 px 是 bbox 量化——它存在 0–1000 整数网格上，每条边约 ±1 px。

**所以同时保有 `images/` 和 `page_images/`，是把同一批像素存两遍。** 于是 `--images` 三选一：

| 模式 | 存什么 | 实测每页 | 相对 |
|---|---|---|---|
| `crops`（默认） | 只有裁剪图 | ~90 KiB | 1.0× |
| `pages` | 只有整页图，图块用 bbox 现裁 | ~311 KiB | 3.4× |
| `none` | 不存像素 | ~5 KiB | — |
| ~~both~~ | ~~两个都存~~ | ~~~401 KiB~~ | ~~4.4×~~ 已禁止 |

（依据：200 dpi 整页 JPEG 311 KiB；裁剪图中位覆盖 7.3% 页面积、均值 11.4%；覆盖 10% 页面积的裁剪图约占页字节的 29%——JPEG 不随面积线性缩放。）

`pages` 模式**在模型无关性上更强**：裁剪图是版面模型决定裁哪儿的产物，换个模型就变；整页光栅不变。它把图像存储压到了 D4 那张表的第 ① 级。换更好的版面模型后重裁即可，不必重渲、不必回 L0。

默认仍是 `crops`，因为纯文本语料没必要为整页像素付 3.4×。

**D6b · 整页图单独一张表，默认不生成**
*理由一*：200 dpi 的整页比裁剪图大一到两个数量级，混在一起会毁掉 row group 尺寸。
*理由二*：**整页图可以从 L0 的不可变 PDF 按任意 DPI 随时重建，而 OCR 文本和版面重算要烧 GPU。** 用途未定时，先把贵的存死、把便宜的留成可重建的槽位。渲染默认 200 dpi——对齐 MinerU 的裁剪分辨率，现裁出来的图和它本来会存的那张在量化误差内一致。

**D6c · 图标记有两种，都是自足的**
`![](img://<sha256>)` 指向 `images/` 里的裁剪图；`![](bbox://x0,y0,x1,y1)` 指向本页光栅的一块矩形。后者只放几何——`doc_id` / `page_index` / `page_image_id` 行上已经有了，每个标记再抄一遍是冗余。两种都能脱离 `blocks` 解析（D3 的性质保住了），`parse_image_ref()` 负责分派。
*边界情况*：`pages` 模式下 bbox 归一化失败的块**保留裁剪图**——否则那张图彻底不可达，比一点冗余糟糕得多。

**D7 · 文档级字段反规范化到每页**
`source_uri` / `provenance` / `doc_lang` / `doc_quality_score` / `router_ocr_prob` / `doc_n_pages` 每页各存一份。
*理由*：dictionary encoding 下重复成本接近零，而最常见的查询（按质量和语种筛文本）因此不需要 join。一条能独立回答问题的页行，就是这个量级上"完整"的含义。

**D7b · 页级质量分与文档级质量分分列，不互相回填**
`quality_score`（页）目前恒为 null，`doc_quality_score` 由 `--meta` 填。
*理由*：现在的打分器是文档级的。把一本书的分抄到每一页上，是把"这本书平均还行"谎报成"这一页还行"。留空是诚实的，等页级打分器上线再填。

**D8 · bbox 要么归一化到 [0,1]，要么为空；越界是拒绝而不是 clamp**
*理由*：缺失的 bbox 是诚实的，错误的 bbox 会静默毁掉下游每一次裁剪。**不 clamp** 是关键——框超出声明的 scale 意味着 scale 判断错了，clamp 会把这个错误藏起来（v1 就是这么踩的，见 §2.1）。

**D9 · `blocks[].idx` 是文档级而非页级**
`mentions`（图号回链）经常跨页——图在第 5 页，讨论它的段落在第 4 页。索引若按页重新编号就接不上。
*代价*：解析 `mentions` 需要读同一 `doc_id` 的兄弟页行，`iter_pairs` 因此接受一整篇文档的页而不是单页。

**D10 · 固定强类型列，不用 JSON 字符串装元数据**
MINT-1T-PDF 和 PMC-InterCPT 都把元数据塞进 JSON 字符串，前者的后果是 schema 漂移 + viewer 永久损坏。唯一的 JSON 逃生舱是 `provenance`（上游 license / 批次），且明确声明 pdfsys 不解析它。

**D11 · `type` / `extractor` / `layout_model` / `lang` 用 dictionary encoding**
十几个取值跑在十亿行上，压缩到接近于零，且保住 `blocks.type == 'image'` 这类谓词下推。

**D12 · schema 版本写进 Parquet file-level key-value metadata**
`pdfsys.schema = "pdfsys.page/2"`。读端不需要旁路信息就能分派。

---

## 5 · Schema

三张表 + 一张可选视图。`pages` 是唯一必需的；另外两张按内容寻址 join，可以独立构建、重建或删除。

规范定义：[`docs/schema/doc_dataset.v2.json`](../../schema/doc_dataset.v2.json)（有测试守着它和 Arrow schema 不漂移）。
一份真实文档的完整样例：[`docs/schema/doc_dataset.v2.sample.md`](../../schema/doc_dataset.v2.sample.md)。
与 MNBVC 既有多模态格式的映射：[`docs/schema/mnbvc-mm-compat.md`](../../schema/mnbvc-mm-compat.md)。

```
dataset/v2/lang=zho_Hans/source=arxiv/qb=high/
├── pages/shard-00000.parquet          # 必需，按 (doc_id, page_index) 有序
├── images/shard-00000.parquet         # 裁剪图        ┐ 互斥：二选一
├── page_images/shard-00000.parquet    # 整页渲染图     ┘ 见 D6
├── pairs/shard-00000.parquet          # 可选物化视图
└── shard-00000.meta.json
```

### 5.1 `pages`

| 列 | 类型 | 说明 |
|---|---|---|
| **身份（来自 PDF，无模型）** | | |
| `doc_id` | string | 源 PDF 的 SHA-256，与 `ExtractedDoc.sha256` 同一身份 |
| `page_index` | int32 | 0 基页码 |
| `width_pt` `height_pt` | float32 | 页面尺寸，PDF 点 |
| `rotation` | int16 | 页面旋转角 |
| **内容** | | |
| `text` | large_string | 页的 Markdown 渲染；图为 `![](img://<sha256>)` 或 `![](bbox://x0,y0,x1,y1)`，**这一列承载交错关系** |
| `image_ids` | list\<string\> | 本页引用的裁剪图，首次出现序；`pages` 模式下为空 |
| `page_image_id` `render_dpi` | string / int16 | 整页渲染图，join `page_images`；默认 null |
| `blocks` | list\<struct\> | 模型派生结构，可为 null，见 5.2 |
| **出处** | | |
| `extractor` | dict\<string\> | `mupdf` \| `pipeline` \| `vlm` |
| `layout_model` | dict\<string\> | 产出 `blocks` 的版面模型标签 |
| **过滤列** | | |
| `n_chars` `n_blocks` `n_images` `n_tables` `n_formulas` | int32 | 便宜的过滤列，`blocks` 丢了也照填 |
| **页级信号** | | |
| `lang` `lang_score` | dict\<string\> / float32 | 页级语种，待页级识别器填 |
| `quality_score` `quality_model` | float32 / dict\<string\> | 页级质量分，待页级打分器填 |
| **文档级，反规范化** | | |
| `doc_n_pages` | int32 | 源文档总页数 |
| `source_uri` `provenance` | string | 来源、上游不透明 JSON |
| `doc_lang` `doc_quality_score` `router_ocr_prob` | | 文档级信号 |

### 5.2 `blocks[]` struct

| 字段 | 类型 | 说明 |
|---|---|---|
| `idx` | int32 | **文档级**阅读序下标（不是页级，见 D9） |
| `type` | dict\<string\> | `text` `title` `list` `code` `table` `formula` `image` `chart` + 四类 furniture：`page_header` `page_footer` `page_number` `aside` |
| `text` | large_string | 沿用 `Segment` 的编码约定：text/title/list = Markdown，table = HTML，formula = LaTeX；image/chart 为 null |
| `level` | int8 | title 的标题层级 |
| `caption` `footnote` | string | 人写的图/表题注与脚注 |
| `alt` | string | 模型生成的图像描述（MinerU VLM `content`），永不进 `text` |
| `bbox` | struct\<x0,y0,x1,y1: float32\> | 归一化 [0,1]，左上原点；不可归一化时为 null |
| `image_id` | string | 裁剪图的 SHA-256，join 键；`pages` 模式下为 null，改由 `bbox` 寻址 |
| `mentions` | list\<int32\> | 正文中按图号引用本图的块 `idx`，**可能在别的页上** |

`page` 字段刻意不进 Arrow struct——它在一行里是常量，等于 `page_index`。

### 5.3 `images` / `page_images`

`images`：`image_id`(PK) · `image: struct<bytes, path>` · `format` · `width` · `height` · `n_bytes`。
`page_images`：同上，外加 `doc_id` · `page_index` · `render_dpi`（同一页可以有多个分辨率）。

`image` 列就是 HF `datasets.Image` 的 wire struct。宽高由容器头解析（不引入 Pillow，`pdfsys-core` 保持零依赖）。

### 5.4 `pairs`（可选物化视图）

`doc_id` · `page_index` · `image_id` · `block_idx` · `text` · `source`。

`source` 由好到差：`content`（题注 + 该块自己的转写，即表格裁剪图配它的 HTML）> `caption`（人写）> `alt`（模型生成）> `mention`（按图号引用该图的正文，可能跨页）> `context`（相邻文本块）。每张图最多一行。

---

## 6 · 三种视图

### 纯文本

```sql
SELECT text FROM 'pages/*.parquet' WHERE doc_quality_score > 2 AND doc_lang = 'zho_Hans'
```

不需要 join。要剥掉图标记用 `pdfsys_core.strip_image_refs()`。

### 图文交错（OBELICS / MINT-1T 形态）

```python
from pdfsys_core import to_interleaved
images, texts = to_interleaved(page["text"])   # 只吃字符串
```

等长、同一位置恰好一侧非空——与 OBELICS 逐字段兼容。**注意它接受的是 `text` 而不是 `blocks`**：这正是 D3 的意义，`--no-blocks` 写出来的 shard 上这个视图照样工作。

整篇文档的交错序列 = 按 `page_index` 拼接各页的 `text` 再投影。

### 图文对

```sql
SELECT p.text, i.image
FROM   'pairs/*.parquet'  p
JOIN   'images/*.parquet' i USING (image_id)
WHERE  p.source IN ('caption', 'content')   -- 只要人写的，不要合成
```

或 `iter_pairs(pages)`——传一整篇文档的页，因为 `mentions` 会跨页。

### （页图, 页文本）对

```sql
SELECT pi.image, p.text
FROM   'pages/*.parquet'       p
JOIN   'page_images/*.parquet' pi USING (doc_id, page_index)
```

olmOCR-mix / DocLayNet 的形态，需要 `--images pages`。同一张光栅还负责按 `bbox` 现裁图块：

```python
from pdfsys_core import IMAGE_REF_RE, crop_region, parse_image_ref
ref = parse_image_ref(IMAGE_REF_RE.search(page["text"]).group(1))
crop = Image.open(io.BytesIO(raster["image"]["bytes"])).crop(
    crop_region(ref.bbox, raster["width"], raster["height"]))
```

---

## 7 · 规模

同一份 4 页文档跑三种模式，实测 shard 总字节：

| `--images` | shard | 每页 | 说明 |
|---|---|---|---|
| `none` | 21.0 KiB | ~5 KiB | 只有文本和结构 |
| `crops`（默认） | 87.7 KiB | ~22 KiB | 加裁剪图 |
| `pages` | 1601.7 KiB | ~400 KiB | 加 200 dpi 整页图，无裁剪图 |

整页渲染图实测（612×792 的学术论文页）：72 dpi ≈ 59 KiB、150 dpi ≈ 204 KiB、200 dpi ≈ 311 KiB、300 dpi ≈ 553 KiB。

按 xsy-01 那 21.8 万份估算——假设平均 20 页/份、约 436 万页——`none` 约 22 GB，`crops` 约 96 GB，`pages` 约 1.7 TB。**平均页数是假设，本地 bench 全是单页 PDF（150/150），量不出真实分布**，跑一次全量统计再定。

分片仍按 PRD §4.7：~1 GB/shard，路径 `v2/lang=/source=/qb=/shard-NNNNN`。

---

## 8 · 与现有 `dataset.parquet` 的关系

**不替换。** 两者是不同层的东西：

- `pdfsys_cli.parquet_writer.ParquetSink` → **L1 运行遥测**：一行一 PDF，记录路由概率、各阶段耗时、错误分类、`kept` 标志。排障和调阈值用的，扁平结构正合适。
- `pdfsys_cli.dataset_writer.DatasetWriter` → **L2 发布数据集**：对外分发给训练框架的东西。

`pdfsys dataset --meta results.jsonl` 按 sha256 把前者的 `quality_score` / `ocr_prob` / `backend` join 进后者的文档级列。

---

## 9 · 实现

| 位置 | 内容 |
|---|---|
| `pdfsys_core/dataset.py` | 格式定义 + 全部纯函数变换（零依赖，stdlib only） |
| `pdfsys_cli/dataset_writer.py` | Arrow schema + `DatasetWriter` |
| `pdfsys_cli/dataset_build.py` | MinerU 目录 / `ExtractedDoc` → 页行；`render_page_images` |
| `pdfsys dataset` | CLI 子命令 |
| `docs/schema/doc_dataset.v2.json` | 规范定义（有测试守同步） |
| `tests/schema/test_dataset_v2.py` · `test_dataset_parquet.py` | 100 个测试 |

```sh
# 常规
pdfsys dataset --from-mineru ./out --to ./dataset/v2 --meta ./out/results.jsonl --pairs

# 改存整页渲染图，图块用 bbox 现裁（需要源 PDF，按 sha256 匹配）
pdfsys dataset --from-mineru ./out --to ./dataset/v2 \
               --images pages --pdf-dir ./data/pdfs --render-dpi 200

# 只要稳定脊柱，丢掉模型派生的 blocks
pdfsys dataset --from-mineru ./out --to ./dataset/v2 --no-blocks
```

---

## 10 · 待办

1. **翻开 `return_images`**（见 §2.2）——图像部分落地的唯一硬阻塞。
2. **mupdf 快车道目前不产图块**：`parser-mupdf` 按设计丢弃 image block。文本类 PDF 里的插图要进语料的话，这条链路要补。
3. **同 sha256 多 backend**：一份 PDF 被 pipeline 和 vlm 各跑一遍会产生两套同 `doc_id` 的页行。发布前需要按 `(doc_id, extractor)` 择优去重——建议按质量分取高。
4. **页级质量分 / 语种**：列已经留好，打分器还是文档级的。这是页级行最直接的收益，值得优先接上。
5. **真实页数分布**：跑一次全量统计，把 §7 的估算换成实测。
6. **v1.1 候选**：`sub_type`（MinerU 的 `natural_image` / `chemical` / …，实测出现 8 次）、图像感知哈希做近重去重、跨页表格合并。

---

## 参考

- [MMC4 (Multimodal C4)](https://github.com/allenai/mmc4) — `text_list` / `image_info.matched_text_index` / `similarity_matrix`
- [OBELICS](https://huggingface.co/datasets/HuggingFaceM4/OBELICS) — 平行 `images[]` / `texts[]`
- [MINT-1T](https://arxiv.org/abs/2406.11271) · [MINT-1T-PDF 数据卡](https://huggingface.co/datasets/mlfoundations/MINT-1T-PDF-CC-2024-18) — PDF 版平行数组 + `image_metadata`，及其 schema 漂移问题
- [OmniCorpus](https://arxiv.org/abs/2406.08418) (ICLR 2025) — 一份存储、三种形态
- [PMC-InterCPT](https://arxiv.org/abs/2606.01049) (2026) — figure-referencing 正文、图后跟 caption 的排布、二级质量过滤
- [FinePDFs](https://huggingface.co/datasets/HuggingFaceFW/finepdfs) (2025) — `extractor` / per-page 分数数组 / `page_ends`
- [DoclingDocument](https://docling-project.github.io/docling/concepts/docling_document/) · [Docling 论文](https://arxiv.org/abs/2501.17887) (2025) — `furniture` 概念
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631) (2025) — book-scale 交错数据构造
- [MinerU 输出格式文档](https://opendatalab.github.io/MinerU/reference/output_files/) — bbox 的 0–1000 网格
- [HF Image Dataset 文档](https://huggingface.co/docs/hub/en/datasets-image) — `struct<bytes, path>` 约定
