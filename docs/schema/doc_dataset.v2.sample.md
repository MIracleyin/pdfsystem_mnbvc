# `pdfsys.page/v2` 样例

一页真实文档在新格式下的完整形态。源文件是《中国水土保持公报》第 174 页——
双栏排版，含一个跨栏表格、一个饼图、页眉页码，正文里同时引用了表和图。
挑它是因为它一次踩到格式的每个特性：题注、脚注、表格 HTML、模型生成的图表描述、
图号回链、页面装饰。

```sh
pdfsys dataset --from-mineru ./out --to ./dataset/v2 --meta ./out/results.jsonl --pairs
```

本页是从一次本地 MinerU 运行的产物导出的（`output/` 在 `.gitignore` 里，不随仓库分发），
所以不能原地复现；换任何一份带 `images/` 的 MinerU 输出目录跑上面那条命令，得到的是同构的结果。

> 规范定义见 [`doc_dataset.v2.json`](doc_dataset.v2.json)，设计取舍见
> [`2026-08-22-page-level-parquet-dataset-design.md`](../superpowers/specs/2026-08-22-page-level-parquet-dataset-design.md)。

---

## 1 · `pages/shard-00000.parquet` — 一行一页

```json
{
  "doc_id": "b8b2757a0c12948c3a3c54b4b6742d4bd40fd47d50d2b0625d25d51780e9d49f",
  "page_index": 0,
  "width_pt": 558.0,
  "height_pt": 773.0,
  "rotation": 0,
  "text": "54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 万 km²，占总面积的 45.7%。…",
  "page_image_id": null,
  "render_dpi": null,
  "extractor": "vlm",
  "layout_model": "pp-doclayoutv3@1.0",
  "n_chars": 1586,
  "n_blocks": 9,
  "n_images": 1,
  "n_tables": 1,
  "n_formulas": 0,
  "lang": null,
  "lang_score": null,
  "quality_score": null,
  "quality_model": "miracleyin/mnbvc-pdf-quality-scorer-modernbert",
  "doc_n_pages": 1,
  "source_uri": "s3://mnbvc-pdf/bench150/中国水土保持公报.pdf",
  "provenance": null,
  "doc_lang": "zho_Hans",
  "doc_quality_score": 2.809999942779541,
  "router_ocr_prob": 0.9300000071525574
}
```

```
image_ids = [
  "236eb687b9d9dbd6252220fafcbc3f13f1f9efbb41c8b017867b1167864d5398",
  "7cf54628ac955693c93e618a8b0720187c8a0b300858899802e93f149be117cc"
]
```

几点值得单独看：

- **主键是 `(doc_id, page_index)`**，两者都来自 PDF 本身。这是整个格式里唯一不依赖
  任何模型的身份，也是 v2 相对 v1（一行一文档 + 模型定义的 `blocks[].idx`）最重要的改动。
- **`width_pt` / `height_pt` 来自 PDF**，不是渲染尺寸。
- **`quality_score` 是 null，`doc_quality_score` 有值**：现在的打分器是文档级的。
  把一本书的分抄到每一页，是把「这本书平均还行」谎报成「这一页还行」，所以留空。
- **`page_image_id` 是 null**：这份 shard 用的是默认的 `--images crops`。整页光栅
  可以从 L0 的不可变 PDF 按任意 DPI 随时重建，而 OCR 文本重算要烧 GPU——见 §5。

---

## 2 · `pages.blocks[]` — 9 个块（可丢弃的增强列）

```jsonc
{ "idx": 0, "type": "text",
  "text": "54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 万 km…",
  "bbox": [0.132, 0.122, 0.872, 0.188]
}
{ "idx": 1, "type": "table",
  "text": "<table><tr><td rowspan=\"2\">侵蚀沟道级别 $^1$ </td><td colspan=\"3\"…",
  "caption": "表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积",
  "footnote": "① 侵蚀沟道级别用侵蚀沟道的长度表示。",
  "bbox": [0.134, 0.222, 0.874, 0.369],
  "image_id": "236eb687b9d9dbd6…",
  "mentions": [0]
}
{ "idx": 2, "type": "text",
  "text": "在西北黄土高原区中，甘肃省侵蚀沟道数量最多，占区域侵蚀沟道总数量的40.3%；其次为陕西省，占21.1%；侵蚀沟道数量…",
  "bbox": [0.132, 0.399, 0.874, 0.557]
}
{ "idx": 3, "type": "chart",
  "caption": "图 6-3-1 西北黄土高原区\n各省（自治区）侵蚀沟道面积占全区沟道面积比例",
  "alt": "| Region | Percentage (%) |\n|---|---|\n| 陕西 | 23.9 |\n| 甘肃 | …",
  "bbox": [0.155, 0.567, 0.460, 0.767],
  "image_id": "7cf54628ac955693…",
  "mentions": [2]
}
{ "idx": 4, "type": "text",
  "text": "按西北黄土高原区侵蚀类型统计，高原沟壑区侵蚀沟道共11.03万条，沟道面积3.05万 $km^{2}$ ；丘陵沟壑区侵…",
  "bbox": [0.474, 0.562, 0.874, 0.698]
}
{ "idx": 5, "type": "text",
  "text": "高原沟壑区侵蚀沟道主要分布于甘肃省东部、陕西省延安南部和渭河以北、山西省南部等地区，平均沟道纵比为20.42%，沟道沟…",
  "bbox": [0.474, 0.702, 0.872, 0.838]
}
{ "idx": 6, "type": "text",
  "text": "西省北部、山西省西北部和内蒙古自治区南部，平均沟道纵比分别为19.93%、14.06%，沟壑密度分别为3.4\\~7.6…",
  "bbox": [0.132, 0.844, 0.874, 0.912]
}
{ "idx": 7, "type": "page_header",
  "text": "第六章 水土流失与治理情况",
  "bbox": [0.136, 0.087, 0.381, 0.103]
}
{ "idx": 8, "type": "page_number",
  "text": "174",
  "bbox": [0.157, 0.926, 0.195, 0.940]
}
```

- **`[1]` 表格块同时有 `text`（HTML）和 `image_id`（裁剪图）**。
  文本视图用 HTML，图文对视图把裁剪图和 HTML 配成一对——正是表格识别的标准配对形态。
- **`[3]` 图表块的 `alt` 是 VLM 生成的数据表**，`caption` 是人写的题注。两者分列，
  `alt` 永远不进 `text`，消费者可以 `WHERE source != 'alt'` 把合成文本滤掉。
- **`mentions` 是文档级 `idx`，不是页级**：图在第 5 页、讨论它的段落在第 4 页是常态，
  按页重编号就接不上了。这里 `[1]` 的 `mentions=[0]` 对应正文「见表 6-3-2」，
  `[3]` 的 `mentions=[2]` 对应「见图 6-3-1」。
- **`[7]` `[8]` 是 furniture**（页眉、页码）。保留在记录里，但不进 `text`。
- **bbox 归一化到 `[0,1]`**：页眉 `y=0.087` 在顶部，页码 `y=0.926` 在底部，
  图表 `x=0.155~0.460` 在左栏、正文 `x=0.474~0.874` 在右栏，双栏版面对得上。
- `blocks[]` 里**没有 `page` 字段**——它在一行里是常量，等于 `page_index`。

整列可以用 `--no-blocks` 写成 null。丢掉它损失的是 bbox、题注、块类型，
**不包括图文交错关系**——那个在 `text` 里。

---

## 3 · `images/shard-00000.parquet` — 2 行

| `image_id` | `format` | `width`×`height` | `n_bytes` |
|---|---|---|---|
| `236eb687b9d9dbd62522…` | jpeg | 1148×318 | 66,915 |
| `7cf54628ac955693c93e…` | jpeg | 473×429 | 19,179 |

`image` 列是 HuggingFace `datasets.Image` 的 wire struct，直接解码：

```python
ds = load_dataset("parquet", data_files="images/*.parquet", split="train")
ds = ds.cast_column("image", Image())
ds[0]["image"]        # -> <PIL.JpegImagePlugin.JpegImageFile 1148x318>
```

`image_id` 是图像字节的 SHA-256。内容寻址意味着扫描件里反复出现的抬头、公章、
logo 在整个 shard 里只存一份。这张表和 `page_images/` **互斥**——见 §5。

---

## 4 · 四种视图

### 视图 1 — 纯文本（`pages.text`）

```markdown
54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 万 km²，占总面积的 45.7%。西北黄土高原区侵蚀沟道数量、长度与面积见表 6-3-2。

表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积

<table><tr><td rowspan="2">侵蚀沟道级别 $^1$ </td><td colspan="3">沟道数量/万条</td><td colspan="3">沟道长度/万km</td><td colspan="3">沟道面积/万km $^2$ </td></tr><tr><td>丘陵沟壑区</td><td>高原沟壑区</td><td>合计</td><td>丘陵沟壑区</td><td>高原沟壑区</td><td>合计</td><td>丘陵沟壑区</td><td>高原沟壑区</td><td>合计</td></tr><tr><td>合计</td><td>55.64</td><td…

① 侵蚀沟道级别用侵蚀沟道的长度表示。

在西北黄土高原区中，甘肃省侵蚀沟道数量最多，占区域侵蚀沟道总数量的40.3%；其次为陕西省，占21.1%；侵蚀沟道数量最少的为宁夏回族自治区，占2.51%。侵蚀沟道面积与数量基本一致，甘肃省和陕西省面积较大，占区域侵蚀沟道总面积的比例分别达到28.9%和23.9%；宁夏回族自治区、河南省及内蒙古自治区侵蚀沟道面积较小，分别占5.3%、6.2%、7.5%。西北黄土高原区各省（自治区）侵蚀沟道数量与面积见附表A32，各省（自治区）侵蚀沟道面积占全区沟道面积比例见图6-3-1。

![](img://7cf54628ac955693c93e618a8b0720187c8a0b300858899802e93f149be117cc)

图 6-3-1 西北黄土高原区
各省（自治区）侵蚀沟道面积占全区沟道面积比例

按西北黄土高原区侵蚀类型统计，高原沟壑区侵蚀沟道共11.03万条，沟道面积3.05万 $km^{2}$ ；丘陵沟壑区侵蚀沟道共55.64万条，沟道面积15.67万 $km^{2}$ 。高原沟壑区侵蚀沟道数量占侵蚀沟道总数的16.5%，丘陵沟壑区占83.5%。

高原沟壑区侵蚀沟道主要分布于甘肃省东部、陕西省延安南部和渭河以北、山西省南部等地区，平均沟道纵比为20.42%，沟道沟壑密度1.25km/km²。丘陵沟壑区依据地形地貌差异分为5个副区。其中，第一、第二副区主要分布于陕

西省北部、山西省西北部和内蒙古自治区南部，平均沟道纵比分别为19.93%、14.06%，沟壑密度分别为3.4\~7.6km/km²、3.0\~5.0km/km²；第三、第四副区主要分布于青海省东部、甘肃省中部、河南省西部，平均沟道纵
```

图渲染成 `![](img://<image_id>)`，题注紧跟其后单独成段。`strip_image_refs()`
剥掉标记后题注仍在正文里，不丢也不重复：

```
54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 万 km²，占总面积的 45.7%。西北黄土高原区侵蚀沟道数量、长度与面积见表 6-3-2。  表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积  <table><tr><td rowspan="2">侵蚀沟道级别 $^1$ </td><td colspan="3">沟道数量/万条…
```

### 视图 2 — 图文交错（OBELICS / MINT-1T 形态）

```python
from pdfsys_core import to_interleaved
images, texts = to_interleaved(page["text"])   # 只吃字符串，不碰 blocks
```

| i | `images[i]` | `texts[i]` |
|---|---|---|
| 0 | `null` | "54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.5…" |
| 1 | `img://7cf54628ac9556…` | `null` |
| 2 | `null` | "图 6-3-1 西北黄土高原区 各省（自治区）侵蚀沟道面积占全区沟道面积比例  按西北黄土高原区侵蚀类型统…" |

等长、同一位置恰好一侧非空——与 OBELICS 逐字段兼容。**注意它接受的是 `text`**：
`--no-blocks` 写出来的 shard 上这个视图照样工作，这正是把交错关系放进字符串的意义。

### 视图 3 — 图文对（`pairs/shard-00000.parquet`）

```jsonc
{
  "doc_id":    "b8b2757a0c12948c…",  "page_index": 0,
  "image_id":  "236eb687b9d9dbd6…",  "block_idx": 1,
  "source":    "content",
  "text":      "表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积\n\n<table><tr><td rowspan=\"2\">侵蚀沟道级别 $^1$ </td><td colspan=\"3\">沟道数量/万条</td><td colspan=\"3\">沟道长度/万km</td…"
}
{
  "doc_id":    "b8b2757a0c12948c…",  "page_index": 0,
  "image_id":  "7cf54628ac955693…",  "block_idx": 3,
  "source":    "caption",
  "text":      "图 6-3-1 西北黄土高原区\n各省（自治区）侵蚀沟道面积占全区沟道面积比例"
}
```

`source` 记录文本来自哪一档（由好到差 `content` > `caption` > `alt` > `mention` >
`context`），每张图最多一行。这里表格走了 `content`（题注 + 自己的 HTML 转写），
饼图走了 `caption`（人写题注）。

```sql
-- 只要人写的图文对，不要合成描述
SELECT p.text, i.image
FROM   'pairs/*.parquet'  p
JOIN   'images/*.parquet' i USING (image_id)
WHERE  p.source IN ('caption', 'content')
```

### 视图 4 —（页图, 页文本）对

olmOCR-mix / DocLayNet 的形态。需要 `--images pages --pdf-dir ...`：

```sql
SELECT pi.image, p.text
FROM   'pages/*.parquet'       p
JOIN   'page_images/*.parquet' pi USING (doc_id, page_index)
```

---

## 5 · 同一页在 `--images pages` 下的样子

MinerU 的裁剪图本身就是 200 dpi 整页光栅的子矩形（19 份文档 172 张裁剪图反推，
中位 201.0 dpi，97% 落在 200±10）。所以裁剪图和整页图**只能二选一**——
两个都存就是把同一批像素存两遍。`--images pages` 丢掉裁剪图，图块改由 bbox 寻址：

```jsonc
{ "idx": 1, "type": "table",
  "image_id": null,   // crops 模式下是 sha256
  "bbox": [0.134, 0.222, 0.874, 0.369]
}
{ "idx": 3, "type": "chart",
  "image_id": null,   // crops 模式下是 sha256
  "bbox": [0.155, 0.567, 0.460, 0.767]
}
```

`text` 里的标记也跟着变：

```markdown
![](bbox://0.1550,0.5670,0.4600,0.7670)
```

```
image_ids = []      # 空：没有裁剪图可引用
page_image_id = "<200dpi 整页光栅的 sha256>"     # 由 --pdf-dir 渲出
```

读的时候把图块裁出来：

```python
from pdfsys_core import IMAGE_REF_RE, crop_region, parse_image_ref
ref = parse_image_ref(IMAGE_REF_RE.search(page["text"]).group(1))
im  = Image.open(io.BytesIO(raster["image"]["bytes"]))
crop = im.crop(crop_region(ref.bbox, im.width, im.height))
```

保真度：这一页 `page_size=[558, 773] pt`，表格 bbox `[0.134, 0.222, 0.874, 0.369]`，
MinerU 存的裁剪图是 `1148×318`；按 200 dpi 渲成 `1550×2147` 再裁得到 `1147×315`。
差的 1–3 px 是 bbox 量化（0–1000 整数网格，每边约 ±1 px），不是信息损失。

三种模式实测（同一份 4 页文档的 shard 总字节）：

| `--images` | shard | 每页 | 
|---|---|---|
| `none` | 21.0 KiB | ~5 KiB |
| `crops`（默认） | 87.7 KiB | ~22 KiB |
| `pages` | 1601.7 KiB | ~400 KiB |

默认是 `crops`：纯文本语料没必要为整页像素付 3.4×。要 (页图, 页文本) 对、
或者想让图像存储彻底不依赖版面模型，就用 `pages`。
