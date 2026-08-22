# `pdfsys.doc/v1` 样例

一份真实文档在新格式下的完整形态。源文件是《中国水土保持公报》第 174 页——
双栏排版，含一个跨栏表格、一个饼图、页眉页码，正文里同时引用了表和图。
挑它是因为它一次踩到格式的每个特性：题注、脚注、表格 HTML、模型生成的图表描述、
图号回链、页面装饰。

生成方式：

```sh
pdfsys dataset --from-mineru ./out --to ./dataset/v1 --meta ./out/results.jsonl --pairs
```

本页是从一次本地 MinerU 运行的产物导出的（`output/` 在 `.gitignore` 里，不随仓库分发），
所以不能原地复现；换任何一份带 `images/` 的 MinerU 输出目录跑上面那条命令，得到的是同构的结果。

> 规范定义见 [`doc_dataset.v1.json`](doc_dataset.v1.json)，设计取舍见
> [`2026-08-22-interleaved-parquet-dataset-design.md`](../superpowers/specs/2026-08-22-interleaved-parquet-dataset-design.md)。

---

## 1 · `documents/shard-00000.parquet` — 标量列

```json
{
  "id": "b8b2757a0c12948c3a3c54b4b6742d4bd40fd47d50d2b0625d25d51780e9d49f",
  "source_uri": "s3://mnbvc-pdf/bench150/中国水土保持公报.pdf",
  "provenance": null,
  "text": "54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 万 km²，占总面积的 45.7%。西北黄土…",
  "n_pages": 1,
  "n_blocks": 9,
  "n_chars": 1586,
  "n_images": 1,
  "n_tables": 1,
  "n_formulas": 0,
  "backend": "vlm",
  "router_ocr_prob": 0.9300000071525574,
  "quality_score": 2.809999942779541,
  "quality_model": "miracleyin/mnbvc-pdf-quality-scorer-modernbert",
  "lang": "zho_Hans",
  "lang_score": 1.0
}
```

```
page_ends = [1586]      # 单页文档：全文 1586 字符都在第 0 页
image_ids = [
  "236eb687b9d9dbd6252220fafcbc3f13f1f9efbb41c8b017867b1167864d5398",
  "7cf54628ac955693c93e618a8b0720187c8a0b300858899802e93f149be117cc",
]
```

`page_ends` 是 FinePDFs 那个技巧：字符偏移，`text[:page_ends[0]]` 就是第 0 页，
`text[page_ends[i-1]:page_ends[i]]` 是第 i 页——不用存第二份文本。

> 除 `source_uri` / `lang` / `lang_score` / `quality_*` / `router_ocr_prob` 五项外，
> 本页所有内容都是真实产出。这五项在生产里由 `--meta results.jsonl` 按 sha256 join 进来，
> 这份样例的运行目录没有对应的 results.jsonl，值是手填的示意。

---

## 2 · `documents.blocks[]` — 9 个块，顺序即交错

```jsonc
{ "idx": 0, "page": 0, "type": "text",
  "text": "54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 万 km²，…",
  "bbox": [0.132, 0.122, 0.872, 0.188]
}
{ "idx": 1, "page": 0, "type": "table",
  "text": "<table><tr><td rowspan=\"2\">侵蚀沟道级别 $^1$ </td><td colspan=\"3\">沟…",
  "caption": "表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积",
  "footnote": "① 侵蚀沟道级别用侵蚀沟道的长度表示。",
  "bbox": [0.134, 0.222, 0.874, 0.369],
  "image_id": "236eb687b9d9dbd6…",
  "mentions": [0]
}
{ "idx": 2, "page": 0, "type": "text",
  "text": "在西北黄土高原区中，甘肃省侵蚀沟道数量最多，占区域侵蚀沟道总数量的40.3%；其次为陕西省，占21.1%；侵蚀沟道数量最少…",
  "bbox": [0.132, 0.399, 0.874, 0.557]
}
{ "idx": 3, "page": 0, "type": "chart",
  "caption": "图 6-3-1 西北黄土高原区\n各省（自治区）侵蚀沟道面积占全区沟道面积比例",
  "alt": "| Region | Percentage (%) |\n|---|---|\n| 陕西 | 23.9 |\n| 甘肃 | 28…",
  "bbox": [0.155, 0.567, 0.460, 0.767],
  "image_id": "7cf54628ac955693…",
  "mentions": [2]
}
{ "idx": 4, "page": 0, "type": "text",
  "text": "按西北黄土高原区侵蚀类型统计，高原沟壑区侵蚀沟道共11.03万条，沟道面积3.05万 $km^{2}$ ；丘陵沟壑区侵蚀沟…",
  "bbox": [0.474, 0.562, 0.874, 0.698]
}
{ "idx": 5, "page": 0, "type": "text",
  "text": "高原沟壑区侵蚀沟道主要分布于甘肃省东部、陕西省延安南部和渭河以北、山西省南部等地区，平均沟道纵比为20.42%，沟道沟壑密…",
  "bbox": [0.474, 0.702, 0.872, 0.838]
}
{ "idx": 6, "page": 0, "type": "text",
  "text": "西省北部、山西省西北部和内蒙古自治区南部，平均沟道纵比分别为19.93%、14.06%，沟壑密度分别为3.4\\~7.6km…",
  "bbox": [0.132, 0.844, 0.874, 0.912]
}
{ "idx": 7, "page": 0, "type": "page_header",
  "text": "第六章 水土流失与治理情况",
  "bbox": [0.136, 0.087, 0.381, 0.103]
}
{ "idx": 8, "page": 0, "type": "page_number",
  "text": "174",
  "bbox": [0.157, 0.926, 0.195, 0.940]
}
```

几个点值得单独看：

- **`[1]` 表格块同时有 `text`（HTML）和 `image_id`（裁剪图）**。
  文本视图用 HTML，图文对视图把裁剪图和 HTML 配成一对——正是表格识别的标准配对形态。
- **`[3]` 图表块的 `alt` 是 VLM 生成的数据表**，`caption` 是人写的题注。两者分列，
  消费者可以 `WHERE source != 'alt'` 把合成文本滤掉。
- **`mentions`**：`[1]` 的 `mentions=[0]` 表示第 0 块正文里写了「见表 6-3-2」；
  `[3]` 的 `mentions=[2]` 对应「见图 6-3-1」。这是 PMC-InterCPT 那条——
  题注之外，引用该图的正文才是解释性内容。
- **`[7]` `[8]` 是 furniture**（页眉「第六章 …」、页码「174」）。保留在记录里，
  但默认不进 `text`——页码进预训练语料是纯污染。
- **bbox 全部归一化到 `[0,1]`**：页眉 `y=0.087` 在顶部，页码 `y=0.926` 在底部，
  图表 `x=0.155~0.460` 在左栏、正文 `x=0.474~0.874` 在右栏，双栏版面对得上。

---

## 3 · `images/shard-00000.parquet` — 2 行

| `image_id` | `format` | `width`×`height` | `n_bytes` | `image` |
|---|---|---|---|---|
| `236eb687b9d9dbd62522…` | jpeg | 1148×318 | 66,915 | `struct<bytes, path>` |
| `7cf54628ac955693c93e…` | jpeg | 473×429 | 19,179 | `struct<bytes, path>` |

`image` 列就是 HuggingFace `datasets.Image` 的 wire struct，直接解码：

```python
ds = load_dataset("parquet", data_files="images/*.parquet", split="train")
ds = ds.cast_column("image", Image())
ds[0]["image"]        # -> <PIL.JpegImagePlugin.JpegImageFile 1148x318>
```

`image_id` 是图像字节的 SHA-256。内容寻址意味着扫描件里反复出现的抬头、公章、
logo 在整个 shard 里只存一份。

---

## 4 · 三种视图

### 视图 1 — 纯文本（`documents.text`）

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

图块渲染成 `![](img://<image_id>)`，题注紧跟其后单独成段——
一条 `!\[\]\(img://[0-9a-f]{64}\)` 正则剥掉图引用就是干净的纯文本，题注不丢也不重复。

### 视图 2 — 图文交错（OBELICS / MINT-1T 形态）

```python
from pdfsys_core import to_interleaved
images, texts = to_interleaved(doc.blocks)
```

| i | `images[i]` | `texts[i]` |
|---|---|---|
| 0 | `null` | "54.3%；长度在 1000m 及以上的侵蚀沟道数量 14.70 万条，占总数的 22.0%，面积 8.56 …" |
| 1 | `236eb687b9d9dbd62522…` | `null` |
| 2 | `null` | "表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积 ① 侵蚀沟道级别用侵蚀沟道的长度表示。" |
| 3 | `null` | "在西北黄土高原区中，甘肃省侵蚀沟道数量最多，占区域侵蚀沟道总数量的40.3%；其次为陕西省，占21.1%；侵蚀…" |
| 4 | `7cf54628ac955693c93e…` | `null` |
| 5 | `null` | "图 6-3-1 西北黄土高原区 各省（自治区）侵蚀沟道面积占全区沟道面积比例" |
| 6 | `null` | "按西北黄土高原区侵蚀类型统计，高原沟壑区侵蚀沟道共11.03万条，沟道面积3.05万 $km^{2}$ ；丘陵…" |
| 7 | `null` | "高原沟壑区侵蚀沟道主要分布于甘肃省东部、陕西省延安南部和渭河以北、山西省南部等地区，平均沟道纵比为20.42%…" |
| 8 | `null` | "西省北部、山西省西北部和内蒙古自治区南部，平均沟道纵比分别为19.93%、14.06%，沟壑密度分别为3.4\\…" |

等长、同一位置恰好一侧非空——与 OBELICS 逐字段兼容，
写给那两个数据集的代码可以原样跑。题注排在图的正后方，跟 PMC-InterCPT 的排布一致。

### 视图 3 — 图文对（`pairs/shard-00000.parquet`）

```jsonc
{
  "doc_id":    "b8b2757a0c12948c…",
  "image_id":  "236eb687b9d9dbd6…",
  "block_idx": 1,  "page": 0,
  "source":    "content",
  "text":      "表 6-3-2 西北黄土高原区侵蚀沟道数量、长度与面积\n\n<table><tr><td rowspan=\"2\">侵蚀沟道级别 $^1$ </td><td colspan=\"3\">沟道数量/万条</td><td colspan=\"3\">沟道长度/万km</td><td colspan=\"3\">沟道面…"
}
{
  "doc_id":    "b8b2757a0c12948c…",
  "image_id":  "7cf54628ac955693…",
  "block_idx": 3,  "page": 0,
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
