# 与 MNBVC 多模态格式的兼容

对象是 [`mm_template_mnbvc`](https://github.com/MIracleyin/mm_template_mnbvc) 里的 `mmDataBlock`
——一行一块、字段名全中文、二进制内联。

> **状态**：本文档里提出的修复已经在
> [PR #4](https://github.com/MIracleyin/mm_template_mnbvc/pull/4) 合并进上游
> （`260b92ee`，2026-08-23）。所以 `--dialect v2` 现在**就是上游的格式**，不再是提案；
> 已发布的标准示例数据集
> [`example_mmdata_mnbvc`](https://huggingface.co/datasets/miracleyin/example_mmdata_mnbvc)
> 也随之更新到 v2.1。`--dialect legacy` 保留给还在读合并前 shard 的消费者。
> 下文里描述的「历史实现如何如何」，指的都是 `260b92ee` 之前。

**结论先说：两个格式在最要紧的那一点上本来就是一致的，不需要改 `pdfsys.page/v2` 去迁就它。**
那个仓库的 `chinaxiv_to_image_text_pair_blocks()` 产出的是**一行一页、每行带该页 PNG 和该页 Markdown**
的 `块类型="image-text-pair"` 块 —— 这正是 `pdfsys.page/v2` 的页行。所以映射是改名，不是重构。

一个直接推论：**导出需要 `--images pages`**。历史格式里「图」的单位就是整页光栅，
而不是图块裁剪图。这也从侧面印证了页级整页图那条路是对的。

```sh
pdfsys dataset --from-mineru ./out --to ./dataset/v2 --images pages --pdf-dir ./data/pdfs
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/chinaxiv_0.parquet
```

走 mupdf 快道的文档没有 MinerU 产物，从 PDF 直接打包，整页光栅是默认：

```sh
pdfsys dataset --from-pdf-dir ./data/pdfs --to ./dataset/v2-mupdf
pdfsys mnbvc-export --from-shard ./dataset/v2-mupdf --to ./mnbvc/chinaxiv_1.parquet
```

`--dialect` 默认 `v2`（上游当前格式）；要合并前那份形态用 `--dialect legacy`。

---

## 1 · 字段映射

| `mmDataBlock` | `pdfsys.page/v2` | 说明 |
|---|---|---|
| `实体ID` | `f"{doc_id}-page-{page_index}"` | 历史实现用的是页图**文件名**。改成 sha256 + 页码：同样的形状，但跨重新入库稳定，也不依赖某个文件曾经存在过 |
| `md5` | 见 §3 | legacy 方言保持「`实体ID` 的 md5」；v2 方言改成内容 md5 |
| `块ID` | 文档内递增序号 | 历史实现恒为 0，见 §2 |
| `块类型` | `"image-text-pair"` | 可用 `--block-type` 改 |
| `扩展字段` | 其余全部 v2 列的 JSON | 见 §4 |
| `时间` | 构建日期 `YYYYMMDD` | 与历史实现一致（是处理时间，不是文档时间） |
| `页ID` | `page_index` | 历史实现从不填，见 §2 |
| `文本` | `text` | 页的 Markdown |
| `图片` | `page_images.image.bytes` | 整页光栅 |
| `OCR文本` | `text`（仅当 `extractor ∈ {pipeline, vlm}`） | mupdf 走的是 PDF 自带文本层，不是 OCR，所以留空 |
| `视频` `音频` `STT文本` | — | PDF 语料用不到，保持 null 但**保留声明类型** |

---

## 2 · 两个直接修掉的 bug（`legacy` 方言也修）

这两条填上不会破坏任何现有读端，所以两种方言都修。**上游已随 PR #4 修复。**

### `块ID` 恒为 0

```python
# chinaxiv_block.py
blocks, block_id = [], 0
for page_id, (img_file, md_file) in enumerate(zip(img_files, md_files)):
    ...
    blocks.append(ChinaxivBlock(..., 块ID=block_id, ...))   # block_id 从未 ++
```

一个文档里所有块的 `块ID` 都是 0。导出时按文档递增、跨文档重置。

### `页ID` 从不填

`page_id` 被塞进 `扩展字段` 的 JSON 里，而专门的 `页ID` 列一直是 `None`。
导出时两处都写：`页ID` 列填上，`扩展字段` 里也保留 `page_id` 键，
这样按历史实现写的读端仍能找到它想要的东西。

---

## 3 · `v2` 方言：四处改动线格式的修复

**这四条已经是上游的默认行为了**（PR #4）。这里保留完整论证，因为它们解释了
`legacy` 方言为什么还要存在，以及迁移时会踩到什么。

### 3.1 `图片` 存二进制，不存 base64

`mmDataBlock.to_dict()` 对 `bytes` 做 `base64.b64encode(...).decode()`，
然后 `pd.DataFrame` → parquet，所以**图片列实际是 string 不是 binary**。

**先纠正一个想当然的结论：这不是省空间。** 实测同一份 shard：

| | 未压缩 | zstd 压缩后 |
|---|---|---|
| base64 string | 2261.5 KiB | 1565.5 KiB |
| binary struct | 1696.7 KiB | 1576.5 KiB |

base64 膨胀 1/3，但 zstd 几乎全部拿回来了——**压缩后 base64 反而小 0.7%**。
改二进制真正买到的是三样别的东西：

1. **HuggingFace 能直接解码。** `cast_column("图片", Image())` 在 v2 上正常出 PIL 对象；
   在 legacy 上抛 `ArrowNotImplementedError: Unsupported cast from large_string to struct`。
2. **未压缩内存 -33%**，影响 row group 缓冲和任何 mmap 的读法。
3. **省掉 base64 解码。** 实测 0.773 ms/页；436 万页跑一趟约 **56 分钟单核**，还不含 JPEG 解码。

### 3.2 `md5` 改成内容哈希

```python
md5=get_md5(img_file.name)   # 文件名的 md5
```

哈希文件名既不能去重也不能校验完整性。v2 方言改成 `md5(图片字节 + 文本)`。
整条 pdfsys 流水线是按 sha256 内容寻址的，这一改让两边对得上。

### 3.3 `页ID` 改成整数

声明是 `Optional[str]`，装的却是页码。v2 方言用 `int32`。

### 3.4 显式声明 Arrow schema

```python
table = pa.Table.from_pandas(df)   # 类型从这一批数据推断
```

某一批 `视频`/`音频` 全是 None → 该列被推成 `null` 类型；下一批有值 → 推成 binary。
**两个 shard 拼不起来。** 这正是 MINT-1T-PDF 官方 viewer 至今损坏的同一个失败模式
（PyArrow schema mismatch）。两种方言都显式声明 schema，全 null 的可选列也保住类型。

---

## 4 · `扩展字段` 的处理

历史格式只有这一个扩展点，所以 `pdfsys.page/v2` 里在 `mmDataBlock` 中无处安放的列
全部进这里，而不是丢掉：`doc_id` `page_index` `doc_n_pages` `width_pt` `height_pt`
`rotation` `extractor` `layout_model` `n_chars` `n_blocks` `n_images` `n_tables`
`n_formulas` `lang` `lang_score` `quality_score` `doc_lang` `doc_quality_score`
`router_ocr_prob` `source_uri` `image_ids`，外加历史实现自己的
`page_id` / `page_image_size` / `page_text_length` / `render_dpi`。

同时修掉一处类型漂移：历史实现在 pdf 路径传的是 **dict**（`扩展字段=json_data`），
在 image-text-pair 路径传的是 **str**（`json.dumps(...)`）——同一列两种类型。
导出恒为 str。

**但要说清楚：JSON 字符串装元数据本身就是个坏形态**，查不了、下推不了谓词。
`pdfsys.page/v2` 里这些都是强类型列。如果 MNBVC 侧愿意加列，把
`extractor` / `quality_score` / `lang` 这几个提成一等列是值得的——
它们是过滤语料最常用的三个。

---

## 5 · 合并后新发现的一处

覆盖已发布的标准示例数据集时才发现：**它和代码在 schema 上也是对不上的**，
而且它才是「标准」。

| | `example_mmdata_mnbvc` v2.0（2024-07） | 代码（合并前） |
|---|---|---|
| md5 列 | `文件md5` | `md5` |
| `视频` 列 | 没有 | 有 |
| 图片/音频 | **`binary`**（对的） | base64 字符串（错的） |
| `块类型` | `文字`/`图片`/`音频` | `pdf`/`image-text-pair`/`视频` |

也就是说 **2024 年那份示例把二进制存对了，2025 年的代码反而退回了 base64**；
`块类型` 是**三套**词汇不是两套。PR 里补了 `text`/`image`/`audio` 常量，
示例数据集也更新到 v2.1 与代码对齐。三处（代码 / 示例 / 本导出器）现在同一个 schema。

## 6 · 还没解决的语义错位

| | 状态 | 说明 |
|---|---|---|
| `图片` 列装 PDF 字节 | **已修**（PR #4） | `chinaxiv_to_pdf_blocks` 原来是 `图片=pdf_data`。现在不塞了——PDF 不是图片，那个块的价值在 docling 解析出的文本。**但合并前落盘的数据里这一列仍混着两种东西**，读历史 shard 时要按 `块类型` 分支 |
| 一份 PDF 多个 backend | **已修**（本仓库侧） | 同一份 PDF 被 pipeline 和 vlm 各跑一遍会产生两套同 `doc_id` 的页行。`pdfsys dataset` 现在按 `(doc_id, extractor)` 择优去重并打印丢弃了哪些，`pdfsys dataset-validate` 会把重复主键报成错误。导出器拿到的 shard 已经是干净的 |
| `时间` 是处理时间 | 未解决 | 不是文档时间。两边一致，但值得在数据卡里写明 |
| `whisperx` 是硬依赖 | 未解决 | `uv sync` 会为了转 PDF 把 torch 拉下来。建议挪到 optional extra，PR #4 没动依赖树 |

---

## 7 · 实现

| 位置 | 内容 |
|---|---|
| `pdfsys_cli/mnbvc_export.py` | 两种方言的 schema + 行映射 + 导出器 |
| `pdfsys mnbvc-export` | CLI 子命令 |
| `tests/schema/test_mnbvc_export.py` | 25 个测试，含 HuggingFace 往返、schema 逐列钉住 |

```sh
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/out.parquet                  # v2，默认
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/out.parquet --dialect legacy
```

没有整页光栅的页仍然导出（文本还是有价值的），但 `图片` 为 null，
并且计数会打到 stderr —— 一次静默产出了一堆没有图的 "image-text-pair" 块，
应该看得见而不是默认没事。
