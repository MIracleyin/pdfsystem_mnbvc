# 与 MNBVC 多模态格式的兼容

对象是 [`mm_template_mnbvc`](https://github.com/MIracleyin/mm_template_mnbvc) 里的 `mmDataBlock`
——一行一块、字段名全中文、二进制内联。

**结论先说：两个格式在最要紧的那一点上本来就是一致的，不需要改 `pdfsys.page/v2` 去迁就它。**
那个仓库的 `chinaxiv_to_image_text_pair_blocks()` 产出的是**一行一页、每行带该页 PNG 和该页 Markdown**
的 `块类型="image-text-pair"` 块 —— 这正是 `pdfsys.page/v2` 的页行。所以映射是改名，不是重构。

一个直接推论：**导出需要 `--images pages`**。历史格式里「图」的单位就是整页光栅，
而不是图块裁剪图。这也从侧面印证了页级整页图那条路是对的。

```sh
pdfsys dataset --from-mineru ./out --to ./dataset/v2 --images pages --pdf-dir ./data/pdfs
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/chinaxiv_0.parquet
```

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

这两条填上不会破坏任何现有读端，所以两种方言都修。

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

用 `--dialect v2` 打开。每条都有代价，所以和 `legacy` 分开。

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

## 5 · 还没解决的语义错位

| | 说明 |
|---|---|
| `图片` 列装 PDF 字节 | `chinaxiv_to_pdf_blocks` 里 `图片=pdf_data`。导出器不产 `块类型="pdf"` 的块，所以没踩到；但历史数据里这一列混着两种东西 |
| 一份 PDF 多个 backend | 同一份 PDF 被 pipeline 和 vlm 各跑一遍会产生两套同 `doc_id` 的页行，导出后就是两套同 `实体ID` 的块。发布前需要按 `(doc_id, extractor)` 择优去重 |
| `时间` 是处理时间 | 不是文档时间。两边一致，但值得在数据卡里写明 |

---

## 6 · 实现

| 位置 | 内容 |
|---|---|
| `pdfsys_cli/mnbvc_export.py` | 两种方言的 schema + 行映射 + 导出器 |
| `pdfsys mnbvc-export` | CLI 子命令 |
| `tests/schema/test_mnbvc_export.py` | 21 个测试，含 HuggingFace 往返 |

```sh
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/out.parquet            # legacy
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/out.parquet --dialect v2
```

没有整页光栅的页仍然导出（文本还是有价值的），但 `图片` 为 null，
并且计数会打到 stderr —— 一次静默产出了一堆没有图的 "image-text-pair" 块，
应该看得见而不是默认没事。
