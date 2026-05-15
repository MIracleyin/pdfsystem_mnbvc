# HF Bucket Pipeline — 使用说明

从 HuggingFace Storage Bucket 读取 PDF，经 pdfsys 流水线处理（XGBoost 路由器 + MuPDF 提取 + 可选质量评分），将结果保存到本地并可选择上传到目标 bucket。

## 环境准备

### 1. 初始化项目依赖

```bash
# 项目根目录执行
cd <project-root>
uv sync
```

> 首次运行会自动下载 XGBoost 路由器权重（~257 KB）。

### 2. 设置环境变量

```bash
# SSL 证书（某些环境需要）
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt

# Python 模块路径
export PYTHONPATH=<project-root>:<project-root>/packages/pdfsys-core/src:\
  <project-root>/packages/pdfsys-router/src:\
  <project-root>/packages/pdfsys-parser-mupdf/src:\
  <project-root>/packages/pdfsys-bench/src
```

### 3. 登录 HuggingFace（需要写权限时）

```bash
huggingface-cli login --token YOUR_TOKEN
```

Token 可以从 https://huggingface.co/settings/tokens 获取，需具备 bucket 的 write 权限。

## 基本用法

```bash
python demo/hf_bucket_pipeline.py --bucket <源-bucket-id>
```

默认配置：
- 源 bucket：`roger1024/raw_doc`
- 输出目录：`./hf_bucket_results`
- 处理全部 PDF
- 不使用质量评分器
- OCR 阈值：0.5

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--bucket` | str | `roger1024/raw_doc` | HuggingFace 源 bucket 标识 |
| `--out` | str | `./hf_bucket_results` | 本地输出目录 |
| `--max-files` | int | 全部处理 | 限制处理的 PDF 数量 |
| `--run-quality` | flag | off | 启用 ModernBERT 质量评分器（首次运行需下载 ~800 MB 模型） |
| `--ocr-threshold` | float | `0.5` | 路由器 OCR 概率阈值（0.0 ~ 1.0） |
| `--list-only` | flag | off | 仅列出 bucket 文件，不处理 |
| `--save-pdfs` | flag | off | 保存原始 PDF 文件 |
| `--use-cache` | flag | off | 使用已缓存 PDF 代替重新下载 |
| `--upload-bucket` | str | 不上传 | 处理后上传结果到目标 bucket（如 `roger1024/extracted_pdf`） |

## 使用示例

### 处理全部 PDF

```bash
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc
```

### 限制处理数量

```bash
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --max-files 50
```

### 自定义输出目录

```bash
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --out ./results
```

### 启用质量评分

```bash
# 首次会下载 ~800 MB 模型
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --run-quality
```

### 使用已缓存的 PDF

```bash
# 跳过下载，直接从缓存目录加载
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --use-cache
```

### 仅查看 bucket 内容

```bash
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc --list-only
```

### 处理并上传结果

```bash
# 需要先登录（见下方注意事项）
python demo/hf_bucket_pipeline.py --bucket roger1024/raw_doc \
  --upload-bucket roger1024/extracted_pdf
```

### 完整流程示例

```bash
python demo/hf_bucket_pipeline.py \
  --bucket roger1024/raw_doc \
  --out ./hf_bucket_results \
  --max-files 10 \
  --save-pdfs \
  --upload-bucket roger1024/extracted_pdf
```

## 输出目录结构

```
<output-dir>/
├── manifest.json              # 批量处理清单（每条记录对应一个 PDF）
├── pdfs/                      # 原始 PDF 缓存（仅当 download 或 --save-pdfs 时存在）
│   ├── 2503.04048v2.pdf
│   └── ...
└── extracted/
    └── <sha256[:16]>/         # 以 SHA256 前 16 位命名的目录
        ├── extracted.md       # 提取的 Markdown 内容
        └── metadata.json      # 路由决策 + 分段数据 + 质量评分
```

### manifest.json 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_path` | str | 原始文件名 |
| `sha256` | str | PDF 的 SHA256 哈希 |
| `extracted` | bool | 是否成功提取 |
| `backend` | str | 使用的后端（mupdf / pipeline / vlm / error） |
| `ocr_prob` | float | 路由器的 OCR 概率 |
| `num_pages` | int | PDF 总页数 |
| `num_segments` | int | 提取的文本段数量 |
| `markdown_chars` | int | Markdown 总字符数 |
| `quality_score` | float 或 null | 质量评分（启用 --run-quality 时） |
| `quality_num_tokens` | int 或 null | 质量评分 token 数 |
| `errors` | list[str] | 处理过程中的错误信息 |
| `wall_ms` | float | 处理耗时（毫秒） |

## 注意事项

1. **首次运行** 会自动下载 XGBoost 路由器权重（~257 KB）。如遇 SSL 错误，设置 `SSL_CERT_FILE` 环境变量。
2. **质量评分器**（`--run-quality`）首次使用会下载 ~800 MB 的 ModernBERT 模型。
3. **上传结果**（`--upload-bucket`）需要先通过 `huggingface-cli login` 登录并具备目标 bucket 的 write 权限。也可通过 `HF_TOKEN` 环境变量传入 token。
4. **SSL 证书问题** 如果在某些环境中遇到证书验证失败，设置 `SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt`（Linux）或系统证书路径。
5. **输出目录中的 manifest.json** 是追加写入的，多次运行同一输出目录会导致重复记录。

## 常见问题

**Q: 提示 "Failed to list bucket"**
A: 检查 HuggingFace 登录状态，运行 `huggingface-cli login`。

**Q: 大量 "MuPDF error" 输出**
A: 这些是 MuPDF 的底层警告（如 Screen annotations），不影响提取结果。已默认在模块中静默处理。

**Q: 上传失败 "unauthorized"**
A: 确保 token 对目标 bucket 有 write 权限，通过 `huggingface-cli login --token TOKEN` 登录。
