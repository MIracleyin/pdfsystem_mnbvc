# 跨机拆分运行手册

CPU 机和 GPU 机不共享磁盘时,整条流水线怎么跑。

面向**执行的人**:命令可以直接抄,每条报错都说明它是什么意思、该做什么。设计上的取舍在 [`../../README.md`](../../README.md) 和各模块 docstring 里,这里不重复。

> 这份文档里的每条命令、每个参数、每条报错都对着代码逐条核过。发现和代码对不上的,请当成 bug 报 —— 手册写错比没有手册更糟。

---

## 0. 三秒确认环境没问题

```bash
pdfsys smoke
```

生成 8 份极小 PDF(外加一个当诱饵的 `.txt`),用进程内桩服务跑完全部四阶段,**不需要 GPU、模型权重或网络**,约三秒。十一项全绿才继续。

确认真实部署:

```bash
pdfsys smoke --mineru-url http://10.253.253.2:8000 \
             --quality-url http://10.253.253.2:8765
```

同一套检查,换成真服务。判断一台 GPU 机接线对不对最快的办法 —— 另一个办法是起一个二十万份的批次然后发现问题。

`--workdir DIR` 保留产物供翻查,`-v` 展开子命令输出(默认收起,失败时自动展开)。

---

## 1. 机器

`~/.ssh/config` 里三台 GPU 机是按**别名**配的:

| 别名 | 主机 | 角色 |
|---|---|---|
| `mnbvcgpu1` | xsy-01 | 语料所在;4×4090 通常空闲 |
| `mnbvcgpu2` | xsy-02 | mineru-api `:8000`、quality `:8765` |
| `mnbvcgpu3` | hgy-01 | 长期高负载,别占 |

> **必须用别名。** `ssh root@10.253.253.2` 会**绕过全部配置**(包括挑 key 的 `IdentitiesOnly`),因为 `Host` 匹配命令行上打的名字,不匹配解析后的 IP。表现为 `Permission denied (publickey)`。用 `ssh mnbvcgpu2`。

隧道是 EasyTier,掉线不会自动重连,要在本机 `sudo` 重启 —— `ping 10.253.253.2` 不通就是这个。

---

## 2. 语料

`/hdd_common/xiaoxin/data/cmn_Hani`(xsy-01)—— 708 GB / 218,297 个文件。

**其中 217,997 份是真 PDF,但只有 199,992 份有 `.pdf` 后缀。** 另外 18,005 份(8.3%)靠文件头才认得出:约 8,400 份完全没后缀,约 9,700 份带着 `.ashx` / `.php` / `.aspx` / `.cgi` / `.jsp` —— 抓取时按 URL 最后一段存的(`download.ashx?id=123`)。

识别规则:**`.pdf` 后缀(任意大小写)直接采信,其余一律看开头五字节是不是 `%PDF-`。** 运行时会打印两者数量:

```
[pdfsys] discovered: 199992 by extension, 18005 extensionless (recognised by %PDF- header)
```

代价是每个非 `.pdf` 文件读 5 字节 —— 该语料冷缓存约 286 秒、热约 6 秒。相对抽取要跑几天,可忽略。

> **磁盘**:`/hdd_common` 3.5 T 已用 82%,**剩 663 GB**。
> - MinerU sidecar 约 250 GB(实测 middle.json 83–325 KB/页)
> - **`--images pages` 会另外存整页光栅,约 311 KiB/页** —— 21.8 万份按 15 页/份算是 **~1 TB,放不下**。不要图的话打包时明确写 `--images none`。

---

## 3. 四个阶段

### Phase 1 · CPU 机:抽出能抽的,把其余排队

```bash
pdfsys run \
  --pdf-dir /hdd_common/xiaoxin/data/cmn_Hani \
  --out-dir ./p1 \
  --stages router,extract \
  --extract-backends mupdf \
  --markdown-dir markdown \
  --ocr-threshold 0.05 \
  --resume
```

`--extract-backends mupdf` 说「这台机器只跑 mupdf」。路由到 OCR 的文档记成 `skip_reason=lane-filter`,带着路径 —— 那就是给 GPU 机的清单。**不加这个参数,OCR 文档会真的去调 MinerU。**

`--resume` 追加到已有 `results.jsonl` 并跳过已完成的。没有它,重启会把整个文件截断 —— 而那个文件同时是另一台机器在等的工作清单。

**按 bucket 分片跑**(21.8 万份不要一把梭)。先拿到全量路径清单 —— `pdfsys` 没有导出清单的子命令,用一趟 router-only 跑生成:

```bash
# 一趟只跑 router,拿到全部 217,997 份的路径(含靠文件头认出来的那 18,005 份)
pdfsys run --pdf-dir /hdd_common/xiaoxin/data/cmn_Hani \
           --out-dir ./inv --stages router
jq -r .pdf_path ./inv/results.jsonl > all_paths.txt

# 按行数均分 256 份。这只是分片和续跑的单位,与 doc_id 无关。
split -n l/256 -d -a 3 all_paths.txt bucket-
```

然后 —— **注意 `--out-dir` 全程不变**:

```bash
for b in bucket-*; do
  pdfsys run --pdf-list $b \
    --out-dir ./p1 --stages router,extract --extract-backends mupdf \
    --markdown-dir markdown --ocr-threshold 0.05 --resume
done
```

> 这里**不需要** `--path-root`:`jq -r .pdf_path` 出来的是绝对路径,而绝对路径原样使用 —— 加了也是空转。`--path-root` 是给**相对**清单用的,下面交给 GPU 机的 `gpu_lane.txt` 才需要(那份被 `sed` 剥成了相对路径,好在挂载点不同的机器上复用)。

`--resume` 会追加并跳过已完成的,所以所有 bucket 汇进**同一份** `./p1/results.jsonl`,下面的交接直接可用。

> 要跨机器并行的话,每个 worker 得有自己的 `--out-dir`(否则两个进程同时追加一个文件),交接前先合:
> `cat ./p1-worker*/results.jsonl > ./p1/results.jsonl`,并且 markdown 分散在各 `--out-dir` 下,Phase 3 要**按 worker 分别打分**再合并。

### 交接

```bash
# 需要 OCR 的
jq -r 'select(.skip_reason == "lane-filter") | .pdf_path' ./p1/results.jsonl \
  | sed 's|^/hdd_common/xiaoxin/data/cmn_Hani/||' > gpu_lane.txt

# 这台机器自己抽完的(Phase 4 打包要用)
jq -r 'select(.extract_backend=="mupdf" and .skip_reason==null and .error_class==null)
       | .pdf_path' ./p1/results.jsonl > cpu_lane.txt

# 先搬 PDF,再搬清单本身
rsync -a --partial --files-from=gpu_lane.txt \
  /hdd_common/xiaoxin/data/cmn_Hani/ mnbvcgpu2:/mnt/lane/
scp gpu_lane.txt mnbvcgpu2:/mnt/lane.txt
```

过网的基本全是 PDF 本身,无法避免 —— 两个 parser 都是 multipart 传字节,没有传路径的模式。

### Phase 2 · GPU 机:跑 MinerU

```bash
MINERU_PIPELINE_URL=http://127.0.0.1:8000 NO_PROXY='*' no_proxy='*' \
pdfsys run \
  --pdf-list /mnt/lane.txt --path-root /mnt/lane \
  --out-dir ./p2 \
  --stages router,extract \
  --extract-backends pipeline \
  --parser-output-dir ./p2/mineru \
  --markdown-dir markdown \
  --ocr-threshold 0.05 \
  --resume
```

> **`MINERU_PIPELINE_URL` 是必须的。** `pdfsys run` **没有** `--mineru-url` 参数(那个只在 `pdfsys smoke` 上有),环境变量是指向外部 mineru-api 的唯一途径。不设的话 parser 会尝试在本机 spawn 一个 `mineru-api` 进程 —— PATH 里没有就报错,有的话会在错误的机器上加载模型。VLM 通道再加 `MINERU_VLM_URL`。

**没有 `layout` 阶段** —— MinerU 内部自己做版面,收到的只有 PDF 字节。在 CPU 机上跑版面等于白付,算出来也传不过去(`LayoutCache` 只写不读,`decide_from_cache` 没有调用者)。

**`--parser-output-dir` 决定产出留不留得住。** MinerU 的 middle.json / content_list.json / 裁剪图只有落在这里才存活 —— mineru-api 自己那份在容器里会被 GC。不给的话运行**不会中止**,只警告一行,markdown 还在,但 `pdfsys dataset --from-mineru` 什么都找不到。

要 VLM 通道的话版面就是必需的了(只有 stage-B 会说 `vlm`):

```bash
  --stages router,layout,extract --vlm --extract-backends pipeline,vlm
```

> **两边的 `--ocr-threshold` 必须一致。** Phase 2 会重跑 stage-A(要 `extract` 就会拉进 `router`),阈值不同会让文档在 GPU 机上重新判成 `mupdf`、被 lane filter 跳掉 —— 而 CPU 机已经把它交出去了,于是**两条道都没有它**。这会打印一条以 `N documents were routed to mupdf here but queued for lane` 开头的警告,正常应该没有。

### Phase 3 · 打分:两条道打到同一个模型

首次在一台机器上起打分服务前,**权重要先在本地缓存好** —— 服务默认走 HF 离线模式(`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` 用的是 `setdefault`,真要联网下载可以显式设 `HF_HUB_OFFLINE=0` 覆盖),没缓存又没覆盖就会在加载模型时退出:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download miracleyin/mnbvc-pdf-quality-scorer-modernbert

CUDA_VISIBLE_DEVICES=0 python -m pdfsys_bench._quality_server \
  --host 0.0.0.0 --port 8765 --device cuda --dtype bfloat16
```

两条道都指过来。**不需要搬 markdown** —— 只有文本过网,截到服务端本来也要截的 40000 字符:

```bash
# GPU 机本地
QUALITY_URL=http://127.0.0.1:8765 pdfsys score \
  --results ./p2/results.jsonl --markdown-dir ./p2/markdown \
  --out ./p2/results.scored.jsonl \
  --model miracleyin/mnbvc-pdf-quality-scorer-modernbert --resume

# CPU 机远程打过来
QUALITY_URL=http://10.253.253.2:8765 NO_PROXY='*' no_proxy='*' pdfsys score \
  --results ./p1/results.jsonl --markdown-dir ./p1/markdown \
  --out ./p1/results.scored.jsonl \
  --model miracleyin/mnbvc-pdf-quality-scorer-modernbert --resume
```

> **过网体积**:客户端用 `json.dumps` 默认的 `ensure_ascii`,汉字会escape 成 `\uXXXX`(6 字节/字)。**中文语料下每篇最多约 240 KB**,纯 ASCII 才接近 40 KB。13 万份 CPU 道按最坏情况估约 30 GB —— 仍远小于搬 PDF。

> **`--model` 只在设了 `QUALITY_URL` 时才核对 `/health`。** 没设的话 pdfsys 会在本机起一个打分子进程,这个参数不做校验 —— 输出里 `model: (local subprocess)` 就是这个情况。两条道打到不同模型上,就是一列里混了两个刻度,而数据里没有任何东西能说明这件事。

`--workers` 默认 4。服务端是 `ThreadingHTTPServer`,加并发确实会并行进请求,但只有一块 GPU 上的一个模型,收益到某处就平了 —— 值得实测,不要当成硬上限。

### Phase 4 · 打包:每条道打包自己的文档

```bash
pdfsys dataset --from-pdf-list cpu_lane.txt --images none \
  --to ./dataset/v2 --shard cpu-00 --meta ./p1/results.scored.jsonl

pdfsys dataset --from-mineru ./p2/mineru --images none \
  --to ./dataset/v2 --shard gpu-00 --meta ./p2/results.scored.jsonl

pdfsys dataset-validate --shard ./dataset/v2
pdfsys mnbvc-export --from-shard ./dataset/v2 --to ./mnbvc/out.parquet --dialect v2
```

> **`--images` 一定要显式写。** 不给的话:`--from-pdf-list` / `--from-pdf-dir` 默认 **`pages`**(把整条道渲染成 200 dpi 整页光栅,约 311 KiB/页,这块盘装不下),`--from-mineru` 默认 **`crops`**。
>
> 要图的话:mupdf 道用 `--images pages`;MinerU 道用 `--images pages --pdf-dir /mnt/lane`(整页光栅要从源 PDF 渲染,按 sha256 找回)。**用 `--images none` 时 `mnbvc-export` 的 `图片` 列会是空的** —— 那个格式的图就是整页光栅。

**CPU 那条道要按清单打包,不能扫目录。** 语料根目录里也躺着 GPU 道的文档,mupdf 会把扫描件抽成一堆空页 —— 那些页带的 doc_id 和 GPU shard 用的是同一个,而 `(doc_id, page_index)` 是主键。给了 `--meta` 会直接报错拦下。

两条道写进**同一个** `--to`,用不同 `--shard` 名。排序是**每个文件内部**的承诺,两条道的 doc_id 区间可以交错。

---

## 4. 参数速查

### `pdfsys run`

| 参数 | 作用 |
|---|---|
| `--stages S` | 逗号分隔。`extract`/`layout` 会自动带上 `router`,`quality`/`parquet` 会带上 `extract` |
| `--pdf-dir DIR` | 扫目录。`.pdf`(任意大小写)+ 文件头是 `%PDF-` 的任何文件 |
| `--pdf-list FILE` | 按清单跑,一行一个路径。顺序保留,缺失和重复都汇报 |
| `--path-root DIR` | 清单里相对路径的锚点。让一份清单在挂载点不同的机器上都能用 |
| `--extract-backends` | 这台机器跑哪些后端:`mupdf` / `pipeline` / `vlm`。不给=全跑 |
| `--resume` | 追加而非截断,跳过已完成的。摘要按整个文件重算 |
| `--limit N` | 只取前 N 份。在 `--resume` 过滤**之前**生效,所以每次都是同一批 |
| `--parser-output-dir DIR` | MinerU sidecar 落在 `DIR/<sha256>/`。不给就丢弃(仅警告,不中止) |
| `--no-parser-images` | 不要裁剪图。分片用 `--images pages/none` 时省约 90 KiB/页传输 |
| `--ocr-threshold F` | Stage-A 阈值。**两台机器必须一致** |
| `--markdown-dir DIR` | 落 `<sha256>.md`(相对 `--out-dir`)。Phase 3 打分要读它 |
| `--vlm` / `--vlm-engine` | 开 VLM 通道 / 选引擎。需要 `layout` 阶段 |

环境变量:`MINERU_PIPELINE_URL`、`MINERU_VLM_URL`、`QUALITY_URL` —— 指向外部服务的**唯一**途径,没有对应的命令行参数。

### `pdfsys score`

| 参数 | 作用 |
|---|---|
| `--results` / `--markdown-dir` / `--out` | 输入 jsonl、markdown 目录、输出 jsonl |
| `--model M` | 核对服务端 `/health`。**只在设了 `QUALITY_URL` 时生效** |
| `--workers N` | 并发请求数,默认 4 |
| `--max-chars N` | 每篇截断,默认 40000(服务端也是这个数) |
| `--resume` / `--rescore` / `--overwrite` | 续跑 / 重打已有分数 / 覆盖输出 |

### `pdfsys dataset`

| 参数 | 作用 |
|---|---|
| `--from-pdf-list` / `--path-root` | 按清单打包 mupdf 道 |
| `--from-pdf-dir` | 扫目录。**拆分场景下会撞主键** |
| `--from-mineru DIR` | 打包 MinerU 道,读 `--parser-output-dir` 那个目录 |
| `--to` / `--shard` | 数据集目录 / 分片名。两条道同目录、不同分片名 |
| `--meta FILE` | 打分后的 jsonl,按 sha256 join 质量列;同时用于跨道检查 |
| `--images crops\|pages\|none` | **默认:mupdf 道 `pages`,MinerU 道 `crops`。** 三者互斥 |
| `--pdf-dir DIR` | `--images pages` 在 MinerU 道上必需,按 sha256 找回源 PDF |
| `--allow-missing-crops` | 源目录没有 `images/` 时仍按 `crops` 打包 |
| `--allow-other-lanes` | 明知故犯地打包别的道的文档 |
| `--overwrite` | 替换同名分片(会删掉该分片的全部文件) |

---

## 5. 报错对照

每条信息都在代码里核过。

| 信息(片段) | 含义 | 处理 |
|---|---|---|
| `no PDFs to process from X` | 路径错,或清单指向这台机器没挂载的地方。清单里是**相对**路径时,`--path-root` 写错也报这个(绝对路径不受 `--path-root` 影响,写错了也不会触发) | 查 `--pdf-dir` / `--path-root`;紧邻的上一行 `warning: N/N listed paths do not exist, e.g. [...]` 会告诉你漏了多少、漏了哪些 |
| `no output_dir for pipeline` | 会跑 MinerU 但没地方放 sidecar。**仅警告,运行继续** | 加 `--parser-output-dir`,否则事后无法打包 |
| `sidecar directory X is not usable` | 路径不可写或是个文件。**开跑前**拦下 | 换路径。不拦的话每份文档抽完之后才失败,连 markdown 一起丢 |
| `N documents were routed to mupdf here but queued for lane ...` | 两台机器路由判断不一致 | 对齐 `--ocr-threshold`,这些文档现在**两条道都没有** |
| `carried rows but skipped nothing` | `--resume` 的路径和上一程不一致,正在重做已完成的工作(**退出码仍是 0**) | 立刻停,核对 `--pdf-dir` / `--path-root` 与上一程一致 |
| `N documents in results.jsonl were filtered out by an earlier lane` | 用 `--resume` 指到了另一条道的输出目录 | 每条道用自己的 `--out-dir` |
| `第 N 行损坏,但后面还有完好的记录` | results.jsonl 中间坏了,不是写到一半 | 人工检查。自动截断会删掉真做过的工作 |
| `could not ask X what it is serving` | 打分服务不可达,或还在加载模型。**EasyTier 掉线也是这个** | `ping 10.253.253.2`;重启隧道;等 `/health` 返回 model |
| `X is serving Y, not the expected Z` | 打分服务和 `--model` 不符 | 别把两个刻度混进一列 |
| `not one row was scored` | markdown 目录不对,或那一程没给 `--markdown-dir` | 查 `--markdown-dir` 下有没有 `<sha256>.md` |
| `N documents failed against M scored` | 打分服务不健康,产出那列大部分是空的 | 别拿去打包,先查服务 |
| `X 已存在。换一个 --shard 名字` | 分片重名 | 换名,或 `--overwrite` |
| `documents are already in another lane's shard` | 在打包别的道的文档 | 用 `--from-pdf-list` 按道打包 |
| `--images crops but no images/ directory` | 那次运行用了 `--no-parser-images` | 改用 `--images none`,或 `--images pages` **并加 `--pdf-dir`** |

---

## 6. 已知没做的

- **版面结果无法跨机复用。** `LayoutCache` 只写不读,`decide_from_cache` 没有调用者。GPU 机跑 VLM 通道时版面要重算,CPU 机上算版面纯属浪费。
- **L1 parquet 与 `--resume` 不兼容。** Parquet 不能追加,续跑会摘掉 `parquet` 阶段并说明原因。要的话最后不带 `--resume` 重跑一次,或直接用 `pdfsys dataset`。
- **`pdfsys dataset` 单份文档失败不影响退出码。** 21.8 万份里挂几份是常态,做成阈值比一挂就停合理,但阈值定多少没定 —— 现在只有 `documents` / `empty` / `failed` 记在分片描述文件里。
- **`page_images` 表没有孤儿检查。** 引用不到的整页光栅和引用不到的裁剪图一样是泄漏,但只有后者会报。
- **打分请求没有关 `ensure_ascii`。** 中文过网体积是应有的约 6 倍;`quality.py` 里加一个参数就能降到约 120 KB/篇。
- **OCR 占比还没实测。** 容量估算里的 40% 是猜的,PRD 说约 10%,差四倍 —— 这个数决定要搬多少 TB 过网。跑 `--stages router` 抽样几千份就能定,不需要 GPU。

---

## 7. 出问题时先做这三件事

```bash
pdfsys smoke                                    # 代码本身还对吗
pdfsys smoke --mineru-url ... --quality-url ... # 服务还活着吗
pdfsys dataset-validate --shard ./dataset/v2    # 产出还合约吗
```

`results.summary.json` 里有账目:`num_pdfs` / `num_extracted` / `num_skipped` / `by_skip_reason` / `num_errors`,以及 `discovery` 块(扫到了什么、怎么扫到的、漏了什么)。

**跑了 `extract` 阶段时**,`num_extracted` / `num_skipped` / `num_errors` 互斥,加起来等于 `num_pdfs` —— 对不上就是 bug,请报。只跑 `--stages router` 时这三个都是 0(没有文档进入抽取),看路由分布要用 `by_backend`。

用了 `--resume` 时,`num_pdfs` 覆盖**所有 leg**(整份文件重算),而 `wall_seconds` 和 `leg_num_pdfs` 只是这一程的。
