# cmn_Hani 全量批跑

21.8 万份 / 708 GB 的具体执行流程 —— 这是 [`ops/split-run/`](../../ops/split-run/) 通用流程的**一个站点**,配置在 `ops/split-run/sites/cmn-hani.sh`。

换机器跑请看 [`ops/split-run/README.md`](../../ops/split-run/README.md):复制 `sites/example.sh`、改成自己的机器和路径、`export PDFSYS_SITE=<名字>`,其余步骤完全一样。

通用的拆分原理、参数含义、报错对照在 [`split-run.md`](split-run.md);这一份只讲**这个语料、这几台机器**怎么跑,以及实测出来的数字。

---

## 机器和角色

| 别名 | 主机 | 内网 | 角色 |
|---|---|---|---|
| `mnbvcgpu1` | xsy-01 | 10.0.49.101 | **语料在这台**;64 核;跑 CPU 通道 |
| `mnbvcgpu2` | xsy-02 | 10.0.49.102 | mineru-api `:8000`、quality `:8765`;跑 GPU 通道 |
| `mnbvcgpu3` | hgy-01 | — | 常年高负载,别占 |

**两台机在同一内网(bond0),实测 441 MB/s。** 传输走 `10.0.49.102`,**不要**走 `10.253.253.2` —— 那是 EasyTier 隧道,数据会绕经一台笔记本。

ssh 必须用别名(`ssh mnbvcgpu1`)。`ssh root@10.253.253.1` 会绕过全部配置,表现为 publickey 被拒。

> xsy-01 上另有两个 checkout:`/root/pdfsys`(旧拷贝,**没有** `--extract-backends` / `--pdf-list` / `--resume` / `score` / `smoke`,用它跑不了拆分)和 `/hdd_common/xiaoxin/0519/...`(同事在跑 demo)。两个都别动,`00-deploy.sh` 会新建 `/root/pdfsys-main`。

---

## 实测数字

全部来自这个语料,不是估算。

| | 值 | 怎么测的 |
|---|---|---|
| 真实 PDF | **217,997** / 218,297 个文件 | 全量 `take_inventory` |
| 其中靠文件头认出的 | **18,005(8.3%)** | 无后缀 + `.ashx`/`.php`/`.aspx` |
| 建清单耗时 | 7 秒热 / 286 秒冷 | 同上 |
| **OCR 占比(份数)** | **36.4%** | 全语料随机 800 份,95% 区间 33–40% |
| **OCR 占比(字节)** | **72.2%** | 同一批样本累加文件大小 |
| 扫描件平均大小 | **9.47 MiB** | 对比电子版 1.85 MiB |
| **要传输的量** | **约 511 GB** | 708 GB × 72.2% |
| CPU 通道速率 | 2.17 份/秒/进程 | 400 份实测 184.6 秒 |
| 32 进程整体 | **15 份/秒 → 约 4 小时** | 实跑 62,330 份后的实测 |
| 路由错误率 | 0.94% | 加密 / 损坏,属正常 |
| xsy-01 可用磁盘 | 658 GB(`/hdd_common`) | |
| xsy-02 可用磁盘 | 1.3 TB(`/hdd_common`) | 够放 511 GB + 约 250 GB sidecar |

### 三个和原计划不一样的结论

**1. OCR 占比是 36%,不是 PRD 说的 10%。** 差三倍半,而这个数直接决定要搬多少数据。

**2. 要传 511 GB,不是按份数推算的 258 GB。** 扫描件平均是电子版的 **5 倍大**,所以 OCR 道占 36% 的份数却占 **72% 的字节**。「只传需要 OCR 的」省下 28% 的流量,不是九成 —— 这个优化仍然值得做(内网 441 MB/s 下约 20 分钟到 1 小时),但它不是原先设想的那种数量级的胜利。

**3. 不需要单独跑一趟分类。** router+extract 一趟(184.6 秒)比只跑 router(222.5 秒)**还快** —— 缓存预热的缘故,而且 router 已经把 PDF 打开解析过了,mupdf 抽取叠在上面几乎零成本。所以第 2 步一趟做完分类和 CPU 抽取,OCR 清单随之产出。

---

## 流程

```bash
# ── xsy-01 ────────────────────────────────────────────────
ssh mnbvcgpu1
cd /root/pdfsys-main/ops/split-run
export PDFSYS_SITE=cmn-hani

./00-deploy.sh        # clone + uv sync + 拉权重 + smoke ×2      ~10 分钟
                      # ↑ 两台机都要跑,再往下走
./preflight.sh        # 两台机、两个服务、两份配置比对,只读       ~30 秒
./01-inventory.sh     # 217,997 条清单,切成 32 片                 ~7 秒
./02-cpu-lane.sh      # 32 路并行:分类 + mupdf 抽取               ~4 小时
./status.sh           # 随时看进度/速率/错误(可重复跑)
./03-handoff.sh       # 合并 results.jsonl,产出两条道的清单        ~1 分钟
./04-transfer.sh      # rsync 511 GB 走内网                       ~20-60 分钟

# ── xsy-02 ────────────────────────────────────────────────
ssh mnbvcgpu2
cd /root/pdfsys-main/ops/split-run
export PDFSYS_SITE=cmn-hani

./05-gpu-lane.sh      # MinerU 跑 7.9 万份扫描件                  ← 时间取决于卡数

# ── 打分:两条道打到同一个模型 ──────────────────────────────
ssh mnbvcgpu1 'cd /root/pdfsys-main/ops/split-run && PDFSYS_SITE=cmn-hani ./06-score.sh cpu'
ssh mnbvcgpu2 'cd /root/pdfsys-main/ops/split-run && PDFSYS_SITE=cmn-hani ./06-score.sh gpu'

# ── 打包:两条道写进同一个数据集,不同 shard ─────────────────
ssh mnbvcgpu1 'cd /root/pdfsys-main/ops/split-run && PDFSYS_SITE=cmn-hani ./07-package.sh cpu /hdd_common/dataset/v2'
ssh mnbvcgpu2 'cd /root/pdfsys-main/ops/split-run && PDFSYS_SITE=cmn-hani ./07-package.sh gpu /hdd_common/dataset/v2'
```

参数改 `ops/split-run/sites/cmn-hani.sh`,或用环境变量覆盖(`WORKERS=16 ./02-cpu-lane.sh`)。

---

## 几条必须守住的

**`OCR_THRESHOLD` 两台机必须一致。** 不一致时,一份文档会被 CPU 机交出去、又被 GPU 机判成不需要 OCR 而跳过 —— **两条道都没有它**。GPU 机上会打印一条以 `documents were routed to mupdf here but queued for lane` 开头的警告,正常应该没有。

**每个 worker 必须有自己的 `--out-dir`。** `results.jsonl` 是追加写的,两个进程写同一个目录会交错成谁都续不了的文件。`--markdown-dir` 可以共享(文件名是 `<sha256>.md`)。

**`--resume` 不能省。** 没有它,重启会**截断**那个文件 —— 而它同时是另一台机器在等的工作清单。

**`--parser-output-dir`(第 5 步)决定产出留不留得住。** mineru-api 自己那份在容器里会被回收。不给的话运行不会中止,只警告一行,markdown 还在,但 `pdfsys dataset --from-mineru` 什么都找不到。

**别在第 2 步还在跑的时候做第 4 步传输。** 两者读同一块 HDD,并行只会互相拖慢。

**第 3 步会拦住没跑完的交接。** 「没有 worker 在跑」不等于「跑完了」—— 可能是被杀掉或机器重启。两种情况产出的清单长得一模一样,而半截的 `gpu_lane.txt` 会让第 4 步传一个子集、第 5 步高高兴兴处理完:全程无报错,只是语料悄悄少了一块。所以第 3 步会拿合并行数和 `all_paths.txt` 比对,不足就拒绝;确实只想交接一部分时用 `ALLOW_PARTIAL=1`。

**`pkill -f pdfsys` 会杀掉你自己。** 你的 ssh 命令行里就含这个字符串。脚本里一律匹配 `--pdf-list $RUN/bucket-`,那是 worker 独有的。

---

## 当前状态(2026-09-05)


xsy-01 上已经部署好 `/root/pdfsys-main`(smoke 11/11 绿),清单和 32 个分片已生成。

**CPU 通道跑到 62,330 / 217,997(28.6%)后被主动停下**,产出留在 `/hdd_common/pdfsys-run/`(`p1/` 里 62,330 行结果,`markdown/` 里 39,137 个文件)。直接跑 `02-cpu-lane.sh` 会用 `--resume` 从这里接着跑,省约 1.2 小时 —— **但只在沿用同一套 32 分片时有效**(每个 worker 的续跑靠自己的 `--out-dir`)。想从头来就先 `rm -rf /hdd_common/pdfsys-run/{p1,markdown}` 再跑 `01-inventory.sh`。

那次部分运行的实测数据和抽样一致:OCR 占比 **35.3%**(21,838 / 61,814),路由错误 516 份(0.83%)。由它产出的半截清单已删除,以免被误当成完整清单使用。

**xsy-02 也已部署好** `/root/pdfsys-main`,两轮 smoke 都 11/11;`/hdd_common/pdfsys-lane/{pdfs,p2}` 已建好,是空的。`preflight.sh` 现在全绿。

> 部署 xsy-02 时踩到一个坑,值得记下来:那台机器的全局 git 配置把 GitHub 重写到了 `ghfast.top` 这个**已经失效**的国内镜像,于是 `git clone` 会对着一个没人输入过的 URL 报 SSL 超时。`00-deploy.sh` 现在会探测这种坏重写并用 `GIT_CONFIG_GLOBAL=/dev/null` 绕过去,**不去改别人机器的全局配置**。

---

## 还没定的两件事

**mineru-api 目前固定用 GPU 2(单卡)。** 7.9 万份扫描件单卡跑会很久,而 xsy-01 的 4 张 4090 常年空着。要不要把它们拉进来,是启动第 5 步前该拍的决定 —— 这也是整条流水线现在最大的时间不确定性。

**`GPU_WORKERS` 该设多少没测过。** 默认 4。观察 `curl $MINERU_URL/health` 的 `queued_tasks`:一直是 0 就说明喂不饱,可以往上加。
