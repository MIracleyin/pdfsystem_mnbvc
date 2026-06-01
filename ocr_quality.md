是的，你们要做的事情可以定义成一个 OCR Release Gate / 数据集发布前质量门禁：上游 PDF/OCR/解析系统产出文本或 Markdown 后，质量系统给每个 page/document 打分，并输出 publish / review / reject，只有通过门禁的数据进入可发布数据集。

但这里有一个关键点：FinePDFs 的做法值得借鉴，但不能照搬成“一个 OCR score 阈值就发布”。 FinePDFs 自己的 OCR quality classifier 主要是 text-only，模型卡也明确说它没有原图、布局和完整上下文，因此无法判断布局保真、表格/公式丢失等问题；他们还写到，不推荐直接用某个 score threshold 做 curation/routing，因为在他们的 downstream eval 里没有可测的收益。 ￼

⸻

1. FinePDFs 这个 case 给我们的启发

FinePDFs 的整体 pipeline 是：先判断 PDF 是否需要 OCR；能直接抽文本的走 CPU/Docling 路径，扫描件或不可抽文本的走 GPU/RolmOCR 路径；之后再做 post-processing、语言识别、去重、过滤和打包。数据集卡里也写明，他们用 XGBoost 训练 OCR routing classifier，使用 7 个文档级特征和 120 个从 8 个随机页采样的页面级特征来决定走哪条提取路径。 ￼

FinePDFs 的 OCR quality prompt 很接近你们要做的事情：它把抽取质量分成 0–3 四档，0 是垃圾/OCR corruption，1 是明显格式问题，2 是轻微格式问题，3 是 clean extraction。它的目标不是判断内容有没有教育价值，而是判断“PDF extraction 是否干净、可读、结构是否基本保留”。 ￼

他们还把大模型 teacher 的标签蒸馏到小模型上：FinePDFs OCR-quality classifier 的模型卡说，它用于判断 OCR quality，基于 Qwen3-235B-A22B-Instruct-2507 生成的约 130 万条 FinePDFs 样本标注训练；模型卡也给出了 ModernBERT + regression/classification head 的实现思路。 ￼

但最重要的反例也来自 FinePDFs：在模型过滤 ablation 里，OCR score ≥ 2.5 会移除 64.2% 数据，但作者发现 OCR filter 和 no filtering 在下游 eval 上几乎不可区分；模型卡进一步写明，不建议直接用该模型的 score threshold 做 curation 或 routing。 ￼

结论：你们应该借鉴 FinePDFs 的“teacher → student → quality score”范式，但把目标函数改成“发布风险控制”，而不是复制它的阈值。

⸻

2. 你们真正要优化的不是 OCR score，而是 false publish rate

对“可发布数据集”来说，最核心指标不是平均分，而是：

有问题的数据被错误发布的比例有多低。

所以建议把系统目标写成：

目标：给上游 OCR/文档解析结果打分，自动决定是否进入 publishable dataset。
优先级：false accept 低于 false reject。
发布策略：宁可多进 review / reprocess，也不要把明显坏样本发布。

这和传统 benchmark 不一样。传统 OCR benchmark 会看 CER/WER、表格准确率、阅读顺序等；你们这里要看的是 “这个样本进入公开/训练数据集后会不会污染数据”。OCR-Quality 这个新数据集也正是朝这个方向做的：它包含 1,000 个 PDF 页面，转成 300 DPI PNG，并由人工按 4 级质量体系标注 Excellent/Good/Fair/Poor，用于 OCR quality assessment 和 OCR verification。 ￼

⸻

3. 推荐的发布门禁架构

建议做成四层，而不是一个单模型分数。

上游输出：
  PDF / page image / OCR text / Markdown / layout metadata
        ↓
Layer 1: deterministic hard checks
        ↓
Layer 2: text-only OCR quality scorer
        ↓
Layer 3: visual verifier / consensus verifier for high-risk samples
        ↓
Layer 4: calibrated document-level release decision
        ↓
publish / review / reprocess / reject

Layer 1：硬规则，先挡掉明显坏样本

这层不需要模型，速度快、可解释，适合作为第一道 gate。

建议 hard blockers：

empty_output
too_short_after_extraction
too_many_replacement_chars
high_garbage_char_ratio
high_symbol_or_punctuation_ratio
repetition_loop_detected
language_mismatch
encoding_artifacts
page_count_text_count_mismatch
extreme_output_length_ratio
ocr_timeout_or_failed_pages

FinePDFs 在 RolmOCR 路径里就遇到过重复生成、耗尽上下文、失败页面、空白/少文本页面 hallucination 等问题，并通过删除出错页面、VLM 检测空白/图形页、FTFY 和 boilerplate removal 等方式处理。 ￼

尤其要注意 fluent hallucination：FinePDFs 观察到 RolmOCR 会在空白页或主要是图像的页面上生成流畅但无关的文字；他们用触发规则筛出候选页，再用 Qwen2.5-VL-7B 判断页面是否空白/图形页并丢弃。 ￼

这说明：text-only OCR quality score 无法发现所有问题。 如果 OCR 输出很流畅但与页面不一致，文本模型可能给高分，所以必须有视觉核验或多模型一致性核验。

⸻

4. 质量分数建议：page score + document score + blocker flags

不要只输出一个 ocr_quality_score。建议输出下面这些字段：

{
  "doc_id": "xxx",
  "decision": "publish | review | reprocess | reject",
  "doc_quality_score": 0.91,
  "doc_quality_grade": "excellent | good | fair | poor",
  "page_quality_p05": 0.82,
  "page_quality_min": 0.44,
  "bad_page_ratio": 0.02,
  "ocr_text_score": 0.88,
  "visual_alignment_score": 0.93,
  "consensus_score": 0.90,
  "blockers": {
    "garbage_text": false,
    "hallucination_risk": false,
    "repetition_loop": false,
    "language_mismatch": false,
    "table_or_formula_broken": true
  },
  "reasons": [
    "minor table formatting issue on page 7"
  ],
  "scorer_version": "ocr-release-gate-v0.3",
  "threshold_profile": "eng_academic_pdf_v2"
}

Docling 的 confidence score 设计也能提供参考：它把 conversion quality 做成 0–1 score 和 poor/fair/good/excellent grade，并区分 layout_score、ocr_score、parse_score、table_score；同时有 page-level 和 document-level confidence，还特别强调 low_grade 用来暴露最差区域，而不是只看平均值。 ￼

发布门禁里，document score 不建议用平均分。 更稳妥的是：

doc_score = min(
  p05(page_scores),
  first_page_score,
  last_page_score,
  visual_alignment_score,
  consensus_score
)

原因很简单：一个 100 页文档里 95 页很好、5 页 hallucination，平均分仍然很高，但作为可发布数据集已经有污染风险。

⸻

5. 打分等级建议

可以沿用 FinePDFs 的 0–3 rubric，但把它映射成发布动作。

OCR 质量等级	含义	发布动作
3	clean extraction；无明显 OCR 垃圾、格式基本保留、可读性好	publish
2	minor formatting problems；轻微空格、换行、局部结构问题	publish 或抽样 review
1	clear formatting issues；表格/公式/列表/段落明显损坏	review / reprocess
0	garbage text；乱码、随机符号、严重 corruption、不可读片段	reject

FinePDFs prompt 对 0/1/2/3 的定义正好覆盖了这些情况：0 是垃圾文本，1 是明显格式问题，2 是轻微格式问题，3 是 clean extraction。 ￼

我建议你们再加一个 visual mismatch / hallucination blocker，因为 FinePDFs 的原始 OCR quality prompt 是看文本片段，不一定能判断“文本是否忠实于页面”。

⸻

6. 三种 scorer 组合使用

A. Text-only OCR quality scorer

这是最便宜的主力模型。输入是 OCR 后文本或 Markdown，输出 0–3 或 0–1。

适合检测：

乱码
编码问题
断词
异常空格
重复文本
表格变乱
公式破碎
列表损坏
OCR 垃圾片段

FinePDFs 的做法就是用 teacher model 标注，再蒸馏成小模型跑大规模数据；它的英文 OCR-quality classifier 也已经公开，可作为 baseline 或参考实现。 ￼

但它不适合单独做最终发布门禁，因为模型卡明确列出限制：它只看 OCR 后文本，不看原图、布局或文档上下文；对语言、脚本、表格、数学、混合代码、手写和非常规排版都可能不稳。 ￼

B. Visual verifier

这层输入是：

rendered page image + upstream extracted text/markdown

让 VLM 判断：

文字是否对应页面
是否凭空生成了不存在的内容
是否漏掉大块正文
表格/公式/标题层级是否严重错
页面是否其实为空白/图像页

不需要全量跑。建议只在这些样本上跑：

text score 处于灰区
hard rule 命中风险
页面文本极少但输出很长
图像/图表/表格/公式密度高
随机抽样审计
准备进入 publish 的高价值样本

olmOCR 的论文也说明，PDF 转 clean linearized text 的难点包括复杂格式、视觉布局、表格、公式、自然阅读顺序等，而 VLM 路线的成本和质量权衡很重要。 ￼

C. Multi-model consensus / self-verification

对于高风险或高价值数据，可以跑第二个 OCR/解析器，计算输出一致性。Consensus Entropy 的核心思想就是：正确 OCR 输出在多个 VLM/OCR 之间更容易收敛，错误输出更容易分散；它是 training-free、model-agnostic 的质量验证方法，并可用于 OCR verification、best-output selection 和 adaptive routing。 ￼

可以定义：

consensus_score = 1 - normalized_pairwise_disagreement

或更细：

text_agreement_score
reading_order_agreement_score
table_agreement_score
math_agreement_score

⸻

7. 阈值不要拍脑袋，要做 calibration set

最重要的落地动作是建立你们自己的 OCR release calibration set。

建议采样维度：

language / script
PDF 来源
文档类型：论文、教材、合同、报告、扫描件、表格密集文档
上游 OCR/解析器版本
页数区间
文本长度区间
score 分桶：高分、中分、低分、边界分
hard rule 命中类型

人工标注不要只标 “好/坏”，建议标：

page_quality: 0/1/2/3
doc_publishable: yes/no
issue_flags:
  garbage_text
  hallucination
  missing_content
  duplicated_content
  broken_table
  broken_math
  reading_order_error
  language_mismatch
  encoding_issue
  boilerplate_pollution
severity:
  minor / major / critical

阈值优化目标建议不是 F1，而是：

maximize release_coverage
subject to:
  false_publish_rate <= target
  precision_publish >= target
  critical_issue_escape_rate <= target

FinePDFs 在语言识别阈值上也采用了 calibration dataset，并用偏向 precision 的 F-beta 策略，同时设 minimum precision、minimum recall 和 minimum score cutoff；发布门禁可以借鉴这个思路，但目标应该换成“发布集 precision / bad sample escape rate”。 ￼

⸻

8. 发布决策规则示例

一个可落地的第一版规则：

def decide_release(doc):
    if doc.hard_blockers.any_critical:
        return "reject"
    if doc.visual_hallucination_risk >= 0.8:
        return "reject"
    if doc.doc_quality_score >= T_publish \
       and doc.page_quality_p05 >= T_page_p05 \
       and doc.bad_page_ratio <= T_bad_page_ratio \
       and not doc.blockers["language_mismatch"]:
        return "publish"
    if doc.doc_quality_score >= T_reprocess:
        return "reprocess"
    return "review"

更严格一点：

publish:
  no critical blocker
  p05(page_score) >= 0.80
  bad_page_ratio <= 1%
  visual_alignment_score >= 0.85
  language confidence >= threshold
  no hallucination flag
review:
  score 灰区
  表格/公式/版面风险高
  多模型不一致
  少量页面异常
reprocess:
  OCR 路由可能错误
  原图质量尚可但上游结果差
  scanned page 被 CPU parser 错误处理
  GPU OCR 出现 repetition/timeout
reject:
  垃圾文本
  大量乱码
  大量 hallucination
  大量缺页/空输出
  语言严重错误

⸻

9. 不建议照搬 FinePDFs 的两个点

第一，不要只看 top/bottom chunk，然后取 max。FinePDFs OCR-quality classifier 的示例代码会从长文本的 top/bottom 创建 chunk，并最终打印 max(scores)。这对“找高质量内容”可能合理，但对发布门禁有风险，因为坏页可能被好 chunk 掩盖。发布场景更适合 min、p05、或 “任一 critical page 触发 blocker”。 ￼

第二，不要把下游模型 eval 当作唯一质量指标。FinePDFs 发现 OCR quality filter 在他们的下游训练评估中没有明显收益，但这不等于 OCR quality 对发布无用；它说明“下游 benchmark 分数”和“数据发布洁净度”不是同一个目标。你们应该直接评估发布集里的坏样本逃逸率，而不是只看训练后模型分数。 ￼

⸻

10. 建议的最小可行实现

v0：不用训练，先做 teacher judge + 规则

输入：

page image
page text / markdown
document metadata
upstream parser name

输出：

0/1/2/3 score
issue flags
decision
reason

Prompt 结构：

你是 OCR/文档解析质量审核器。你的任务不是评价内容价值，而是评价抽取结果能否进入可发布数据集。
请根据原始页面图像和抽取文本判断：
1. 是否有乱码、随机字符、编码污染；
2. 是否有明显 hallucination，即文本中出现页面上不存在的内容；
3. 是否有大块漏识别；
4. 阅读顺序是否严重错误；
5. 表格、公式、列表、标题层级是否破坏到影响使用；
6. 是否存在重复生成、循环输出、页眉页脚污染。
返回 JSON：
{
  "score": 0|1|2|3,
  "publishable": true|false,
  "severity": "none|minor|major|critical",
  "issues": [...],
  "reason": "..."
}

v1：训练 text-only student

用 teacher/VLM 给样本打标签，再训练小模型：

input: OCR text / markdown
target: score 0-3 + issue flags
model: ModernBERT / XLM-R / mDeBERTa / small encoder

FinePDFs 已经证明这条路线可以规模化：teacher 标注，student 模型大规模推理。 ￼

v2：加入 visual verifier 和 consensus

只对灰区样本、高风险样本和抽样审计样本使用 VLM/多 OCR 共识：

cheap scorer 全量跑
expensive verifier 只跑 5–20% 风险样本
human audit 只看边界样本和发布抽样

这样成本可控，同时能挡住 text-only scorer 最容易漏掉的 hallucination 和视觉不一致问题。

⸻

11. 外部 benchmark 怎么用

建议把外部 benchmark 用作 sanity check，而不是最终阈值来源。

Benchmark / 资源	用途
OCR-Quality	评估你们的 quality scorer 是否能复现人工 4 级质量判断；它有 1,000 个 PDF 页面、300 DPI PNG 和人工质量分。 ￼
olmOCR-Bench	检查 PDF linearization 的细粒度能力，包括公式、表格、阅读顺序、页眉页脚、tiny text、老扫描件等；数据卡写明它有 1,402 个 PDF 页面和 7,010 个 binary pass/fail unit tests。 ￼
Docling confidence design	参考如何拆成 layout/ocr/parse/table component score，以及 page-level/document-level confidence。 ￼
Consensus Entropy	参考如何用多模型一致性做无监督 OCR quality verification 和 adaptive routing。 ￼

⸻

12. 我会这样定义你们的最终系统

一句话定义：

一个面向数据集发布的 OCR 质量门禁系统，对上游文档解析/OCR 输出进行 page-level 和 document-level 评分，优先控制 false publish，并输出可解释的 publish/review/reprocess/reject 决策。

核心原则：

1. 分数服务于发布决策，不服务于好看指标。
2. text-only scorer 只能作为主力低成本筛选器，不能单独担任最终审核。
3. 文档级聚合看低分位/最差页，不看平均分。
4. 阈值必须用你们自己的 calibration set 校准。
5. 发布集必须保留 score、flags、scorer version、threshold profile，方便追溯。
6. 高风险样本用 VLM visual verifier 或 multi-OCR consensus 兜底。

这样做出来的东西才是真正的 publishable dataset quality gate，而不是一个“看起来像 OCR quality score 的模型分数”。