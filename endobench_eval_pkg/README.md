# EndoBench 消融评测包（A100 服务器一键运行）

本目录是旧版 u50 controller 的 EndoBench 评测包，不是 P3 Qwen3.5 SFT/GRPO 训练包。
完整 test 题库共 6,832 题；默认使用 `stratified_1000.jsonl` 中的 1,002 题固定分层子集。
A/B/E/F 为 2×2 记忆消融；C/D/G 保留为历史机制实验。保持现有推理协议和模型不变。

本次补齐：离线题库、原始查询图、全量 66,291 张语料证据图、图片路径映射与校验清单。
旧 Release 只有模型/数据库/缓存，没有题库和真实证据图；候选缓存里的旧绝对路径不能在另一台机器直接读图。
现在读图时通过映射定位本机文件，保留原始候选字段及 ID，不修改数据库、候选顺序或检索分数。
修复了 `download_assets.sh` 解包到错误目录的问题。

## 前置条件

- 2× A100 80GB（单卡即可放下整套模型，每卡跑一个配置）
- Python 3.10+，GitHub CLI（`gh`）、tar、sha256sum；Python 环境需有 Pillow。
- 首次下载需访问 GitHub Release；公开基座模型仍从 HF 获取。数据就位后不再依赖 HF 下载题库。
- 当前仓库为公开仓库，不必更改可见性；若 `gh` 要求认证，可执行 `gh auth login`。
- 建议至少 100GB 可用磁盘（下载分片与解包文件都会保留；可选 SFT/RFT 权重需要额外空间）。
- Milvus SQLite 索引固定使用 `pymilvus==2.6.10`、`milvus-lite==2.5.1`、`setuptools<81`。

## 首次安装（含模型和数据下载）

```bash
git clone https://github.com/yds-sharks/agentic.git
cd agentic/endobench_eval_pkg
bash setup_env.sh
```

`setup_env.sh` 会建 venv、安装依赖、下载公开模型（Qwen3-VL-8B / bge-m3 /
mengzi-bert-base）、下载并重组 u50 checkpoint / Milvus / 权重模块 / 离线评测数据、
写入路径配置，最后仅做 CPU 完整性检查。默认不占 GPU；需要自动冒烟时设 `RUN_SMOKE=1`。

## 已部署环境：只更新缺失的评测数据

在仓库根目录执行 `git pull --ff-only`，进入 `endobench_eval_pkg/` 后：

```bash
source .venv/bin/activate
pip install 'pymilvus==2.6.10' 'milvus-lite==2.5.1' 'setuptools<81'
bash download_assets.sh --benchmark-only
python check_eval_data.py
```

该命令只下载题库及图片（约 5GB 以内，准确大小见 Release），不下载 controller、
不启动评测或训练。已下载的分片保留在 `assets/dl/`；重复执行先检查 SHA256，再解包。
若已有错误分片，校验会报出文件名并停止，不会拿损坏数据继续评测。

## 数据究竟在哪里

代码仓库：<https://github.com/yds-sharks/agentic>，分支 `main`。
大数据下载位置：<https://github.com/yds-sharks/agentic/releases/tag/eval-assets-v1>。
Git clone 只包含代码、README 和固定 QID 子集，**不包含大数据文件**。

以下路径均相对仓库内的 `endobench_eval_pkg/`：

| 内容 | 下载后实际位置 | 规模/作用 |
|---|---|---|
| 完整原始题库 | `assets/benchmark/endobench_test.jsonl` | 6,832 题，含问题、A–F 选项、answer、index、scene/task 等；不是结果文件 |
| 原始查询图片 | `assets/benchmark/query_images/eb_<index>.jpg` | 6,832 张，从 HF base64 解码，字节不重压缩 |
| 中文检索查询 | `assets/benchmark/endobench_translated_queries.jsonl` | 全量 6,832 题 |
| 首轮候选缓存 | `assets/benchmark/cand_cache_v2.jsonl` | 全量 6,832 题，保留既有排序和分数 |
| 语料证据图片 | `assets/corpus_images/` | 66,291 张，覆盖样本库的全部图片及候选缓存图片；支持后续轮重新检索 |
| 旧路径→本机路径 | `assets/benchmark/image_path_map.json` | 读图时映射，不必创建原机 `/mnt/...` 目录或修改 Milvus |
| 题库来源/数量/校验 | `assets/benchmark/manifest.json` | test、上游 revision、数量和 JSON 文件 SHA256 |
| 固定分层子集 | `stratified_1000.jsonl` | 1,002 个 QID；只用于选题，不能替代完整题库 |
| 冒烟子集 | `smoke10.jsonl` | 10 个 QID |
| 多模态样本库 | `assets/multimodal_samples.db` | 旧 Release 原有资产；数据库并不包含图片字节 |
| 向量索引 | `assets/multimodal_vector_indexes.db` | 由 `milvus.db.part_*` 合并，评测无需重建 |
| u50 controller | `models/ckpt_grpo_v4_u50/` | 由 `u50.tar.part_*` 合并解包 |
| 权重模块 checkpoint | `weight_module/checkpoints/` | 由 `weight_ckpts.tar` 解包 |

新增 Release 资产：`endobench_data_v1.tar.gz`（题库/查询图/缓存/映射）、
`evidence_images_v1.tar.gz.part00` 至 `part08`（9 片证据图）、`EVAL_DATA_SHA256SUMS.txt`（逐资产校验）。
数据和代码必须一起更新。`check_eval_data.py` 不加载模型，核验全部查询/证据图片可读、
缓存和翻译覆盖所有题、两个固定子集 QID 有效，并检查 JSON 的 SHA256。
`python check_eval_data.py --runtime` 另外检查模型和检索文件的位置。

## 启动夜间任务

```bash
nohup bash run_overnight.sh > logs/overnight.log 2>&1 &
```

- 夜间脚本按仓库现有 B/E、F/A 两波安排运行；每个并行 worker 使用独立 Milvus DB 副本。
- 耗时依赖硬件和缓存状态，不保证固定完成时间。
- 结果：`results/<CFG>/agentic_samples.jsonl` + `agentic_summary.json`，日志在 `logs/`

## 手动单跑 / 换 controller

```bash
bash run_config.sh SMOKE 0                                # GPU0，10题
bash run_config.sh A 0                                    # GPU0，固定1002题
QID_FILE=ALL OUT_SUFFIX=_full bash run_config.sh A 0        # 完整6832题
CTRL_CKPT=models/ckpt_ctrl_rft_v3 OUT_SUFFIX=_rft bash run_config.sh A 0
```

配置定义（对应论文消融表）：

| 配置 | 含义 | 开关 |
|---|---|---|
| A | Full Memory（deterministic M+ 与 M-） | `--mplus-det-topk 3` |
| B | No Memory | `--no-mplus --no-mminus` |
| C | Selection-only（禁 REWRITE） | `--selection-only` |
| D | Rewrite/Search-only（固定 Top-k） | `--rewrite-only` |
| E | M+ Only | `--no-mminus --mplus-det-topk 3` |
| F | M- Only | `--no-mplus` |
| G | Sequential rewrite→retrieval→score-rerank | `--sequential` |

`run_config.sh` 明确传入本地 `--benchmark-file`，不会重新从 HF 加载题库。
如只跑 closed-book baseline（无需 u50、检索编码器、Milvus）：

```bash
source .venv/bin/activate
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python eval_endobench.py \
  --mode baseline --benchmark-file assets/benchmark/endobench_test.jsonl \
  --gen-model models/Qwen3-VL-8B-Instruct --gen-device cuda:0 \
  --qid-file smoke10.jsonl --out-dir results/baseline_smoke
```

换模型需给 `--gen-model` 或 `CTRL_CKPT` 的实际本机路径；对比结果需保持相同 QID、
生成器、缓存和协议。`results/<配置>/...` 是评测输出，不是评测输入。不同设置请使用新的
`OUT_SUFFIX`，避免旧结果断点续跑造成混算。完整数据检查通过不等于端到端 GPU 评测已通过。

## 来源与使用边界

EndoBench 原始来源：<https://huggingface.co/datasets/Saint-lsy/EndoBench>，
本次导出使用 revision `96e79dab4e7f8b1f966019764a8ed5e85aaf56ce` 的 test split。
论文：<https://arxiv.org/abs/2505.23601>。本地转换只将 base64 图片拆为文件，
保留问题/答案/题号；不把翻译查询当作原题，不混入训练 samples。
原始数据和语料图片的权利归各上游来源所有；使用及再分发须遵循来源许可，
本项目代码的许可不自动授予这些图片额外权利。

## 目录结构

```
endobench_eval_pkg/
├── eval_endobench.py      # 主评测脚本
├── code/                  # controller/检索适配/打分器等模块
├── retrieval_mm/          # Milvus 多模态检索引擎
│   └── build_multimodal_milvus.py  # （可选）从 assets/multimodal_samples.db 重建索引；评测无需运行，DB 已由 Release 直接提供
├── weight_module/         # v2 权重模块（checkpoints 由 Release 提供）
├── assets/                # 下载后生成；大数据不在 Git 提交内
│   ├── benchmark/         # 完整题库、query_images、缓存、manifest、图片映射
│   ├── corpus_images/     # 全量语料证据图片
│   └── dl/                # 下载分片与 SHA256 清单（保留以支持复用）
├── models/                # u50 与公开模型（下载后生成）
├── stratified_1000.jsonl  # 固定评测子集（QID 与本机其他配置完全一致）
├── smoke10.jsonl          # 10 题冒烟子集
└── check_eval_data.py     # CPU 全量数据检查
```
