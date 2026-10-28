# Agentic / EndoBench 评测资源

EndoBench 的部署、数据位置及离线评测说明见 [endobench_eval_pkg/README.md](endobench_eval_pkg/README.md)。

- 代码：本仓库 `main` 分支的 `endobench_eval_pkg/`。
- 完整评测数据及运行资产：[Release eval-assets-v1](https://github.com/yds-sharks/agentic/releases/tag/eval-assets-v1)。大数据不随 Git clone 下载。
- 完整中文题库：`endobench_eval_pkg/assets/benchmark/endobench_test_zh.jsonl`，共 6,832 题，含中文题干、选项及答案文本；正确选项字母不变。
- 英文对照：`endobench_eval_pkg/assets/benchmark/endobench_test.jsonl`；中文检索 query 缓存不能替代完整中文题库。
- 查询图：`endobench_eval_pkg/assets/benchmark/query_images/`。
- 全量语料证据图：`endobench_eval_pkg/assets/corpus_images/`，共 66,291 张。
- `stratified_1000.jsonl` 是 1,002 题的 QID 子集，不是完整题库；历史 `results/` 是评测输出，不是输入。

已部署环境补数据：

```bash
git pull --ff-only
cd endobench_eval_pkg
source .venv/bin/activate
bash download_assets.sh --benchmark-only
python check_eval_data.py
```

如果上一轮英文题库和图片已经下载完成，可将上面的下载命令换成 `bash download_assets.sh --zh-only`，仅补中文增量。
首次完整部署运行 `bash setup_env.sh`；默认仅检查数据，不启动 GPU 任务。
明确开始评测时运行 `bash run_config.sh SMOKE 0`（中文 10 题）或 `bash run_config.sh A 0`（中文 1,002 题）。
默认使用中文并保存到 `results/<CFG>_zh/`；`BENCHMARK_LANG=en` 可切回英文，避免混用旧结果。
中文是机器翻译派生版本，保留英文对照并经结构/答案对齐校验，未逐题人工医学审校。

本评测包使用历史 u50 controller，与另外发布的 P3 Qwen3.5 SFT/GRPO 训练包不同，不应混用配置或权重。
