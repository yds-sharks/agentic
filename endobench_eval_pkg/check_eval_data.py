#!/usr/bin/env python3
"""仅 CPU 检查离线题库、全部图片、候选缓存和固定评测子集；不加载模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
from portable_data import load_local_benchmark, read_jsonl


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def collect_paths(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "image_path" and value:
                yield value
            elif isinstance(value, (dict, list)):
                yield from collect_paths(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from collect_paths(value)


def check(package, benchmark, runtime=False):
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if digest(benchmark / name) != expected:
            raise ValueError(f"数据 SHA256 不一致: {name}")
    rows = load_local_benchmark(benchmark / "endobench_test.jsonl")
    qids = {f"eb_{row['index']}" for row in rows}
    if len(rows) != 6832 or len(qids) != 6832:
        raise ValueError(f"题库数量异常: {len(rows)} / {len(qids)}")
    query_paths = [Path(row["query_image_path"]) for row in rows if row.get("query_image_path")]
    for row in rows:
        if not row.get("question") or not row.get(str(row.get("answer", ""))):
            raise ValueError(f"题目/答案选项缺失: {row['index']}")
    cache = list(read_jsonl(benchmark / "cand_cache_v2.jsonl"))
    if {row["qid"] for row in cache} != qids:
        raise ValueError("候选缓存未覆盖完整 6832 题")
    translations = list(read_jsonl(benchmark / "endobench_translated_queries.jsonl"))
    if {f"eb_{row['index']}" for row in translations if row.get("query_zh")} != qids:
        raise ValueError("中文检索查询缓存不完整")
    mapping = json.loads((benchmark / "image_path_map.json").read_text(encoding="utf-8"))
    corpus_paths = [package / relative for relative in mapping.values()]
    if len(mapping) != manifest["corpus_images"]:
        raise ValueError("证据图片数量与清单不一致")
    missing_refs = set(collect_paths(cache)) - mapping.keys()
    if missing_refs:
        raise ValueError(f"候选缓存中有 {len(missing_refs)} 个图片路径未映射")
    for path in query_paths + corpus_paths:
        if path.is_symlink():
            raise ValueError(f"图片包不应依赖原机器软链接: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"缺少图片: {path}")
        with Image.open(path) as image:
            image.verify()
    if len(query_paths) != manifest["query_images"]:
        raise ValueError("查询图片数量与清单不一致")
    subsets = {}
    for name, expected in (("stratified_1000.jsonl", 1002), ("smoke10.jsonl", 10)):
        selected = [row["qid"] for row in read_jsonl(HERE / name)]
        if len(selected) != expected or len(set(selected)) != expected or set(selected) - qids:
            raise ValueError(f"固定子集异常: {name}")
        subsets[name] = len(selected)
    if runtime:
        for relative in (
            "assets/multimodal_vector_indexes.db", "assets/multimodal_samples.db",
            "models/ckpt_grpo_v4_u50/config.json", "models/Qwen3-VL-8B-Instruct/config.json",
            "models/bge-m3/config.json", "models/mengzi-bert-base/config.json",
            "weight_module/checkpoints",
        ):
            if not (package / relative).exists():
                raise FileNotFoundError(f"缺少运行资产: {package / relative}")
    result = {
        "questions": len(rows), "query_images": len(query_paths),
        "candidate_cache_questions": len(cache), "translation_rows": len(translations),
        "corpus_images": len(corpus_paths), "subsets": subsets,
        "missing_images": 0, "benchmark_directory": str(benchmark),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root", type=Path, default=HERE)
    ap.add_argument("--benchmark-dir", type=Path)
    ap.add_argument("--runtime", action="store_true", help="额外检查模型与检索资产位置")
    args = ap.parse_args()
    package = args.package_root.resolve()
    benchmark = (args.benchmark_dir or package / "assets/benchmark").resolve()
    check(package, benchmark, args.runtime)


if __name__ == "__main__":
    main()
